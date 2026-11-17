use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};
use gpui::{
    AnyElement, App, Context, Entity, EventEmitter, FocusHandle, Focusable, ImageSource,
    IntoElement, ParentElement, Render, RenderImage, SharedString, Styled, Task, WeakEntity,
    Window, div, img, px,
};
use image::{Frame, RgbaImage};
use pdfium_render::prelude::*;
use project::{Project, ProjectEntryId, ProjectPath};
use smallvec::smallvec;
use ui::{Button, ButtonStyle, IconButton, IconSize, Label, prelude::*};
use workspace::{
    Pane,
    item::{Item, ItemEvent, ProjectItem as WorkspaceProjectItem, TabContentParams},
};

const PDFIUM_LIBRARY_DIRECTORY_ENV: &str = "PDFIUM_LIBRARY_DIRECTORY";
const BASE_PAGE_WIDTH: f32 = 900.0;
const ZOOM_FACTOR: f32 = 1.25;
const MIN_ZOOM: f32 = 0.5;
const MAX_ZOOM: f32 = 4.0;

pub fn init(cx: &mut App) {
    workspace::register_project_item::<PdfView>(cx);
}

/// Loads the pdfium shared library from `directory`, which must contain the
/// platform library file (for example `libpdfium.so`).
pub fn bind_pdfium(directory: &Path) -> Result<Pdfium> {
    let library_path = Pdfium::pdfium_platform_library_name_at_path(directory);
    let bindings =
        Pdfium::bind_to_library(library_path).context("failed to load the pdfium library")?;
    Ok(Pdfium::new(bindings))
}

/// Uses `PDFIUM_LIBRARY_DIRECTORY` when set, otherwise the system pdfium library.
fn bind_default_pdfium() -> Result<Pdfium> {
    match std::env::var_os(PDFIUM_LIBRARY_DIRECTORY_ENV) {
        Some(directory) => bind_pdfium(Path::new(&directory)),
        None => {
            let bindings = Pdfium::bind_to_system_library().with_context(|| {
                format!(
                    "pdfium library not found; set {PDFIUM_LIBRARY_DIRECTORY_ENV} to the directory that contains it"
                )
            })?;
            Ok(Pdfium::new(bindings))
        }
    }
}

/// pdfium can only be bound once per process, so every document shares this instance.
static PDFIUM: OnceLock<std::result::Result<Pdfium, String>> = OnceLock::new();

fn shared_pdfium() -> Result<&'static Pdfium> {
    PDFIUM
        .get_or_init(|| bind_default_pdfium().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map_err(|message| anyhow::anyhow!("{message}"))
}

/// Renders one page of an open document to an image `target_width` pixels wide,
/// keeping the page's aspect ratio.
pub fn render_page(
    document: &PdfDocument,
    page_index: i32,
    target_width: i32,
) -> Result<Arc<RenderImage>> {
    let page = document
        .pages()
        .get(page_index)
        .context("the PDF has no page at this index")?;
    let configuration = PdfRenderConfig::new().set_target_width(target_width);
    let rendered = page
        .render_with_config(&configuration)
        .context("failed to render the PDF page")?
        .as_image()
        .context("failed to convert the rendered page to pixels")?
        .into_rgba8();

    let width = rendered.width();
    let height = rendered.height();
    // gpui expects BGRA pixels, so the red and blue channels are swapped.
    let mut pixels = rendered.into_raw();
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    let buffer = RgbaImage::from_raw(width, height, pixels)
        .context("rendered page has an unexpected pixel buffer size")?;
    Ok(Arc::new(RenderImage::new(smallvec![Frame::new(buffer)])))
}

fn render_all_pages(bytes: &[u8], target_width: i32) -> Result<Vec<Arc<RenderImage>>> {
    let pdfium = shared_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_slice(bytes, None)
        .context("failed to open the PDF document")?;
    let page_count = document.pages().len();
    (0..page_count)
        .map(|page_index| render_page(&document, page_index.into(), target_width))
        .collect()
}

fn is_pdf_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
}

pub struct PdfItem {
    project_path: ProjectPath,
    entry_id: Option<ProjectEntryId>,
    bytes: Arc<Vec<u8>>,
}

impl project::ProjectItem for PdfItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<Result<Entity<Self>>>> {
        if !is_pdf_path(path.path.as_unix_str()) {
            return None;
        }
        let worktree = project.read(cx).worktree_for_id(path.worktree_id, cx)?;
        let project_path = path.clone();
        Some(cx.spawn(async move |cx| -> Result<Entity<Self>> {
            let loaded = worktree
                .update(cx, |worktree, cx| {
                    worktree.load_binary_file(&project_path.path, cx)
                })
                .await?;
            Ok(cx.new(|_| PdfItem {
                project_path,
                entry_id: loaded.file.entry_id,
                bytes: Arc::new(loaded.content),
            }))
        }))
    }

    fn entry_id(&self, _: &App) -> Option<ProjectEntryId> {
        self.entry_id
    }

    fn project_path(&self, _: &App) -> Option<ProjectPath> {
        Some(self.project_path.clone())
    }

    fn is_dirty(&self) -> bool {
        false
    }
}

enum PagesState {
    Loading,
    Rendered(Vec<Arc<RenderImage>>),
    Failed(SharedString),
}

pub struct PdfView {
    pdf_item: Entity<PdfItem>,
    focus_handle: FocusHandle,
    pages: PagesState,
    zoom: f32,
    _render_task: Task<()>,
}

impl PdfView {
    fn new(pdf_item: Entity<PdfItem>, cx: &mut Context<Self>) -> Self {
        let mut view = Self {
            pdf_item,
            focus_handle: cx.focus_handle(),
            pages: PagesState::Loading,
            zoom: 1.0,
            _render_task: Task::ready(()),
        };
        view.render_pages(cx);
        view
    }

    fn render_pages(&mut self, cx: &mut Context<Self>) {
        let bytes = self.pdf_item.read(cx).bytes.clone();
        let target_width = (BASE_PAGE_WIDTH * self.zoom).round() as i32;
        self._render_task = cx.spawn(async move |this: WeakEntity<Self>, cx| {
            let rendered = cx
                .background_spawn(async move { render_all_pages(&bytes, target_width) })
                .await;
            this.update(cx, |this, cx| {
                this.pages = match rendered {
                    Ok(pages) => PagesState::Rendered(pages),
                    Err(error) => PagesState::Failed(format!("{error:#}").into()),
                };
                cx.notify();
            })
            .ok();
        });
    }

    fn set_zoom(&mut self, zoom: f32, cx: &mut Context<Self>) {
        let zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        if (zoom - self.zoom).abs() < f32::EPSILON {
            return;
        }
        self.zoom = zoom;
        self.render_pages(cx);
        cx.notify();
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .w_full()
            .p_1()
            .gap_1()
            .justify_center()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                IconButton::new("pdf-zoom-out", IconName::Dash)
                    .icon_size(IconSize::Small)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.set_zoom(this.zoom / ZOOM_FACTOR, cx);
                    })),
            )
            .child(
                Button::new("pdf-zoom-reset", format!("{:.0}%", self.zoom * 100.))
                    .style(ButtonStyle::Subtle)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.set_zoom(1.0, cx);
                    })),
            )
            .child(
                IconButton::new("pdf-zoom-in", IconName::Plus)
                    .icon_size(IconSize::Small)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.set_zoom(this.zoom * ZOOM_FACTOR, cx);
                    })),
            )
    }
}

impl EventEmitter<()> for PdfView {}

impl Focusable for PdfView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for PdfView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let content: AnyElement = match &self.pages {
            PagesState::Loading => Label::new("Rendering PDF…")
                .color(Color::Muted)
                .into_any_element(),
            PagesState::Failed(message) => Label::new(message.clone())
                .color(Color::Error)
                .into_any_element(),
            PagesState::Rendered(pages) => {
                let display_width = BASE_PAGE_WIDTH * self.zoom;
                div()
                    .id("pdf-pages")
                    .size_full()
                    .overflow_scroll()
                    .p_4()
                    .gap_4()
                    .flex()
                    .flex_col()
                    .items_center()
                    .children(pages.iter().map(|page| {
                        let size = page.size(0);
                        let aspect_ratio = size.height.0 as f32 / size.width.0 as f32;
                        img(ImageSource::Render(page.clone()))
                            .w(px(display_width))
                            .h(px(display_width * aspect_ratio))
                            .flex_shrink_0()
                            .into_any_element()
                    }))
                    .into_any_element()
            }
        };
        div()
            .track_focus(&self.focus_handle)
            .size_full()
            .flex()
            .flex_col()
            .bg(cx.theme().colors().editor_background)
            .child(self.render_toolbar(cx))
            .child(div().flex_1().min_h_0().w_full().child(content))
    }
}

impl Item for PdfView {
    type Event = ();

    fn to_item_events(_: &(), _: &mut dyn FnMut(ItemEvent)) {}

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(gpui::EntityId, &dyn project::ProjectItem),
    ) {
        f(self.pdf_item.entity_id(), self.pdf_item.read(cx))
    }

    fn tab_content(&self, params: TabContentParams, _: &Window, cx: &App) -> AnyElement {
        Label::new(self.tab_content_text(params.detail.unwrap_or_default(), cx))
            .single_line()
            .color(params.text_color())
            .into_any_element()
    }

    fn tab_content_text(&self, _: usize, cx: &App) -> SharedString {
        self.pdf_item
            .read(cx)
            .project_path
            .path
            .file_name()
            .unwrap_or_default()
            .to_string()
            .into()
    }
}

impl WorkspaceProjectItem for PdfView {
    type Item = PdfItem;

    fn for_project_item(
        _: Entity<Project>,
        _: Option<&Pane>,
        item: Entity<Self::Item>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(item, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs `PDFIUM_LIBRARY_DIRECTORY` and `PDF_VIEWER_TEST_PDF` to point at a
    /// pdfium library directory and any PDF file; run with `--ignored`.
    #[test]
    #[ignore = "needs PDFIUM_LIBRARY_DIRECTORY and PDF_VIEWER_TEST_PDF"]
    fn test_render_first_page_of_pdf() {
        let library_directory = std::env::var("PDFIUM_LIBRARY_DIRECTORY")
            .expect("PDFIUM_LIBRARY_DIRECTORY must be set");
        let pdf_path =
            std::env::var("PDF_VIEWER_TEST_PDF").expect("PDF_VIEWER_TEST_PDF must be set");
        let pdfium = bind_pdfium(Path::new(&library_directory)).expect("bind pdfium");
        let bytes = std::fs::read(&pdf_path).expect("read test PDF");
        let document = pdfium
            .load_pdf_from_byte_slice(&bytes, None)
            .expect("open test PDF");

        let image = render_page(&document, 0, 600).expect("render first page");
        let size = image.size(0);
        assert_eq!(size.width.0, 600);
        assert!(size.height.0 > 0);
    }
}
