mod synctex;

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context as _, Result};
use editor::Editor;
use gpui::{
    AnyElement, App, Bounds, ClipboardItem, Context, Entity, EventEmitter, FocusHandle, Focusable,
    ImageSource, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    ParentElement, Pixels, Point, Render, RenderImage, ScrollHandle, SharedString, Styled, Task,
    WeakEntity, Window, actions, div, img, point, px, red,
};
use image::{Frame, RgbaImage};
use language::Point as TextPoint;
use pdfium_render::prelude::*;
use project::{Project, ProjectEntryId, ProjectPath};
use smallvec::smallvec;
use ui::{
    Button, ButtonStyle, IconButton, IconSize, Label, ToggleButtonGroup, ToggleButtonSimple,
    prelude::*,
};
use workspace::{
    OpenOptions, Pane, Workspace,
    item::{Item, ItemEvent, ProjectItem as WorkspaceProjectItem, TabContentParams},
};

use crate::synctex::PdfPosition;

actions!(
    pdf_viewer,
    [
        /// Scrolls the PDF to the position of the cursor in the active source file.
        ForwardSearch
    ]
);

const PDFIUM_LIBRARY_DIRECTORY_ENV: &str = "PDFIUM_LIBRARY_DIRECTORY";
const BASE_PAGE_WIDTH: f32 = 900.0;
const PAGE_SPACING: f32 = 16.0;
const REVEAL_TOP_MARGIN: f32 = 120.0;
const ZOOM_FACTOR: f32 = 1.25;
const MIN_ZOOM: f32 = 0.5;
const MAX_ZOOM: f32 = 4.0;
const SIDEBAR_WIDTH: f32 = 220.0;
const SCROLLBAR_WIDTH: f32 = 8.0;
const SCROLLBAR_TRACK_WIDTH: f32 = 16.0;
const SCROLLBAR_MIN_THUMB_HEIGHT: f32 = 32.0;
const THUMBNAIL_WIDTH: f32 = 140.0;
const THUMBNAIL_RENDER_WIDTH: i32 = 280;
const PDF_VIEWER_KEY_CONTEXT: &str = "PdfViewer";

pub fn init(cx: &mut App) {
    workspace::register_project_item::<PdfView>(cx);
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(forward_search);
    })
    .detach();
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

struct CharacterBox {
    character: char,
    /// Bounds in PDF points with the origin at the top left of the page.
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

struct RenderedPage {
    image: Arc<RenderImage>,
    thumbnail: Arc<RenderImage>,
    /// The page size in PDF points, used to map between PDF and screen coordinates.
    width_points: f32,
    height_points: f32,
    characters: Vec<CharacterBox>,
}

struct OutlineEntry {
    title: SharedString,
    depth: usize,
    page_index: Option<usize>,
}

fn read_outline(document: &PdfDocument) -> Vec<OutlineEntry> {
    document
        .bookmarks()
        .iter()
        .map(|bookmark| OutlineEntry {
            title: bookmark
                .title()
                .filter(|title| !title.trim().is_empty())
                .unwrap_or_else(|| "Untitled".to_string())
                .into(),
            depth: std::iter::successors(bookmark.parent(), |parent| parent.parent()).count(),
            page_index: bookmark
                .destination()
                .and_then(|destination| destination.page_index().ok())
                .and_then(|index| usize::try_from(index).ok()),
        })
        .collect()
}

fn render_all_pages(
    bytes: &[u8],
    target_width: i32,
) -> Result<(Vec<RenderedPage>, Vec<OutlineEntry>)> {
    let pdfium = shared_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_slice(bytes, None)
        .context("failed to open the PDF document")?;
    let outline = read_outline(&document);
    let page_count = document.pages().len();
    let pages = (0..page_count)
        .map(|page_index| {
            let page = document
                .pages()
                .get(page_index.into())
                .context("the PDF has no page at this index")?;
            let height_points = page.height().value;
            let characters = page
                .text()
                .map(|text| {
                    text.chars()
                        .iter()
                        .filter_map(|character| {
                            let character_value = character.unicode_char()?;
                            let bounds = character.loose_bounds().ok()?;
                            Some(CharacterBox {
                                character: character_value,
                                left: bounds.left().value,
                                top: height_points - bounds.top().value,
                                right: bounds.right().value,
                                bottom: height_points - bounds.bottom().value,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(RenderedPage {
                image: render_page(&document, page_index.into(), target_width)?,
                thumbnail: render_page(&document, page_index.into(), THUMBNAIL_RENDER_WIDTH)?,
                width_points: page.width().value,
                height_points,
                characters,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((pages, outline))
}

pub struct PdfItem {
    project_path: ProjectPath,
    abs_path: Option<PathBuf>,
    entry_id: Option<ProjectEntryId>,
    bytes: Arc<Vec<u8>>,
}

impl project::ProjectItem for PdfItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<Result<Entity<Self>>>> {
        let worktree = project.read(cx).worktree_for_id(path.worktree_id, cx)?;
        // A file opened on its own gets a worktree rooted at the file, so its relative
        // path is empty and the extension has to come from the worktree.
        let worktree_abs_path = worktree.read(cx).abs_path();
        let extension = path
            .path
            .extension()
            .or_else(|| worktree_abs_path.extension()?.to_str());
        if !extension.is_some_and(|extension| extension.eq_ignore_ascii_case("pdf")) {
            return None;
        }
        let project_path = path.clone();
        let abs_path = project.read(cx).absolute_path(path, cx);
        Some(cx.spawn(async move |cx| -> Result<Entity<Self>> {
            let loaded = worktree
                .update(cx, |worktree, cx| {
                    worktree.load_binary_file(&project_path.path, cx)
                })
                .await?;
            Ok(cx.new(|_| PdfItem {
                project_path,
                abs_path,
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
    Rendered(Vec<RenderedPage>),
    Failed(SharedString),
}

fn scrollbar_thumb_height(viewport_height: f32, max_offset_y: f32) -> f32 {
    let content_height = viewport_height + max_offset_y;
    (viewport_height * viewport_height / content_height).clamp(
        SCROLLBAR_MIN_THUMB_HEIGHT.min(viewport_height),
        viewport_height,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SidebarMode {
    Hidden,
    Outline,
    Thumbnails,
}

pub struct PdfView {
    pdf_item: Entity<PdfItem>,
    focus_handle: FocusHandle,
    pages: PagesState,
    outline: Vec<OutlineEntry>,
    sidebar_mode: SidebarMode,
    zoom: f32,
    scroll_handle: ScrollHandle,
    pending_reveal: Option<PdfPosition>,
    highlight: Option<PdfPosition>,
    selection: Option<(Anchor, Anchor)>,
    drag_anchor: Option<Anchor>,
    scrollbar_dragging: bool,
    _render_task: Task<()>,
    _highlight_task: Task<()>,
}

type Anchor = (usize, usize);

impl PdfView {
    fn new(pdf_item: Entity<PdfItem>, cx: &mut Context<Self>) -> Self {
        let mut view = Self {
            pdf_item,
            focus_handle: cx.focus_handle(),
            pages: PagesState::Loading,
            outline: Vec::new(),
            sidebar_mode: SidebarMode::Hidden,
            zoom: 1.0,
            scroll_handle: ScrollHandle::new(),
            pending_reveal: None,
            highlight: None,
            selection: None,
            drag_anchor: None,
            scrollbar_dragging: false,
            _render_task: Task::ready(()),
            _highlight_task: Task::ready(()),
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
                    Ok((pages, outline)) => {
                        this.outline = outline;
                        PagesState::Rendered(pages)
                    }
                    Err(error) => PagesState::Failed(format!("{error:#}").into()),
                };
                this.apply_pending_reveal(cx);
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

    fn display_width(&self) -> f32 {
        BASE_PAGE_WIDTH * self.zoom
    }

    /// Scrolls to `position` once the pages are rendered, so a request made while
    /// the document is still loading is not lost.
    fn reveal(&mut self, position: PdfPosition, cx: &mut Context<Self>) {
        self.pending_reveal = Some(position);
        self.apply_pending_reveal(cx);
    }

    fn apply_pending_reveal(&mut self, cx: &mut Context<Self>) {
        let PagesState::Rendered(pages) = &self.pages else {
            return;
        };
        let Some(position) = self.pending_reveal else {
            return;
        };
        let display_width = self.display_width();
        let mut page_top = PAGE_SPACING;
        for (index, page) in pages.iter().enumerate() {
            let page_height = display_width * page.height_points / page.width_points;
            if index + 1 == position.page as usize {
                let y = page_top + position.y * display_width / page.width_points;
                self.scroll_handle
                    .set_offset(point(px(0.), px(-(y - REVEAL_TOP_MARGIN).max(0.))));
                self.pending_reveal = None;
                self.highlight = Some(position);
                self._highlight_task = cx.spawn(async move |this: WeakEntity<Self>, cx| {
                    cx.background_executor().timer(Duration::from_secs(2)).await;
                    this.update(cx, |this, cx| {
                        this.highlight = None;
                        cx.notify();
                    })
                    .ok();
                });
                cx.notify();
                return;
            }
            page_top += page_height + PAGE_SPACING;
        }
        self.pending_reveal = None;
    }

    fn current_page(&self) -> usize {
        let PagesState::Rendered(pages) = &self.pages else {
            return 0;
        };
        if pages.is_empty() {
            return 0;
        }
        let viewport_height = f32::from(self.scroll_handle.bounds().size.height);
        let visible_top = -f32::from(self.scroll_handle.offset().y) + viewport_height / 3.0;
        let display_width = self.display_width();
        let mut page_top = PAGE_SPACING;
        let mut current_page = 1;
        for (index, page) in pages.iter().enumerate() {
            if page_top > visible_top {
                break;
            }
            current_page = index + 1;
            page_top += display_width * page.height_points / page.width_points + PAGE_SPACING;
        }
        current_page
    }

    fn fit_width(&mut self, cx: &mut Context<Self>) {
        let viewport_width = f32::from(self.scroll_handle.bounds().size.width);
        if viewport_width <= 0.0 {
            return;
        }
        // The scroll area has p_4 padding on both sides.
        self.set_zoom((viewport_width - 32.0) / BASE_PAGE_WIDTH, cx);
    }

    fn fit_page(&mut self, cx: &mut Context<Self>) {
        let viewport_height = f32::from(self.scroll_handle.bounds().size.height);
        if viewport_height <= 0.0 {
            return;
        }
        let PagesState::Rendered(pages) = &self.pages else {
            return;
        };
        let Some(page) = pages.first() else {
            return;
        };
        let page_height = BASE_PAGE_WIDTH * page.height_points / page.width_points;
        self.set_zoom((viewport_height - 32.0) / page_height, cx);
    }

    /// Scrolls so the thumb's centre follows `pointer_y`, the pointer's window y coordinate.
    fn scroll_to_pointer(&mut self, pointer_y: Pixels, cx: &mut Context<Self>) {
        let viewport = self.scroll_handle.bounds();
        let viewport_height = f32::from(viewport.size.height);
        let max_offset_y = f32::from(self.scroll_handle.max_offset().y);
        if viewport_height <= 0.0 || max_offset_y <= 0.0 {
            return;
        }
        let thumb_height = scrollbar_thumb_height(viewport_height, max_offset_y);
        let thumb_travel = (viewport_height - thumb_height).max(f32::EPSILON);
        let thumb_top = f32::from(pointer_y - viewport.origin.y) - thumb_height / 2.0;
        let ratio = (thumb_top / thumb_travel).clamp(0.0, 1.0);
        self.scroll_handle
            .set_offset(point(px(0.), px(-ratio * max_offset_y)));
        cx.notify();
    }

    fn scroll_to_page(&mut self, page_index: usize, cx: &mut Context<Self>) {
        let PagesState::Rendered(pages) = &self.pages else {
            return;
        };
        let display_width = self.display_width();
        let mut page_top = PAGE_SPACING;
        for page in pages.iter().take(page_index) {
            page_top += display_width * page.height_points / page.width_points + PAGE_SPACING;
        }
        self.scroll_handle
            .set_offset(point(px(0.), px(-page_top.max(0.))));
        cx.notify();
    }

    fn toggle_sidebar(&mut self, mode: SidebarMode, cx: &mut Context<Self>) {
        self.sidebar_mode = if self.sidebar_mode == mode {
            SidebarMode::Hidden
        } else {
            mode
        };
        cx.notify();
    }

    fn render_sidebar(&self, pages: &[RenderedPage], cx: &mut Context<Self>) -> AnyElement {
        match self.sidebar_mode {
            SidebarMode::Hidden => div().into_any_element(),
            SidebarMode::Outline if self.outline.is_empty() => {
                Label::new("This PDF has no outline")
                    .color(Color::Muted)
                    .into_any_element()
            }
            SidebarMode::Outline => v_flex()
                .id("pdf-outline")
                .size_full()
                .overflow_y_scroll()
                .p_1()
                .children(self.outline.iter().enumerate().map(|(index, entry)| {
                    let page_index = entry.page_index;
                    div()
                        .id(("pdf-outline-entry", index))
                        .w_full()
                        .py_0p5()
                        .pl(px(8.0 + 12.0 * entry.depth as f32))
                        .pr_2()
                        .rounded_sm()
                        .hover(|style| style.bg(cx.theme().colors().ghost_element_hover))
                        .when_some(page_index, |this, page_index| {
                            this.cursor_pointer()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.scroll_to_page(page_index, cx);
                                }))
                        })
                        .child(Label::new(entry.title.clone()).single_line())
                }))
                .into_any_element(),
            SidebarMode::Thumbnails => {
                let current_page = self.current_page();
                v_flex()
                    .id("pdf-thumbnails")
                    .size_full()
                    .overflow_y_scroll()
                    .p_2()
                    .gap_2()
                    .items_center()
                    .children(pages.iter().enumerate().map(|(index, page)| {
                        let height = THUMBNAIL_WIDTH * page.height_points / page.width_points;
                        let border_color = if index + 1 == current_page {
                            cx.theme().colors().border_focused
                        } else {
                            cx.theme().colors().border
                        };
                        v_flex()
                            .id(("pdf-thumbnail", index))
                            .gap_1()
                            .items_center()
                            .cursor_pointer()
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.scroll_to_page(index, cx);
                            }))
                            .child(
                                div().border_1().border_color(border_color).child(
                                    img(ImageSource::Render(page.thumbnail.clone()))
                                        .w(px(THUMBNAIL_WIDTH))
                                        .h(px(height)),
                                ),
                            )
                            .child(Label::new((index + 1).to_string()).color(Color::Muted))
                    }))
                    .into_any_element()
            }
        }
    }

    /// Returns the page under `position` and its on-screen bounds. The scroll handle
    /// reports child bounds before scrolling, so the scroll offset is added here.
    fn page_at(
        &self,
        page_count: usize,
        position: Point<Pixels>,
    ) -> Option<(usize, Bounds<Pixels>)> {
        let offset = self.scroll_handle.offset();
        (0..page_count).find_map(|index| {
            let bounds = self.scroll_handle.bounds_for_item(index)?;
            let bounds = Bounds::new(bounds.origin + offset, bounds.size);
            bounds.contains(&position).then_some((index, bounds))
        })
    }

    fn hit_test(&self, position: Point<Pixels>) -> Option<Anchor> {
        let PagesState::Rendered(pages) = &self.pages else {
            return None;
        };
        let (page_index, bounds) = self.page_at(pages.len(), position)?;
        let page = pages.get(page_index)?;
        let scale = f32::from(bounds.size.width) / page.width_points;
        if scale <= 0.0 {
            return None;
        }
        let x = f32::from(position.x - bounds.origin.x) / scale;
        let y = f32::from(position.y - bounds.origin.y) / scale;
        let character_index = page
            .characters
            .iter()
            .position(|character| {
                x >= character.left
                    && x <= character.right
                    && y >= character.top
                    && y <= character.bottom
            })
            .or_else(|| {
                page.characters
                    .iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        let left_center_x = (left.left + left.right) / 2.0;
                        let left_center_y = (left.top + left.bottom) / 2.0;
                        let right_center_x = (right.left + right.right) / 2.0;
                        let right_center_y = (right.top + right.bottom) / 2.0;
                        let left_distance =
                            (left_center_x - x).powi(2) + (left_center_y - y).powi(2);
                        let right_distance =
                            (right_center_x - x).powi(2) + (right_center_y - y).powi(2);
                        left_distance.total_cmp(&right_distance)
                    })
                    .map(|(index, _)| index)
            })?;
        Some((page_index, character_index))
    }

    fn copy_selection(&mut self, cx: &mut Context<Self>) {
        let Some((first, second)) = self.selection else {
            return;
        };
        let PagesState::Rendered(pages) = &self.pages else {
            return;
        };
        let (start, end) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
        let mut text = String::new();
        let mut has_text = false;
        for page_index in start.0..=end.0 {
            let Some(page) = pages.get(page_index) else {
                continue;
            };
            let start_character = if page_index == start.0 {
                start.1.min(page.characters.len())
            } else {
                0
            };
            let end_character = if page_index == end.0 {
                end.1.saturating_add(1).min(page.characters.len())
            } else {
                page.characters.len()
            };
            if start_character >= end_character {
                continue;
            }
            if has_text {
                text.push('\n');
            }
            for character in &page.characters[start_character..end_character] {
                text.push(character.character);
            }
            has_text = true;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
    }

    fn inverse_search(
        &mut self,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let PagesState::Rendered(pages) = &self.pages else {
            log::warn!("pdf inverse search: the PDF is not rendered yet");
            return;
        };
        let Some((index, bounds)) = self.page_at(pages.len(), position) else {
            log::warn!("pdf inverse search: click {position:?} is outside every page");
            return;
        };
        let Some(page) = pages.get(index) else {
            log::warn!("pdf inverse search: no rendered page at index {index}");
            return;
        };
        let scale = f32::from(bounds.size.width) / page.width_points;
        let x = f32::from(position.x - bounds.origin.x) / scale;
        let y = f32::from(position.y - bounds.origin.y) / scale;
        let Some(pdf_path) = self.pdf_item.read(cx).abs_path.clone() else {
            log::warn!("pdf inverse search: the PDF has no absolute path");
            return;
        };
        let Some(workspace) = Workspace::for_window(window, cx) else {
            log::warn!("pdf inverse search: no workspace for this window");
            return;
        };
        log::info!(
            "pdf inverse search: page {} at ({x:.1}, {y:.1}) pt",
            index + 1
        );
        let pdf_position = PdfPosition {
            page: index as u32 + 1,
            x,
            y,
            width: 0.0,
        };
        cx.spawn_in(window, async move |_, cx| -> Result<()> {
            let source = cx
                .background_spawn(async move { synctex::inverse_search(&pdf_path, pdf_position) })
                .await?;
            let open_task = workspace.update_in(cx, |workspace, window, cx| {
                workspace.open_abs_path(source.path, OpenOptions::default(), window, cx)
            })?;
            let item = open_task.await?;
            let editor = cx.update(|_, cx| item.act_as::<Editor>(cx))?;
            if let Some(editor) = editor {
                editor.update_in(cx, |editor, window, cx| {
                    editor.go_to_singleton_buffer_point(
                        TextPoint::new(source.line.saturating_sub(1), 0),
                        window,
                        cx,
                    );
                })?;
            }
            Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let page_count = match &self.pages {
            PagesState::Rendered(pages) => pages.len(),
            PagesState::Loading | PagesState::Failed(_) => 0,
        };
        let current_page = self.current_page();
        h_flex()
            .w_full()
            .p_1()
            .gap_1()
            .justify_center()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(ToggleButtonGroup::single_row(
                "pdf-sidebar-mode",
                [
                    ToggleButtonSimple::new(
                        "Outline",
                        cx.listener(|this, _, _, cx| {
                            this.toggle_sidebar(SidebarMode::Outline, cx);
                        }),
                    )
                    .selected(self.sidebar_mode == SidebarMode::Outline),
                    ToggleButtonSimple::new(
                        "Thumbnails",
                        cx.listener(|this, _, _, cx| {
                            this.toggle_sidebar(SidebarMode::Thumbnails, cx);
                        }),
                    )
                    .selected(self.sidebar_mode == SidebarMode::Thumbnails),
                ],
            ))
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
            .child(
                Button::new("pdf-fit-width", "Fit width")
                    .style(ButtonStyle::Subtle)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.fit_width(cx);
                    })),
            )
            .child(
                Button::new("pdf-fit-page", "Fit page")
                    .style(ButtonStyle::Subtle)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.fit_page(cx);
                    })),
            )
            .child(Label::new(format!("{current_page} / {page_count}")))
    }
}

impl EventEmitter<()> for PdfView {}

impl Focusable for PdfView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for PdfView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selection = self.selection;
        let highlight = self.highlight;
        let selection_color = cx.theme().colors().text_accent.opacity(0.3);
        let content: AnyElement = match &self.pages {
            PagesState::Loading => Label::new("Rendering PDF…")
                .color(Color::Muted)
                .into_any_element(),
            PagesState::Failed(message) => Label::new(message.clone())
                .color(Color::Error)
                .into_any_element(),
            PagesState::Rendered(pages) => {
                let display_width = self.display_width();
                let viewport_height = f32::from(self.scroll_handle.bounds().size.height);
                let max_offset_y = f32::from(self.scroll_handle.max_offset().y);
                let scrollbar = (viewport_height > 0.0 && max_offset_y > 0.0).then(|| {
                    let thumb_height = scrollbar_thumb_height(viewport_height, max_offset_y);
                    let scrolled_ratio =
                        (-f32::from(self.scroll_handle.offset().y) / max_offset_y).clamp(0.0, 1.0);
                    let thumb_top = (viewport_height - thumb_height) * scrolled_ratio;
                    div()
                        .absolute()
                        .top(px(thumb_top))
                        .right_0()
                        .w(px(SCROLLBAR_WIDTH))
                        .h(px(thumb_height))
                        .rounded_full()
                        .bg(cx.theme().colors().scrollbar_thumb_background)
                });
                let sidebar = (self.sidebar_mode != SidebarMode::Hidden)
                    .then(|| self.render_sidebar(pages, cx));
                let pages_view = div()
                    .id("pdf-pages")
                    .relative()
                    .size_full()
                    .overflow_scroll()
                    .track_scroll(&self.scroll_handle)
                    .on_scroll_wheel(cx.listener(|_, _, _, cx| {
                        cx.notify();
                    }))
                    .p_4()
                    .gap_4()
                    .flex()
                    .flex_col()
                    .items_center()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseDownEvent, window, cx| {
                            if event.click_count == 2 {
                                this.inverse_search(event.position, window, cx);
                            } else if event.click_count == 1 {
                                window.focus(&this.focus_handle, cx);
                                this.selection = None;
                                this.drag_anchor = this.hit_test(event.position);
                                cx.notify();
                            }
                        }),
                    )
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                        if event.pressed_button != Some(MouseButton::Left) {
                            return;
                        }
                        let Some(anchor) = this.drag_anchor else {
                            return;
                        };
                        // Positions between pages hit nothing; keeping the last selection
                        // avoids it flickering off while the drag crosses a gap.
                        if let Some(current) = this.hit_test(event.position) {
                            this.selection = Some((anchor, current));
                            cx.notify();
                        }
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _: &MouseUpEvent, _, _cx| {
                            this.drag_anchor = None;
                        }),
                    )
                    .children(pages.iter().enumerate().map(|(page_index, page)| {
                        let height = display_width * page.height_points / page.width_points;
                        let scale = display_width / page.width_points;
                        let mut overlays: Vec<AnyElement> = Vec::new();
                        if let Some(position) =
                            highlight.filter(|position| position.page as usize == page_index + 1)
                        {
                            let left = position.x * scale;
                            let top = position.y * scale;
                            let width = position.width * scale;
                            overlays.push(
                                div()
                                    .absolute()
                                    .left(px(left))
                                    .top(px(top))
                                    .w(px(width))
                                    .h(px(2.0))
                                    .bg(red())
                                    .into_any_element(),
                            );
                        }
                        if let Some((first, second)) = selection {
                            let (start, end) = if first <= second {
                                (first, second)
                            } else {
                                (second, first)
                            };
                            if page_index >= start.0 && page_index <= end.0 {
                                for (character_index, character) in
                                    page.characters.iter().enumerate()
                                {
                                    let anchor = (page_index, character_index);
                                    if anchor < start || anchor > end {
                                        continue;
                                    }
                                    let left = character.left * scale;
                                    let top = character.top * scale;
                                    let width = (character.right - character.left).max(0.0) * scale;
                                    let height =
                                        (character.bottom - character.top).max(0.0) * scale;
                                    overlays.push(
                                        div()
                                            .absolute()
                                            .left(px(left))
                                            .top(px(top))
                                            .w(px(width))
                                            .h(px(height))
                                            .bg(selection_color)
                                            .into_any_element(),
                                    );
                                }
                            }
                        }
                        div()
                            .relative()
                            .w(px(display_width))
                            .h(px(height))
                            .flex_shrink_0()
                            .child(img(ImageSource::Render(page.image.clone())).size_full())
                            .children(overlays)
                            .into_any_element()
                    }));
                h_flex()
                    .size_full()
                    .children(sidebar.map(|sidebar| {
                        div()
                            .w(px(SIDEBAR_WIDTH))
                            .h_full()
                            .flex_shrink_0()
                            .border_r_1()
                            .border_color(cx.theme().colors().border)
                            .child(sidebar)
                    }))
                    .child(
                        div()
                            .relative()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                                if this.scrollbar_dragging
                                    && event.pressed_button == Some(MouseButton::Left)
                                {
                                    this.scroll_to_pointer(event.position.y, cx);
                                }
                            }))
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, _cx| {
                                    this.scrollbar_dragging = false;
                                }),
                            )
                            .child(pages_view)
                            .children(scrollbar)
                            .child(
                                div()
                                    .absolute()
                                    .top_0()
                                    .bottom_0()
                                    .right_0()
                                    .w(px(SCROLLBAR_TRACK_WIDTH))
                                    .cursor_pointer()
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, event: &MouseDownEvent, _, cx| {
                                            this.scrollbar_dragging = true;
                                            this.scroll_to_pointer(event.position.y, cx);
                                        }),
                                    ),
                            ),
                    )
                    .into_any_element()
            }
        };
        div()
            .key_context(PDF_VIEWER_KEY_CONTEXT)
            .track_focus(&self.focus_handle)
            .size_full()
            .flex()
            .flex_col()
            .bg(cx.theme().colors().editor_background)
            .on_action(cx.listener(|this, _: &editor::actions::Copy, _, cx| {
                this.copy_selection(cx);
            }))
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

/// Moves the PDF next to the active source file to the cursor position.
fn forward_search(
    workspace: &mut Workspace,
    _: &ForwardSearch,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let Some(active_item) = workspace.active_item(cx) else {
        return;
    };
    let Some(editor) = active_item.act_as::<Editor>(cx) else {
        return;
    };
    let Some(project_path) = active_item.project_path(cx) else {
        return;
    };
    let Some(source_path) = workspace
        .project()
        .read(cx)
        .absolute_path(&project_path, cx)
    else {
        return;
    };
    let cursor = editor.update(cx, |editor, cx| {
        let snapshot = editor.snapshot(window, cx);
        editor.selections.newest::<TextPoint>(&snapshot).head()
    });
    let pdf_path = source_path.with_extension("pdf");

    cx.spawn_in(window, async move |workspace, cx| -> Result<()> {
        let line = cursor.row + 1;
        let column = cursor.column;
        let position = cx
            .background_spawn({
                let source_path = source_path.clone();
                let pdf_path = pdf_path.clone();
                async move { synctex::forward_search(&source_path, line, column, &pdf_path) }
            })
            .await?;

        let open_task = workspace.update_in(cx, |workspace, window, cx| {
            workspace.open_abs_path(pdf_path, OpenOptions::default(), window, cx)
        })?;
        let item = open_task.await?;
        let view = cx
            .update(|_, cx| item.act_as::<PdfView>(cx))?
            .context("the opened file is not a PDF view")?;
        view.update(cx, |view, cx| view.reveal(position, cx));
        Ok(())
    })
    .detach_and_log_err(cx);
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
