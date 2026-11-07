use std::sync::Arc;
use std::time::{Duration, Instant};

use editor::{
    DisplayPoint, Editor, EditorMode, MultiBuffer, SelectionEffects, SizingBehavior,
    actions::{
        Copy, CopyAndTrim, CopyPermalinkToLine, Cut, FindAllReferences,
        Format, FormatSelections, GoToDeclaration, GoToDefinition, GoToImplementation,
        GoToTypeDefinition, Paste, Rename, ToggleCodeActions,
    },
};
use futures::{StreamExt as _, future::Shared};
use gpui::{
    App, Context, Entity, EventEmitter, Focusable, Hsla, InteractiveElement, RetainAllImageCache,
    StatefulInteractiveElement, Task, Window, prelude::*,
};
use language::{Buffer, Language, LanguageRegistry};
use markdown::{Markdown, MarkdownElement, MarkdownFont, MarkdownOptions, MarkdownStyle};
use nbformat::v4::{CellId, CellMetadata, CellType, JupyterCellMetadata};
use project::{DisableAiSettings, Project};
use workspace::OpenInTerminal;
use runtimelib::{JupyterMessage, JupyterMessageContent};
use settings::{NotebookCellNumbersContent, Settings as _};
use ui::{
    CommonAnimationExt, ContextMenu, IconButtonShape, Tooltip, prelude::*, utils::WithRemSize,
};
use util::ResultExt;
use zed_actions::notebook::{InterruptKernel, ToggleCellOutput as NotebookToggleOutput};

use crate::{
    jupyter_settings::JupyterSettings,
    notebook::{CODE_BLOCK_INSET, GUTTER_WIDTH},
    outputs::{Output, plain, plain::TerminalOutput, user_error::ErrorView},
    repl_settings::ReplSettings,
};

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum CellPosition {
    First,
    Middle,
    Last,
}

pub enum CellControlType {
    RunCell,
    RerunCell,
    StopCell,
    ClearCell,
    CellOptions,
    CollapseCell,
    ExpandCell,
}

pub enum CellEvent {
    Run(CellId),
    FocusedIn(CellId),
}

pub enum MarkdownCellEvent {
    FinishedEditing,
    Run(CellId),
}

impl CellControlType {
    fn icon_name(&self) -> IconName {
        match self {
            CellControlType::RunCell => IconName::PlayFilled,
            CellControlType::RerunCell => IconName::ArrowCircle,
            CellControlType::StopCell => IconName::Stop,
            CellControlType::ClearCell => IconName::ListX,
            CellControlType::CellOptions => IconName::Ellipsis,
            CellControlType::CollapseCell => IconName::ChevronDown,
            CellControlType::ExpandCell => IconName::ChevronRight,
        }
    }
    fn id(&self) -> &'static str {
        match self {
            CellControlType::RunCell => "CellControlType::RunCell",
            CellControlType::RerunCell => "CellControlType::RerunCell",
            CellControlType::StopCell => "CellControlType::StopCell",
            CellControlType::ClearCell => "CellControlType::ClearCell",
            CellControlType::CellOptions => "CellControlType::CellOptions",
            CellControlType::CollapseCell => "CellControlType::CollapseCelln",
            CellControlType::ExpandCell => "CellControlType::ExpandCell",
        }
    }
}

pub struct CellControl {
    button: IconButton,
}

impl CellControl {
    fn new(id: impl Into<SharedString>, control_type: CellControlType) -> Self {
        let icon_name = control_type.icon_name();
        let id = id.into();
        let button = IconButton::new(id, icon_name)
            .icon_size(IconSize::Small)
            .shape(IconButtonShape::Square);
        Self { button }
    }
}

impl Clickable for CellControl {
    fn on_click(
        self,
        handler: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let button = self.button.on_click(handler);
        Self { button }
    }

    fn cursor_style(self, _cursor_style: gpui::CursorStyle) -> Self {
        self
    }
}

/// A notebook cell
#[derive(Clone)]
pub enum Cell {
    Code(Entity<CodeCell>),
    Markdown(Entity<MarkdownCell>),
    Raw(Entity<RawCell>),
}

pub(crate) enum MovementDirection {
    Start,
    End,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct GutterLabels {
    pub position: Option<String>,
    pub execution_count: Option<String>,
}

pub(crate) fn format_execution_count(
    execution_count: Option<i32>,
    is_executing: bool,
) -> String {
    if is_executing {
        "[*]".to_string()
    } else {
        match execution_count.filter(|&count| count > 0) {
            Some(count) => format!("[{count}]"),
            None => "[ ]".to_string(),
        }
    }
}

pub(crate) fn cell_gutter_labels(
    numbers: NotebookCellNumbersContent,
    cell_number: Option<usize>,
    execution_count: Option<i32>,
    is_executing: bool,
    is_code_cell: bool,
) -> GutterLabels {
    let show_position = matches!(
        numbers,
        NotebookCellNumbersContent::Both | NotebookCellNumbersContent::Position
    );
    let show_execution_count = matches!(
        numbers,
        NotebookCellNumbersContent::Both | NotebookCellNumbersContent::ExecutionCount
    );

    GutterLabels {
        position: if show_position {
            cell_number.map(|number| number.to_string())
        } else {
            None
        },
        execution_count: if show_execution_count && is_code_cell {
            Some(format_execution_count(execution_count, is_executing))
        } else {
            None
        },
    }
}

fn set_editor_source(editor: &Entity<Editor>, source: &str, cx: &mut App) {
    let multi_buffer = editor.read(cx).buffer().clone();
    let Some(buffer) = multi_buffer.read(cx).as_singleton() else {
        log::warn!("notebook: cell editor buffer is no longer a singleton");
        return;
    };
    buffer.update(cx, |buffer, cx| {
        let current = buffer.text();
        if current == source {
            return;
        }
        let (current_range, new_range) = differing_byte_ranges(&current, source);
        buffer.edit([(current_range, source[new_range].to_string())], None, cx);
    });
}

/// Finds the smallest byte ranges that differ between two strings, keeping the
/// ranges on UTF-8 character boundaries.
fn differing_byte_ranges(
    current: &str,
    new: &str,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let current_bytes = current.as_bytes();
    let new_bytes = new.as_bytes();
    let mut start = current_bytes
        .iter()
        .zip(new_bytes)
        .take_while(|(left, right)| left == right)
        .count();
    while start > 0 && (!current.is_char_boundary(start) || !new.is_char_boundary(start)) {
        start -= 1;
    }

    let max_suffix = current.len().min(new.len()) - start;
    let mut suffix = current_bytes
        .iter()
        .rev()
        .zip(new_bytes.iter().rev())
        .take(max_suffix)
        .take_while(|(left, right)| left == right)
        .count();
    while suffix > 0
        && (!current.is_char_boundary(current.len() - suffix)
            || !new.is_char_boundary(new.len() - suffix))
    {
        suffix -= 1;
    }

    (start..current.len() - suffix, start..new.len() - suffix)
}

fn json_values_equal<T, U>(left: &T, right: &U) -> bool
where
    T: serde::Serialize + ?Sized,
    U: serde::Serialize + ?Sized,
{
    match (serde_json::to_value(left), serde_json::to_value(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

pub(crate) fn convert_outputs(
    outputs: &[nbformat::v4::Output],
    window: &mut Window,
    cx: &mut App,
) -> Vec<Output> {
    outputs
        .iter()
        .map(|output| match output {
            nbformat::v4::Output::Stream { name, text } => Output::Stream {
                content: cx.new(|cx| TerminalOutput::from(&text.0, window, cx)),
                name: name.clone(),
                nbformat: Some(output.clone()),
            },
            nbformat::v4::Output::DisplayData(display_data) => {
                Output::new(&display_data.data, None, output.clone(), window, cx)
            }
            nbformat::v4::Output::ExecuteResult(execute_result) => {
                Output::new(&execute_result.data, None, output.clone(), window, cx)
            }
            nbformat::v4::Output::Error(error) => Output::ErrorOutput {
                content: ErrorView::new(
                    error.ename.clone(),
                    error.evalue.clone(),
                    &error.traceback.join("\n"),
                    window,
                    cx,
                ),
                nbformat: Some(output.clone()),
            },
        })
        .collect()
}

#[derive(Clone, Copy)]
struct CellContextMenuState {
    has_selection: bool,
    runs_code: bool,
    format_selections: bool,
    ai_enabled: bool,
    has_file_target: bool,
    has_git_repo: bool,
}

enum CellContextMenuEntry {
    Action {
        label: &'static str,
        action: Box<dyn gpui::Action>,
        disabled: bool,
    },
    Separator,
}

impl CellContextMenuEntry {
    fn action(label: &'static str, action: impl gpui::Action + 'static) -> Self {
        Self::Action {
            label,
            action: Box::new(action),
            disabled: false,
        }
    }

    fn disabled_action(
        label: &'static str,
        action: impl gpui::Action + 'static,
        disabled: bool,
    ) -> Self {
        Self::Action {
            label,
            action: Box::new(action),
            disabled,
        }
    }
}

fn cell_context_menu_entries(state: CellContextMenuState) -> Vec<CellContextMenuEntry> {
    let mut entries = Vec::new();

    entries.extend([
        CellContextMenuEntry::action("Go to Definition", GoToDefinition::default()),
        CellContextMenuEntry::action("Go to Declaration", GoToDeclaration::default()),
        CellContextMenuEntry::action("Go to Type Definition", GoToTypeDefinition::default()),
        CellContextMenuEntry::action("Go to Implementation", GoToImplementation::default()),
        CellContextMenuEntry::action("Find All References", FindAllReferences::default()),
        // The call hierarchy asks the project's language servers about the cell buffer, which
        // they only know as a virtual document, so these stay disabled.
        CellContextMenuEntry::disabled_action(
            "Show Incoming Calls",
            zed_actions::ShowIncomingCalls,
            true,
        ),
        CellContextMenuEntry::disabled_action(
            "Show Outgoing Calls",
            zed_actions::ShowOutgoingCalls,
            true,
        ),
        CellContextMenuEntry::Separator,
        CellContextMenuEntry::action("Rename Symbol", Rename),
        CellContextMenuEntry::action("Format Buffer", Format),
    ]);
    if state.format_selections {
        entries.push(CellContextMenuEntry::action(
            "Format Selections",
            FormatSelections,
        ));
    }
    entries.push(CellContextMenuEntry::action(
        "Show Code Actions",
        ToggleCodeActions {
            deployed_from: None,
            quick_launch: false,
        },
    ));
    if state.ai_enabled && state.has_selection {
        entries.push(CellContextMenuEntry::action(
            "Add to Agent Thread",
            zed_actions::agent::AddSelectionToThread,
        ));
    }

    entries.extend([
        CellContextMenuEntry::Separator,
        CellContextMenuEntry::disabled_action("Cut", Cut, !state.has_selection),
        CellContextMenuEntry::disabled_action("Copy", Copy, !state.has_selection),
        CellContextMenuEntry::action("Copy and Trim", CopyAndTrim),
        CellContextMenuEntry::action("Paste", Paste),
        CellContextMenuEntry::Separator,
    ]);

    // Cell buffers are anonymous, so the file- and git-backed entries have no
    // target and remain disabled, matching an editor opened without a file.
    entries.extend([
        CellContextMenuEntry::disabled_action(
            ui::utils::reveal_in_file_manager_label(false),
            editor::RevealInFileManager,
            !state.has_file_target,
        ),
        CellContextMenuEntry::disabled_action(
            "Open in Terminal",
            OpenInTerminal,
            !state.has_file_target,
        ),
        CellContextMenuEntry::disabled_action(
            "Copy Permalink to Line",
            CopyPermalinkToLine,
            !state.has_git_repo,
        ),
        CellContextMenuEntry::disabled_action(
            "View File History",
            git::FileHistory,
            !state.has_git_repo,
        ),
        CellContextMenuEntry::Separator,
        CellContextMenuEntry::action("Split Cell", zed_actions::notebook::SplitCell),
        CellContextMenuEntry::action("Merge Cell Above", zed_actions::notebook::MergeCellAbove),
        CellContextMenuEntry::action("Merge Cell Below", zed_actions::notebook::MergeCellBelow),
        CellContextMenuEntry::action(
            "Convert Cell to Markdown",
            zed_actions::notebook::ConvertCellToMarkdown,
        ),
        CellContextMenuEntry::action(
            "Convert Cell to Code",
            zed_actions::notebook::ConvertCellToCode,
        ),
    ]);

    if state.runs_code {
        entries.extend([
            CellContextMenuEntry::Separator,
            CellContextMenuEntry::action("Run Cell", zed_actions::notebook::Run),
            CellContextMenuEntry::action("Run All Above", zed_actions::notebook::RunAbove),
            CellContextMenuEntry::action("Run All Below", zed_actions::notebook::RunBelow),
            CellContextMenuEntry::action(
                "Clear Outputs",
                zed_actions::notebook::ClearCellOutputs,
            ),
            CellContextMenuEntry::action(
                "Toggle Outputs",
                zed_actions::notebook::ToggleCellOutput,
            ),
        ]);
    }

    entries
}

fn move_cursor_to_point_if_outside_selection(
    editor: &mut Editor,
    point: DisplayPoint,
    window: &mut Window,
    cx: &mut Context<Editor>,
) {
    let display_snapshot = editor.display_snapshot(cx);
    let is_inside_selection = editor
        .selections
        .all_display(&display_snapshot)
        .iter()
        .any(|selection| (selection.start..selection.end).contains(&point));
    if is_inside_selection {
        return;
    }

    editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
        selections.clear_disjoint();
        selections.select_display_ranges([point..point]);
    });
}

fn install_cell_context_menu(editor: &Entity<Editor>, runs_code: bool, cx: &mut App) {
    editor.update(cx, |editor, _cx| {
        editor.set_custom_context_menu(move |editor, point, window, cx| {
            move_cursor_to_point_if_outside_selection(editor, point, window, cx);

            let has_selection = editor.has_non_empty_selection(&editor.display_snapshot(cx));
            let focus_handle = editor.focus_handle(cx);
            let state = CellContextMenuState {
                has_selection,
                runs_code,
                format_selections: window.is_action_available(&FormatSelections, cx),
                ai_enabled: !DisableAiSettings::is_ai_disabled_for_buffer(
                    editor.buffer().read(cx).as_singleton().as_ref(),
                    cx,
                ),
                has_file_target: false,
                has_git_repo: false,
            };
            let entries = cell_context_menu_entries(state);

            Some(ContextMenu::build(window, cx, move |menu, _, _| {
                let mut menu = menu.context(focus_handle);
                for entry in entries {
                    menu = match entry {
                        CellContextMenuEntry::Action {
                            label,
                            action,
                            disabled,
                        } => {
                            if disabled {
                                menu.action_disabled_when(true, label, action)
                            } else {
                                menu.action(label, action)
                            }
                        }
                        CellContextMenuEntry::Separator => menu.separator(),
                    };
                }
                menu
            }))
        });
    });
}

impl Cell {
    pub fn id(&self, cx: &App) -> CellId {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).id().clone(),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).id().clone(),
            Cell::Raw(raw_cell) => raw_cell.read(cx).id().clone(),
        }
    }

    pub fn current_source(&self, cx: &App) -> String {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).current_source(cx),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).current_source(cx),
            Cell::Raw(raw_cell) => raw_cell.read(cx).source.clone(),
        }
    }

    pub fn to_nbformat_cell(&self, cx: &App) -> nbformat::v4::Cell {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).to_nbformat_cell(cx),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).to_nbformat_cell(cx),
            Cell::Raw(raw_cell) => raw_cell.read(cx).to_nbformat_cell(),
        }
    }

    pub fn is_dirty(&self, cx: &App) -> bool {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).is_dirty(cx),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).is_dirty(cx),
            Cell::Raw(_) => false,
        }
    }

    pub(crate) fn cell_type(&self) -> CellType {
        match self {
            Cell::Code(_) => CellType::Code,
            Cell::Markdown(_) => CellType::Markdown,
            Cell::Raw(_) => CellType::Raw,
        }
    }

    pub(crate) fn metadata<'a>(&'a self, cx: &'a App) -> &'a CellMetadata {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).metadata(),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).metadata(),
            Cell::Raw(raw_cell) => raw_cell.read(cx).metadata(),
        }
    }

    /// Updates an existing cell in place from a cell read back from disk.
    ///
    /// Returns `false` when the cell type changed, since the editor, output
    /// views, and rendering differ per type and the entity has to be replaced.
    pub(crate) fn update_from_notebook_cell(
        &self,
        notebook_cell: &nbformat::v4::Cell,
        window: &mut gpui::Window,
        cx: &mut App,
    ) -> bool {
        match (self, notebook_cell) {
            (
                Cell::Code(code_cell),
                nbformat::v4::Cell::Code {
                    metadata,
                    execution_count,
                    source,
                    outputs,
                    ..
                },
            ) => {
                code_cell.update(cx, |cell, cx| {
                    cell.apply_notebook_update(
                        metadata,
                        *execution_count,
                        &source.concat(),
                        outputs,
                        window,
                        cx,
                    );
                });
                true
            }
            (
                Cell::Markdown(markdown_cell),
                nbformat::v4::Cell::Markdown {
                    metadata, source, ..
                },
            ) => {
                markdown_cell.update(cx, |cell, cx| {
                    cell.apply_notebook_update(metadata, &source.concat(), cx);
                });
                true
            }
            (
                Cell::Raw(raw_cell),
                nbformat::v4::Cell::Raw {
                    metadata, source, ..
                },
            ) => {
                raw_cell.update(cx, |cell, cx| {
                    cell.apply_notebook_update(metadata, &source.concat(), cx);
                });
                true
            }
            _ => false,
        }
    }

    pub fn load(
        cell: &nbformat::v4::Cell,
        languages: &Arc<LanguageRegistry>,
        notebook_language: Shared<Task<Option<Arc<Language>>>>,
        project: &Entity<Project>,
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        match cell {
            nbformat::v4::Cell::Markdown {
                id,
                metadata,
                source,
                ..
            } => {
                let source = source.concat();

                let entity = cx.new(|cx| {
                    MarkdownCell::new(
                        id.clone(),
                        metadata.clone(),
                        source,
                        languages.clone(),
                        window,
                        cx,
                    )
                });

                Cell::Markdown(entity)
            }
            nbformat::v4::Cell::Code {
                id,
                metadata,
                execution_count,
                source,
                outputs,
            } => {
                let text = source.concat();
                let outputs = convert_outputs(outputs, window, cx);

                Cell::Code(cx.new(|cx| {
                    CodeCell::new(
                        CellSource::Existing {
                            execution_count: *execution_count,
                            outputs,
                        },
                        id.clone(),
                        metadata.clone(),
                        text,
                        languages.clone(),
                        notebook_language,
                        project,
                        window,
                        cx,
                    )
                }))
            }
            nbformat::v4::Cell::Raw {
                id,
                metadata,
                source,
            } => Cell::Raw(cx.new(|_| RawCell {
                id: id.clone(),
                metadata: metadata.clone(),
                source: source.concat(),
                selected: false,
                cell_position: None,
                cell_number: None,
            })),
        }
    }

    pub(crate) fn move_to(&self, direction: MovementDirection, window: &mut Window, cx: &mut App) {
        fn move_in_editor(
            editor: &Entity<Editor>,
            direction: MovementDirection,
            window: &mut Window,
            cx: &mut App,
        ) {
            editor.update(cx, |editor, cx| {
                match direction {
                    MovementDirection::Start => {
                        editor.move_to_beginning(&Default::default(), window, cx);
                    }
                    MovementDirection::End => {
                        editor.move_to_end(&Default::default(), window, cx);
                    }
                }
                editor.focus_handle(cx).focus(window, cx);
            })
        }

        match self {
            Cell::Code(cell) => {
                cell.update(cx, |cell, cx| {
                    move_in_editor(&cell.editor, direction, window, cx)
                });
            }
            Cell::Markdown(cell) => {
                cell.update(cx, |cell, cx| {
                    cell.set_editing(true);
                    move_in_editor(&cell.editor, direction, window, cx);

                    cx.notify();
                });
            }
            _ => {}
        }
    }

    pub(crate) fn editor<'a>(&'a self, cx: &'a App) -> Option<&'a Entity<Editor>> {
        match self {
            Cell::Code(cell) => Some(cell.read(cx).editor()),
            Cell::Markdown(cell) => Some(cell.read(cx).editor()),
            _ => None,
        }
    }

    pub(crate) fn buffer(&self, cx: &App) -> Option<Entity<Buffer>> {
        let editor = self.editor(cx)?;
        editor.read(cx).buffer().read(cx).as_singleton()
    }

    pub(crate) fn toggle_outputs_collapsed(&self, cx: &mut App) {
        if let Cell::Code(code_cell) = self {
            code_cell.update(cx, |code_cell, cx| {
                code_cell.toggle_outputs_collapsed(cx);
            });
        }
    }
}

pub trait RenderableCell: Render {
    const CELL_TYPE: CellType;

    fn id(&self) -> &CellId;
    fn cell_type(&self) -> CellType;
    fn metadata(&self) -> &CellMetadata;
    fn source(&self) -> &String;
    fn selected(&self) -> bool;
    fn set_selected(&mut self, selected: bool) -> &mut Self;
    fn cell_number(&self) -> Option<usize>;
    fn set_cell_number(&mut self, cell_number: usize) -> &mut Self;
    fn selected_bg_color(&self, _window: &mut Window, cx: &mut Context<Self>) -> Hsla {
        if self.selected() {
            let mut color = cx.theme().colors().element_hover;
            color.fade_out(0.5);
            color
        } else {
            // Not sure if this is correct, previous was TODO: this is wrong
            gpui::transparent_black()
        }
    }
    fn control(&self, _window: &mut Window, _cx: &mut Context<Self>) -> Option<CellControl> {
        None
    }

    fn cell_position_spacer(
        &self,
        is_first: bool,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        let cell_position = self.cell_position();

        if (cell_position == Some(&CellPosition::First) && is_first)
            || (cell_position == Some(&CellPosition::Last) && !is_first)
        {
            Some(div().flex().w_full().h(DynamicSpacing::Base12.px(cx)))
        } else {
            None
        }
    }

    fn gutter(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.selected();
        let position = cell_gutter_labels(
            JupyterSettings::get_global(cx).notebook_cell_numbers,
            self.cell_number(),
            None,
            false,
            false,
        )
        .position;

        div()
            .relative()
            .h_full()
            .w(px(GUTTER_WIDTH))
            .child(
                div()
                    .w(px(GUTTER_WIDTH))
                    .flex()
                    .flex_none()
                    .justify_center()
                    .h_full()
                    .child(
                        div()
                            .flex_none()
                            .w(px(1.))
                            .h_full()
                            .when(is_selected, |this| this.bg(cx.theme().colors().icon_accent))
                            .when(!is_selected, |this| this.bg(cx.theme().colors().border)),
                    ),
            )
            .when_some(self.control(window, cx), |this, control| {
                this.child(
                    div()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        .left_0()
                        .flex()
                        .flex_none()
                        .w(px(GUTTER_WIDTH))
                        .h(px(GUTTER_WIDTH + 12.0))
                        .items_center()
                        .justify_center()
                        .bg(cx.theme().colors().tab_bar_background)
                        .child(control.button),
                )
            })
            .when_some(position, |this, position| {
                this.child(
                    div()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        .left_0()
                        .flex()
                        .flex_none()
                        .w(px(GUTTER_WIDTH))
                        .justify_center()
                        .text_xs()
                        .text_color(cx.theme().colors().text_muted)
                        .child(position),
                )
            })
    }

    fn cell_position(&self) -> Option<&CellPosition>;
    fn set_cell_position(&mut self, position: CellPosition) -> &mut Self;
}

pub trait RunnableCell: RenderableCell {
    fn execution_count(&self) -> Option<i32>;
    fn set_execution_count(&mut self, count: i32) -> &mut Self;
    fn run(&mut self, window: &mut Window, cx: &mut Context<Self>) -> ();
}

/// Creates a cell's buffer along with a task that keeps its injections resolved.
/// This task is subscribed to `LanguageRegistry`'s changes, so that the cell is re-parsed
/// whenever loaded languages change. This is a simpler version of what
/// `LspStore::maintain_buffer_languages` does for other project buffers, but this way does not
/// require registering each cell as a project buffer.
fn new_cell_buffer(
    source: &str,
    languages: Arc<LanguageRegistry>,
    cx: &mut App,
) -> (Entity<Buffer>, Task<()>) {
    let buffer = cx.new(|cx| {
        let buffer = Buffer::local(source, cx);
        buffer.set_language_registry(languages.clone());
        buffer
    });

    let weak_buffer = buffer.downgrade();
    let mut registry_changes = languages.subscribe();
    let reparse_task = cx.spawn(async move |cx| {
        while registry_changes.next().await.is_some() {
            let updated = weak_buffer.update(cx, |buffer, cx| {
                if buffer.contains_unknown_injections() {
                    buffer.reparse(cx, false);
                }
            });
            if updated.is_err() {
                break;
            }
        }
    });

    (buffer, reparse_task)
}

pub struct MarkdownCell {
    id: CellId,
    metadata: CellMetadata,
    image_cache: Entity<RetainAllImageCache>,
    source: String,
    editor: Entity<Editor>,
    markdown: Entity<Markdown>,
    editing: bool,
    selected: bool,
    cell_position: Option<CellPosition>,
    cell_number: Option<usize>,
    _editor_subscription: gpui::Subscription,
    _reparse_task: Task<()>,
}

impl EventEmitter<MarkdownCellEvent> for MarkdownCell {}

impl MarkdownCell {
    pub fn new(
        id: CellId,
        metadata: CellMetadata,
        source: String,
        languages: Arc<LanguageRegistry>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (buffer, reparse_task) = new_cell_buffer(&source, languages.clone(), cx);
        let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer.clone(), cx));

        let markdown_language = languages.language_for_name("Markdown");
        cx.spawn_in(window, async move |_this, cx| {
            if let Some(markdown) = markdown_language.await.log_err() {
                buffer.update(cx, |buffer, cx| {
                    buffer.set_language(Some(markdown), cx);
                });
            }
        })
        .detach();

        let editor = cx.new(|cx| {
            let mut editor = Editor::new(
                EditorMode::Full {
                    scale_ui_elements_with_buffer_font_size: false,
                    show_active_line_background: false,
                    sizing_behavior: SizingBehavior::SizeByContent,
                },
                multi_buffer,
                None,
                window,
                cx,
            );

            editor.set_show_gutter(false, cx);
            editor.set_use_modal_editing(true);
            editor.disable_mouse_wheel_zoom();
            editor.disable_scrollbars_and_minimap(window, cx);
            editor
        });

        install_cell_context_menu(&editor, false, cx);

        let markdown = cx.new(|cx| {
            Markdown::new_with_options(
                source.clone().into(),
                None,
                None,
                MarkdownOptions {
                    render_math: true,
                    ..Default::default()
                },
                cx,
            )
        });

        let editor_subscription =
            cx.subscribe(&editor, move |this, _editor, event, cx| match event {
                editor::EditorEvent::Blurred => {
                    // Opening the context menu blurs the editor while the menu
                    // holds focus. Without this guard the cell would exit edit
                    // mode and the menu would dismiss on the same click.
                    if this.editing && !this.editor.read(cx).has_mouse_context_menu() {
                        this.editing = false;
                        cx.emit(MarkdownCellEvent::FinishedEditing);
                        cx.notify();
                    }
                }
                _ => {}
            });

        let start_editing = source.is_empty();
        Self {
            id,
            metadata,
            image_cache: RetainAllImageCache::new(cx),
            source,
            editor,
            markdown,
            editing: start_editing,
            selected: false,
            cell_position: None,
            cell_number: None,
            _editor_subscription: editor_subscription,
            _reparse_task: reparse_task,
        }
    }

    pub fn editor(&self) -> &Entity<Editor> {
        &self.editor
    }

    pub fn current_source(&self, cx: &App) -> String {
        let editor = self.editor.read(cx);
        let buffer = editor.buffer().read(cx);
        buffer
            .as_singleton()
            .map(|b| b.read(cx).text())
            .unwrap_or_default()
    }

    pub fn is_dirty(&self, cx: &App) -> bool {
        self.editor.read(cx).buffer().read(cx).is_dirty(cx)
    }

    pub fn to_nbformat_cell(&self, cx: &App) -> nbformat::v4::Cell {
        let source = self.current_source(cx);
        let source_lines: Vec<String> = source.lines().map(|l| format!("{}\n", l)).collect();

        nbformat::v4::Cell::Markdown {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            source: source_lines,
            attachments: None,
        }
    }

    pub fn is_editing(&self) -> bool {
        self.editing
    }

    pub fn set_editing(&mut self, editing: bool) {
        self.editing = editing;
    }

    fn apply_notebook_update(
        &mut self,
        metadata: &CellMetadata,
        source: &str,
        cx: &mut Context<Self>,
    ) {
        self.metadata = metadata.clone();
        if self.current_source(cx) != source {
            set_editor_source(&self.editor, source, cx);
            self.source = source.to_string();
            self.reparse_markdown(cx);
        }
        cx.notify();
    }

    pub fn reparse_markdown(&mut self, cx: &mut Context<Self>) {
        let editor = self.editor.read(cx);
        let buffer = editor.buffer().read(cx);
        let source = buffer
            .as_singleton()
            .map(|b| b.read(cx).text())
            .unwrap_or_default();

        self.source = source.clone();
        self.markdown.update(cx, |markdown, cx| {
            markdown.reset(source.into(), cx);
        });
    }

    /// Called when user presses Shift+Enter or Ctrl+Enter while editing.
    /// Finishes editing and signals to move to the next cell.
    pub fn run(&mut self, cx: &mut Context<Self>) {
        if self.editing {
            self.editing = false;
            cx.emit(MarkdownCellEvent::FinishedEditing);
            cx.emit(MarkdownCellEvent::Run(self.id.clone()));
            cx.notify();
        }
    }
}

impl RenderableCell for MarkdownCell {
    const CELL_TYPE: CellType = CellType::Markdown;

    fn id(&self) -> &CellId {
        &self.id
    }

    fn cell_type(&self) -> CellType {
        CellType::Markdown
    }

    fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    fn source(&self) -> &String {
        &self.source
    }

    fn selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    fn cell_number(&self) -> Option<usize> {
        self.cell_number
    }

    fn set_cell_number(&mut self, cell_number: usize) -> &mut Self {
        self.cell_number = Some(cell_number);
        self
    }

    fn control(&self, _window: &mut Window, _: &mut Context<Self>) -> Option<CellControl> {
        None
    }

    fn cell_position(&self) -> Option<&CellPosition> {
        self.cell_position.as_ref()
    }

    fn set_cell_position(&mut self, cell_position: CellPosition) -> &mut Self {
        self.cell_position = Some(cell_position);
        self
    }
}

impl Render for MarkdownCell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // If editing, show the editor
        if self.editing {
            return v_flex()
                .size_full()
                .children(self.cell_position_spacer(true, window, cx))
                .child(
                    h_flex()
                        .w_full()
                        .pr_6()
                        .rounded_xs()
                        .items_start()
                        .gap(DynamicSpacing::Base08.rems(cx))
                        .bg(self.selected_bg_color(window, cx))
                        .child(self.gutter(window, cx))
                        .child(
                            div()
                                .flex_1()
                                .p_3()
                                .bg(cx.theme().colors().editor_background)
                                .rounded_sm()
                                .child(self.editor.clone())
                                .on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|_this, _event, _window, _cx| {
                                        // Prevent the click from propagating
                                    }),
                                ),
                        ),
                )
                .children(self.cell_position_spacer(false, window, cx));
        }

        // Preview mode - show rendered markdown

        let style = MarkdownStyle::themed(MarkdownFont::Preview, window, cx);
        let markdown_font_size =
            theme_settings::ThemeSettings::get_global(cx).markdown_preview_font_size(cx);

        v_flex()
            .size_full()
            .children(self.cell_position_spacer(true, window, cx))
            .child(
                h_flex()
                    .w_full()
                    .pr_6()
                    .rounded_xs()
                    .items_start()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .bg(self.selected_bg_color(window, cx))
                    .child(self.gutter(window, cx))
                    .child(
                        v_flex()
                            .image_cache(self.image_cache.clone())
                            .id("markdown-content")
                            .size_full()
                            .flex_1()
                            .p_3()
                            .font_ui(cx)
                            .text_size(TextSize::Default.rems(cx))
                            .cursor_pointer()
                            .on_click(cx.listener(|this, _event, window, cx| {
                                this.editing = true;
                                window.focus(&this.editor.focus_handle(cx), cx);
                                cx.notify();
                            }))
                            .child(
                                WithRemSize::new(markdown_font_size)
                                    .child(MarkdownElement::new(self.markdown.clone(), style)),
                            ),
                    ),
            )
            .children(self.cell_position_spacer(false, window, cx))
    }
}

pub struct CodeCell {
    id: CellId,
    metadata: CellMetadata,
    execution_count: Option<i32>,
    source: String,
    editor: Entity<editor::Editor>,
    outputs: Vec<Output>,
    selected: bool,
    cell_position: Option<CellPosition>,
    cell_number: Option<usize>,
    _language_task: Task<()>,
    _reparse_task: Task<()>,
    execution_start_time: Option<Instant>,
    execution_duration: Option<Duration>,
    is_executing: bool,
    outputs_collapsed: bool,
}

impl EventEmitter<CellEvent> for CodeCell {}

pub(super) enum CellSource {
    /// Crate a new empty cell
    None,
    /// Backed by an existing notebook cell
    Existing {
        execution_count: Option<i32>,
        outputs: Vec<Output>,
    },
}

impl CellSource {
    fn into_outputs(self) -> (Option<i32>, Vec<Output>) {
        match self {
            CellSource::Existing {
                execution_count,
                outputs,
            } => (execution_count, outputs),
            CellSource::None => Default::default(),
        }
    }
}

impl CodeCell {
    pub(super) fn new(
        cell_source: CellSource,
        id: CellId,
        metadata: CellMetadata,
        source: String,
        languages: Arc<LanguageRegistry>,
        notebook_language: Shared<Task<Option<Arc<Language>>>>,
        project: &Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (buffer, reparse_task) = new_cell_buffer(&source, languages, cx);
        let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer.clone(), cx));

        let editor = cx.new(|cx| {
            let mut editor = Editor::new(
                EditorMode::Full {
                    scale_ui_elements_with_buffer_font_size: false,
                    show_active_line_background: false,
                    sizing_behavior: SizingBehavior::SizeByContent,
                },
                multi_buffer,
                Some(project.clone()),
                window,
                cx,
            );

            editor.disable_mouse_wheel_zoom();
            editor.disable_scrollbars_and_minimap(window, cx);
            editor.set_text(source.clone(), window, cx);
            editor.set_show_gutter(false, cx);
            editor.set_use_modal_editing(true);
            editor
        });

        install_cell_context_menu(&editor, true, cx);

        let language_task = cx.spawn_in(window, async move |_this, cx| {
            let language = notebook_language.await;
            buffer.update(cx, |buffer, cx| {
                buffer.set_language(language.clone(), cx);
            });
        });

        let (execution_count, outputs) = cell_source.into_outputs();
        let outputs_collapsed = metadata.collapsed.unwrap_or(false)
            || metadata
                .jupyter
                .as_ref()
                .and_then(|jupyter| jupyter.outputs_hidden)
                .unwrap_or(false);

        Self {
            id,
            metadata,
            execution_count,
            source,
            editor,
            outputs,
            selected: false,
            cell_position: None,
            cell_number: None,
            execution_start_time: None,
            execution_duration: None,
            is_executing: false,
            outputs_collapsed,
            _language_task: language_task,
            _reparse_task: reparse_task,
        }
    }

    pub fn set_language(&mut self, language: Option<Arc<Language>>, cx: &mut Context<Self>) {
        self.editor.update(cx, |editor, cx| {
            editor.buffer().update(cx, |buffer, cx| {
                if let Some(buffer) = buffer.as_singleton() {
                    buffer.update(cx, |buffer, cx| {
                        buffer.set_language(language, cx);
                    });
                }
            });
        });
    }

    fn apply_notebook_update(
        &mut self,
        metadata: &CellMetadata,
        execution_count: Option<i32>,
        source: &str,
        outputs: &[nbformat::v4::Output],
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        self.metadata = metadata.clone();
        self.execution_count = execution_count;

        if self.current_source(cx) != source {
            set_editor_source(&self.editor, source, cx);
            self.source = source.to_string();
        }

        if !json_values_equal(&self.outputs_to_nbformat(cx), &outputs) {
            self.outputs = convert_outputs(outputs, window, cx);
            self.execution_duration = None;
            self.is_executing = false;
        }

        self.outputs_collapsed = metadata.collapsed.unwrap_or(false)
            || metadata
                .jupyter
                .as_ref()
                .and_then(|jupyter| jupyter.outputs_hidden)
                .unwrap_or(false);
        cx.notify();
    }

    pub fn editor(&self) -> &Entity<editor::Editor> {
        &self.editor
    }

    pub fn current_source(&self, cx: &App) -> String {
        let editor = self.editor.read(cx);
        let buffer = editor.buffer().read(cx);
        buffer
            .as_singleton()
            .map(|b| b.read(cx).text())
            .unwrap_or_default()
    }

    pub fn is_dirty(&self, cx: &App) -> bool {
        self.editor.read(cx).buffer().read(cx).is_dirty(cx)
    }

    pub fn to_nbformat_cell(&self, cx: &App) -> nbformat::v4::Cell {
        let source = self.current_source(cx);
        let source_lines: Vec<String> = source.lines().map(|l| format!("{}\n", l)).collect();

        let outputs = self.outputs_to_nbformat(cx);

        nbformat::v4::Cell::Code {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            execution_count: self.execution_count,
            source: source_lines,
            outputs,
        }
    }

    fn outputs_to_nbformat(&self, cx: &App) -> Vec<nbformat::v4::Output> {
        self.outputs
            .iter()
            .filter_map(|output| output.to_nbformat(cx))
            .collect()
    }

    pub fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }

    pub fn output_summary(&self) -> Option<String> {
        if self.outputs.is_empty() {
            return None;
        }

        let mut kinds: Vec<&str> = Vec::new();
        for output in &self.outputs {
            let kind = match output {
                Output::Plain { .. } => "text",
                Output::Stream { .. } => "stream",
                Output::Image { .. } => "image",
                Output::ErrorOutput { .. } => "error",
                Output::Message { .. } => "message",
                Output::Table { .. } => "table",
                Output::Markdown { .. } => "markdown",
                Output::Json { .. } => "json",
                Output::ClearOutputWaitMarker => continue,
            };
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
        Some(format!(
            "outputs: {} ({})",
            self.outputs.len(),
            kinds.join(", ")
        ))
    }

    pub fn clear_outputs(&mut self) {
        self.outputs.clear();
        self.execution_duration = None;
    }

    pub fn outputs_collapsed(&self) -> bool {
        self.outputs_collapsed
    }

    pub fn toggle_outputs_collapsed(&mut self, cx: &mut Context<Self>) {
        self.set_outputs_collapsed(!self.outputs_collapsed, cx);
    }

    pub fn set_outputs_collapsed(&mut self, collapsed: bool, cx: &mut Context<Self>) {
        self.outputs_collapsed = collapsed;
        self.metadata.collapsed = Some(collapsed);
        let jupyter = self.metadata.jupyter.get_or_insert_with(|| JupyterCellMetadata {
            source_hidden: None,
            outputs_hidden: None,
            additional: Default::default(),
        });
        jupyter.outputs_hidden = Some(collapsed);
        cx.notify();
    }

    pub fn start_execution(&mut self) {
        self.execution_start_time = Some(Instant::now());
        self.execution_duration = None;
        self.is_executing = true;
    }

    pub fn finish_execution(&mut self) {
        if let Some(start_time) = self.execution_start_time.take() {
            self.execution_duration = Some(start_time.elapsed());
        }
        self.is_executing = false;
    }

    pub fn is_executing(&self) -> bool {
        self.is_executing
    }

    /// Displays a kernel-level failure (e.g. the kernel failed to launch because
    /// Python is not installed) as an error output on this cell, so the user gets
    /// feedback instead of a spinner that never resolves.
    pub fn show_kernel_error(
        &mut self,
        error_message: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.outputs.push(Output::ErrorOutput {
            content: ErrorView::new(
                "Kernel Error".to_string(),
                "cell could not be executed".to_string(),
                error_message,
                window,
                cx,
            ),
            nbformat: Some(nbformat::v4::Output::Error(nbformat::v4::ErrorOutput {
                ename: "Kernel Error".to_string(),
                evalue: "cell could not be executed".to_string(),
                traceback: error_message.lines().map(|line| line.to_string()).collect(),
            })),
        });
        self.execution_start_time = None;
        self.is_executing = false;
        cx.notify();
    }

    pub fn execution_duration(&self) -> Option<Duration> {
        self.execution_duration
    }

    fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs_f64();
        if total_secs < 1.0 {
            format!("{:.0}ms", duration.as_millis())
        } else if total_secs < 60.0 {
            format!("{:.1}s", total_secs)
        } else {
            let minutes = (total_secs / 60.0).floor() as u64;
            let secs = total_secs % 60.0;
            format!("{}m {:.1}s", minutes, secs)
        }
    }

    pub fn handle_message(
        &mut self,
        message: &JupyterMessage,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match &message.content {
            JupyterMessageContent::StreamContent(stream) => {
                let (name, text) = match &stream.name {
                    runtimelib::Stdio::Stdout => ("stdout".to_string(), stream.text.clone()),
                    runtimelib::Stdio::Stderr => ("stderr".to_string(), stream.text.clone()),
                };
                self.outputs.push(Output::Stream {
                    content: cx.new(|cx| TerminalOutput::from(&stream.text, window, cx)),
                    nbformat: Some(nbformat::v4::Output::Stream {
                        name: name.clone(),
                        text: nbformat::v4::MultilineString(text),
                    }),
                    name,
                });
            }
            JupyterMessageContent::DisplayData(display_data) => {
                self.outputs.push(Output::new(
                    &display_data.data,
                    None,
                    nbformat::v4::Output::DisplayData(nbformat::v4::DisplayData {
                        data: display_data.data.clone(),
                        metadata: display_data.metadata.clone(),
                    }),
                    window,
                    cx,
                ));
            }
            JupyterMessageContent::ExecuteResult(execute_result) => {
                self.outputs.push(Output::new(
                    &execute_result.data,
                    None,
                    nbformat::v4::Output::ExecuteResult(nbformat::v4::ExecuteResult {
                        execution_count: execute_result.execution_count,
                        data: execute_result.data.clone(),
                        metadata: execute_result.metadata.clone(),
                    }),
                    window,
                    cx,
                ));
            }
            JupyterMessageContent::ExecuteInput(input) => {
                self.execution_count = serde_json::to_value(&input.execution_count)
                    .ok()
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32);
            }
            JupyterMessageContent::ExecuteReply(_) => {
                self.finish_execution();
            }
            JupyterMessageContent::ErrorOutput(error) => {
                self.outputs.push(Output::ErrorOutput {
                    content: ErrorView::new(
                        error.ename.clone(),
                        error.evalue.clone(),
                        &error.traceback.join("\n"),
                        window,
                        cx,
                    ),
                    nbformat: Some(nbformat::v4::Output::Error(nbformat::v4::ErrorOutput {
                        ename: error.ename.clone(),
                        evalue: error.evalue.clone(),
                        traceback: error.traceback.clone(),
                    })),
                });
            }
            _ => {}
        }
        cx.notify();
    }

    pub fn gutter_output(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.selected();

        div()
            .relative()
            .h_full()
            .w(px(GUTTER_WIDTH))
            .child(
                div()
                    .w(px(GUTTER_WIDTH))
                    .flex()
                    .flex_none()
                    .justify_center()
                    .h_full()
                    .child(
                        div()
                            .flex_none()
                            .w(px(1.))
                            .h_full()
                            .when(is_selected, |this| this.bg(cx.theme().colors().icon_accent))
                            .when(!is_selected, |this| this.bg(cx.theme().colors().border)),
                    ),
            )
            .when(self.has_outputs(), |this| {
                let icon = if self.outputs_collapsed {
                    IconName::ChevronRight
                } else {
                    IconName::ChevronDown
                };
                this.child(
                    div()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        .left_0()
                        .flex()
                        .flex_none()
                        .w(px(GUTTER_WIDTH))
                        .h(px(GUTTER_WIDTH + 12.0))
                        .items_center()
                        .justify_center()
                        .bg(cx.theme().colors().tab_bar_background)
                        .child(
                            IconButton::new("toggle-outputs", icon)
                                .icon_size(IconSize::Small)
                                .tooltip(|_window, cx| {
                                    Tooltip::for_action("Toggle Outputs", &NotebookToggleOutput, cx)
                                })
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.toggle_outputs_collapsed(cx);
                                })),
                        ),
                )
            })
    }
}

impl RenderableCell for CodeCell {
    const CELL_TYPE: CellType = CellType::Code;

    fn id(&self) -> &CellId {
        &self.id
    }

    fn cell_type(&self) -> CellType {
        CellType::Code
    }

    fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    fn source(&self) -> &String {
        &self.source
    }

    fn control(&self, _window: &mut Window, cx: &mut Context<Self>) -> Option<CellControl> {
        let control_type = if self.is_executing {
            CellControlType::StopCell
        } else if self.has_outputs() {
            CellControlType::RerunCell
        } else {
            CellControlType::RunCell
        };

        Some(
            CellControl::new(control_type.id(), control_type).on_click(cx.listener(
                move |this, _, window, cx| {
                    if this.is_executing {
                        window.dispatch_action(Box::new(InterruptKernel), cx);
                    } else {
                        this.run(window, cx);
                    }
                },
            )),
        )
    }

    fn selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    fn cell_number(&self) -> Option<usize> {
        self.cell_number
    }

    fn set_cell_number(&mut self, cell_number: usize) -> &mut Self {
        self.cell_number = Some(cell_number);
        self
    }

    fn cell_position(&self) -> Option<&CellPosition> {
        self.cell_position.as_ref()
    }

    fn set_cell_position(&mut self, cell_position: CellPosition) -> &mut Self {
        self.cell_position = Some(cell_position);
        self
    }

    fn gutter(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.selected();
        let GutterLabels {
            position,
            execution_count,
        } = cell_gutter_labels(
            JupyterSettings::get_global(cx).notebook_cell_numbers,
            self.cell_number(),
            self.execution_count,
            self.is_executing,
            true,
        );

        div()
            .relative()
            .h_full()
            .w(px(GUTTER_WIDTH))
            .child(
                div()
                    .w(px(GUTTER_WIDTH))
                    .flex()
                    .flex_none()
                    .justify_center()
                    .h_full()
                    .child(
                        div()
                            .flex_none()
                            .w(px(1.))
                            .h_full()
                            .when(is_selected, |this| this.bg(cx.theme().colors().icon_accent))
                            .when(!is_selected, |this| this.bg(cx.theme().colors().border)),
                    ),
            )
            .when_some(self.control(window, cx), |this, control| {
                this.child(
                    div()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        .left_0()
                        .flex()
                        .flex_col()
                        .w(px(GUTTER_WIDTH))
                        .items_center()
                        .justify_center()
                        .bg(cx.theme().colors().tab_bar_background)
                        .when_some(position, |this, position| {
                            this.child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(position),
                            )
                        })
                        .child(control.button)
                        .when_some(execution_count, |this, execution_count| {
                            this.child(
                                div()
                                    .mt_0p5()
                                    .text_xs()
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(execution_count),
                            )
                        }),
                )
            })
    }
}

impl RunnableCell for CodeCell {
    fn run(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(CellEvent::Run(self.id.clone()));
    }

    fn execution_count(&self) -> Option<i32> {
        self.execution_count.filter(|&count| count > 0)
    }

    fn set_execution_count(&mut self, count: i32) -> &mut Self {
        self.execution_count = Some(count);
        self
    }
}

impl Render for CodeCell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let output_max_height = ReplSettings::get_global(cx).output_max_height_lines;
        let output_max_height = if output_max_height > 0 {
            Some(window.line_height() * output_max_height as f32)
        } else {
            None
        };
        let output_max_width =
            plain::max_width_for_columns(ReplSettings::get_global(cx).max_columns, window, cx);
        // get the language from the editor's buffer
        let language_name = self
            .editor
            .read(cx)
            .buffer()
            .read(cx)
            .as_singleton()
            .and_then(|buffer| buffer.read(cx).language())
            .map(|lang| lang.name().to_string());

        v_flex()
            .size_full()
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(true, window, cx))
            // Editor portion
            .child(
                h_flex()
                    .w_full()
                    .pr_6()
                    .rounded_xs()
                    .items_start()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .bg(self.selected_bg_color(window, cx))
                    .child(self.gutter(window, cx))
                    .child(
                        div().py_1p5().w_full().child(
                            div()
                                .relative()
                                .flex()
                                .size_full()
                                .flex_1()
                                .py_3()
                                .px_5()
                                .rounded_lg()
                                .border_1()
                                .border_color(cx.theme().colors().border)
                                .bg(cx.theme().colors().editor_background)
                                .child(div().w_full().child(self.editor.clone()))
                                // lang badge in top-right corner
                                .when_some(language_name, |this, name| {
                                    this.child(
                                        div()
                                            .absolute()
                                            .top_1()
                                            .right_2()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_md()
                                            .bg(cx.theme().colors().element_background.opacity(0.7))
                                            .text_xs()
                                            .text_color(cx.theme().colors().text_muted)
                                            .child(name),
                                    )
                                }),
                        ),
                    ),
            )
            .when(
                self.has_outputs() || self.execution_duration.is_some() || self.is_executing,
                |this| {
                    let execution_time_label = self.execution_duration.map(Self::format_duration);
                    let is_executing = self.is_executing;
                    this.child(
                        h_flex()
                            .w_full()
                            .pr_6()
                            .rounded_xs()
                            .items_start()
                            .gap(DynamicSpacing::Base08.rems(cx))
                            .bg(self.selected_bg_color(window, cx))
                            .child(self.gutter_output(window, cx))
                            .child(
                                div().py_1p5().w_full().child(
                                    v_flex()
                                        .size_full()
                                        .flex_1()
                                        .py_3()
                                        .px_5()
                                        .rounded_lg()
                                        .border_1()
                                        // execution status/time at the TOP
                                        .when(
                                            is_executing || execution_time_label.is_some(),
                                            |this| {
                                                let time_element = if is_executing {
                                                    h_flex()
                                                        .gap_1()
                                                        .items_center()
                                                        .child(
                                                            Icon::new(IconName::ArrowCircle)
                                                                .size(IconSize::XSmall)
                                                                .color(Color::Warning)
                                                                .with_rotate_animation(2)
                                                                .into_any_element(),
                                                        )
                                                        .child(
                                                            div()
                                                                .text_xs()
                                                                .text_color(
                                                                    cx.theme().colors().text_muted,
                                                                )
                                                                .child("Running..."),
                                                        )
                                                        .into_any_element()
                                                } else if let Some(duration_text) =
                                                    execution_time_label.clone()
                                                {
                                                    h_flex()
                                                        .gap_1()
                                                        .items_center()
                                                        .child(
                                                            Icon::new(IconName::Check)
                                                                .size(IconSize::XSmall)
                                                                .color(Color::Success),
                                                        )
                                                        .child(
                                                            div()
                                                                .text_xs()
                                                                .text_color(
                                                                    cx.theme().colors().text_muted,
                                                                )
                                                                .child(duration_text),
                                                        )
                                                        .into_any_element()
                                                } else {
                                                    div().into_any_element()
                                                };
                                                this.child(div().mb_2().child(time_element))
                                            },
                                        )
                                        // output at bottom
                                        .child(
                                            div()
                                                .id((
                                                    ElementId::from(self.id.to_string()),
                                                    "output-scroll",
                                                ))
                                                .w_full()
                                                .when_some(output_max_width, |div, max_width| {
                                                    div.max_w(max_width).overflow_x_scroll()
                                                })
                                                .when_some(output_max_height, |div, max_height| {
                                                    div.max_h(max_height).overflow_y_scroll()
                                                })
                                                .when(self.outputs_collapsed, |this| {
                                                    this.child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(
                                                                cx.theme().colors().text_muted,
                                                            )
                                                            .child("Outputs hidden"),
                                                    )
                                                })
                                                .when(!self.outputs_collapsed, |this| {
                                                    this.children(self.outputs.iter().map(
                                                        |output| {
                                                            div().children(
                                                                output.content(window, cx),
                                                            )
                                                        },
                                                    ))
                                                }),
                                        ),
                                ),
                            ),
                    )
                },
            )
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(false, window, cx))
    }
}

pub struct RawCell {
    id: CellId,
    metadata: CellMetadata,
    source: String,
    selected: bool,
    cell_position: Option<CellPosition>,
    cell_number: Option<usize>,
}

impl RawCell {
    fn apply_notebook_update(
        &mut self,
        metadata: &CellMetadata,
        source: &str,
        cx: &mut Context<Self>,
    ) {
        self.metadata = metadata.clone();
        if self.source != source {
            self.source = source.to_string();
        }
        cx.notify();
    }

    pub fn to_nbformat_cell(&self) -> nbformat::v4::Cell {
        let source_lines: Vec<String> = self.source.lines().map(|l| format!("{}\n", l)).collect();

        nbformat::v4::Cell::Raw {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            source: source_lines,
        }
    }
}

impl RenderableCell for RawCell {
    const CELL_TYPE: CellType = CellType::Raw;

    fn id(&self) -> &CellId {
        &self.id
    }

    fn cell_type(&self) -> CellType {
        CellType::Raw
    }

    fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    fn source(&self) -> &String {
        &self.source
    }

    fn selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    fn cell_number(&self) -> Option<usize> {
        self.cell_number
    }

    fn set_cell_number(&mut self, cell_number: usize) -> &mut Self {
        self.cell_number = Some(cell_number);
        self
    }

    fn cell_position(&self) -> Option<&CellPosition> {
        self.cell_position.as_ref()
    }

    fn set_cell_position(&mut self, cell_position: CellPosition) -> &mut Self {
        self.cell_position = Some(cell_position);
        self
    }
}

impl Render for RawCell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(true, window, cx))
            .child(
                h_flex()
                    .w_full()
                    .pr_2()
                    .rounded_xs()
                    .items_start()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .bg(self.selected_bg_color(window, cx))
                    .child(self.gutter(window, cx))
                    .child(
                        div()
                            .flex()
                            .size_full()
                            .flex_1()
                            .p_3()
                            .font_ui(cx)
                            .text_size(TextSize::Default.rems(cx))
                            .child(self.source.clone()),
                    ),
            )
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(false, window, cx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{TestAppContext, VisualTestContext};
    use serde_json::json;
    use settings::SettingsStore;

    const TEST_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

    fn init_test(cx: &mut TestAppContext) -> &mut VisualTestContext {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        cx.add_empty_window()
    }

    fn round_trip(
        outputs: &Vec<nbformat::v4::Output>,
        cx: &mut VisualTestContext,
    ) -> Vec<serde_json::Value> {
        let converted = cx.update(|window, cx| convert_outputs(outputs, window, cx));
        cx.update(|_, cx| {
            converted
                .iter()
                .filter_map(|output| output.to_nbformat(cx))
                .map(|output| serde_json::to_value(output).expect("serializable"))
                .collect()
        })
    }

    #[gpui::test]
    async fn test_convert_outputs_round_trip_preserves_every_output(
        cx: &mut TestAppContext,
    ) {
        let cx = init_test(cx);

        let original = json!([
            { "output_type": "stream", "name": "stdout", "text": ["hello\n"] },
            { "output_type": "stream", "name": "stderr", "text": ["warning\n"] },
            {
                "output_type": "execute_result",
                "execution_count": 7,
                "data": {
                    "image/png": TEST_PNG,
                    "text/plain": ["<Figure size 200x120>"]
                },
                "metadata": { "needs_background": "light" }
            },
            {
                "output_type": "execute_result",
                "execution_count": 8,
                "data": { "application/json": { "values": [1, 2, 3] } },
                "metadata": {}
            },
            {
                "output_type": "display_data",
                "data": {
                    "application/vnd.jupyter.widget-view+json": {
                        "model_id": "abc",
                        "version_major": 2,
                        "version_minor": 0
                    },
                    "text/plain": ["HBox(children=(...))"]
                },
                "metadata": {}
            },
            {
                "output_type": "display_data",
                "data": {
                    "text/html": ["<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"],
                    "text/plain": ["   A\n", "0  1"]
                },
                "metadata": {}
            },
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "bad",
                "traceback": ["Traceback (most recent call last):\n", "ValueError: bad\n"]
            }
        ]);

        let outputs: Vec<nbformat::v4::Output> =
            serde_json::from_value(original.clone()).expect("valid outputs");

        let round_tripped = round_trip(&outputs, cx);

        assert_eq!(serde_json::Value::Array(round_tripped), original);
    }

    #[gpui::test]
    async fn test_notebook_cell_outputs_round_trip(cx: &mut TestAppContext) {
        let cx = init_test(cx);

        let notebook_json = json!({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "id": "cell-1",
                    "metadata": {},
                    "execution_count": 1,
                    "source": ["print('hi')\n"],
                    "outputs": [
                        { "output_type": "stream", "name": "stdout", "text": ["hi\n"] },
                        {
                            "output_type": "execute_result",
                            "execution_count": 1,
                            "data": { "text/plain": ["42"] },
                            "metadata": {}
                        }
                    ]
                },
                {
                    "cell_type": "code",
                    "id": "cell-2",
                    "metadata": {},
                    "execution_count": 2,
                    "source": ["frame\n"],
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/html": ["<table><tr><td>1</td></tr></table>"],
                                "text/plain": ["   1\n0  1"]
                            },
                            "metadata": {}
                        }
                    ]
                },
                {
                    "cell_type": "markdown",
                    "id": "cell-3",
                    "metadata": {},
                    "source": ["# title"]
                }
            ]
        });

        let notebook: nbformat::v4::Notebook =
            serde_json::from_value(notebook_json).expect("valid notebook");

        for cell in &notebook.cells {
            let nbformat::v4::Cell::Code { outputs, .. } = cell else {
                continue;
            };

            let round_tripped = round_trip(outputs, cx);
            assert_eq!(
                serde_json::Value::Array(round_tripped),
                serde_json::to_value(outputs).expect("serializable"),
                "code cell outputs must survive a load/save round trip"
            );
        }
    }

    #[gpui::test]
    async fn test_mime_selection_prefers_richer_rendering(cx: &mut TestAppContext) {
        let cx = init_test(cx);

        let png_and_html: runtimelib::MimeBundle = serde_json::from_value(json!({
            "image/png": TEST_PNG,
            "text/html": ["<b>bold</b>"],
            "text/plain": ["bold"]
        }))
        .expect("valid bundle");

        let output = cx.update(|window, cx| {
            Output::new(
                &png_and_html,
                None,
                nbformat::v4::Output::DisplayData(nbformat::v4::DisplayData::default()),
                window,
                cx,
            )
        });
        assert!(matches!(output, Output::Image { .. }), "png should win");

        let html_and_plain: runtimelib::MimeBundle = serde_json::from_value(json!({
            "text/html": ["<b>bold</b>"],
            "text/plain": ["bold"]
        }))
        .expect("valid bundle");

        let output = cx.update(|window, cx| {
            Output::new(
                &html_and_plain,
                None,
                nbformat::v4::Output::DisplayData(nbformat::v4::DisplayData::default()),
                window,
                cx,
            )
        });
        assert!(
            matches!(output, Output::Markdown { .. }),
            "html should win over plain"
        );

        let interactive_only: runtimelib::MimeBundle = serde_json::from_value(json!({
            "application/vnd.jupyter.widget-view+json": {
                "model_id": "abc",
                "version_major": 2,
                "version_minor": 0
            }
        }))
        .expect("valid bundle");

        let output = cx.update(|window, cx| {
            Output::new(
                &interactive_only,
                None,
                nbformat::v4::Output::DisplayData(nbformat::v4::DisplayData::default()),
                window,
                cx,
            )
        });
        assert!(
            matches!(output, Output::Message { .. }),
            "interactive-only output should fall back to a placeholder"
        );
    }

    fn menu_state() -> CellContextMenuState {
        CellContextMenuState {
            has_selection: true,
            runs_code: true,
            format_selections: true,
            ai_enabled: true,
            has_file_target: false,
            has_git_repo: false,
        }
    }

    fn action_labels(entries: &[CellContextMenuEntry]) -> Vec<&'static str> {
        entries
            .iter()
            .filter_map(|entry| match entry {
                CellContextMenuEntry::Action { label, .. } => Some(*label),
                CellContextMenuEntry::Separator => None,
            })
            .collect()
    }

    fn entry<'a>(entries: &'a [CellContextMenuEntry], label: &str) -> &'a CellContextMenuEntry {
        entries
            .iter()
            .find(|entry| matches!(entry, CellContextMenuEntry::Action { label: entry_label, .. } if *entry_label == label))
            .expect("menu entry should be present")
    }

    fn is_disabled(entry: &CellContextMenuEntry) -> bool {
        match entry {
            CellContextMenuEntry::Action { disabled, .. } => *disabled,
            CellContextMenuEntry::Separator => false,
        }
    }

    #[test]
    fn context_menu_matches_editor_menu_then_notebook_block_in_order() {
        let entries = cell_context_menu_entries(menu_state());
        let expected = vec![
            "Go to Definition",
            "Go to Declaration",
            "Go to Type Definition",
            "Go to Implementation",
            "Find All References",
            "Show Incoming Calls",
            "Show Outgoing Calls",
            "Rename Symbol",
            "Format Buffer",
            "Format Selections",
            "Show Code Actions",
            "Add to Agent Thread",
            "Cut",
            "Copy",
            "Copy and Trim",
            "Paste",
            ui::utils::reveal_in_file_manager_label(false),
            "Open in Terminal",
            "Copy Permalink to Line",
            "View File History",
            "Split Cell",
            "Merge Cell Above",
            "Merge Cell Below",
            "Convert Cell to Markdown",
            "Convert Cell to Code",
            "Run Cell",
            "Run All Above",
            "Run All Below",
            "Clear Outputs",
            "Toggle Outputs",
        ];
        assert_eq!(action_labels(&entries), expected);

        for label in [
            ui::utils::reveal_in_file_manager_label(false),
            "Open in Terminal",
            "Copy Permalink to Line",
            "View File History",
        ] {
            assert!(
                is_disabled(entry(&entries, label)),
                "{label} should be disabled without a file target"
            );
        }

        for label in ["Show Incoming Calls", "Show Outgoing Calls"] {
            assert!(is_disabled(entry(&entries, label)), "{label} should be disabled");
        }
    }

    #[test]
    fn markdown_cell_context_menu_omits_run_entries() {
        let entries = cell_context_menu_entries(CellContextMenuState {
            runs_code: false,
            ..menu_state()
        });
        let labels = action_labels(&entries);
        assert!(!labels.contains(&"Run Cell"));
        assert!(!labels.contains(&"Run All Above"));
        assert!(!labels.contains(&"Run All Below"));
        assert!(!labels.contains(&"Clear Outputs"));
        assert!(!labels.contains(&"Toggle Outputs"));
        assert!(labels.contains(&"Split Cell"));
    }

    #[test]
    fn context_menu_disables_cut_and_copy_without_a_selection() {
        let entries = cell_context_menu_entries(CellContextMenuState {
            has_selection: false,
            ..menu_state()
        });
        assert!(is_disabled(entry(&entries, "Cut")));
        assert!(is_disabled(entry(&entries, "Copy")));
        assert!(!action_labels(&entries).contains(&"Add to Agent Thread"));

        let entries = cell_context_menu_entries(menu_state());
        assert!(!is_disabled(entry(&entries, "Cut")));
        assert!(!is_disabled(entry(&entries, "Copy")));
        assert!(action_labels(&entries).contains(&"Add to Agent Thread"));
    }

    #[gpui::test]
    async fn right_click_outside_selection_moves_cursor(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        cx.update(|_, cx| editor::init(cx));

        let editor = cx.update(|window, cx| {
            let buffer = cx.new(|cx| Buffer::local("hello world", cx));
            let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
            cx.new(|cx| {
                Editor::new(
                    EditorMode::Full {
                        scale_ui_elements_with_buffer_font_size: false,
                        show_active_line_background: false,
                        sizing_behavior: SizingBehavior::SizeByContent,
                    },
                    multi_buffer,
                    None,
                    window,
                    cx,
                )
            })
        });

        let mut selection_start = DisplayPoint::zero();
        let mut selection_end = DisplayPoint::zero();
        *selection_end.column_mut() = 5;
        let mut clicked = DisplayPoint::zero();
        *clicked.column_mut() = 8;

        editor.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_display_ranges([selection_start..selection_end]);
            });
            assert_eq!(
                editor.selections.newest_display(&editor.display_snapshot(cx)).end,
                selection_end
            );

            move_cursor_to_point_if_outside_selection(editor, clicked, window, cx);

            let snapshot = editor.display_snapshot(cx);
            let selection = editor.selections.newest_display(&snapshot);
            assert!(selection.is_empty());
            assert_eq!(selection.head(), clicked);

            selection_start = DisplayPoint::zero();
            *selection_start.column_mut() = 2;
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_display_ranges([selection_start..selection_end]);
            });
            let mut inside = DisplayPoint::zero();
            *inside.column_mut() = 3;
            move_cursor_to_point_if_outside_selection(editor, inside, window, cx);

            let snapshot = editor.display_snapshot(cx);
            let selection = editor.selections.newest_display(&snapshot);
            assert_eq!(selection.start, selection_start);
            assert_eq!(selection.end, selection_end);
        });
    }
}

#[cfg(test)]
mod gutter_tests {
    use super::*;

    #[test]
    fn execution_count_is_bracketed_and_distinguishes_pending_and_running() {
        assert_eq!(format_execution_count(None, false), "[ ]");
        assert_eq!(format_execution_count(Some(0), false), "[ ]");
        assert_eq!(format_execution_count(Some(3), false), "[3]");
        assert_eq!(format_execution_count(None, true), "[*]");
        assert_eq!(format_execution_count(Some(3), true), "[*]");
    }

    #[test]
    fn code_cell_labels_follow_each_configuration() {
        let both = cell_gutter_labels(
            NotebookCellNumbersContent::Both,
            Some(2),
            Some(3),
            false,
            true,
        );
        assert_eq!(both.position.as_deref(), Some("2"));
        assert_eq!(both.execution_count.as_deref(), Some("[3]"));

        let position = cell_gutter_labels(
            NotebookCellNumbersContent::Position,
            Some(2),
            Some(3),
            false,
            true,
        );
        assert_eq!(position.position.as_deref(), Some("2"));
        assert_eq!(position.execution_count, None);

        let execution_count = cell_gutter_labels(
            NotebookCellNumbersContent::ExecutionCount,
            Some(2),
            Some(3),
            false,
            true,
        );
        assert_eq!(execution_count.position, None);
        assert_eq!(execution_count.execution_count.as_deref(), Some("[3]"));

        let none = cell_gutter_labels(
            NotebookCellNumbersContent::None,
            Some(2),
            Some(3),
            false,
            true,
        );
        assert_eq!(none.position, None);
        assert_eq!(none.execution_count, None);
    }

    #[test]
    fn non_code_cells_only_show_a_position() {
        let both = cell_gutter_labels(
            NotebookCellNumbersContent::Both,
            Some(4),
            Some(3),
            false,
            false,
        );
        assert_eq!(both.position.as_deref(), Some("4"));
        assert_eq!(both.execution_count, None);

        let execution_count = cell_gutter_labels(
            NotebookCellNumbersContent::ExecutionCount,
            Some(4),
            Some(3),
            false,
            false,
        );
        assert_eq!(execution_count.position, None);
        assert_eq!(execution_count.execution_count, None);
    }

    #[test]
    fn missing_positions_are_not_formatted() {
        let labels = cell_gutter_labels(
            NotebookCellNumbersContent::Both,
            None,
            Some(3),
            false,
            true,
        );
        assert_eq!(labels.position, None);
        assert_eq!(labels.execution_count.as_deref(), Some("[3]"));
    }
}

#[cfg(test)]
mod injection_tests {
    use super::*;

    use collections::HashSet;
    use futures::FutureExt as _;
    use gpui::TestAppContext;
    use indoc::indoc;
    use language::LanguageConfig;
    use settings::SettingsStore;

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
    }

    /// Returns markdown-inline language for immediate registration into a buffer's language repo.
    fn markdown_inline_lang() -> Language {
        Language::new(
            LanguageConfig {
                name: "Markdown-Inline".into(),
                hidden: true,
                ..LanguageConfig::default()
            },
            Some(tree_sitter_md::INLINE_LANGUAGE.into()),
        )
        .with_highlights_query("(emphasis) @emphasis")
        .unwrap()
    }

    /// Registers Markdown-Inline the way the app does: available by name, but
    /// only loaded once a buffer's injection first asks for it.
    fn register_markdown_inline_lazily(languages: &LanguageRegistry) {
        languages.register_native_grammars([("markdown-inline", tree_sitter_md::INLINE_LANGUAGE)]);
        languages.register_test_language(LanguageConfig {
            name: "Markdown-Inline".into(),
            grammar: Some("markdown-inline".into()),
            hidden: true,
            ..LanguageConfig::default()
        });
    }

    fn assert_layer_languages(editor: &Entity<Editor>, expected_langs: &[&str], cx: &App) {
        let buffer = editor.read(cx).buffer().read(cx).as_singleton().unwrap();
        let snapshot = buffer.read(cx).snapshot();
        let actual = snapshot
            .syntax_layers_languages()
            .map(|language| language.name().to_string())
            .collect::<HashSet<String>>();

        let expected = expected_langs
            .iter()
            .map(|name| name.to_string())
            .collect::<HashSet<String>>();
        assert_eq!(
            expected, actual,
            "cell languages mismatch: expected {expected:?}, got {actual:?}"
        )
    }

    fn run_cell_language_test(cx: &mut TestAppContext, cell_source: &str, expected_langs: &[&str]) {
        init_test(cx);

        let languages = Arc::new(LanguageRegistry::test(cx.executor()));

        languages.add(languages::json_lang());
        languages.add(languages::rust_lang());
        languages.add(languages::markdown_lang());
        languages.add(Arc::new(markdown_inline_lang()));

        let metadata: CellMetadata = serde_json::from_str("{}").unwrap();

        let cx = cx.add_empty_window();
        let cell = cx.update(|window, cx| {
            cx.new(|cx| {
                MarkdownCell::new(
                    uuid::Uuid::new_v4().into(),
                    metadata,
                    cell_source.to_string(),
                    languages,
                    window,
                    cx,
                )
            })
        });
        cx.run_until_parked();

        cell.read_with(cx, |cell, cx| {
            assert_layer_languages(&cell.editor, expected_langs, cx)
        });
    }

    #[gpui::test]
    fn test_markdown_cell_parses_injections(cx: &mut TestAppContext) {
        run_cell_language_test(cx, "Hello, World!", &["Markdown", "Markdown-Inline"]);
        run_cell_language_test(
            cx,
            indoc! {r#"
                Some *emphasized* text
                ```json
                { "foo": "bar" }
                ```
            "#},
            &["Markdown", "Markdown-Inline", "JSON"],
        );
        run_cell_language_test(
            cx,
            indoc! {r#"
                ```json
                [ "a", "b", { "foo": "bar" }]
                ```
                ```rs
                fn main() {
                    println!("Hello, World!");
                }
                ```
            "#},
            &["Markdown", "JSON", "Rust"],
        );
    }

    #[gpui::test]
    fn test_markdown_cell_parses_lazily_loaded_injections(cx: &mut TestAppContext) {
        init_test(cx);

        let languages = Arc::new(LanguageRegistry::test(cx.executor()));
        languages.add(languages::markdown_lang());
        register_markdown_inline_lazily(&languages);

        let metadata: CellMetadata = serde_json::from_str("{}").unwrap();

        let cx = cx.add_empty_window();
        let cell = cx.update(|window, cx| {
            cx.new(|cx| {
                MarkdownCell::new(
                    uuid::Uuid::new_v4().into(),
                    metadata,
                    "Some *emphasized* text".to_string(),
                    languages,
                    window,
                    cx,
                )
            })
        });
        cx.run_until_parked();

        cell.read_with(cx, |cell, cx| {
            assert_layer_languages(&cell.editor, &["Markdown", "Markdown-Inline"], cx)
        });
    }

    #[gpui::test]
    async fn test_code_cell_parses_lazily_loaded_injections(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = project::FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;

        let languages = Arc::new(LanguageRegistry::test(cx.executor()));
        languages.add(languages::markdown_lang());
        register_markdown_inline_lazily(&languages);

        // Any kernel language can inject others; Markdown is the injecting
        // language available to these tests.
        let notebook_language = Task::ready(Some(languages::markdown_lang())).shared();
        let metadata: CellMetadata = serde_json::from_str("{}").unwrap();

        let cx = cx.add_empty_window();
        let cell = cx.update(|window, cx| {
            cx.new(|cx| {
                CodeCell::new(
                    CellSource::None,
                    uuid::Uuid::new_v4().into(),
                    metadata,
                    "Some *emphasized* text".to_string(),
                    languages,
                    notebook_language,
                    &project,
                    window,
                    cx,
                )
            })
        });
        cx.run_until_parked();

        cell.read_with(cx, |cell, cx| {
            assert_layer_languages(&cell.editor, &["Markdown", "Markdown-Inline"], cx)
        });
    }
}
