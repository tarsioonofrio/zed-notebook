#![allow(unused, dead_code)]
use std::ffi::OsStr;
use std::future::Future;
use std::path::Path;
use std::rc::Rc;
use std::{any::TypeId, ops::RangeInclusive, path::PathBuf, sync::Arc};

use anyhow::{Context as _, Result, anyhow};
use client::proto::ViewId;
use collections::HashMap;
use editor::{Anchor, DisplayPoint, Editor};
use feature_flags::{FeatureFlagAppExt as _, NotebookFeatureFlag};
use futures::FutureExt;
use futures::future::Shared;
use gpui::{
    Action as _, AnyElement, AnyEntity, AnyWindowHandle, App, Entity, EventEmitter, FocusHandle,
    Focusable, KeyContext, ListOffset, ListScrollEvent, ListState, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Point, Subscription, Task, TaskExt, actions, canvas, list,
    prelude::*,
};
use jupyter_protocol::{ExecuteReply, ExpressionResult, JupyterKernelspec, MediaType, ReplyStatus};
use language::{Language, LanguageRegistry};
use log;
use multi_buffer::MultiBufferRow;
use notebook_agent_tools::{
    NotebookCellSnapshot, NotebookCellType, NotebookLiveCommand, NotebookLiveResult,
    NotebookSnapshot,
};
use notebook_lsp::{NotebookCell, NotebookLsp};
use project::search::SearchQuery;
use project::{Fs, Project, ProjectEntryId, ProjectPath};
use settings::{Settings as _, SeedQuerySetting};
use ui::{CommonAnimationExt, ContextMenu, KeyBinding, PopoverMenu, Tooltip, prelude::*};
use util::ResultExt as _;
use workspace::item::{ItemEvent, SaveOptions, TabContentParams};
use workspace::searchable::{
    Direction, SearchEvent, SearchOptions, SearchToken, SearchableItem, SearchableItemHandle,
};
use workspace::{Item, ItemHandle, Pane, ProjectItem, ToolbarItemLocation, Workspace};

use super::{
    Cell, CellEvent, CellPosition, MarkdownCellEvent, NotebookKernelCompleter, RenderableCell,
};

use nbformat::v4::CellId;
use nbformat::v4::Metadata as NotebookMetadata;
use serde_json;
use uuid::Uuid;

use crate::components::{KernelPickerDelegate, KernelSelector};
use crate::jupyter_settings::JupyterSettings;
use crate::kernels::{
    Kernel, KernelSession, KernelSpecification, KernelStatus, LocalKernelSpecification,
    NativeRunningKernel, PythonEnvKernelSpecification, RemoteRunningKernel, SshRunningKernel,
    VENV_DIR_NAMES, WslRunningKernel, probe_python_environment_interpreter,
};
use crate::notebook::MovementDirection;
use crate::repl_store::ReplStore;
use notebook_variables::{
    DATA_PAGE_EXPRESSION, DataPageRequest, RefreshVariables, ToggleVariablesPanel,
    ToggleVariablesPanelPosition, VARIABLES_INTROSPECTION_CODE, VARIABLES_INTROSPECTION_EXPRESSION,
    VARIABLES_NAME_COLUMN_WIDTH, VARIABLES_PAGE_SIZE, VARIABLES_REQUEST_TIMEOUT,
    VARIABLES_TYPE_COLUMN_WIDTH, VariablesPanel, VariablesPanelEvent, VariablesPanelPosition,
    clamp_variables_bottom_height, clamp_variables_right_width, data_page_code, is_python_language,
    parse_data_page_json, parse_variables_json, variables_bottom_min_height,
    variables_right_min_width,
};

use picker::Picker;
use runtimelib::{ExecuteRequest, JupyterMessage, JupyterMessageContent};
use ui::PopoverMenuHandle;
use zed_actions::editor::{MoveDown, MoveUp};
use zed_actions::notebook::{
    AddCodeBlock, AddMarkdownBlock, ClearCellOutputs, ClearOutputs, ConvertCellToCode,
    ConvertCellToMarkdown, DeleteCell, EnterCommandMode, EnterEditMode, InterruptKernel,
    MergeCellAbove, MergeCellBelow, MoveCellDown, MoveCellUp, NotebookMoveDown, NotebookMoveUp,
    OpenNotebook, Redo, RestartKernel, Run, RunAbove, RunAll, RunAndAdvance, RunBelow, SplitCell,
    ToggleCellOutput, Undo,
};

/// Whether the notebook is in command mode (navigating cells) or edit mode (editing a cell).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotebookMode {
    Command,
    Edit,
}

#[derive(PartialEq, Eq)]
enum SelectionMode {
    SelectOnly,
    SelectAndMove,
}

pub(crate) const MAX_TEXT_BLOCK_WIDTH: f32 = 9999.0;
pub(crate) const SMALL_SPACING_SIZE: f32 = 8.0;
pub(crate) const MEDIUM_SPACING_SIZE: f32 = 12.0;
pub(crate) const LARGE_SPACING_SIZE: f32 = 16.0;
pub(crate) const GUTTER_WIDTH: f32 = 32.0;
pub(crate) const CODE_BLOCK_INSET: f32 = MEDIUM_SPACING_SIZE;
pub(crate) const CONTROL_SIZE: f32 = 20.0;

const NOTEBOOK_EXTENSION: &str = "ipynb";

pub fn init(cx: &mut App) {
    if cx.has_flag::<NotebookFeatureFlag>() || std::env::var("LOCAL_NOTEBOOK_DEV").is_ok() {
        workspace::register_project_item::<NotebookEditor>(cx);
    }

    cx.observe_flag::<NotebookFeatureFlag, _>({
        move |flag, cx| {
            if *flag {
                workspace::register_project_item::<NotebookEditor>(cx);
            } else {
                // todo: there is no way to unregister a project item, so if the feature flag
                // gets turned off they need to restart Zed.
            }
        }
    })
    .detach();
}

/// The outcome of deciding which kernel a notebook should run.
enum KernelAdoption {
    /// The running kernel already matches the decision; leave it alone.
    Keep,
    /// Replace the running kernel with this discovered environment.
    Switch(KernelSpecification),
    /// The closest `.venv` cannot run notebooks because ipykernel is missing.
    MissingIpykernel(PathBuf),
}

/// A snapshot of the notebook's cell structure, used by the notebook-level
/// undo stack. Cell handles are cheap to clone and keep removed cells alive, so
/// restoring a snapshot never depends on re-reading the notebook from disk.
#[derive(Clone)]
struct NotebookStructure {
    cell_order: Vec<CellId>,
    cell_map: HashMap<CellId, Cell>,
    selected_cell_index: usize,
}

/// Identifies the cell at the top of the viewport so a rebuilt list can keep the
/// user looking at the same place even when cells are inserted or removed.
#[derive(Clone)]
struct ScrollAnchor {
    cell_id: CellId,
    offset_in_item: Pixels,
}

pub struct NotebookEditor {
    languages: Arc<LanguageRegistry>,
    project: Entity<Project>,
    worktree_id: project::WorktreeId,
    focus_handle: FocusHandle,
    notebook_item: Entity<NotebookItem>,
    notebook_path: Option<PathBuf>,
    agent_window: Option<AnyWindowHandle>,
    external_change_subscription: Option<Subscription>,
    focused_cell_id: Option<CellId>,
    registered_cell_buffers: Vec<text::BufferId>,
    notebook_lsp: Option<Entity<NotebookLsp>>,
    notebook_language: Shared<Task<Option<Arc<Language>>>>,
    remote_id: Option<ViewId>,
    cell_list: ListState,
    notebook_mode: NotebookMode,
    selected_cell_index: usize,
    cell_order: Vec<CellId>,
    original_cell_order: Vec<CellId>,
    cell_map: HashMap<CellId, Cell>,
    undo_stack: Vec<NotebookStructure>,
    redo_stack: Vec<NotebookStructure>,
    kernel: Kernel,
    kernel_specification: Option<KernelSpecification>,
    deferred_kernel_adoption: Option<KernelAdoption>,
    parent_environment_search_started: bool,
    kernel_generation: u64,
    execution_requests: HashMap<String, PendingExecution>,
    kernel_completer: Rc<NotebookKernelCompleter>,
    variables_panel: Entity<VariablesPanel>,
    variables_bottom_height: Option<Pixels>,
    variables_resize_start: Option<VariablesResizeStart>,
    variables_right_width: Option<Pixels>,
    variables_width_resize_start: Option<VariablesWidthResizeStart>,
    variable_requests: HashMap<String, VariableRequest>,
    kernel_picker_handle: PopoverMenuHandle<Picker<KernelPickerDelegate>>,
    repl_store_subscription: Option<Subscription>,
    /// The workspace hosting this notebook. Cell editors are not workspace items
    /// themselves, so they need it handed to them to open reference
    /// multibuffers and apply code actions there.
    workspace: Option<(gpui::WeakEntity<Workspace>, Option<workspace::WorkspaceId>)>,
}

#[derive(Clone)]
enum VariableRequest {
    Introspection,
    DataPage { variable: String },
}

#[derive(Clone, Copy)]
struct VariablesResizeStart {
    mouse_y: Pixels,
    height: Pixels,
}

#[derive(Clone, Copy)]
struct VariablesWidthResizeStart {
    mouse_x: Pixels,
    width: Pixels,
}

/// An in-flight execute request. Shell replies and iopub messages are read from
/// independent sockets, so a request may keep emitting messages after its reply
/// has been processed. We only stop routing to its cell once both the reply and
/// the idle status have been seen, which is when the request is truly finished.
struct PendingExecution {
    cell_id: CellId,
    reply_received: bool,
    idle_received: bool,
}

impl PendingExecution {
    fn is_complete(&self) -> bool {
        self.reply_received && self.idle_received
    }
}

enum SaveDestination {
    CurrentPath,
    NewPath(ProjectPath),
}

impl NotebookEditor {
    pub fn new(
        project: Entity<Project>,
        notebook_item: Entity<NotebookItem>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();

        let languages = project.read(cx).languages().clone();
        let language_name = notebook_item.read(cx).language_name();
        let worktree_id = notebook_item.read(cx).project_path.worktree_id;

        let notebook_language = notebook_item.read(cx).notebook_language();
        let notebook_language = cx
            .spawn_in(window, async move |_, _| notebook_language.await)
            .shared();

        let mut cell_order = vec![]; // Vec<CellId>
        let mut cell_map = HashMap::default(); // HashMap<CellId, Cell>

        let cell_count = notebook_item.read(cx).notebook.cells.len();
        for index in 0..cell_count {
            let cell = notebook_item.read(cx).notebook.cells[index].clone();
            let cell_id = cell.id();
            cell_order.push(cell_id.clone());
            let cell_entity = Cell::load(
                &cell,
                &languages,
                notebook_language.clone(),
                &project,
                window,
                cx,
            );

            match &cell_entity {
                Cell::Code(code_cell) => {
                    let cell_id_for_focus = cell_id.clone();
                    cx.subscribe_in(code_cell, window, move |this, _cell, event, window, cx| {
                        match event {
                            CellEvent::Run(cell_id) => {
                                this.execute_cell(cell_id.clone(), window, cx)
                            }
                            CellEvent::FocusedIn(_) => {
                                this.select_cell_by_id(&cell_id_for_focus, cx)
                            }
                        }
                    })
                    .detach();

                    let cell_id_for_editor = cell_id.clone();
                    let editor = code_cell.read(cx).editor().clone();
                    cx.subscribe(&editor, move |this, _editor, event, cx| {
                        if let editor::EditorEvent::Focused = event {
                            this.select_cell_by_id(&cell_id_for_editor, cx);
                        }
                    })
                    .detach();
                }
                Cell::Markdown(markdown_cell) => {
                    cx.subscribe(
                        markdown_cell,
                        move |_this, cell, event: &MarkdownCellEvent, cx| {
                            match event {
                                MarkdownCellEvent::FinishedEditing => {
                                    cell.update(cx, |cell, cx| {
                                        cell.reparse_markdown(cx);
                                    });
                                }
                                MarkdownCellEvent::Run(_cell_id) => {
                                    // run is handled separately by move_to_next_cell
                                    // Just reparse here
                                    cell.update(cx, |cell, cx| {
                                        cell.reparse_markdown(cx);
                                    });
                                }
                            }
                        },
                    )
                    .detach();

                    let cell_id_for_editor = cell_id.clone();
                    let editor = markdown_cell.read(cx).editor().clone();
                    cx.subscribe(&editor, move |this, _editor, event, cx| {
                        if let editor::EditorEvent::Focused = event {
                            this.select_cell_by_id(&cell_id_for_editor, cx);
                        }
                    })
                    .detach();
                }
                Cell::Raw(_) => {}
            }

            cell_map.insert(cell_id.clone(), cell_entity);
        }

        let notebook_handle = cx.entity().downgrade();
        let cell_count = cell_order.len();

        let this = cx.entity();
        let cell_list = ListState::new(cell_count, gpui::ListAlignment::Top, px(1000.));

        let (variables_visible, variables_position) = match JupyterSettings::get_global(cx)
            .notebook_variables_panel
        {
            settings::NotebookVariablesPanelContent::Right => (true, VariablesPanelPosition::Right),
            settings::NotebookVariablesPanelContent::Bottom => {
                (true, VariablesPanelPosition::Bottom)
            }
            settings::NotebookVariablesPanelContent::Hidden => {
                (false, VariablesPanelPosition::Right)
            }
        };
        let variables_panel = cx.new(|cx| {
            let mut panel = VariablesPanel::new();
            panel.set_position(variables_position, cx);
            panel.set_visible(variables_visible, cx);
            panel
        });
        cx.subscribe(&variables_panel, |this, _panel, event, cx| match event {
            VariablesPanelEvent::RefreshRequested => this.refresh_variables(cx),
            VariablesPanelEvent::OpenVariable { name } => {
                this.load_variable_page(name.clone(), 0, None, true, cx)
            }
            VariablesPanelEvent::LoadPage {
                variable,
                offset,
                sort_column,
                ascending,
            } => this.load_variable_page(
                variable.clone(),
                *offset,
                sort_column.clone(),
                *ascending,
                cx,
            ),
        })
        .detach();

        let notebook_path = {
            let project_path = notebook_item.read(cx).project_path.clone();
            project.read(cx).absolute_path(&project_path, cx)
        };

        let kernel_completer = NotebookKernelCompleter::new(cx.background_executor().clone());

        let mut editor = Self {
            project,
            languages: languages.clone(),
            worktree_id,
            focus_handle,
            notebook_item: notebook_item.clone(),
            notebook_path,
            agent_window: Some(window.window_handle()),
            external_change_subscription: None,
            focused_cell_id: None,
            registered_cell_buffers: Vec::new(),
            notebook_lsp: None,
            notebook_language,
            remote_id: None,
            cell_list,
            notebook_mode: NotebookMode::Command,
            selected_cell_index: 0,
            cell_order: cell_order.clone(),
            original_cell_order: cell_order.clone(),
            cell_map: cell_map.clone(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            kernel: Kernel::Shutdown,
            kernel_specification: None,
            deferred_kernel_adoption: None,
            parent_environment_search_started: false,
            kernel_generation: 0,
            execution_requests: HashMap::default(),
            kernel_completer,
            variables_panel,
            variables_bottom_height: None,
            variables_resize_start: None,
            variables_right_width: None,
            variables_width_resize_start: None,
            variable_requests: HashMap::default(),
            kernel_picker_handle: PopoverMenuHandle::default(),
            repl_store_subscription: None,
            workspace: None,
        };
        editor.launch_kernel(window, cx);
        editor.refresh_language(cx);
        editor.refresh_kernelspecs(window, cx);
        editor.refresh_cell_registry(cx);
        editor.register_agent_live_editing(cx);

        // Discovery is asynchronous and may still be in flight when this
        // notebook opens (the store de-duplicates it per worktree), so adoption
        // also has to run whenever the store finishes publishing environments.
        editor.repl_store_subscription = Some(cx.observe_in(
            &ReplStore::global(cx),
            window,
            |this, _store, window, cx| {
                this.adopt_project_environment_kernel(window, cx);
            },
        ));

        // The cell editors are built from the notebook on disk, so record them
        // as the saved baseline: without this their freshly created buffers
        // count as unsaved edits and every external change is dismissed.
        editor.mark_as_saved(cx);

        // Only this editor's registration is removed on close, so another
        // window presenting the same notebook keeps answering the agent.
        let registered_entity_id = cx.entity_id();
        cx.on_release(move |_editor, cx| {
            notebook_agent_tools::unregister_live(cx, registered_entity_id);
        })
        .detach();

        editor.observe_notebook_buffer(cx);

        cx.subscribe(&notebook_item, |this, _item, _event, cx| {
            this.refresh_language(cx);
        })
        .detach();

        editor
    }

    fn refresh_kernelspecs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let store = ReplStore::global(cx);
        let project = self.project.clone();
        let project_path = self.notebook_item.read(cx).project_path.clone();

        let refresh_task = store.update(cx, |store, cx| {
            store.refresh_python_kernelspecs(project_path, &project, cx)
        });

        // Python environments are discovered asynchronously, so the kernel that
        // `launch_kernel` picked at creation time cannot depend on them yet.
        // Once discovery finishes, adopt the notebook's nearest virtual
        // environment when the user has not chosen a kernel explicitly.
        cx.spawn_in(window, async move |this, cx| {
            if let Err(error) = refresh_task.await {
                log::warn!("notebook: could not discover python environments: {error:#}");
                return;
            }
            this.update_in(cx, |this, window, cx| {
                this.adopt_project_environment_kernel(window, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Picks the kernel a notebook should run after environment discovery.
    ///
    /// An explicit user selection always wins. Otherwise the Python environment
    /// inside the `.venv` closest to the notebook (walking up to the worktree
    /// root) wins, even over a working Conda base kernel. If there is no
    /// virtual environment, the previous recommendation/fallback behavior is
    /// kept unchanged.
    fn adopt_project_environment_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.project.read(cx).is_via_remote_server() {
            self.adopt_remote_kernel(window, cx);
            return;
        }
        let Some(notebook_path) = self.notebook_path.clone() else {
            return;
        };
        let Some(worktree_root) = self
            .project
            .read(cx)
            .worktree_for_id(self.worktree_id, cx)
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
        else {
            return;
        };

        let store = ReplStore::global(cx);
        let worktree_id = self.worktree_id;
        let explicit_selection_is_configured = JupyterSettings::try_get(cx)
            .is_some_and(|settings| settings.kernel_selections.contains_key("python"));
        if explicit_selection_is_configured {
            self.deferred_kernel_adoption = None;
            return;
        }

        // Environments inside the worktree are already discovered by the store.
        // Only when none of them covers the notebook do we need to look at the
        // notebook's parent directories, which the worktree scan cannot see.
        let nearest_venv_is_discovered = store.read_with(cx, |store, _| {
            let discovered = store
                .kernel_specifications_for_worktree(worktree_id)
                .cloned()
                .collect::<Vec<_>>();
            nearest_virtual_environment(&notebook_path, &worktree_root, &discovered).is_some()
        });

        let is_remote = self.project.read(cx).is_remote();
        if is_remote || nearest_venv_is_discovered || self.parent_environment_search_started {
            self.apply_project_environment_adoption(window, cx);
            return;
        }

        self.parent_environment_search_started = true;
        let fs = self.project.read(cx).fs().clone();
        cx.spawn_in(window, async move |this, cx| {
            let interpreter =
                nearest_virtual_environment_on_disk(fs.as_ref(), &notebook_path).await;
            if let Some(interpreter) = interpreter {
                let specification =
                    probe_python_environment_interpreter(interpreter, Some("venv".to_string()))
                        .await;
                let specification = KernelSpecification::PythonEnv(specification);
                store.update(cx, |store, cx| {
                    store.add_discovered_kernel_specification(worktree_id, specification, cx);
                });
            }
            this.update_in(cx, |this, window, cx| {
                this.apply_project_environment_adoption(window, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Starts the notebook's kernel on the remote machine once its Python
    /// environments are known. Nothing is launched before that, because the
    /// local fallback kernel cannot run in a project that lives elsewhere.
    fn adopt_remote_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(
            self.kernel_specification,
            Some(KernelSpecification::SshRemote(_))
        ) {
            return;
        }

        let store = ReplStore::global(cx);
        let worktree_id = self.worktree_id;
        let notebook_path = self.notebook_path.clone();
        let candidate = store.read_with(cx, |store, _| {
            notebook_path
                .as_deref()
                .and_then(|path| store.notebook_kernelspec(path))
                .or_else(|| store.selected_kernel(worktree_id))
                .or_else(|| {
                    store
                        .kernel_specifications_for_worktree(worktree_id)
                        .find(|specification| {
                            matches!(specification, KernelSpecification::SshRemote(_))
                        })
                })
                .cloned()
        });

        match candidate {
            Some(specification) => self.change_kernel(specification, window, cx),
            None => self.kernel_errored(
                "No Python environment with ipykernel was found on the remote machine. \
                 Install it there (for example `python -m pip install ipykernel` inside a \
                 .venv next to the notebook) and reopen the notebook."
                    .to_string(),
                cx,
            ),
        }
    }

    fn apply_project_environment_adoption(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(notebook_path) = self.notebook_path.clone() else {
            return;
        };
        let Some(worktree_root) = self
            .project
            .read(cx)
            .worktree_for_id(self.worktree_id, cx)
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
        else {
            return;
        };

        let store = ReplStore::global(cx);
        let worktree_id = self.worktree_id;
        let current = self.kernel_specification.clone();
        let explicit_selection_is_configured = JupyterSettings::try_get(cx)
            .is_some_and(|settings| settings.kernel_selections.contains_key("python"));

        let adoption = if explicit_selection_is_configured {
            KernelAdoption::Keep
        } else {
            store.read_with(cx, |store, _| {
                let discovered = store
                    .kernel_specifications_for_worktree(worktree_id)
                    .cloned()
                    .collect::<Vec<_>>();
                let recommended_path = store
                    .active_python_toolchain_path(worktree_id)
                    .map(|path| path.to_string());

                choose_kernel_for_notebook(
                    &notebook_path,
                    &worktree_root,
                    current.as_ref(),
                    store
                        .notebook_kernelspec(&notebook_path)
                        .or_else(|| store.selected_kernel(worktree_id)),
                    recommended_path.as_deref(),
                    &discovered,
                )
            })
        };

        match adoption {
            KernelAdoption::Keep => {
                self.deferred_kernel_adoption = None;
            }
            adoption if matches!(self.kernel.status(), KernelStatus::Busy) => {
                log::info!(
                    "notebook: deferring kernel adoption because the running kernel is busy; \
                     it will be applied once the kernel becomes idle"
                );
                self.deferred_kernel_adoption = Some(adoption);
            }
            adoption => self.apply_kernel_adoption(adoption, window, cx),
        }
    }

    fn apply_kernel_adoption(
        &mut self,
        adoption: KernelAdoption,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.deferred_kernel_adoption = None;

        match adoption {
            KernelAdoption::Keep => {}
            KernelAdoption::Switch(specification) => {
                log::info!(
                    "notebook: adopting python environment '{}' ({}) as the notebook kernel",
                    specification.name(),
                    specification.path()
                );
                self.change_kernel(specification, window, cx);
            }
            KernelAdoption::MissingIpykernel(environment) => {
                let message = format!(
                    "The Python environment at {} does not have ipykernel installed. \
                     Run `python -m pip install ipykernel` from inside that environment, \
                     then restart the kernel.",
                    environment.display()
                );
                log::warn!("notebook: {message}");
                self.kernel_errored(message, cx);
            }
        }
    }

    fn refresh_language(&mut self, cx: &mut Context<Self>) {
        let notebook_language = self.notebook_item.read(cx).notebook_language();
        let task = cx.spawn(async move |this, cx| {
            let language = notebook_language.await;
            if let Some(this) = this.upgrade() {
                this.update(cx, |this, cx| {
                    for cell in this.cell_map.values() {
                        if let Cell::Code(code_cell) = cell {
                            code_cell.update(cx, |cell, cx| {
                                cell.set_language(language.clone(), cx);
                            });
                        }
                    }
                    this.ensure_notebook_lsp(language.clone(), cx);
                });
            }
            language
        });
        self.notebook_language = task.shared();
    }

    fn ensure_notebook_lsp(&mut self, language: Option<Arc<Language>>, cx: &mut Context<Self>) {
        if self.notebook_lsp.is_some() {
            return;
        }
        let Some(notebook_path) = self.notebook_path.clone() else {
            return;
        };
        let project = self.project.clone();
        let notebook_lsp = cx.new(|cx| NotebookLsp::new(project, notebook_path, language, cx));
        notebook_lsp.update(cx, |notebook_lsp, _cx| {
            notebook_lsp.set_kernel_completer(self.kernel_completer.clone());
        });
        self.notebook_lsp = Some(notebook_lsp);
        self.sync_notebook_lsp(cx);
        if let Some(spec) = self.kernel_specification.clone() {
            self.sync_kernel_interpreter(&spec, cx);
        }
    }

    /// Publishes the current kernel's shell channel and idle status to the
    /// kernel completer, so cell completion and hover can reach the live
    /// interpreter. Called whenever the kernel process or its execution state
    /// changes.
    fn sync_kernel_completer(&self) {
        let (request_tx, idle) = match &self.kernel {
            Kernel::RunningKernel(kernel) => (
                Some(kernel.request_tx()),
                matches!(kernel.execution_state(), runtimelib::ExecutionState::Idle),
            ),
            _ => (None, false),
        };
        // `set_kernel` replaces the whole state; the idle flag above is already
        // the current one, so a process swap cannot leave it stale.
        self.kernel_completer.set_kernel(request_tx, idle);
    }

    /// Tells the notebook's language-server bridge which Python interpreter the
    /// active kernel runs, so the virtual document resolves imports against the
    /// same environment the cells execute in.
    fn sync_kernel_interpreter(&mut self, spec: &KernelSpecification, cx: &mut Context<Self>) {
        let Some(interpreter) = python_interpreter_for_spec(spec) else {
            return;
        };
        if let Some(notebook_lsp) = self.notebook_lsp.clone() {
            notebook_lsp.update(cx, |notebook_lsp, cx| {
                notebook_lsp.set_python_interpreter(interpreter, cx);
            });
        }
    }

    fn sync_notebook_lsp(&mut self, cx: &mut Context<Self>) {
        self.apply_workspace_to_cells(cx);
        let Some(notebook_lsp) = self.notebook_lsp.clone() else {
            return;
        };
        let cells: Vec<NotebookCell> = self
            .cell_order
            .iter()
            .filter_map(|cell_id| {
                let cell = self.cell_map.get(cell_id)?;
                let Cell::Code(code_cell) = cell else {
                    return None;
                };
                Some(NotebookCell {
                    id: cell_id.clone(),
                    editor: code_cell.read(cx).editor().clone(),
                    buffer: cell.buffer(cx)?,
                })
            })
            .collect();
        notebook_lsp.update(cx, |notebook_lsp, cx| notebook_lsp.set_cells(cells, cx));
    }

    /// Hands the hosting workspace to every code cell editor, including cells
    /// created after the notebook was opened. Rename and formatting only need the
    /// project, but references and code actions open UI in the workspace.
    fn apply_workspace_to_cells(&self, cx: &mut Context<Self>) {
        let Some((workspace, workspace_id)) = self.workspace.clone() else {
            return;
        };
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                let editor = code_cell.read(cx).editor().clone();
                editor.update(cx, |editor, _cx| {
                    editor.set_workspace(workspace.clone(), workspace_id);
                });
            }
        }
    }

    fn has_structural_changes(&self) -> bool {
        self.cell_order != self.original_cell_order
    }

    fn has_content_changes(&self, cx: &App) -> bool {
        self.cell_map.values().any(|cell| cell.is_dirty(cx))
    }

    pub fn to_notebook(&self, cx: &App) -> nbformat::v4::Notebook {
        let cells: Vec<nbformat::v4::Cell> = self
            .cell_order
            .iter()
            .filter_map(|cell_id| {
                self.cell_map
                    .get(cell_id)
                    .map(|cell| cell.to_nbformat_cell(cx))
            })
            .collect();

        let metadata = self.notebook_item.read(cx).notebook.metadata.clone();

        nbformat::v4::Notebook {
            metadata,
            nbformat: 4,
            nbformat_minor: 5,
            cells,
        }
    }

    pub fn mark_as_saved(&mut self, cx: &mut Context<Self>) {
        self.original_cell_order = self.cell_order.clone();

        for cell in self.cell_map.values() {
            match cell {
                Cell::Code(code_cell) => {
                    code_cell.update(cx, |code_cell, cx| {
                        let editor = code_cell.editor();
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                if let Some(buf) = buffer.as_singleton() {
                                    buf.update(cx, |b, cx| {
                                        let version = b.version();
                                        b.did_save(version, None, cx);
                                    });
                                }
                            });
                        });
                    });
                }
                Cell::Markdown(markdown_cell) => {
                    markdown_cell.update(cx, |markdown_cell, cx| {
                        let editor = markdown_cell.editor();
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                if let Some(buf) = buffer.as_singleton() {
                                    buf.update(cx, |b, cx| {
                                        let version = b.version();
                                        b.did_save(version, None, cx);
                                    });
                                }
                            });
                        });
                    });
                }
                Cell::Raw(_) => {}
            }
        }
        cx.notify();
    }

    fn save_impl(
        &mut self,
        destination: SaveDestination,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook = self.to_notebook(cx);
        let project_path = self.notebook_item.read(cx).project_path.clone();

        self.mark_as_saved(cx);

        cx.spawn(async move |this, cx| {
            let json =
                serde_json::to_string_pretty(&notebook).context("Failed to serialize notebook")?;
            let buffer = project
                .update(cx, |project, cx| project.open_buffer(project_path, cx))
                .await?;
            buffer.update(cx, |buffer, cx| buffer.set_text(json, cx));

            match destination {
                SaveDestination::CurrentPath => {
                    project
                        .update(cx, |project, cx| project.save_buffer(buffer, cx))
                        .await
                }
                SaveDestination::NewPath(new_path) => {
                    project
                        .update(cx, |project, cx| {
                            project.save_buffer_as(buffer, new_path.clone(), cx)
                        })
                        .await?;

                    // The buffer now lives at the new path, so the notebook has
                    // to follow it or the next save writes to the old file.
                    let entry_id = project.read_with(cx, |project, cx| {
                        project.entry_for_path(&new_path, cx).map(|entry| entry.id)
                    });
                    this.update(cx, |this, cx| {
                        this.notebook_item.update(cx, |notebook_item, _| {
                            notebook_item.project_path = new_path;
                            if let Some(entry_id) = entry_id {
                                notebook_item.id = entry_id;
                            }
                        });
                        let project_path = this.notebook_item.read(cx).project_path.clone();
                        this.notebook_path = this.project.read(cx).absolute_path(&project_path, cx);
                        let entity_id = cx.entity().entity_id();
                        notebook_agent_tools::unregister_live(cx, entity_id);
                        this.register_agent_live_editing(cx);
                        // The notebook now lives at a new path, so the external
                        // change watcher has to follow it.
                        this.observe_notebook_buffer(cx);
                    })
                }
            }
        })
    }

    fn launch_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let spec = self.kernel_specification.clone().or_else(|| {
            ReplStore::global(cx)
                .read(cx)
                .active_kernelspec(self.worktree_id, None, cx)
        });

        if spec.is_none() && self.project.read(cx).is_via_remote_server() {
            // Environment discovery on the remote machine starts the kernel.
            return;
        }

        let spec = spec.unwrap_or_else(|| {
            KernelSpecification::Jupyter(LocalKernelSpecification {
                name: "python3".to_string(),
                path: PathBuf::from("python3"),
                kernelspec: JupyterKernelspec {
                    argv: vec![
                        "python3".to_string(),
                        "-m".to_string(),
                        "ipykernel_launcher".to_string(),
                        "-f".to_string(),
                        "{connection_file}".to_string(),
                    ],
                    display_name: "Python 3".to_string(),
                    language: "python".to_string(),
                    interrupt_mode: None,
                    metadata: None,
                    env: None,
                },
            })
        });

        self.launch_kernel_with_spec(spec, window, cx);
    }

    fn launch_kernel_with_spec(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let entity_id = cx.entity_id();
        let working_directory = {
            let worktree = self
                .project
                .read(cx)
                .worktree_for_id(self.worktree_id, cx)
                .map(|worktree| {
                    let worktree = worktree.read(cx);
                    (worktree.abs_path().to_path_buf(), worktree.is_single_file())
                });
            kernel_working_directory(
                self.notebook_path.as_deref(),
                worktree.as_ref().map(|(path, _)| path.as_path()),
                worktree
                    .as_ref()
                    .is_some_and(|(_, is_single_file)| *is_single_file),
            )
        };
        let fs = self.project.read(cx).fs().clone();
        let view = cx.entity();

        self.kernel_specification = Some(spec.clone());
        self.sync_kernel_interpreter(&spec, cx);

        self.notebook_item.update(cx, |item, cx| {
            let kernel_name = spec.name().to_string();
            let language = spec.language().to_string();

            let display_name = match &spec {
                KernelSpecification::Jupyter(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::PythonEnv(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::JupyterServer(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::SshRemote(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::WslRemote(s) => s.kernelspec.display_name.clone(),
            };

            let kernelspec_json = serde_json::json!({
                "display_name": display_name,
                "name": kernel_name,
                "language": language
            });

            if let Ok(k) = serde_json::from_value(kernelspec_json) {
                item.notebook.metadata.kernelspec = Some(k);
                cx.emit(());
            }
        });

        let kernel_task = match spec {
            KernelSpecification::Jupyter(local_spec) => NativeRunningKernel::new(
                local_spec,
                entity_id,
                working_directory,
                fs,
                view,
                window,
                cx,
            ),
            KernelSpecification::PythonEnv(env_spec) => NativeRunningKernel::new(
                env_spec.as_local_spec(),
                entity_id,
                working_directory,
                fs,
                view,
                window,
                cx,
            ),
            KernelSpecification::JupyterServer(remote_spec) => {
                RemoteRunningKernel::new(remote_spec, working_directory, view, window, cx)
            }

            KernelSpecification::SshRemote(spec) => {
                let project = self.project.clone();
                SshRunningKernel::new(spec, working_directory, project, view, window, cx)
            }
            KernelSpecification::WslRemote(spec) => {
                WslRunningKernel::new(spec, entity_id, working_directory, fs, view, window, cx)
            }
        };

        // A later `launch_kernel_with_spec` may supersede this launch while it is
        // still starting (for example when a discovered venv replaces the
        // fallback kernel). Stamp each launch so a stale task cannot overwrite
        // the kernel that the user is now running.
        self.kernel_generation += 1;
        let generation = self.kernel_generation;

        let pending_kernel = cx
            .spawn(async move |this, cx| {
                let kernel = kernel_task.await;

                match kernel {
                    Ok(mut kernel) => {
                        this.update(cx, |editor, cx| {
                            if editor.kernel_generation != generation {
                                kernel.kill();
                                return;
                            }
                            editor.kernel = Kernel::RunningKernel(kernel);
                            editor.sync_kernel_completer();
                            cx.notify();
                        })
                        .ok();
                    }
                    Err(err) => {
                        log::error!("Kernel failed to start: {:?}", err);
                        this.update(cx, |editor, cx| {
                            if editor.kernel_generation != generation {
                                return;
                            }
                            editor.kernel = Kernel::ErroredLaunch(err.to_string());
                            editor.sync_kernel_completer();
                            cx.notify();
                        })
                        .ok();
                    }
                }
            })
            .shared();

        self.kernel = Kernel::StartingKernel(pending_kernel);
        self.sync_kernel_completer();
        cx.notify();
    }

    // Note: Python environments are only detected as kernels if ipykernel is installed.
    // Users need to run `pip install ipykernel` (or `uv pip install ipykernel`) in their
    // virtual environment for it to appear in the kernel selector.
    // This happens because we have an ipykernel check inside the function python_env_kernel_specification in mod.rs L:121

    fn change_kernel(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Kernel::RunningKernel(kernel) = &mut self.kernel {
            kernel.force_shutdown(window, cx).detach();
        }

        self.execution_requests.clear();
        self.variable_requests.clear();

        self.launch_kernel_with_spec(spec, window, cx);
    }

    /// Applies a kernel the user picked explicitly. Recording it in the store
    /// makes it survive reopening this notebook and stops environment
    /// discovery from overriding it.
    fn select_kernel(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.deferred_kernel_adoption = None;
        let store = ReplStore::global(cx);
        let worktree_id = self.worktree_id;
        let notebook_path = self.notebook_path.clone();
        store.update(cx, |store, cx| match notebook_path {
            Some(notebook_path) => store.set_notebook_kernelspec(notebook_path, spec.clone()),
            None => store.set_active_kernelspec(worktree_id, spec.clone(), cx),
        });
        self.change_kernel(spec, window, cx);
    }

    fn restart_kernel(&mut self, _: &RestartKernel, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(spec) = self.kernel_specification.clone() {
            if let Kernel::RunningKernel(kernel) = &mut self.kernel {
                kernel.force_shutdown(window, cx).detach();
            }

            self.kernel = Kernel::Restarting;
            cx.notify();

            self.launch_kernel_with_spec(spec, window, cx);
        }
    }

    fn interrupt_kernel(
        &mut self,
        _: &InterruptKernel,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Kernel::RunningKernel(kernel) = &self.kernel {
            let interrupt_request = runtimelib::InterruptRequest {};
            let message: JupyterMessage = interrupt_request.into();
            kernel.request_tx().try_send(message).ok();
            cx.notify();
        }
    }

    fn execute_cell(&mut self, cell_id: CellId, window: &mut Window, cx: &mut Context<Self>) {
        let code = if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
            let editor = cell.read(cx).editor().clone();
            let buffer = editor.read(cx).buffer().read(cx);
            buffer
                .as_singleton()
                .map(|b| b.read(cx).text())
                .unwrap_or_default()
        } else {
            return;
        };

        let request = ExecuteRequest {
            code,
            ..Default::default()
        };
        let message: JupyterMessage = request.into();
        let msg_id = message.header.msg_id.clone();

        let send_result = match &mut self.kernel {
            Kernel::RunningKernel(kernel) => kernel
                .request_tx()
                .try_send(message)
                .map_err(|err| format!("failed to send execute request to kernel (the kernel process may have died): {err}")),
            Kernel::StartingKernel(_) => Err("the kernel is still starting".to_string()),
            Kernel::ErroredLaunch(error) => Err(format!("the kernel failed to launch: {error}")),
            Kernel::ShuttingDown | Kernel::Shutdown => Err("the kernel is shut down".to_string()),
            Kernel::Restarting => Err("the kernel is restarting".to_string()),
        };

        if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
            cell.update(cx, |cell, cx| {
                if cell.has_outputs() {
                    cell.clear_outputs();
                }
                if let Err(error) = &send_result {
                    cell.show_kernel_error(error, window, cx);
                } else {
                    cell.start_execution();
                }
                cx.notify();
            });
        }

        if let Err(error) = send_result {
            log::error!("notebook: cannot execute cell: {error}");
        } else {
            self.execution_requests.insert(
                msg_id,
                PendingExecution {
                    cell_id: cell_id.clone(),
                    reply_received: false,
                    idle_received: false,
                },
            );
        }
    }

    fn kernel_language(&self) -> Option<String> {
        self.kernel_specification
            .as_ref()
            .map(|specification| specification.language().to_string())
            .or_else(|| match &self.kernel {
                Kernel::RunningKernel(kernel) => kernel
                    .kernel_info()
                    .map(|info| info.language_info.name.clone()),
                _ => None,
            })
    }

    fn toggle_variables_panel(&mut self, cx: &mut Context<Self>) {
        let visible = self
            .variables_panel
            .update(cx, |panel, cx| panel.toggle_visibility(cx));
        if visible {
            self.refresh_variables(cx);
        }
        cx.notify();
    }

    fn toggle_variables_panel_position(&mut self, cx: &mut Context<Self>) {
        self.variables_panel
            .update(cx, |panel, cx| panel.toggle_position(cx));
        cx.notify();
    }

    fn effective_variables_bottom_height(&self, viewport_height: Pixels) -> Pixels {
        let height = self
            .variables_bottom_height
            .unwrap_or(viewport_height * 0.3);
        px(clamp_variables_bottom_height(
            f32::from(height),
            f32::from(viewport_height),
        ))
    }

    fn effective_variables_right_width(&self, viewport_width: Pixels) -> Pixels {
        let width = self.variables_right_width.unwrap_or(px(280.));
        px(clamp_variables_right_width(
            f32::from(width),
            f32::from(viewport_width),
        ))
    }

    fn render_variables_divider(&self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let view = cx.entity();
        let viewport_height = window.viewport_size().height;

        div()
            .id("notebook-variables-divider")
            .h(px(6.))
            .w_full()
            .flex_none()
            .cursor_row_resize()
            .hover(|this| this.bg(cx.theme().colors().border))
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, cx| {
                        window.on_mouse_event({
                            let view = view.clone();
                            move |event: &MouseDownEvent, phase, _, cx| {
                                if !phase.bubble()
                                    || event.button != MouseButton::Left
                                    || !bounds.contains(&event.position)
                                {
                                    return;
                                }

                                view.update(cx, |this, cx| {
                                    this.variables_resize_start = Some(VariablesResizeStart {
                                        mouse_y: event.position.y,
                                        height: this
                                            .effective_variables_bottom_height(viewport_height),
                                    });
                                    cx.notify();
                                });
                                cx.stop_propagation();
                            }
                        });
                        window.on_mouse_event({
                            let view = view.clone();
                            move |event: &MouseMoveEvent, phase, _, cx| {
                                if !phase.bubble() || !event.dragging() {
                                    return;
                                }

                                let Some(start) = view.read(cx).variables_resize_start else {
                                    return;
                                };
                                let height = px(clamp_variables_bottom_height(
                                    f32::from(start.height + (start.mouse_y - event.position.y)),
                                    f32::from(viewport_height),
                                ));
                                view.update(cx, |this, cx| {
                                    if this.variables_bottom_height != Some(height) {
                                        this.variables_bottom_height = Some(height);
                                        cx.notify();
                                    }
                                });
                            }
                        });
                        window.on_mouse_event({
                            let view = view.clone();
                            move |event: &MouseUpEvent, _, _, cx| {
                                if event.button != MouseButton::Left {
                                    return;
                                }
                                view.update(cx, |this, cx| {
                                    if this.variables_resize_start.take().is_some() {
                                        cx.notify();
                                    }
                                });
                            }
                        });
                    },
                )
                .size_full(),
            )
            .into_any_element()
    }

    fn render_variables_right_divider(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let view = cx.entity();
        let viewport_width = window.viewport_size().width;

        div()
            .id("notebook-variables-right-divider")
            .w(px(6.))
            .h_full()
            .flex_none()
            .cursor_col_resize()
            .hover(|this| this.bg(cx.theme().colors().border))
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, cx| {
                        window.on_mouse_event({
                            let view = view.clone();
                            move |event: &MouseDownEvent, phase, _, cx| {
                                if !phase.bubble()
                                    || event.button != MouseButton::Left
                                    || !bounds.contains(&event.position)
                                {
                                    return;
                                }

                                view.update(cx, |this, cx| {
                                    this.variables_width_resize_start =
                                        Some(VariablesWidthResizeStart {
                                            mouse_x: event.position.x,
                                            width: this
                                                .effective_variables_right_width(viewport_width),
                                        });
                                    cx.notify();
                                });
                                cx.stop_propagation();
                            }
                        });
                        window.on_mouse_event({
                            let view = view.clone();
                            move |event: &MouseMoveEvent, phase, _, cx| {
                                if !phase.bubble() || !event.dragging() {
                                    return;
                                }

                                let Some(start) = view.read(cx).variables_width_resize_start else {
                                    return;
                                };
                                let width = px(clamp_variables_right_width(
                                    f32::from(start.width + (start.mouse_x - event.position.x)),
                                    f32::from(viewport_width),
                                ));
                                view.update(cx, |this, cx| {
                                    if this.variables_right_width != Some(width) {
                                        this.variables_right_width = Some(width);
                                        cx.notify();
                                    }
                                });
                            }
                        });
                        window.on_mouse_event({
                            let view = view.clone();
                            move |event: &MouseUpEvent, _, _, cx| {
                                if event.button != MouseButton::Left {
                                    return;
                                }
                                view.update(cx, |this, cx| {
                                    if this.variables_width_resize_start.take().is_some() {
                                        cx.notify();
                                    }
                                });
                            }
                        });
                    },
                )
                .size_full(),
            )
            .into_any_element()
    }

    fn refresh_variables(&mut self, cx: &mut Context<Self>) {
        if !self.variables_panel.read(cx).is_visible() {
            return;
        }

        match self.kernel_language() {
            Some(language) if is_python_language(&language) => {}
            Some(language) => {
                self.variables_panel.update(cx, |panel, cx| {
                    panel.set_message(
                        format!(
                            "Variables are only available for Python kernels (current kernel: {language})."
                        ),
                        cx,
                    )
                });
                return;
            }
            None => {
                self.variables_panel.update(cx, |panel, cx| {
                    panel.set_message("Select a Python kernel to inspect variables.", cx)
                });
                return;
            }
        }

        if !matches!(self.kernel, Kernel::RunningKernel(_)) {
            self.variables_panel.update(cx, |panel, cx| {
                panel.set_message("The kernel is not running.", cx)
            });
            return;
        }

        self.variables_panel
            .update(cx, |panel, cx| panel.begin_loading(cx));
        self.send_variable_request(
            VariableRequest::Introspection,
            VARIABLES_INTROSPECTION_CODE.to_string(),
            VARIABLES_INTROSPECTION_EXPRESSION,
            cx,
        );
    }

    fn load_variable_page(
        &mut self,
        variable: String,
        offset: usize,
        sort_column: Option<String>,
        ascending: bool,
        cx: &mut Context<Self>,
    ) {
        if !matches!(self.kernel, Kernel::RunningKernel(_)) {
            self.variables_panel.update(cx, |panel, cx| {
                panel.set_data_error(&variable, "The kernel is not running.", cx)
            });
            return;
        }

        let request = DataPageRequest {
            variable: variable.clone(),
            offset,
            limit: VARIABLES_PAGE_SIZE,
            sort_column,
            ascending,
        };
        let code = match data_page_code(&request) {
            Ok(code) => code,
            Err(error) => {
                log::warn!("notebook_variables: could not build the data page request: {error}");
                self.variables_panel.update(cx, |panel, cx| {
                    panel.set_data_error(
                        &variable,
                        format!("Could not build the request: {error}"),
                        cx,
                    )
                });
                return;
            }
        };

        self.send_variable_request(
            VariableRequest::DataPage {
                variable: variable.clone(),
            },
            code,
            DATA_PAGE_EXPRESSION,
            cx,
        );
    }

    fn send_variable_request(
        &mut self,
        request: VariableRequest,
        code: String,
        expression: &str,
        cx: &mut Context<Self>,
    ) {
        let user_expressions =
            std::collections::HashMap::from([("payload".to_string(), expression.to_string())]);
        let execute_request = ExecuteRequest {
            code,
            silent: true,
            store_history: false,
            user_expressions: Some(user_expressions),
            ..Default::default()
        };
        let message: JupyterMessage = execute_request.into();
        let message_id = message.header.msg_id.clone();

        let send_result = match &mut self.kernel {
            Kernel::RunningKernel(kernel) => {
                kernel.request_tx().try_send(message).map_err(|error| {
                    format!("failed to send the variables request to the kernel: {error}")
                })
            }
            _ => Err("the kernel is not running".to_string()),
        };

        if let Err(error) = send_result {
            log::warn!("notebook_variables: {error}");
            match request {
                VariableRequest::Introspection => self.variables_panel.update(cx, |panel, cx| {
                    panel.set_error(error, cx);
                }),
                VariableRequest::DataPage { variable } => {
                    self.variables_panel.update(cx, |panel, cx| {
                        panel.set_data_error(&variable, error, cx);
                    })
                }
            }
            return;
        }

        self.variable_requests.insert(message_id.clone(), request);
        self.watch_variable_request(message_id, cx);
    }

    fn watch_variable_request(&mut self, message_id: String, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(VARIABLES_REQUEST_TIMEOUT)
                .await;
            if let Err(error) = this.update(cx, |this, cx| {
                let Some(request) = this.variable_requests.remove(&message_id) else {
                    return;
                };
                log::warn!("notebook_variables: request timed out");
                let message = format!(
                    "Timed out after {} seconds waiting for the kernel.",
                    VARIABLES_REQUEST_TIMEOUT.as_secs()
                );
                match request {
                    VariableRequest::Introspection => {
                        this.variables_panel.update(cx, |panel, cx| {
                            panel.set_error(message, cx);
                        });
                    }
                    VariableRequest::DataPage { variable } => {
                        this.variables_panel.update(cx, |panel, cx| {
                            panel.set_data_error(&variable, message, cx);
                        });
                    }
                }
                cx.notify();
            }) {
                log::warn!("notebook_variables: could not report the request timeout: {error}");
            }
        })
        .detach();
    }

    fn handle_variable_reply(
        &mut self,
        request: VariableRequest,
        reply: &ExecuteReply,
        cx: &mut Context<Self>,
    ) {
        let result = variable_reply_text(reply);
        match request {
            VariableRequest::Introspection => {
                match result.and_then(|text| parse_variables_json(&text)) {
                    Ok(variables) => {
                        self.variables_panel
                            .update(cx, |panel, cx| panel.set_variables(variables, cx));
                    }
                    Err(error) => {
                        log::warn!("notebook_variables: could not read variables: {error}");
                        self.variables_panel.update(cx, |panel, cx| {
                            panel.set_error(format!("Could not read variables: {error}"), cx)
                        });
                    }
                }
            }
            VariableRequest::DataPage { variable } => match result
                .and_then(|text| parse_data_page_json(&text))
            {
                Ok(page) => {
                    self.variables_panel
                        .update(cx, |panel, cx| panel.set_data_page(&variable, page, cx));
                }
                Err(error) => {
                    log::warn!("notebook_variables: could not load data for {variable}: {error}");
                    self.variables_panel.update(cx, |panel, cx| {
                        panel.set_data_error(
                            &variable,
                            format!("Could not load {variable}: {error}"),
                            cx,
                        )
                    });
                }
            },
        }
    }

    fn get_selected_cell(&self) -> Option<&Cell> {
        self.cell_order
            .get(self.selected_cell_index)
            .and_then(|cell_id| self.cell_map.get(cell_id))
    }

    fn focused_cell(&self) -> Option<&Cell> {
        let cell_id = self.focused_cell_id.as_ref()?;
        self.cell_map.get(cell_id)
    }

    fn capture_scroll_anchor(&self) -> Option<ScrollAnchor> {
        let last_index = self.cell_order.len().checked_sub(1)?;
        let scroll_top = self.cell_list.logical_scroll_top();
        let cell_id = self
            .cell_order
            .get(scroll_top.item_ix.min(last_index))?
            .clone();
        Some(ScrollAnchor {
            cell_id,
            offset_in_item: scroll_top.offset_in_item,
        })
    }

    fn restore_scroll_anchor(
        &self,
        anchor: Option<ScrollAnchor>,
        previous_cell_order: &[CellId],
    ) {
        let Some(last_index) = self.cell_order.len().checked_sub(1) else {
            return;
        };
        let Some(anchor) = anchor else {
            return;
        };

        let item_ix = self
            .cell_order
            .iter()
            .position(|cell_id| cell_id == &anchor.cell_id)
            .or_else(|| {
                nearest_surviving_cell_index(
                    &anchor.cell_id,
                    previous_cell_order,
                    &self.cell_order,
                )
            })
            .unwrap_or(0)
            .min(last_index);

        self.cell_list.scroll_to(ListOffset {
            item_ix,
            offset_in_item: anchor.offset_in_item,
        });
    }

    fn nearest_cell_index(
        cell_id: &CellId,
        previous_cell_order: &[CellId],
        new_cell_order: &[CellId],
    ) -> Option<usize> {
        new_cell_order
            .iter()
            .position(|candidate| candidate == cell_id)
            .or_else(|| nearest_surviving_cell_index(cell_id, previous_cell_order, new_cell_order))
    }

    fn focus_cell_editor(
        &self,
        cell_id: &CellId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(cell) = self.cell_map.get(cell_id).cloned() else {
            return;
        };
        match cell {
            Cell::Code(code_cell) => {
                let editor = code_cell.read(cx).editor().clone();
                window.focus(&editor.focus_handle(cx), cx);
            }
            Cell::Markdown(markdown_cell) => {
                markdown_cell.update(cx, |markdown_cell, cx| {
                    if !markdown_cell.is_editing() {
                        markdown_cell.set_editing(true);
                        cx.notify();
                    }
                });
                let editor = markdown_cell.read(cx).editor().clone();
                window.focus(&editor.focus_handle(cx), cx);
            }
            Cell::Raw(_) => {}
        }
    }

    fn has_outputs(&self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                code_cell.read(cx).has_outputs()
            } else {
                false
            }
        })
    }

    fn clear_outputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, cx| {
                    cell.clear_outputs();
                    cx.notify();
                });
            }
        }
        cx.notify();
    }

    fn run_cells(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for cell_id in self.cell_order.clone() {
            self.execute_cell(cell_id, window, cx);
        }
    }

    fn run_current_cell(&mut self, _: &Run, window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id) else {
            return;
        };
        match cell {
            Cell::Code(_) => {
                self.execute_cell(cell_id, window, cx);
            }
            Cell::Markdown(markdown_cell) => {
                // for markdown, finish editing and move to next cell
                let is_editing = markdown_cell.read(cx).is_editing();
                if is_editing {
                    markdown_cell.update(cx, |cell, cx| {
                        cell.run(cx);
                    });
                    self.enter_command_mode(window, cx);
                }
            }
            Cell::Raw(_) => {}
        }
    }

    fn run_and_advance(&mut self, _: &RunAndAdvance, window: &mut Window, cx: &mut Context<Self>) {
        let mut create_markdown_cell = false;
        if let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() {
            if let Some(cell) = self.cell_map.get(&cell_id) {
                match cell {
                    Cell::Code(_) => {
                        self.execute_cell(cell_id, window, cx);
                    }
                    Cell::Markdown(markdown_cell) => {
                        create_markdown_cell = true;
                        if markdown_cell.read(cx).is_editing() {
                            markdown_cell.update(cx, |cell, cx| {
                                cell.run(cx);
                            });
                        }
                    }
                    Cell::Raw(_) => {}
                }
            }
        }

        let is_last_cell = self.selected_cell_index == self.cell_count().saturating_sub(1);
        if is_last_cell {
            if create_markdown_cell {
                self.add_markdown_block(window, cx);
            } else {
                self.add_code_block(window, cx);
            }
            self.enter_command_mode(window, cx);
        } else {
            self.advance_in_command_mode(window, cx);
        }
    }

    fn enter_edit_mode(&mut self, _: &EnterEditMode, window: &mut Window, cx: &mut Context<Self>) {
        self.notebook_mode = NotebookMode::Edit;
        if let Some(cell_id) = self.cell_order.get(self.selected_cell_index) {
            if let Some(cell) = self.cell_map.get(cell_id) {
                match cell {
                    Cell::Code(code_cell) => {
                        let editor = code_cell.read(cx).editor().clone();
                        window.focus(&editor.focus_handle(cx), cx);
                    }
                    Cell::Markdown(markdown_cell) => {
                        markdown_cell.update(cx, |cell, cx| {
                            cell.set_editing(true);
                            cx.notify();
                        });
                        let editor = markdown_cell.read(cx).editor().clone();
                        window.focus(&editor.focus_handle(cx), cx);
                    }
                    Cell::Raw(_) => {}
                }
            }
        }
        cx.notify();
    }

    fn enter_command_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    fn handle_enter_command_mode(
        &mut self,
        _: &EnterCommandMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.enter_command_mode(window, cx);
    }

    /// Advances to the next cell while staying in command mode (used by RunAndAdvance and shift-enter).
    fn advance_in_command_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let count = self.cell_count();
        if count == 0 {
            return;
        }
        if self.selected_cell_index < count - 1 {
            self.selected_cell_index += 1;
        }
        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    // Discussion can be done on this default implementation
    /// Moves focus to the next cell editor (used when already in edit mode).
    fn move_to_next_cell(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.cell_order.is_empty() && self.selected_cell_index < self.cell_order.len() - 1 {
            self.selected_cell_index += 1;
            // focus the new cell's editor
            if let Some(cell_id) = self.cell_order.get(self.selected_cell_index) {
                if let Some(cell) = self.cell_map.get(cell_id) {
                    match cell {
                        Cell::Code(code_cell) => {
                            let editor = code_cell.read(cx).editor();
                            window.focus(&editor.focus_handle(cx), cx);
                        }
                        Cell::Markdown(markdown_cell) => {
                            // Don't auto-enter edit mode for next markdown cell
                            // Just select it
                        }
                        Cell::Raw(_) => {}
                    }
                }
            }
            cx.notify();
        } else {
            // in the end, could optionally create a new cell
            // For now, just stay on the current cell
        }
    }

    fn open_notebook(&mut self, _: &OpenNotebook, _window: &mut Window, _cx: &mut Context<Self>) {
        println!("Open notebook triggered");
    }

    fn move_cell_up(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_cell_index > 0 {
            self.push_undo_snapshot();
            self.cell_order
                .swap(self.selected_cell_index, self.selected_cell_index - 1);
            self.selected_cell_index -= 1;
            cx.notify();
        }
        self.refresh_cell_registry(cx);
    }

    fn move_cell_down(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.cell_order.is_empty() && self.selected_cell_index < self.cell_order.len() - 1 {
            self.push_undo_snapshot();
            self.cell_order
                .swap(self.selected_cell_index, self.selected_cell_index + 1);
            self.selected_cell_index += 1;
            cx.notify();
        }
        self.refresh_cell_registry(cx);
    }

    fn delete_cell(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.cell_order.is_empty() {
            return;
        }
        let index = self.selected_cell_index.min(self.cell_order.len() - 1);
        self.push_undo_snapshot();
        let cell_id = self.cell_order.remove(index);
        self.cell_map.remove(&cell_id);
        self.cell_list.splice(index..index + 1, 0);

        if self.cell_order.is_empty() {
            self.selected_cell_index = 0;
        } else {
            self.selected_cell_index = index.min(self.cell_order.len() - 1);
            self.cell_list
                .scroll_to_reveal_item(self.selected_cell_index);
        }
        self.notebook_mode = NotebookMode::Command;
        window.focus(&self.focus_handle, cx);
        cx.notify();
        self.refresh_cell_registry(cx);
    }

    fn insert_cell_at_current_position(&mut self, cell_id: CellId, cell: Cell) {
        let insert_index = if self.cell_order.is_empty() {
            0
        } else {
            self.selected_cell_index + 1
        };
        self.cell_order.insert(insert_index, cell_id.clone());
        self.cell_map.insert(cell_id, cell);
        self.selected_cell_index = insert_index;
        self.cell_list.splice(insert_index..insert_index, 1);
        self.cell_list.scroll_to_reveal_item(insert_index);
    }

    fn snapshot_structure(&self) -> NotebookStructure {
        NotebookStructure {
            cell_order: self.cell_order.clone(),
            cell_map: self.cell_map.clone(),
            selected_cell_index: self.selected_cell_index,
        }
    }

    fn push_undo_snapshot(&mut self) {
        const MAX_UNDO_STEPS: usize = 100;
        self.undo_stack.push(self.snapshot_structure());
        if self.undo_stack.len() > MAX_UNDO_STEPS {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    fn restore_structure(
        &mut self,
        structure: NotebookStructure,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let scroll_anchor = self.capture_scroll_anchor();
        let previously_focused_cell_id = focused_cell_editor_in_window(self, window, cx);
        let previous_cell_order = std::mem::take(&mut self.cell_order);

        self.cell_order = structure.cell_order;
        self.cell_map = structure.cell_map;
        self.selected_cell_index = structure
            .selected_cell_index
            .min(self.cell_order.len().saturating_sub(1));
        self.cell_list.reset(self.cell_order.len());
        self.restore_scroll_anchor(scroll_anchor, &previous_cell_order);

        let focus_cell_id = previously_focused_cell_id.and_then(|cell_id| {
            if self.cell_order.contains(&cell_id) {
                Some(cell_id)
            } else {
                self.cell_order.get(self.selected_cell_index).cloned()
            }
        });
        if let Some(cell_id) = focus_cell_id {
            self.focused_cell_id = Some(cell_id.clone());
            self.notebook_mode = NotebookMode::Edit;
            self.focus_cell_editor(&cell_id, window, cx);
        } else {
            self.focused_cell_id = None;
            self.notebook_mode = NotebookMode::Command;
            window.focus(&self.focus_handle, cx);
        }
        self.refresh_cell_registry(cx);
        cx.notify();
    }

    fn undo_structure(&mut self, _: &Undo, window: &mut Window, cx: &mut Context<Self>) {
        let Some(previous) = self.undo_stack.pop() else {
            return;
        };
        let current = self.snapshot_structure();
        self.redo_stack.push(current);
        self.restore_structure(previous, window, cx);
    }

    fn redo_structure(&mut self, _: &Redo, window: &mut Window, cx: &mut Context<Self>) {
        let Some(next) = self.redo_stack.pop() else {
            return;
        };
        let current = self.snapshot_structure();
        self.undo_stack.push(current);
        self.restore_structure(next, window, cx);
    }

    fn build_cell(
        &mut self,
        cell: nbformat::v4::Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Cell {
        let cell_id = cell.id().clone();
        let loaded = Cell::load(
            &cell,
            &self.languages,
            self.notebook_language.clone(),
            &self.project,
            window,
            cx,
        );
        match &loaded {
            Cell::Code(code_cell) => {
                self.attach_code_cell(cell_id, code_cell, window, cx);
            }
            Cell::Markdown(markdown_cell) => {
                self.attach_markdown_cell(cell_id, markdown_cell, cx);
            }
            Cell::Raw(_) => {}
        }
        loaded
    }

    fn cursor_offset_in_cell(cell: &Cell, cx: &mut App) -> Option<usize> {
        let editor = cell.editor(cx)?.clone();
        editor.update(cx, |editor, cx| {
            let display_snapshot = editor.display_snapshot(cx);
            let point = editor
                .selections
                .newest::<language::Point>(&display_snapshot)
                .head();
            let buffer = editor.buffer().read(cx);
            let snapshot = buffer.snapshot(cx);
            Some(snapshot.point_to_offset(point).0)
        })
    }

    fn split_cell(&mut self, _: &SplitCell, window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id).cloned() else {
            return;
        };
        if !matches!(cell, Cell::Code(_) | Cell::Markdown(_)) {
            return;
        }

        let source = cell.current_source(cx);
        let mut offset = Self::cursor_offset_in_cell(&cell, cx).unwrap_or(source.len());
        offset = offset.min(source.len());
        while offset > 0 && !source.is_char_boundary(offset) {
            offset -= 1;
        }
        let (before, after) = source.split_at(offset);
        let original = cell.to_nbformat_cell(cx);
        let second_id: CellId = Uuid::new_v4().into();

        let (first_cell, second_cell) = match original {
            nbformat::v4::Cell::Code {
                id,
                metadata,
                execution_count,
                outputs,
                ..
            } => (
                nbformat::v4::Cell::Code {
                    id,
                    metadata: metadata.clone(),
                    execution_count,
                    source: source_lines(before),
                    outputs,
                },
                nbformat::v4::Cell::Code {
                    id: second_id.clone(),
                    metadata,
                    execution_count: None,
                    source: source_lines(after),
                    outputs: Vec::new(),
                },
            ),
            nbformat::v4::Cell::Markdown { id, metadata, .. } => (
                nbformat::v4::Cell::Markdown {
                    id,
                    metadata: metadata.clone(),
                    source: source_lines(before),
                    attachments: None,
                },
                nbformat::v4::Cell::Markdown {
                    id: second_id.clone(),
                    metadata,
                    source: source_lines(after),
                    attachments: None,
                },
            ),
            nbformat::v4::Cell::Raw { .. } => return,
        };

        self.push_undo_snapshot();
        let first = self.build_cell(first_cell, window, cx);
        let second = self.build_cell(second_cell, window, cx);
        self.cell_map.insert(cell_id, first);
        self.cell_order
            .insert(self.selected_cell_index + 1, second_id.clone());
        self.cell_map.insert(second_id, second);
        self.cell_list.splice(
            self.selected_cell_index + 1..self.selected_cell_index + 1,
            1,
        );
        self.cell_list
            .scroll_to_reveal_item(self.selected_cell_index);
        self.refresh_cell_registry(cx);
        cx.notify();
    }

    fn merge_cell(&mut self, above: bool, window: &mut Window, cx: &mut Context<Self>) {
        let current_index = self.selected_cell_index;
        let (first_index, second_index) = if above {
            let Some(first_index) = current_index.checked_sub(1) else {
                return;
            };
            (first_index, current_index)
        } else {
            let second_index = current_index + 1;
            if second_index >= self.cell_order.len() {
                return;
            }
            (current_index, second_index)
        };

        let first_id = self.cell_order[first_index].clone();
        let second_id = self.cell_order[second_index].clone();
        let (Some(first), Some(second)) = (
            self.cell_map.get(&first_id).cloned(),
            self.cell_map.get(&second_id).cloned(),
        ) else {
            return;
        };

        let mergeable = matches!(
            (&first, &second),
            (Cell::Code(_), Cell::Code(_)) | (Cell::Markdown(_), Cell::Markdown(_))
        );
        if !mergeable {
            return;
        }

        let first_source = first.current_source(cx);
        let second_source = second.current_source(cx);
        let mut merged_source = first_source;
        if !merged_source.is_empty() && !merged_source.ends_with('\n') {
            merged_source.push('\n');
        }
        merged_source.push_str(&second_source);

        let merged_cell = match (first.to_nbformat_cell(cx), second.to_nbformat_cell(cx)) {
            (
                nbformat::v4::Cell::Code {
                    id,
                    metadata,
                    execution_count,
                    outputs,
                    ..
                },
                nbformat::v4::Cell::Code {
                    execution_count: second_execution_count,
                    outputs: second_outputs,
                    ..
                },
            ) => nbformat::v4::Cell::Code {
                id,
                metadata,
                execution_count: execution_count.or(second_execution_count),
                source: source_lines(&merged_source),
                outputs: outputs.into_iter().chain(second_outputs).collect(),
            },
            (
                nbformat::v4::Cell::Markdown { id, metadata, .. },
                nbformat::v4::Cell::Markdown { .. },
            ) => nbformat::v4::Cell::Markdown {
                id,
                metadata,
                source: source_lines(&merged_source),
                attachments: None,
            },
            _ => return,
        };

        self.push_undo_snapshot();
        let merged = self.build_cell(merged_cell, window, cx);
        self.cell_map.insert(first_id, merged);
        self.cell_order.remove(second_index);
        self.cell_map.remove(&second_id);
        self.cell_list.splice(second_index..second_index + 1, 0);
        self.selected_cell_index = first_index;
        self.cell_list
            .scroll_to_reveal_item(self.selected_cell_index);
        self.refresh_cell_registry(cx);
        cx.notify();
    }

    fn merge_cell_above_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.merge_cell(true, window, cx);
    }

    fn merge_cell_below_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.merge_cell(false, window, cx);
    }

    fn convert_cell_to_markdown(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id).cloned() else {
            return;
        };
        let converted = match cell.to_nbformat_cell(cx) {
            nbformat::v4::Cell::Code {
                id,
                metadata,
                source,
                ..
            } => nbformat::v4::Cell::Markdown {
                id,
                metadata,
                source,
                attachments: None,
            },
            nbformat::v4::Cell::Markdown { .. } | nbformat::v4::Cell::Raw { .. } => return,
        };

        self.push_undo_snapshot();
        let converted_cell = self.build_cell(converted, window, cx);
        self.cell_map.insert(cell_id, converted_cell);
        self.refresh_cell_registry(cx);
        cx.notify();
    }

    fn convert_cell_to_code(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id).cloned() else {
            return;
        };
        let converted = match cell.to_nbformat_cell(cx) {
            nbformat::v4::Cell::Markdown {
                id,
                metadata,
                source,
                ..
            } => nbformat::v4::Cell::Code {
                id,
                metadata,
                execution_count: None,
                source,
                outputs: Vec::new(),
            },
            nbformat::v4::Cell::Code { .. } | nbformat::v4::Cell::Raw { .. } => return,
        };

        self.push_undo_snapshot();
        let converted_cell = self.build_cell(converted, window, cx);
        self.cell_map.insert(cell_id, converted_cell);
        self.refresh_cell_registry(cx);
        cx.notify();
    }

    fn run_above(&mut self, _: &RunAbove, window: &mut Window, cx: &mut Context<Self>) {
        for index in 0..self.selected_cell_index {
            if let Some(cell_id) = self.cell_order.get(index).cloned() {
                self.execute_cell(cell_id, window, cx);
            }
        }
    }

    fn run_below(&mut self, _: &RunBelow, window: &mut Window, cx: &mut Context<Self>) {
        for index in self.selected_cell_index + 1..self.cell_order.len() {
            if let Some(cell_id) = self.cell_order.get(index).cloned() {
                self.execute_cell(cell_id, window, cx);
            }
        }
    }

    fn clear_selected_cell_outputs(
        &mut self,
        _: &ClearCellOutputs,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        if let Some(Cell::Code(code_cell)) = self.cell_map.get(&cell_id) {
            code_cell.update(cx, |cell, cx| {
                cell.clear_outputs();
                cx.notify();
            });
        }
        cx.notify();
    }

    fn toggle_selected_cell_output(
        &mut self,
        _: &ToggleCellOutput,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        if let Some(cell) = self.cell_map.get(&cell_id) {
            cell.toggle_outputs_collapsed(cx);
        }
    }

    fn add_markdown_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.push_undo_snapshot();
        let new_cell_id: CellId = Uuid::new_v4().into();
        let languages = self.languages.clone();
        let metadata: nbformat::v4::CellMetadata =
            serde_json::from_str("{}").expect("empty object should parse");

        let markdown_cell = cx.new(|cx| {
            super::MarkdownCell::new(
                new_cell_id.clone(),
                metadata,
                String::new(),
                languages,
                window,
                cx,
            )
        });

        cx.subscribe(
            &markdown_cell,
            move |_this, cell, event: &MarkdownCellEvent, cx| match event {
                MarkdownCellEvent::FinishedEditing | MarkdownCellEvent::Run(_) => {
                    cell.update(cx, |cell, cx| {
                        cell.reparse_markdown(cx);
                    });
                }
            },
        )
        .detach();

        let cell_id_for_editor = new_cell_id.clone();
        let editor = markdown_cell.read(cx).editor().clone();
        cx.subscribe(&editor, move |this, _editor, event, cx| {
            if let editor::EditorEvent::Focused = event {
                this.select_cell_by_id(&cell_id_for_editor, cx);
            }
        })
        .detach();

        self.insert_cell_at_current_position(new_cell_id, Cell::Markdown(markdown_cell.clone()));
        markdown_cell.update(cx, |cell, cx| {
            cell.set_editing(true);
            cx.notify();
        });
        let editor = markdown_cell.read(cx).editor().clone();
        window.focus(&editor.focus_handle(cx), cx);
        self.notebook_mode = NotebookMode::Edit;
        cx.notify();
        self.refresh_cell_registry(cx);
    }

    fn add_code_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.push_undo_snapshot();
        let new_cell_id: CellId = Uuid::new_v4().into();
        let languages = self.languages.clone();
        let notebook_language = self.notebook_language.clone();
        let metadata: nbformat::v4::CellMetadata =
            serde_json::from_str("{}").expect("empty object should parse");

        let code_cell = cx.new(|cx| {
            super::CodeCell::new(
                super::CellSource::None,
                new_cell_id.clone(),
                metadata,
                String::new(),
                languages,
                notebook_language,
                &self.project,
                window,
                cx,
            )
        });

        let cell_id_for_run = new_cell_id.clone();
        cx.subscribe_in(
            &code_cell,
            window,
            move |this, _cell, event, window, cx| match event {
                CellEvent::Run(cell_id) => this.execute_cell(cell_id.clone(), window, cx),
                CellEvent::FocusedIn(_) => this.select_cell_by_id(&cell_id_for_run, cx),
            },
        )
        .detach();

        let cell_id_for_editor = new_cell_id.clone();
        let editor = code_cell.read(cx).editor().clone();
        cx.subscribe(&editor, move |this, _editor, event, cx| {
            if let editor::EditorEvent::Focused = event {
                this.select_cell_by_id(&cell_id_for_editor, cx);
            }
        })
        .detach();

        self.insert_cell_at_current_position(new_cell_id, Cell::Code(code_cell.clone()));
        let editor = code_cell.read(cx).editor().clone();
        window.focus(&editor.focus_handle(cx), cx);
        self.notebook_mode = NotebookMode::Edit;
        cx.notify();
        self.refresh_cell_registry(cx);
    }

    fn cell_count(&self) -> usize {
        self.cell_map.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_cell_index
    }

    fn select_cell_by_id(&mut self, cell_id: &CellId, cx: &mut Context<Self>) {
        if let Some(index) = self.cell_order.iter().position(|id| id == cell_id) {
            self.selected_cell_index = index;
            self.focused_cell_id = Some(cell_id.clone());
            self.notebook_mode = NotebookMode::Edit;
            cx.notify();
        }
    }

    fn update_cell_numbers(&mut self, cx: &mut Context<Self>) {
        for (index, cell_id) in self.cell_order.iter().enumerate() {
            let Some(cell) = self.cell_map.get(cell_id) else {
                continue;
            };
            let cell_number = index + 1;
            match cell {
                Cell::Code(cell) => cell.update(cx, |cell, _| {
                    cell.set_cell_number(cell_number);
                }),
                Cell::Markdown(cell) => cell.update(cx, |cell, _| {
                    cell.set_cell_number(cell_number);
                }),
                Cell::Raw(cell) => cell.update(cx, |cell, _| {
                    cell.set_cell_number(cell_number);
                }),
            }
        }
    }

    /// Mirrors the notebook's cells into the global registry so the agent
    /// message editor can recognise pasted cell contents.
    fn refresh_cell_registry(&mut self, cx: &mut Context<Self>) {
        self.update_cell_numbers(cx);
        let Some(notebook_path) = self.notebook_path.clone() else {
            return;
        };
        let entries: Vec<(text::BufferId, usize, u32)> = self
            .cell_order
            .iter()
            .enumerate()
            .filter_map(|(index, cell_id)| {
                let cell = self.cell_map.get(cell_id)?;
                let buffer = cell.buffer(cx)?;
                let buffer_id = buffer.read(cx).remote_id();
                let cell_line_count = buffer.read(cx).snapshot().max_point().row + 1;
                Some((buffer_id, index + 1, cell_line_count))
            })
            .collect();

        let previously_registered = std::mem::take(&mut self.registered_cell_buffers);
        let mut registered_cell_buffers = Vec::with_capacity(entries.len());
        let registry = notebook_cell_context::registry(cx);
        for buffer_id in previously_registered {
            registry.unregister(buffer_id);
        }
        for (buffer_id, cell_number, cell_line_count) in entries {
            registry.register(
                buffer_id,
                notebook_cell_context::NotebookCellReference {
                    notebook_path: notebook_path.clone(),
                    cell_number,
                    cell_line_count,
                },
            );
            registered_cell_buffers.push(buffer_id);
        }
        self.registered_cell_buffers = registered_cell_buffers;
        self.sync_notebook_lsp(cx);
    }

    fn register_agent_live_editing(&mut self, cx: &mut Context<Self>) {
        let Some(notebook_path) = self.notebook_path.clone() else {
            return;
        };
        let Some(window_handle) = self.agent_window else {
            return;
        };
        let weak_editor = cx.entity().downgrade();
        let entity: gpui::AnyWeakEntity = weak_editor.clone().into();
        let handler: notebook_agent_tools::LiveHandler = Rc::new(move |command, cx| {
            window_handle
                .update(cx, |_, window, cx| {
                    weak_editor
                        .update(cx, |editor, cx| {
                            editor.apply_agent_command(command, window, cx)
                        })
                        .map_err(|error| error.to_string())?
                })
                .map_err(|error| error.to_string())?
        });
        notebook_agent_tools::register_live(cx, notebook_path, entity, handler);
    }

    /// Rebuilds the open notebook when the `.ipynb` on disk changes externally
    /// (for example when an ACP agent edits it) and the user has no local edits
    /// to preserve.
    ///
    /// This reacts to the worktree entry change itself rather than to the
    /// notebook buffer's `Reloaded` event: the raw JSON buffer the item is
    /// backed by is not guaranteed to be reloaded by the project when the file
    /// changes on disk, so waiting for that event can leave the notebook stale.
    /// `Item::reload` reads the file back from disk, and the on-disk content is
    /// compared against the open buffer first so the notebook's own saves do not
    /// needlessly rebuild every cell.
    fn observe_notebook_buffer(&mut self, cx: &mut Context<Self>) {
        let project = self.project.clone();
        let worktree_id = self.notebook_item.read(cx).project_path.worktree_id;
        let relative_path = self.notebook_item.read(cx).project_path.path.clone();

        log::info!(
            "notebook: observing '{}' for external changes",
            relative_path
        );

        let subscription = cx.subscribe(&project, move |this, _project, event, cx| {
            let project::Event::WorktreeUpdatedEntries(updated_worktree_id, changes) = event else {
                return;
            };
            if *updated_worktree_id != worktree_id
                || !changes
                    .iter()
                    .any(|(path, _, _)| path.as_ref() == relative_path.as_ref())
            {
                return;
            }

            log::info!(
                "notebook: received change event for '{}'",
                relative_path
            );

            if this.has_structural_changes() || this.has_content_changes(cx) {
                log::info!(
                    "notebook: ignoring external change to '{}' because there are unsaved local edits",
                    relative_path
                );
                return;
            }

            let Some(window_handle) = this.agent_window else {
                log::info!(
                    "notebook: ignoring external change to '{}' because no window is attached",
                    relative_path
                );
                return;
            };
            let weak_editor = cx.entity().downgrade();
            let project = this.project.clone();
            let project_path = this.notebook_item.read(cx).project_path.clone();
            cx.spawn(async move |_this, cx| {
                let (fs, absolute_path) = project.read_with(cx, |project, cx| {
                    (
                        project.fs().clone(),
                        project.absolute_path(&project_path, cx),
                    )
                });
                let Some(absolute_path) = absolute_path else {
                    return anyhow::Ok(());
                };
                let disk_content = fs.load(&absolute_path).await?;

                // Only reload when the file on disk actually differs from what
                // the editor is showing; the notebook's own saves write exactly
                // what it already displays and must not rebuild every cell.
                let incoming = parse_notebook_content(&disk_content);
                let current = weak_editor.update(cx, |this, cx| {
                    serde_json::to_value(this.to_notebook(cx))
                })??;
                if let Ok(incoming) = incoming
                    && serde_json::to_value(&incoming)? == current
                {
                    log::info!(
                        "notebook: change event for '{}' left the notebook content unchanged",
                        absolute_path.display()
                    );
                    return anyhow::Ok(());
                }

                log::info!(
                    "notebook: reloading '{}' after an external change",
                    absolute_path.display()
                );
                let reload = window_handle.update(cx, |_, window, cx| {
                    weak_editor.update(cx, |this, cx| this.reload(project, window, cx))
                })??;
                reload.await?;
                anyhow::Ok(())
            })
            .detach_and_log_err(cx);
        });
        self.external_change_subscription = Some(subscription);
    }

    fn agent_snapshot(&self, cx: &App) -> NotebookSnapshot {
        let cells = self
            .cell_order
            .iter()
            .enumerate()
            .filter_map(|(index, cell_id)| {
                let cell = self.cell_map.get(cell_id)?;
                let cell_type = match cell {
                    Cell::Code(_) => NotebookCellType::Code,
                    Cell::Markdown(_) => NotebookCellType::Markdown,
                    Cell::Raw(_) => NotebookCellType::Raw,
                };
                let output_summary = match cell {
                    Cell::Code(code_cell) => code_cell.read(cx).output_summary(),
                    Cell::Markdown(_) | Cell::Raw(_) => None,
                };
                Some(NotebookCellSnapshot {
                    cell_number: index + 1,
                    id: cell.id(cx).to_string(),
                    cell_type,
                    source: cell.current_source(cx),
                    output_summary,
                })
            })
            .collect();

        NotebookSnapshot { cells }
    }

    fn apply_agent_command(
        &mut self,
        command: NotebookLiveCommand,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<NotebookLiveResult, String> {
        let message = match command {
            NotebookLiveCommand::Read => "Read notebook cells.".to_string(),
            NotebookLiveCommand::EditCell {
                cell_number,
                new_source,
                line_range,
            } => {
                let index = agent_cell_index(cell_number, self.cell_order.len())?;
                let cell_id = self.cell_order[index].clone();
                let cell = self
                    .cell_map
                    .get(&cell_id)
                    .cloned()
                    .ok_or_else(|| format!("Cell {cell_number} does not exist."))?;
                let source = match line_range {
                    Some(line_range) => notebook_agent_tools::apply_line_edit(
                        &cell.current_source(cx),
                        line_range,
                        &new_source,
                    )?,
                    None => new_source,
                };
                let buffer = cell
                    .buffer(cx)
                    .ok_or_else(|| format!("Cell {cell_number} has no editable buffer."))?;
                buffer.update(cx, |buffer, cx| {
                    buffer.set_text(source, cx);
                });
                if let Cell::Markdown(markdown_cell) = &cell {
                    markdown_cell.update(cx, |markdown_cell, cx| {
                        markdown_cell.reparse_markdown(cx);
                    });
                }
                format!("Edited cell {cell_number}.")
            }
            NotebookLiveCommand::InsertCell {
                position,
                cell_type,
                source,
            } => {
                if position == 0 || position > self.cell_order.len() + 1 {
                    return Err(format!(
                        "Cannot insert a cell at position {position}: the notebook has {} cells.",
                        self.cell_order.len()
                    ));
                }
                let index = position - 1;
                let new_cell_id: CellId = Uuid::new_v4().into();
                let metadata: nbformat::v4::CellMetadata =
                    serde_json::from_str("{}").map_err(|error| error.to_string())?;
                let project = self.project.clone();
                let notebook_language = self.notebook_language.clone();
                let languages = self.languages.clone();

                let cell = match cell_type {
                    NotebookCellType::Code => {
                        let code_cell = cx.new(|cx| {
                            super::CodeCell::new(
                                super::CellSource::None,
                                new_cell_id.clone(),
                                metadata,
                                source,
                                languages.clone(),
                                notebook_language,
                                &project,
                                window,
                                cx,
                            )
                        });
                        self.attach_code_cell(new_cell_id.clone(), &code_cell, window, cx);
                        Cell::Code(code_cell)
                    }
                    NotebookCellType::Markdown => {
                        let markdown_cell = cx.new(|cx| {
                            super::MarkdownCell::new(
                                new_cell_id.clone(),
                                metadata,
                                source,
                                languages,
                                window,
                                cx,
                            )
                        });
                        self.attach_markdown_cell(new_cell_id.clone(), &markdown_cell, cx);
                        Cell::Markdown(markdown_cell)
                    }
                    NotebookCellType::Raw => {
                        return Err("Only code and markdown cells can be inserted by the agent."
                            .to_string());
                    }
                };

                self.push_undo_snapshot();
                self.cell_order.insert(index, new_cell_id.clone());
                self.cell_map.insert(new_cell_id, cell);
                self.cell_list.splice(index..index, 1);
                if index <= self.selected_cell_index {
                    self.selected_cell_index += 1;
                }
                format!(
                    "Inserted a {} cell at position {position}.",
                    cell_type.as_str()
                )
            }
            NotebookLiveCommand::DeleteCell { cell_number } => {
                let index = agent_cell_index(cell_number, self.cell_order.len())?;
                self.push_undo_snapshot();
                let cell_id = self.cell_order.remove(index);
                self.cell_map.remove(&cell_id);
                self.cell_list.splice(index..index + 1, 0);
                if self.selected_cell_index > index {
                    self.selected_cell_index -= 1;
                }
                if self.selected_cell_index >= self.cell_order.len() {
                    self.selected_cell_index = self.cell_order.len().saturating_sub(1);
                }
                format!("Deleted cell {cell_number}.")
            }
            NotebookLiveCommand::MoveCell { from_cell, to_cell } => {
                let from_index = agent_cell_index(from_cell, self.cell_order.len())?;
                let to_index = agent_cell_index(to_cell, self.cell_order.len())?;
                if from_index == to_index {
                    format!("Cell {from_cell} is already at position {to_cell}.")
                } else {
                    self.push_undo_snapshot();
                    let selected_cell_id = self.cell_order.get(self.selected_cell_index).cloned();
                    let cell_id = self.cell_order.remove(from_index);
                    self.cell_order.insert(to_index, cell_id);
                    self.cell_list.splice(from_index..from_index + 1, 0);
                    self.cell_list.splice(to_index..to_index, 1);
                    if let Some(selected_cell_id) = selected_cell_id
                        && let Some(selected_index) = self
                            .cell_order
                            .iter()
                            .position(|cell_id| cell_id == &selected_cell_id)
                    {
                        self.selected_cell_index = selected_index;
                    }
                    format!("Moved cell {from_cell} to position {to_cell}.")
                }
            }
        };

        self.refresh_cell_registry(cx);
        cx.notify();
        Ok(NotebookLiveResult {
            snapshot: self.agent_snapshot(cx),
            message,
        })
    }

    fn attach_code_cell(
        &mut self,
        cell_id: CellId,
        code_cell: &Entity<super::CodeCell>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let cell_id_for_run = cell_id.clone();
        cx.subscribe_in(
            code_cell,
            window,
            move |this, _cell, event, window, cx| match event {
                CellEvent::Run(cell_id) => this.execute_cell(cell_id.clone(), window, cx),
                CellEvent::FocusedIn(_) => this.select_cell_by_id(&cell_id_for_run, cx),
            },
        )
        .detach();

        let cell_id_for_editor = cell_id;
        let editor = code_cell.read(cx).editor().clone();
        cx.subscribe(&editor, move |this, _editor, event, cx| {
            if let editor::EditorEvent::Focused = event {
                this.select_cell_by_id(&cell_id_for_editor, cx);
            }
        })
        .detach();
    }

    fn attach_markdown_cell(
        &mut self,
        cell_id: CellId,
        markdown_cell: &Entity<super::MarkdownCell>,
        cx: &mut Context<Self>,
    ) {
        cx.subscribe(
            markdown_cell,
            move |_this, cell, event: &MarkdownCellEvent, cx| match event {
                MarkdownCellEvent::FinishedEditing | MarkdownCellEvent::Run(_) => {
                    cell.update(cx, |cell, cx| {
                        cell.reparse_markdown(cx);
                    });
                }
            },
        )
        .detach();

        let cell_id_for_editor = cell_id;
        let editor = markdown_cell.read(cx).editor().clone();
        cx.subscribe(&editor, move |this, _editor, event, cx| {
            if let editor::EditorEvent::Focused = event {
                this.select_cell_by_id(&cell_id_for_editor, cx);
            }
        })
        .detach();
    }

    fn capture_copy(&mut self, _: &editor::actions::Copy, _: &mut Window, cx: &mut Context<Self>) {
        self.record_cell_copy(cx);
        cx.propagate();
    }

    fn capture_cut(&mut self, _: &editor::actions::Cut, _: &mut Window, cx: &mut Context<Self>) {
        self.record_cell_copy(cx);
        cx.propagate();
    }

    /// Records the text the focused cell editor is about to copy, so pasting it
    /// back can be recognised as a notebook cell. Mirrors `Editor::do_copy`
    /// without leading-indent stripping.
    fn record_cell_copy(&mut self, cx: &mut Context<Self>) {
        let Some(cell_id) = self.focused_cell_id.clone() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id) else {
            return;
        };
        let Some(editor) = cell.editor(cx).cloned() else {
            return;
        };
        let Some(buffer) = cell.buffer(cx) else {
            return;
        };
        let Some(notebook_path) = self.notebook_path.clone() else {
            return;
        };
        let Some(cell_number) = self
            .cell_order
            .iter()
            .position(|id| id == &cell_id)
            .map(|i| i + 1)
        else {
            return;
        };
        let cell_line_count = buffer.read(cx).snapshot().max_point().row + 1;
        let reference = notebook_cell_context::NotebookCellReference {
            notebook_path,
            cell_number,
            cell_line_count,
        };

        let copied = editor.update(cx, |editor, cx| {
            let selections = editor
                .selections
                .all::<language::Point>(&editor.display_snapshot(cx));
            let buffer = editor.buffer().read(cx).snapshot(cx);
            let max_point = buffer.max_point();
            let line_mode = editor.selections.line_mode();

            let mut text = String::new();
            let mut line_range: Option<RangeInclusive<u32>> = None;
            let mut is_first = true;
            let mut previous_was_entire_line = false;
            for selection in &selections {
                let mut start = selection.start;
                let mut end = selection.end;
                let is_entire_line = selection.is_empty() || line_mode;
                let mut add_trailing_newline = false;
                if is_entire_line {
                    start = language::Point::new(start.row, 0);
                    let next_line_start = language::Point::new(end.row + 1, 0);
                    if next_line_start <= max_point {
                        end = next_line_start;
                    } else {
                        end =
                            language::Point::new(end.row, buffer.line_len(MultiBufferRow(end.row)));
                        add_trailing_newline = true;
                    }
                }

                if is_first {
                    is_first = false;
                } else if !previous_was_entire_line {
                    text.push('\n');
                }
                for chunk in buffer.text_for_range(start..end) {
                    text.push_str(chunk);
                }
                if add_trailing_newline {
                    text.push('\n');
                }
                line_range = Some(match line_range {
                    Some(existing) => *existing.start()..=end.row,
                    None => start.row..=end.row,
                });
                previous_was_entire_line = is_entire_line;
            }

            if text.is_empty() {
                None
            } else {
                line_range.map(|line_range| (text, line_range))
            }
        });

        let Some((text, line_range)) = copied else {
            return;
        };
        notebook_cell_context::registry(cx).record_copy(text, reference, line_range);
    }

    pub fn set_selected_index(
        &mut self,
        index: usize,
        jump_to_index: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // let previous_index = self.selected_cell_index;
        self.selected_cell_index = index;
        let current_index = self.selected_cell_index;

        // in the future we may have some `on_cell_change` event that we want to fire here

        if jump_to_index {
            self.jump_to_cell(current_index, window, cx);
        }
    }

    fn select_next(
        &mut self,
        _: &menu::SelectNext,
        selection_mode: SelectionMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == count - 1 {
                count - 1
            } else {
                index + 1
            };
            self.set_selected_index(ix, true, window, cx);

            if selection_mode == SelectionMode::SelectAndMove
                && let Some(cell) = self.get_selected_cell()
            {
                cell.move_to(MovementDirection::Start, window, cx);
            }

            cx.notify();
        }
    }

    fn select_previous(
        &mut self,
        _: &menu::SelectPrevious,
        selection_mode: SelectionMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == 0 { 0 } else { index - 1 };
            self.set_selected_index(ix, true, window, cx);

            if selection_mode == SelectionMode::SelectAndMove
                && let Some(cell) = self.get_selected_cell()
            {
                cell.move_to(MovementDirection::End, window, cx);
            }

            cx.notify();
        }
    }

    pub fn select_first(
        &mut self,
        _: &menu::SelectFirst,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(0, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_last(
        &mut self,
        _: &menu::SelectLast,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(count - 1, true, window, cx);
            cx.notify();
        }
    }

    fn jump_to_cell(&mut self, index: usize, _window: &mut Window, _cx: &mut Context<Self>) {
        self.cell_list.scroll_to_reveal_item(index);
    }

    fn button_group(window: &mut Window, cx: &mut Context<Self>) -> Div {
        v_flex()
            .gap(DynamicSpacing::Base04.rems(cx))
            .items_center()
            .w(px(CONTROL_SIZE + 4.0))
            .overflow_hidden()
            .rounded(px(5.))
            .bg(cx.theme().colors().title_bar_background)
            .p_px()
            .border_1()
            .border_color(cx.theme().colors().border)
    }

    fn render_notebook_control(
        id: impl Into<SharedString>,
        icon: IconName,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> IconButton {
        let id: ElementId = ElementId::Name(id.into());
        IconButton::new(id, icon).width(px(CONTROL_SIZE))
    }

    fn render_notebook_controls(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let has_outputs = self.has_outputs(window, cx);
        let variables_visible = self.variables_panel.read(cx).is_visible();
        let variables_position = self.variables_panel.read(cx).position();

        v_flex()
            .max_w(px(CONTROL_SIZE + 4.0))
            .items_center()
            .gap(DynamicSpacing::Base16.rems(cx))
            .justify_between()
            .flex_none()
            .h_full()
            .py(DynamicSpacing::Base12.px(cx))
            .child(
                v_flex()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "run-all-cells",
                                    IconName::PlayFilled,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Execute all cells", &RunAll, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(RunAll), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "clear-all-outputs",
                                    IconName::ListX,
                                    window,
                                    cx,
                                )
                                .disabled(!has_outputs)
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Clear all outputs", &ClearOutputs, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(ClearOutputs), cx);
                                }),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-up",
                                    IconName::ArrowUp,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Move cell up", &MoveCellUp, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(MoveCellUp), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-down",
                                    IconName::ArrowDown,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Move cell down", &MoveCellDown, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(MoveCellDown), cx);
                                }),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "new-markdown-cell",
                                    IconName::Plus,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Add markdown block", &AddMarkdownBlock, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(AddMarkdownBlock), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "new-code-cell",
                                    IconName::Code,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Add code block", &AddCodeBlock, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(AddCodeBlock), cx);
                                }),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx).child(
                            Self::render_notebook_control(
                                "delete-cell",
                                IconName::Trash,
                                window,
                                cx,
                            )
                            .disabled(self.cell_order.is_empty())
                            .tooltip(move |window, cx| {
                                Tooltip::for_action("Delete cell", &DeleteCell, cx)
                            })
                            .on_click(|_, window, cx| {
                                window.dispatch_action(Box::new(DeleteCell), cx);
                            }),
                        ),
                    ),
            )
            .child(
                v_flex()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .items_center()
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "toggle-variables-panel",
                                    if variables_visible {
                                        IconName::Eye
                                    } else {
                                        IconName::EyeOff
                                    },
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::text(if variables_visible {
                                        "Hide variables panel"
                                    } else {
                                        "Show variables panel"
                                    })(window, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(ToggleVariablesPanel), cx);
                                }),
                            )
                            .child(
                                PopoverMenu::new("notebook-more-menu")
                                    .trigger_with_tooltip(
                                        Self::render_notebook_control(
                                            "more-menu",
                                            IconName::Ellipsis,
                                            window,
                                            cx,
                                        ),
                                        Tooltip::text("More options"),
                                    )
                                    .anchor(gpui::Anchor::TopRight)
                                    .menu(move |window, cx| {
                                        Some(ContextMenu::build(window, cx, move |menu, _, _| {
                                            menu.action(
                                                if variables_visible {
                                                    "Hide Variables Panel"
                                                } else {
                                                    "Show Variables Panel"
                                                },
                                                ToggleVariablesPanel.boxed_clone(),
                                            )
                                            .action(
                                                if variables_position
                                                    == VariablesPanelPosition::Bottom
                                                {
                                                    "Move Variables Panel to Right"
                                                } else {
                                                    "Move Variables Panel Below"
                                                },
                                                ToggleVariablesPanelPosition.boxed_clone(),
                                            )
                                        }))
                                    }),
                            ),
                    )
                    .child(Self::button_group(window, cx).child({
                        let kernel_status = self.kernel.status();
                        let (icon, icon_color) = match &kernel_status {
                            KernelStatus::Idle => (IconName::ReplNeutral, Color::Success),
                            KernelStatus::Busy => (IconName::ReplNeutral, Color::Warning),
                            KernelStatus::Starting => (IconName::ReplNeutral, Color::Muted),
                            KernelStatus::Error => (IconName::ReplNeutral, Color::Error),
                            KernelStatus::ShuttingDown => (IconName::ReplNeutral, Color::Muted),
                            KernelStatus::Shutdown => (IconName::ReplNeutral, Color::Disabled),
                            KernelStatus::Restarting => (IconName::ReplNeutral, Color::Warning),
                        };
                        let kernel_name = self
                            .kernel_specification
                            .as_ref()
                            .map(|spec| spec.name().to_string())
                            .unwrap_or_else(|| "Select Kernel".to_string());
                        IconButton::new("repl", icon)
                            .icon_color(icon_color)
                            .tooltip(move |window, cx| {
                                Tooltip::text(format!(
                                    "{} ({}). Click to change kernel.",
                                    kernel_name,
                                    kernel_status.to_string()
                                ))(window, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.kernel_picker_handle.toggle(window, cx);
                            }))
                    })),
            )
    }

    fn render_kernel_status_bar(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let kernel_status = self.kernel.status();
        let kernel_name = self
            .kernel_specification
            .as_ref()
            .map(|spec| spec.name().to_string())
            .unwrap_or_else(|| "Select Kernel".to_string());

        let (status_icon, status_color) = match &kernel_status {
            KernelStatus::Idle => (IconName::Circle, Color::Success),
            KernelStatus::Busy => (IconName::ArrowCircle, Color::Warning),
            KernelStatus::Starting => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Error => (IconName::XCircle, Color::Error),
            KernelStatus::ShuttingDown => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Shutdown => (IconName::Circle, Color::Muted),
            KernelStatus::Restarting => (IconName::ArrowCircle, Color::Warning),
        };

        let is_spinning = matches!(
            kernel_status,
            KernelStatus::Busy
                | KernelStatus::Starting
                | KernelStatus::ShuttingDown
                | KernelStatus::Restarting
        );

        let status_icon_element = if is_spinning {
            Icon::new(status_icon)
                .size(IconSize::Small)
                .color(status_color)
                .with_rotate_animation(2)
                .into_any_element()
        } else {
            Icon::new(status_icon)
                .size(IconSize::Small)
                .color(status_color)
                .into_any_element()
        };

        let worktree_id = self.worktree_id;
        let kernel_picker_handle = self.kernel_picker_handle.clone();
        let view = cx.entity().downgrade();

        h_flex()
            .w_full()
            .px_3()
            .py_1()
            .gap_2()
            .items_center()
            .justify_between()
            .bg(cx.theme().colors().status_bar_background)
            .child(
                KernelSelector::new(
                    Box::new(move |spec: KernelSpecification, window, cx| {
                        if let Some(view) = view.upgrade() {
                            view.update(cx, |this, cx| {
                                this.select_kernel(spec, window, cx);
                            });
                        }
                    }),
                    worktree_id,
                    Button::new("kernel-selector", kernel_name.clone())
                        .label_size(LabelSize::Small)
                        .start_icon(
                            Icon::new(status_icon)
                                .size(IconSize::Small)
                                .color(status_color),
                        ),
                    Tooltip::text(format!(
                        "Kernel: {} ({}). Click to change.",
                        kernel_name,
                        kernel_status.to_string()
                    )),
                )
                .with_selected_kernel(self.kernel_specification.clone())
                .with_handle(kernel_picker_handle),
            )
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        IconButton::new("restart-kernel", IconName::RotateCw)
                            .icon_size(IconSize::Small)
                            .tooltip(|window, cx| {
                                Tooltip::for_action("Restart Kernel", &RestartKernel, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.restart_kernel(&RestartKernel, window, cx);
                            })),
                    )
                    .child(
                        IconButton::new("interrupt-kernel", IconName::Stop)
                            .icon_size(IconSize::Small)
                            .disabled(!matches!(kernel_status, KernelStatus::Busy))
                            .tooltip(|window, cx| {
                                Tooltip::for_action("Interrupt Kernel", &InterruptKernel, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.interrupt_kernel(&InterruptKernel, window, cx);
                            })),
                    ),
            )
    }

    fn cell_list(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        list(self.cell_list.clone(), move |index, window, cx| {
            view.update(cx, |this, cx| {
                let cell_id = &this.cell_order[index];
                let cell = this.cell_map.get(cell_id).unwrap();
                this.render_cell(index, cell, window, cx).into_any_element()
            })
        })
        .size_full()
    }

    fn render_empty_state(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            .items_center()
            .justify_center()
            .gap_3()
            .child(Label::new("This notebook is empty.").color(Color::Muted))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("empty-state-add-code", "Add code cell")
                            .start_icon(Icon::new(IconName::Code))
                            .key_binding(KeyBinding::for_action_in(
                                &AddCodeBlock,
                                &self.focus_handle,
                                cx,
                            ))
                            .on_click(
                                cx.listener(|this, _, window, cx| this.add_code_block(window, cx)),
                            ),
                    )
                    .child(
                        Button::new("empty-state-add-markdown", "Add markdown cell")
                            .style(ButtonStyle::Subtle)
                            .start_icon(Icon::new(IconName::FileMarkdown))
                            .key_binding(KeyBinding::for_action_in(
                                &AddMarkdownBlock,
                                &self.focus_handle,
                                cx,
                            ))
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.add_markdown_block(window, cx)
                            })),
                    ),
            )
    }

    fn cell_position(&self, index: usize) -> CellPosition {
        match index {
            0 => CellPosition::First,
            index if index == self.cell_count() - 1 => CellPosition::Last,
            _ => CellPosition::Middle,
        }
    }

    fn render_cell(
        &self,
        index: usize,
        cell: &Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let cell_position = self.cell_position(index);

        let is_selected = index == self.selected_cell_index;

        match cell {
            Cell::Code(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
            Cell::Markdown(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
            Cell::Raw(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
        }
    }
}

fn source_lines(source: &str) -> Vec<String> {
    source.lines().map(|line| format!("{line}\n")).collect()
}

/// The focused cell only counts as focused if its editor currently holds the
/// window focus, so a stale `focused_cell_id` left behind by a blur does not
/// cause a rebuild to steal focus back.
fn focused_cell_editor_in_window(
    editor: &NotebookEditor,
    window: &Window,
    cx: &App,
) -> Option<CellId> {
    let cell_id = editor.focused_cell_id.as_ref()?;
    let cell = editor.cell_map.get(cell_id)?;
    let cell_editor = cell.editor(cx)?;
    let focus_handle = cell_editor.focus_handle(cx);
    focus_handle.is_focused(window).then(|| cell_id.clone())
}

/// Finds the cell closest to `cell_id`'s old position that still exists after a
/// rebuild, so a deleted anchor falls back to its neighbour rather than the top.
fn nearest_surviving_cell_index(
    cell_id: &CellId,
    previous_cell_order: &[CellId],
    new_cell_order: &[CellId],
) -> Option<usize> {
    let previous_index = previous_cell_order
        .iter()
        .position(|candidate| candidate == cell_id)?;
    for distance in 1..previous_cell_order.len() {
        if let Some(previous_cell_id) = previous_index
            .checked_sub(distance)
            .and_then(|index| previous_cell_order.get(index))
            && let Some(index) = new_cell_order
                .iter()
                .position(|candidate| candidate == previous_cell_id)
        {
            return Some(index);
        }
        if let Some(previous_cell_id) = previous_cell_order.get(previous_index + distance)
            && let Some(index) = new_cell_order
                .iter()
                .position(|candidate| candidate == previous_cell_id)
        {
            return Some(index);
        }
    }
    None
}

fn notebook_cells_have_same_type(cell: &Cell, notebook_cell: &nbformat::v4::Cell) -> bool {
    matches!(
        (cell, notebook_cell),
        (Cell::Code(_), nbformat::v4::Cell::Code { .. })
            | (Cell::Markdown(_), nbformat::v4::Cell::Markdown { .. })
            | (Cell::Raw(_), nbformat::v4::Cell::Raw { .. })
    )
}

fn notebook_cells_have_same_content(
    cell: &Cell,
    notebook_cell: &nbformat::v4::Cell,
    cx: &App,
) -> bool {
    if !notebook_cells_have_same_type(cell, notebook_cell) {
        return false;
    }
    if cell.current_source(cx) != notebook_cell_source(notebook_cell) {
        return false;
    }
    if !json_values_equal(cell.metadata(cx), notebook_cell.metadata()) {
        return false;
    }
    match (cell, notebook_cell) {
        (
            Cell::Code(_),
            nbformat::v4::Cell::Code {
                execution_count,
                outputs,
                ..
            },
        ) => match cell.to_nbformat_cell(cx) {
            nbformat::v4::Cell::Code {
                execution_count: current_execution_count,
                outputs: current_outputs,
                ..
            } => {
                current_execution_count == *execution_count
                    && json_values_equal(&current_outputs, outputs)
            }
            _ => false,
        },
        _ => true,
    }
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

/// Matches each cell read from disk to an existing cell id.
///
/// Cells with a stable id match directly. Cells whose file entry has no id are
/// aligned by content and position against the unmatched existing cells, since
/// parsing gives them a fresh random id on every reload.
fn match_notebook_cells(
    notebook: &nbformat::v4::Notebook,
    missing_cell_ids: &[bool],
    previous_cell_order: &[CellId],
    previous_cell_map: &HashMap<CellId, Cell>,
    cx: &App,
) -> Vec<Option<CellId>> {
    let mut matched: Vec<Option<CellId>> = Vec::with_capacity(notebook.cells.len());
    let mut used_cell_ids: std::collections::HashSet<CellId> = std::collections::HashSet::default();

    for cell in &notebook.cells {
        let cell_id = cell.id().clone();
        if previous_cell_map.contains_key(&cell_id) && used_cell_ids.insert(cell_id.clone()) {
            matched.push(Some(cell_id));
        } else {
            matched.push(None);
        }
    }

    for (index, notebook_cell) in notebook.cells.iter().enumerate() {
        if matched[index].is_some() || !missing_cell_ids.get(index).copied().unwrap_or(false) {
            continue;
        }
        let cell_type = notebook_cell_type(notebook_cell);
        let source = notebook_cell_source(notebook_cell);
        let candidate = previous_cell_order
            .iter()
            .enumerate()
            .filter(|(_, cell_id)| !used_cell_ids.contains(*cell_id))
            .filter_map(|(previous_index, cell_id)| {
                let cell = previous_cell_map.get(cell_id)?;
                (cell.cell_type() == cell_type && cell.current_source(cx) == source)
                    .then(|| (previous_index.abs_diff(index), cell_id.clone()))
            })
            .min_by_key(|(distance, _)| *distance)
            .map(|(_, cell_id)| cell_id);
        if let Some(cell_id) = candidate {
            used_cell_ids.insert(cell_id.clone());
            matched[index] = Some(cell_id);
        }
    }

    matched
}

fn notebook_cell_type(cell: &nbformat::v4::Cell) -> nbformat::v4::CellType {
    match cell {
        nbformat::v4::Cell::Code { .. } => nbformat::v4::CellType::Code,
        nbformat::v4::Cell::Markdown { .. } => nbformat::v4::CellType::Markdown,
        nbformat::v4::Cell::Raw { .. } => nbformat::v4::CellType::Raw,
    }
}

fn set_notebook_cell_id(cell: &mut nbformat::v4::Cell, cell_id: CellId) {
    match cell {
        nbformat::v4::Cell::Code { id, .. }
        | nbformat::v4::Cell::Markdown { id, .. }
        | nbformat::v4::Cell::Raw { id, .. } => *id = cell_id,
    }
}

/// Computes the minimal splice operations that turn `previous` into `new`.
///
/// The longest common subsequence stays in place, removed cells are deleted
/// from the end, and every other new cell is inserted at its final position.
/// This treats moved cells as a remove plus insert while keeping all other list
/// items untouched.
fn cell_order_edits(
    previous: &[CellId],
    new: &[CellId],
) -> (Vec<usize>, std::collections::HashSet<CellId>) {
    let mut lengths = vec![vec![0usize; new.len() + 1]; previous.len() + 1];
    for (previous_index, previous_cell_id) in previous.iter().enumerate().rev() {
        for (new_index, new_cell_id) in new.iter().enumerate().rev() {
            lengths[previous_index][new_index] = if previous_cell_id == new_cell_id {
                lengths[previous_index + 1][new_index + 1] + 1
            } else {
                lengths[previous_index + 1][new_index].max(lengths[previous_index][new_index + 1])
            };
        }
    }

    let mut kept_cell_ids: std::collections::HashSet<CellId> = std::collections::HashSet::default();
    let mut previous_index = 0;
    let mut new_index = 0;
    while previous_index < previous.len() && new_index < new.len() {
        if previous[previous_index] == new[new_index] {
            kept_cell_ids.insert(previous[previous_index].clone());
            previous_index += 1;
            new_index += 1;
        } else if lengths[previous_index + 1][new_index] >= lengths[previous_index][new_index + 1] {
            previous_index += 1;
        } else {
            new_index += 1;
        }
    }

    let removed_indices = (0..previous.len())
        .rev()
        .filter(|index| !kept_cell_ids.contains(&previous[*index]))
        .collect();
    (removed_indices, kept_cell_ids)
}

fn notebook_cell_source(cell: &nbformat::v4::Cell) -> String {
    match cell {
        nbformat::v4::Cell::Code { source, .. }
        | nbformat::v4::Cell::Markdown { source, .. }
        | nbformat::v4::Cell::Raw { source, .. } => source.concat(),
    }
}

/// Converts a 1-based agent-facing cell number into a zero-based index.
fn agent_cell_index(cell_number: usize, cell_count: usize) -> Result<usize, String> {
    if cell_number == 0 || cell_number > cell_count {
        return Err(format!(
            "Cell positions are 1-based and this notebook has {cell_count} cells, so {cell_number} is not a valid cell number."
        ));
    }
    Ok(cell_number - 1)
}

/// Parses `.ipynb` text into a v4 notebook, filling in any missing cell ids and
/// upgrading legacy and v3 notebooks, the same way a freshly opened notebook is
/// loaded.
fn parse_notebook_content(content: &str) -> Result<nbformat::v4::Notebook> {
    Ok(parse_notebook_content_with_missing_ids(content)?.0)
}

/// An empty file is how a new notebook starts: it opens as one empty Python
/// code cell and only becomes valid JSON on disk at the first save.
fn new_notebook() -> Result<nbformat::v4::Notebook> {
    let template = serde_json::json!({
        "cells": [{
            "cell_type": "code",
            "id": Uuid::new_v4().to_string(),
            "metadata": {},
            "execution_count": null,
            "outputs": [],
            "source": [],
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    });
    match nbformat::parse_notebook(&template.to_string()) {
        Ok(nbformat::Notebook::V4(notebook)) => Ok(notebook),
        Ok(_) => Err(anyhow::anyhow!("new notebook is not nbformat 4")),
        Err(error) => Err(anyhow::anyhow!("Failed to build new notebook: {error:?}")),
    }
}

fn parse_notebook_content_with_missing_ids(
    content: &str,
) -> Result<(nbformat::v4::Notebook, Vec<bool>)> {
    if content.trim().is_empty() {
        return Ok((new_notebook()?, vec![false]));
    }
    let mut json: serde_json::Value = serde_json::from_str(content)?;
    let mut missing_cell_ids = Vec::new();
    if let Some(cells) = json.get_mut("cells").and_then(|cells| cells.as_array_mut()) {
        for cell in cells {
            let missing_id = cell.get("id").is_none();
            missing_cell_ids.push(missing_id);
            if missing_id {
                cell["id"] = serde_json::Value::String(Uuid::new_v4().to_string());
            }
        }
    }
    let content = serde_json::to_string(&json)?;
    let notebook = match nbformat::parse_notebook(&content) {
        Ok(nbformat::Notebook::V4(notebook)) => notebook,
        Ok(nbformat::Notebook::Legacy(legacy_notebook)) => {
            nbformat::upgrade_legacy_notebook(legacy_notebook)?
        }
        Ok(nbformat::Notebook::V3(v3_notebook)) => nbformat::upgrade_v3_notebook(v3_notebook)?,
        Err(error) => return Err(anyhow::anyhow!("Failed to parse notebook: {error:?}")),
    };
    missing_cell_ids.resize(notebook.cells.len(), false);
    missing_cell_ids.truncate(notebook.cells.len());
    Ok((notebook, missing_cell_ids))
}

/// The directory of the virtual environment a Python environment runs from,
/// when the interpreter lives in a conventional `<venv>/bin/python` or
/// `<venv>/Scripts/python.exe` layout and the environment directory has a
/// recognized venv name. Conda environments and global interpreters yield
/// `None`.
fn virtual_environment_root(specification: &KernelSpecification) -> Option<PathBuf> {
    let KernelSpecification::PythonEnv(environment) = specification else {
        return None;
    };
    let interpreter_directory = environment.path.parent()?;
    let environment_root = interpreter_directory.parent()?;
    let environment_name = environment_root.file_name()?;
    VENV_DIR_NAMES
        .iter()
        .any(|venv_name| OsStr::new(venv_name) == environment_name)
        .then(|| environment_root.to_path_buf())
}

/// Finds the Python environment inside the virtual environment closest to the
/// notebook, walking up from the notebook directory to the worktree root.
///
/// "Closest" is measured by how deep the environment's parent directory is
/// inside the worktree, so `<notebook-dir>/.venv` beats `<parent>/.venv`.
/// Ties are broken by interpreter path to keep the result independent of
/// environment discovery order.
fn nearest_virtual_environment<'a>(
    notebook_path: &Path,
    worktree_root: &Path,
    discovered: &'a [KernelSpecification],
) -> Option<&'a KernelSpecification> {
    let notebook_directory = notebook_path.parent()?;

    discovered
        .iter()
        .filter_map(|specification| {
            let environment_root = virtual_environment_root(specification)?;
            let environment_parent = environment_root.parent()?;
            if !notebook_directory.starts_with(environment_parent) {
                return None;
            }
            // Environments outside the worktree (a single-file worktree root is
            // the notebook itself) still compete by how close they are.
            let depth = environment_root
                .strip_prefix(worktree_root)
                .map(|relative_root| relative_root.components().count())
                .unwrap_or_else(|_| environment_root.components().count());
            Some((depth, specification))
        })
        .max_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.path().to_string().cmp(&right.1.path().to_string()))
        })
        .map(|(_, specification)| specification)
}

/// The directory a notebook's kernel runs in. Jupyter starts kernels in the
/// notebook's folder; a worktree made of a single file has that file as its
/// root, so its parent is the folder.
fn kernel_working_directory(
    notebook_path: Option<&Path>,
    worktree_path: Option<&Path>,
    worktree_is_single_file: bool,
) -> PathBuf {
    if let Some(parent_directory) = notebook_path.and_then(Path::parent) {
        return parent_directory.to_path_buf();
    }
    if let Some(worktree_path) = worktree_path {
        if worktree_is_single_file {
            return worktree_path
                .parent()
                .unwrap_or(worktree_path)
                .to_path_buf();
        }
        return worktree_path.to_path_buf();
    }
    std::env::temp_dir()
}

/// Finds the closest `.venv` in the notebook's directory or any parent on the
/// local disk. Worktree-based discovery cannot see environments outside the
/// worktree, which for a single-file worktree excludes the notebook's folder.
async fn nearest_virtual_environment_on_disk(fs: &dyn Fs, notebook_path: &Path) -> Option<PathBuf> {
    let mut directory = notebook_path.parent();
    while let Some(current_directory) = directory {
        let interpreter = if cfg!(windows) {
            current_directory
                .join(".venv")
                .join("Scripts")
                .join("python.exe")
        } else {
            current_directory.join(".venv").join("bin").join("python")
        };
        if fs.is_file(&interpreter).await {
            return Some(interpreter);
        }
        directory = current_directory.parent();
    }
    None
}

/// Decides which kernel a notebook should run.
///
/// Explicit user selections are never overridden. When a `.venv` exists near
/// the notebook, its Python environment wins even if the current kernel (for
/// example a Conda base environment) already has ipykernel. Without any
/// `.venv`, the previous recommended/fallback selection is preserved.
fn choose_kernel_for_notebook(
    notebook_path: &Path,
    worktree_root: &Path,
    current: Option<&KernelSpecification>,
    explicit_selection: Option<&KernelSpecification>,
    recommended_path: Option<&str>,
    discovered: &[KernelSpecification],
) -> KernelAdoption {
    if explicit_selection.is_some() {
        return KernelAdoption::Keep;
    }

    if let Some(environment) = nearest_virtual_environment(notebook_path, worktree_root, discovered)
    {
        if !environment.has_ipykernel() {
            return KernelAdoption::MissingIpykernel(
                virtual_environment_root(environment)
                    .unwrap_or_else(|| PathBuf::from(environment.path().to_string())),
            );
        }
        if current == Some(environment) {
            return KernelAdoption::Keep;
        }
        return KernelAdoption::Switch(environment.clone());
    }

    let fallback = discovered
        .iter()
        .find(|specification| {
            recommended_path.is_some_and(|path| specification.path().as_ref() == path)
                && specification.has_ipykernel()
                && specification.language().as_ref() == "python"
        })
        .or_else(|| {
            discovered.iter().find(|specification| {
                matches!(specification, KernelSpecification::PythonEnv(_))
                    && specification.has_ipykernel()
                    && specification.language().as_ref() == "python"
            })
        });

    let Some(fallback) = fallback else {
        return KernelAdoption::Keep;
    };

    if let Some(current) = current {
        if current == fallback {
            return KernelAdoption::Keep;
        }
        if current.has_ipykernel() && !matches!(current, KernelSpecification::Jupyter(_)) {
            return KernelAdoption::Keep;
        }
    }

    KernelAdoption::Switch(fallback.clone())
}

/// The interpreter a kernel specification will run, when it points at an
/// absolute Python binary (a project virtual environment). Kernels addressed by
/// a bare program name (`python3`) have no interpreter to hand to a language
/// server and yield `None`.
fn python_interpreter_for_spec(spec: &KernelSpecification) -> Option<PathBuf> {
    match spec {
        KernelSpecification::PythonEnv(environment) => Some(environment.path.clone()),
        KernelSpecification::Jupyter(local) => {
            let program = local.kernelspec.argv.first()?;
            let path = PathBuf::from(program);
            path.is_absolute().then_some(path)
        }
        _ => None,
    }
}

impl Render for NotebookEditor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut key_context = KeyContext::new_with_defaults();
        key_context.add("NotebookEditor");
        key_context.set(
            "notebook_mode",
            match self.notebook_mode {
                NotebookMode::Command => "command",
                NotebookMode::Edit => "edit",
            },
        );

        let variables_visible = self.variables_panel.read(cx).is_visible();
        let variables_position = self.variables_panel.read(cx).position();
        let variables_panel = self.variables_panel.clone();
        let variables_bottom_height =
            self.effective_variables_bottom_height(window.viewport_size().height);
        let variables_right_width =
            self.effective_variables_right_width(window.viewport_size().width);

        v_flex()
            .size_full()
            .key_context(key_context)
            .capture_action(cx.listener(Self::capture_copy))
            .capture_action(cx.listener(Self::capture_cut))
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &OpenNotebook, window, cx| {
                this.open_notebook(&OpenNotebook, window, cx)
            }))
            .on_action(
                cx.listener(|this, _: &ClearOutputs, window, cx| this.clear_outputs(window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &Run, window, cx| this.run_current_cell(&Run, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.run_and_advance(action, window, cx)),
            )
            .on_action(cx.listener(|this, _: &RunAll, window, cx| this.run_cells(window, cx)))
            .on_action(
                cx.listener(|this, _: &RunAbove, window, cx| this.run_above(&RunAbove, window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &RunBelow, window, cx| this.run_below(&RunBelow, window, cx)),
            )
            .on_action(cx.listener(|this, _: &SplitCell, window, cx| {
                this.split_cell(&SplitCell, window, cx)
            }))
            .on_action(cx.listener(|this, _: &MergeCellAbove, window, cx| {
                this.merge_cell_above_mode(window, cx)
            }))
            .on_action(cx.listener(|this, _: &MergeCellBelow, window, cx| {
                this.merge_cell_below_mode(window, cx)
            }))
            .on_action(cx.listener(|this, _: &ConvertCellToMarkdown, window, cx| {
                this.convert_cell_to_markdown(window, cx)
            }))
            .on_action(cx.listener(|this, _: &ConvertCellToCode, window, cx| {
                this.convert_cell_to_code(window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.clear_selected_cell_outputs(action, window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.toggle_selected_cell_output(action, window, cx)
            }))
            .on_action(
                cx.listener(|this, action, window, cx| this.undo_structure(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.redo_structure(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &MoveCellUp, window, cx| this.move_cell_up(window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &MoveCellDown, window, cx| this.move_cell_down(window, cx)),
            )
            .on_action(cx.listener(|this, _: &AddMarkdownBlock, window, cx| {
                this.add_markdown_block(window, cx)
            }))
            .on_action(
                cx.listener(|this, _: &AddCodeBlock, window, cx| this.add_code_block(window, cx)),
            )
            .on_action(cx.listener(|this, _: &DeleteCell, window, cx| this.delete_cell(window, cx)))
            .on_action(
                cx.listener(|this, action, window, cx| this.enter_edit_mode(action, window, cx)),
            )
            .on_action(cx.listener(|this, action, window, cx| {
                this.handle_enter_command_mode(action, window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.select_next(action, SelectionMode::SelectOnly, window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.select_previous(action, SelectionMode::SelectOnly, window, cx)
            }))
            .on_action(cx.listener(Self::select_first))
            .on_action(cx.listener(Self::select_last))
            .on_action(cx.listener(|this, _: &MoveDown, window, cx| {
                this.select_next(
                    &Default::default(),
                    SelectionMode::SelectAndMove,
                    window,
                    cx,
                );
            }))
            .on_action(cx.listener(|this, _: &MoveUp, window, cx| {
                this.select_previous(
                    &Default::default(),
                    SelectionMode::SelectAndMove,
                    window,
                    cx,
                );
            }))
            .on_action(cx.listener(|this, _: &NotebookMoveDown, window, cx| {
                let Some(cell) = this.get_selected_cell() else {
                    return;
                };

                let Some(editor) = cell.editor(cx).cloned() else {
                    return;
                };

                let is_at_last_line = editor.update(cx, |editor, cx| {
                    let display_snapshot = editor.display_snapshot(cx);
                    let selections = editor.selections.all_display(&display_snapshot);
                    if let Some(selection) = selections.last() {
                        let head = selection.head();
                        let cursor_row = head.row();
                        let max_row = display_snapshot.max_point().row();

                        cursor_row >= max_row
                    } else {
                        false
                    }
                });

                if is_at_last_line {
                    this.select_next(
                        &Default::default(),
                        SelectionMode::SelectAndMove,
                        window,
                        cx,
                    );
                } else {
                    editor.update(cx, |editor, cx| {
                        editor.move_down(&Default::default(), window, cx);
                    });
                }
            }))
            .on_action(cx.listener(|this, _: &NotebookMoveUp, window, cx| {
                let Some(cell) = this.get_selected_cell() else {
                    return;
                };

                let Some(editor) = cell.editor(cx).cloned() else {
                    return;
                };

                let is_at_first_line = editor.update(cx, |editor, cx| {
                    let display_snapshot = editor.display_snapshot(cx);
                    let selections = editor.selections.all_display(&display_snapshot);
                    if let Some(selection) = selections.first() {
                        let head = selection.head();
                        let cursor_row = head.row();

                        cursor_row.0 == 0
                    } else {
                        false
                    }
                });

                if is_at_first_line {
                    this.select_previous(
                        &Default::default(),
                        SelectionMode::SelectAndMove,
                        window,
                        cx,
                    );
                } else {
                    editor.update(cx, |editor, cx| {
                        editor.move_up(&Default::default(), window, cx);
                    });
                }
            }))
            .on_action(
                cx.listener(|this, action, window, cx| this.restart_kernel(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.interrupt_kernel(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &RefreshVariables, _window, cx| this.refresh_variables(cx)),
            )
            .on_action(cx.listener(|this, _: &ToggleVariablesPanel, _window, cx| {
                this.toggle_variables_panel(cx)
            }))
            .on_action(
                cx.listener(|this, _: &ToggleVariablesPanelPosition, _window, cx| {
                    this.toggle_variables_panel_position(cx)
                }),
            )
            .child(
                v_flex()
                    .flex_1()
                    .w_full()
                    .h_full()
                    .child(
                        h_flex()
                            .flex_1()
                            .w_full()
                            .h_full()
                            .gap_2()
                            .child(
                                div()
                                    .flex_1()
                                    .h_full()
                                    .child(if self.cell_order.is_empty() {
                                        self.render_empty_state(cx).into_any_element()
                                    } else {
                                        self.cell_list(window, cx).into_any_element()
                                    }),
                            )
                            .when(
                                variables_visible
                                    && variables_position == VariablesPanelPosition::Right,
                                |this| {
                                    this.child(self.render_variables_right_divider(window, cx))
                                        .child(
                                            div()
                                                .w(variables_right_width)
                                                .h_full()
                                                .flex_none()
                                                .child(variables_panel.clone()),
                                        )
                                },
                            )
                            .child(self.render_notebook_controls(window, cx)),
                    )
                    .when(
                        variables_visible && variables_position == VariablesPanelPosition::Bottom,
                        |this| {
                            this.child(self.render_variables_divider(window, cx)).child(
                                div()
                                    .w_full()
                                    .h(variables_bottom_height)
                                    .flex_none()
                                    .child(variables_panel.clone()),
                            )
                        },
                    ),
            )
            .child(self.render_kernel_status_bar(window, cx))
    }
}

impl Focusable for NotebookEditor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

// Intended to be a NotebookBuffer
pub struct NotebookItem {
    project_path: ProjectPath,
    languages: Arc<LanguageRegistry>,
    // Raw notebook data
    notebook: nbformat::v4::Notebook,
    // Store our version of the notebook in memory (cell_order, cell_map)
    id: ProjectEntryId,
}

impl project::ProjectItem for NotebookItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        let path = path.clone();
        let project = project.clone();
        let languages = project.read(cx).languages().clone();

        // For single-file worktrees the relative path is empty, so fall back
        // to the absolute path to detect notebooks opened directly.
        let abs_path = project.read(cx).absolute_path(&path, cx);
        let is_notebook = path.path.extension().unwrap_or_default() == NOTEBOOK_EXTENSION
            || abs_path
                .as_ref()
                .and_then(|abs_path| abs_path.extension())
                .is_some_and(|extension| extension == NOTEBOOK_EXTENSION);

        if is_notebook {
            Some(cx.spawn(async move |cx| {
                // todo: watch for changes to the file
                let buffer = project
                    .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
                    .await?;
                let file_content = buffer.read_with(cx, |buffer, _| buffer.text());

                let notebook = if file_content.trim().is_empty() {
                    new_notebook()?
                } else {
                    let notebook = match nbformat::parse_notebook(&file_content) {
                        Ok(nb) => nb,
                        Err(_) => {
                            // Pre-process to ensure IDs exist
                            let mut json: serde_json::Value = serde_json::from_str(&file_content)?;
                            if let Some(cells) =
                                json.get_mut("cells").and_then(|c| c.as_array_mut())
                            {
                                for cell in cells {
                                    if cell.get("id").is_none() {
                                        cell["id"] =
                                            serde_json::Value::String(Uuid::new_v4().to_string());
                                    }
                                }
                            }
                            let file_content = serde_json::to_string(&json)?;
                            nbformat::parse_notebook(&file_content)?
                        }
                    };

                    match notebook {
                        nbformat::Notebook::V4(notebook) => notebook,
                        // 4.1 - 4.4 are converted to 4.5
                        nbformat::Notebook::Legacy(legacy_notebook) => {
                            // TODO: Decide if we want to mutate the notebook by including Cell IDs
                            // and any other conversions

                            nbformat::upgrade_legacy_notebook(legacy_notebook)?
                        }
                        nbformat::Notebook::V3(v3_notebook) => {
                            nbformat::upgrade_v3_notebook(v3_notebook)?
                        }
                    }
                };

                let id = project
                    .update(cx, |project, cx| {
                        project.entry_for_path(&path, cx).map(|entry| entry.id)
                    })
                    .context("Entry not found")?;

                Ok(cx.new(|_| NotebookItem {
                    project_path: path,
                    languages,
                    notebook,
                    id,
                }))
            }))
        } else {
            None
        }
    }

    fn entry_id(&self, _: &App) -> Option<ProjectEntryId> {
        Some(self.id)
    }

    fn project_path(&self, _: &App) -> Option<ProjectPath> {
        Some(self.project_path.clone())
    }

    fn is_dirty(&self) -> bool {
        // TODO: Track if notebook metadata or structure has changed
        false
    }
}

impl NotebookItem {
    pub fn language_name(&self) -> Option<String> {
        self.notebook
            .metadata
            .language_info
            .as_ref()
            .map(|l| l.name.clone())
            .or(self
                .notebook
                .metadata
                .kernelspec
                .as_ref()
                .and_then(|spec| spec.language.clone()))
    }

    pub fn notebook_language(&self) -> impl Future<Output = Option<Arc<Language>>> + use<> {
        let language_name = self.language_name();
        let languages = self.languages.clone();

        async move {
            if let Some(language_name) = language_name {
                languages.language_for_name(&language_name).await.ok()
            } else {
                None
            }
        }
    }
}

impl EventEmitter<()> for NotebookItem {}

impl EventEmitter<()> for NotebookEditor {}

impl EventEmitter<SearchEvent> for NotebookEditor {}

/// A match inside one cell's editor. Matches are ordered by cell, then by
/// position in the cell.
#[derive(Clone)]
pub struct NotebookSearchMatch {
    cell_id: CellId,
    range: std::ops::Range<Anchor>,
}

impl NotebookEditor {
    fn searchable_cell_editors(&self, cx: &App) -> Vec<(CellId, Entity<Editor>)> {
        self.cell_order
            .iter()
            .filter_map(|cell_id| {
                let editor = self.cell_map.get(cell_id)?.editor(cx)?.clone();
                Some((cell_id.clone(), editor))
            })
            .collect()
    }

    fn cell_editor(&self, cell_id: &CellId, cx: &App) -> Option<Entity<Editor>> {
        self.cell_map.get(cell_id)?.editor(cx).cloned()
    }
}

impl SearchableItem for NotebookEditor {
    type Match = NotebookSearchMatch;

    fn supported_options(&self) -> SearchOptions {
        SearchOptions {
            case: true,
            word: true,
            regex: true,
            replacement: true,
            selection: false,
            select_all: false,
            find_in_results: false,
        }
    }

    fn clear_matches(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for (_, editor) in self.searchable_cell_editors(cx) {
            editor.update(cx, |editor, cx| editor.clear_matches(window, cx));
        }
    }

    fn update_matches(
        &mut self,
        matches: &[Self::Match],
        active_match_index: Option<usize>,
        token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for (cell_id, editor) in self.searchable_cell_editors(cx) {
            let ranges: Vec<_> = matches
                .iter()
                .filter(|search_match| search_match.cell_id == cell_id)
                .map(|search_match| search_match.range.clone())
                .collect();
            let active_in_cell = active_match_index
                .filter(|index| {
                    matches
                        .get(*index)
                        .is_some_and(|search_match| search_match.cell_id == cell_id)
                })
                .map(|index| {
                    matches[..index]
                        .iter()
                        .filter(|search_match| search_match.cell_id == cell_id)
                        .count()
                });
            editor.update(cx, |editor, cx| {
                editor.update_matches(&ranges, active_in_cell, token, window, cx)
            });
        }
    }

    fn query_suggestion(
        &mut self,
        seed_query_override: Option<SeedQuerySetting>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> String {
        let Some(editor) = self
            .focused_cell()
            .and_then(|cell| cell.editor(cx))
            .cloned()
        else {
            return String::new();
        };
        editor.update(cx, |editor, cx| {
            editor.query_suggestion(seed_query_override, window, cx)
        })
    }

    fn activate_match(
        &mut self,
        index: usize,
        matches: &[Self::Match],
        token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(active_match) = matches.get(index) else {
            return;
        };
        let cell_id = active_match.cell_id.clone();
        let Some(cell_index) = self
            .cell_order
            .iter()
            .position(|candidate| *candidate == cell_id)
        else {
            return;
        };
        let Some(editor) = self.cell_editor(&cell_id, cx) else {
            return;
        };

        if let Some(Cell::Markdown(markdown_cell)) = self.cell_map.get(&cell_id) {
            markdown_cell.update(cx, |markdown_cell, cx| {
                if !markdown_cell.is_editing() {
                    markdown_cell.set_editing(true);
                    cx.notify();
                }
            });
        }

        let ranges: Vec<_> = matches
            .iter()
            .filter(|search_match| search_match.cell_id == cell_id)
            .map(|search_match| search_match.range.clone())
            .collect();
        let index_in_cell = matches[..index]
            .iter()
            .filter(|search_match| search_match.cell_id == cell_id)
            .count();

        self.set_selected_index(cell_index, true, window, cx);
        editor.update(cx, |editor, cx| {
            editor.activate_match(index_in_cell, &ranges, token, window, cx)
        });
        cx.notify();
    }

    fn select_matches(
        &mut self,
        _matches: &[Self::Match],
        _token: SearchToken,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) {
    }

    fn replace(
        &mut self,
        search_match: &Self::Match,
        query: &SearchQuery,
        token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(editor) = self.cell_editor(&search_match.cell_id, cx) else {
            return;
        };
        editor.update(cx, |editor, cx| {
            editor.replace(&search_match.range, query, token, window, cx)
        });
    }

    fn find_matches(
        &mut self,
        query: Arc<SearchQuery>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Vec<Self::Match>> {
        let searches: Vec<_> = self
            .searchable_cell_editors(cx)
            .into_iter()
            .map(|(cell_id, editor)| {
                let task = editor.update(cx, |editor, cx| {
                    editor.find_matches(query.clone(), window, cx)
                });
                (cell_id, task)
            })
            .collect();

        cx.spawn(async move |_, _| {
            let mut all_matches = Vec::new();
            for (cell_id, task) in searches {
                for range in task.await {
                    all_matches.push(NotebookSearchMatch {
                        cell_id: cell_id.clone(),
                        range,
                    });
                }
            }
            all_matches
        })
    }

    fn active_match_index(
        &mut self,
        direction: Direction,
        matches: &[Self::Match],
        _token: SearchToken,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<usize> {
        if matches.is_empty() {
            return None;
        }

        let selected_cell_index = self.selected_cell_index;
        let cursor = self
            .cell_order
            .get(selected_cell_index)
            .and_then(|cell_id| self.cell_editor(cell_id, cx))
            .map(|editor| {
                let editor = editor.read(cx);
                (
                    *editor.selections.newest_anchor(),
                    editor.buffer().read(cx).snapshot(cx),
                )
            });

        let order_against_cursor = |search_match: &NotebookSearchMatch| {
            let match_cell_index = self
                .cell_order
                .iter()
                .position(|cell_id| *cell_id == search_match.cell_id)?;
            Some(match match_cell_index.cmp(&selected_cell_index) {
                std::cmp::Ordering::Equal => match &cursor {
                    Some((selection, snapshot)) => {
                        let ordering = search_match.range.start.cmp(&selection.start, snapshot);
                        if ordering.is_eq() && selection.start != selection.end {
                            std::cmp::Ordering::Less
                        } else {
                            ordering
                        }
                    }
                    None => std::cmp::Ordering::Equal,
                },
                ordering => ordering,
            })
        };

        match direction {
            Direction::Next => matches
                .iter()
                .position(|search_match| {
                    order_against_cursor(search_match).is_some_and(|ordering| ordering.is_ge())
                })
                .or(Some(0)),
            Direction::Prev => matches
                .iter()
                .rposition(|search_match| {
                    order_against_cursor(search_match).is_some_and(|ordering| ordering.is_lt())
                })
                .or(Some(matches.len() - 1)),
        }
    }
}

// pub struct NotebookControls {
//     pane_focused: bool,
//     active_item: Option<Box<dyn ItemHandle>>,
//     // subscription: Option<Subscription>,
// }

// impl NotebookControls {
//     pub fn new() -> Self {
//         Self {
//             pane_focused: false,
//             active_item: Default::default(),
//             // subscription: Default::default(),
//         }
//     }
// }

// impl EventEmitter<ToolbarItemEvent> for NotebookControls {}

// impl Render for NotebookControls {
//     fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
//         div().child("notebook controls")
//     }
// }

// impl ToolbarItemView for NotebookControls {
//     fn set_active_pane_item(
//         &mut self,
//         active_pane_item: Option<&dyn workspace::ItemHandle>,
//         window: &mut Window, cx: &mut Context<Self>,
//     ) -> workspace::ToolbarItemLocation {
//         cx.notify();
//         self.active_item = None;

//         let Some(item) = active_pane_item else {
//             return ToolbarItemLocation::Hidden;
//         };

//         ToolbarItemLocation::PrimaryLeft
//     }

//     fn pane_focus_update(&mut self, pane_focused: bool, _window: &mut Window, _cx: &mut Context<Self>) {
//         self.pane_focused = pane_focused;
//     }
// }

impl Item for NotebookEditor {
    type Event = ();

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>>
    where
        Self: Sized,
    {
        Task::ready(Some(cx.new(|cx| {
            Self::new(self.project.clone(), self.notebook_item.clone(), window, cx)
        })))
    }

    fn buffer_kind(&self, _: &App) -> workspace::item::ItemBufferKind {
        workspace::item::ItemBufferKind::Singleton
    }

    fn added_to_workspace(
        &mut self,
        workspace: &mut Workspace,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.workspace = Some((workspace.weak_handle(), workspace.database_id()));
        self.apply_workspace_to_cells(cx);
    }

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(gpui::EntityId, &dyn project::ProjectItem),
    ) {
        f(self.notebook_item.entity_id(), self.notebook_item.read(cx))
    }

    fn act_as_type<'a>(
        &'a self,
        type_id: TypeId,
        self_handle: &'a Entity<Self>,
        cx: &'a App,
    ) -> Option<AnyEntity> {
        if type_id == TypeId::of::<Self>() {
            Some(self_handle.clone().into())
        } else if type_id == TypeId::of::<Editor>() {
            self.focused_cell()
                .or_else(|| self.get_selected_cell())
                .and_then(|cell| cell.editor(cx))
                .map(|editor| editor.clone().into())
        } else {
            None
        }
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        self.notebook_item
            .read(cx)
            .project_path
            .path
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_default()
            .into()
    }

    fn tab_content(&self, params: TabContentParams, window: &Window, cx: &App) -> AnyElement {
        Label::new(self.tab_content_text(params.detail.unwrap_or(0), cx))
            .single_line()
            .color(params.text_color())
            .when(params.preview, |this| this.italic())
            .into_any_element()
    }

    fn tab_icon(&self, _window: &Window, _cx: &App) -> Option<Icon> {
        Some(IconName::Book.into())
    }

    fn show_toolbar(&self) -> bool {
        true
    }

    // TODO
    fn pixel_position_of_cursor(&self, _: &App) -> Option<Point<Pixels>> {
        None
    }

    fn as_searchable(
        &self,
        handle: &Entity<Self>,
        _: &App,
    ) -> Option<Box<dyn SearchableItemHandle>> {
        Some(Box::new(handle.clone()))
    }

    fn set_nav_history(
        &mut self,
        _: workspace::ItemNavHistory,
        _window: &mut Window,
        _: &mut Context<Self>,
    ) {
        // TODO
    }

    fn can_save(&self, _cx: &App) -> bool {
        true
    }

    fn save(
        &mut self,
        _options: SaveOptions,
        project: Entity<Project>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if let Some(notebook_lsp) = self.notebook_lsp.clone() {
            notebook_lsp.update(cx, |notebook_lsp, cx| notebook_lsp.save(cx));
        }
        self.save_impl(SaveDestination::CurrentPath, project, cx)
    }

    fn save_as(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.save_impl(SaveDestination::NewPath(path), project, cx)
    }

    fn reload(
        &mut self,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let project_path = self.notebook_item.read(cx).project_path.clone();
        let languages = self.languages.clone();
        let notebook_language = self.notebook_language.clone();

        cx.spawn_in(window, async move |this, cx| {
            let buffer = this
                .update(cx, |this, cx| {
                    this.project
                        .update(cx, |project, cx| project.open_buffer(project_path, cx))
                })?
                .await?;

            // The project does not necessarily reload this buffer on its own
            // when the file changes on disk, so read it back from disk before
            // rebuilding the cells from its text.
            this.update(cx, |this, cx| {
                this.project.update(cx, |project, cx| {
                    project.reload_buffers([buffer.clone()].into_iter().collect(), false, cx)
                })
            })?
            .await?;

            let file_content = buffer.read_with(cx, |buffer, _| buffer.text());
            let (mut notebook, missing_cell_ids) =
                parse_notebook_content_with_missing_ids(&file_content)?;

            this.update_in(cx, |this, window, cx| {
                let scroll_anchor = this.capture_scroll_anchor();
                let previous_selected_index = this.selected_cell_index;
                let previous_selected_cell_id =
                    this.cell_order.get(previous_selected_index).cloned();
                let previously_focused_cell_id = focused_cell_editor_in_window(this, window, cx);
                let previous_cell_order = this.cell_order.clone();
                let previous_cell_map = std::mem::take(&mut this.cell_map);
                this.focused_cell_id = None;

                let matched_cell_ids = match_notebook_cells(
                    &notebook,
                    &missing_cell_ids,
                    &previous_cell_order,
                    &previous_cell_map,
                    cx,
                );
                let new_cell_order: Vec<CellId> = matched_cell_ids
                    .iter()
                    .enumerate()
                    .map(|(index, matched)| {
                        let cell_id = matched
                            .clone()
                            .unwrap_or_else(|| notebook.cells[index].id().clone());
                        set_notebook_cell_id(&mut notebook.cells[index], cell_id.clone());
                        cell_id
                    })
                    .collect();

                let mut cell_map = HashMap::default();
                let mut remeasure_indices = Vec::new();
                for (index, notebook_cell) in notebook.cells.iter().enumerate() {
                    let cell_id = new_cell_order[index].clone();
                    let existing_cell = previous_cell_map.get(&cell_id);
                    let cell_entity = match existing_cell {
                        Some(existing_cell)
                            if notebook_cells_have_same_type(existing_cell, notebook_cell) =>
                        {
                            if !notebook_cells_have_same_content(existing_cell, notebook_cell, cx) {
                                existing_cell.update_from_notebook_cell(notebook_cell, window, cx);
                                remeasure_indices.push(index);
                            }
                            existing_cell.clone()
                        }
                        _ => {
                            remeasure_indices.push(index);
                            Cell::load(
                                notebook_cell,
                                &languages,
                                notebook_language.clone(),
                                &this.project,
                                window,
                                cx,
                            )
                        }
                    };
                    cell_map.insert(cell_id, cell_entity);
                }

                let (removed_indices, kept_cell_ids) =
                    cell_order_edits(&previous_cell_order, &new_cell_order);
                for index in removed_indices {
                    this.cell_order.remove(index);
                    this.cell_list.splice(index..index + 1, 0);
                }
                let mut position = 0;
                for cell_id in &new_cell_order {
                    if kept_cell_ids.contains(cell_id) {
                        position += 1;
                    } else {
                        this.cell_order.insert(position, cell_id.clone());
                        this.cell_list.splice(position..position, 1);
                        position += 1;
                    }
                }

                this.original_cell_order = this.cell_order.clone();
                this.cell_map = cell_map;
                this.undo_stack.clear();
                this.redo_stack.clear();
                this.selected_cell_index = previous_selected_cell_id
                    .as_ref()
                    .and_then(|cell_id| {
                        Self::nearest_cell_index(cell_id, &previous_cell_order, &this.cell_order)
                    })
                    .unwrap_or_else(|| {
                        previous_selected_index.min(this.cell_order.len().saturating_sub(1))
                    });
                for index in remeasure_indices {
                    this.cell_list.remeasure_items(index..index + 1);
                }
                this.restore_scroll_anchor(scroll_anchor, &previous_cell_order);
                // Keep the raw notebook in sync so a later save preserves
                // metadata that changed on disk while the item was reloaded.
                this.notebook_item.update(cx, |item, _cx| {
                    item.notebook = notebook.clone();
                });
                this.refresh_cell_registry(cx);
                // The cells were rebuilt from disk, so they are the new clean
                // baseline until the user edits them again.
                this.mark_as_saved(cx);

                let focus_cell_id = previously_focused_cell_id.and_then(|cell_id| {
                    if this.cell_map.contains_key(&cell_id) {
                        Some(cell_id)
                    } else {
                        this.cell_order.get(this.selected_cell_index).cloned()
                    }
                });
                if let Some(cell_id) = focus_cell_id {
                    this.focused_cell_id = Some(cell_id.clone());
                    this.notebook_mode = NotebookMode::Edit;
                    this.focus_cell_editor(&cell_id, window, cx);
                } else {
                    this.notebook_mode = NotebookMode::Command;
                }
            })?;

            Ok(())
        })
    }

    fn is_dirty(&self, cx: &App) -> bool {
        self.has_structural_changes() || self.has_content_changes(cx)
    }
}

fn variable_reply_text(reply: &ExecuteReply) -> Result<String> {
    if reply.status != ReplyStatus::Ok {
        let error = reply
            .error
            .as_ref()
            .map(|error| format!("{}: {}", error.ename, error.evalue))
            .unwrap_or_else(|| "the kernel rejected the request".to_string());
        return Err(anyhow!(error));
    }

    let expressions = reply
        .user_expressions
        .as_ref()
        .ok_or_else(|| anyhow!("the kernel did not return the requested expression"))?;
    let expression = expressions
        .get("payload")
        .ok_or_else(|| anyhow!("the kernel omitted the variables expression"))?;
    match expression {
        ExpressionResult::Ok { data, .. } => data
            .content
            .iter()
            .find_map(|media| match media {
                MediaType::Plain(text) => Some(text.clone()),
                _ => None,
            })
            .ok_or_else(|| anyhow!("the kernel response did not contain plain text")),
        ExpressionResult::Error { ename, evalue, .. } => Err(anyhow!("{ename}: {evalue}")),
    }
}

impl ProjectItem for NotebookEditor {
    type Item = NotebookItem;

    fn for_project_item(
        project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(project, item, window, cx)
    }
}

impl KernelSession for NotebookEditor {
    fn route(&mut self, message: &JupyterMessage, window: &mut Window, cx: &mut Context<Self>) {
        // Completion and inspection replies are not handled anywhere else, so
        // resolve them here before falling through to the execution routing.
        self.kernel_completer.resolve(message);

        if let JupyterMessageContent::ExecuteReply(reply) = &message.content {
            if let Some(parent_header) = &message.parent_header {
                if let Some(request) = self.variable_requests.remove(&parent_header.msg_id) {
                    self.handle_variable_reply(request, reply, cx);
                    cx.notify();
                    return;
                }
            }
        }

        // Handle kernel status updates (these are broadcast to all)
        if let JupyterMessageContent::Status(status) = &message.content {
            self.kernel.set_execution_state(&status.execution_state);
            self.sync_kernel_completer();
            cx.notify();

            if self.deferred_kernel_adoption.is_some()
                && matches!(self.kernel.status(), KernelStatus::Idle)
            {
                log::info!(
                    "notebook: applying the deferred kernel adoption now that the kernel is idle"
                );
                self.adopt_project_environment_kernel(window, cx);
            }
        }

        if let JupyterMessageContent::KernelInfoReply(reply) = &message.content {
            self.kernel.set_kernel_info(reply);

            if let Ok(language_info) = serde_json::from_value::<nbformat::v4::LanguageInfo>(
                serde_json::to_value(&reply.language_info).unwrap(),
            ) {
                self.notebook_item.update(cx, |item, cx| {
                    item.notebook.metadata.language_info = Some(language_info);
                    cx.emit(());
                });
            }
            cx.notify();
        }

        let received_reply = matches!(message.content, JupyterMessageContent::ExecuteReply(_));
        let received_idle = matches!(
            &message.content,
            JupyterMessageContent::Status(status)
                if matches!(status.execution_state, runtimelib::ExecutionState::Idle)
        );

        if let Some(parent_message_id) = message.parent_header.as_ref().map(|header| &header.msg_id)
        {
            let cell_id = if let Some(pending) = self.execution_requests.get_mut(parent_message_id)
            {
                if received_reply {
                    pending.reply_received = true;
                }
                if received_idle {
                    pending.idle_received = true;
                }
                Some(pending.cell_id.clone())
            } else {
                None
            };

            if let Some(cell_id) = cell_id {
                if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                    cell.update(cx, |cell, cx| {
                        cell.handle_message(message, window, cx);
                    });
                }

                if self
                    .execution_requests
                    .get(parent_message_id)
                    .is_some_and(PendingExecution::is_complete)
                {
                    self.execution_requests.remove(parent_message_id);
                    if self.execution_requests.is_empty() {
                        self.refresh_variables(cx);
                    }
                }
            }
        }
    }

    fn kernel_errored(&mut self, error_message: String, cx: &mut Context<Self>) {
        self.kernel = Kernel::ErroredLaunch(error_message);
        self.sync_kernel_completer();
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notebook::cell::RunnableCell;
    use editor::{MultiBufferOffset, SelectionEffects};
    use gpui::{TestAppContext, VisualTestContext};
    use project::{FakeFs, Fs as _, Project, ProjectItem as _};
    use serde_json::json;
    use settings::{SettingsContent, SettingsStore};
    use util::path;
    use util::rel_path::rel_path;

    const NOTEBOOK_WITH_ONE_CODE_CELL: &str = r#"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-one",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["print('hello')"]
            }
        ]
    }"#;

    /// When the configured interpreter doesn't exist (e.g. Python isn't installed),
    /// running a cell must not leave it stuck in the executing state. It should
    /// instead surface the kernel launch error as an error output on the cell.
    #[gpui::test]
    async fn test_run_cell_with_missing_interpreter_shows_error(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        // Select a kernel whose interpreter doesn't exist, simulating a machine
        // where Python isn't installed properly. This is the same path the
        // kernel picker uses.
        let missing_interpreter = path!("/nonexistent/python3");
        let broken_spec = KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from(missing_interpreter),
            kernelspec: JupyterKernelspec {
                argv: vec![
                    missing_interpreter.to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        });
        cx.update(|cx| {
            ReplStore::global(cx).update(cx, |store, cx| {
                store.set_active_kernelspec(worktree_id, broken_spec, cx);
            })
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("test.ipynb").into(),
                    },
                    cx,
                )
                .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        // Don't render the notebook UI itself: its animated kernel status icon
        // schedules a new frame on every render, which makes `run_until_parked`
        // spin forever in tests. The editor entity is created inside an empty
        // window instead; we are testing execution behavior, not rendering.
        let cx = cx.add_empty_window();

        // Launching a kernel probes real TCP ports on localhost, which the
        // deterministic test scheduler cannot drive.
        cx.executor().allow_parking();

        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        // Creating the editor launches the kernel. Wait for the actual launch
        // task, which fails because the interpreter cannot be spawned.
        let pending_kernel = editor.read_with(cx, |editor, _| match &editor.kernel {
            Kernel::StartingKernel(task) => task.clone(),
            _ => panic!("kernel should be starting right after the editor is created"),
        });
        pending_kernel.await;

        editor.read_with(cx, |editor, _| {
            assert!(
                matches!(editor.kernel, Kernel::ErroredLaunch(_)),
                "kernel launch should fail, instead status is: {}",
                editor.kernel.status().to_string()
            );
        });

        // Run the (only) cell via the production action handler.
        editor.update_in(cx, |editor, window, cx| {
            editor.run_current_cell(&Run, window, cx);
        });

        editor.read_with(cx, |editor, cx| {
            let cell_id = editor.cell_order.first().expect("notebook has one cell");
            let Some(Cell::Code(cell)) = editor.cell_map.get(cell_id) else {
                panic!("expected a code cell");
            };
            let cell = cell.read(cx);

            assert!(
                !cell.is_executing(),
                "cell must not be stuck in the executing state when the kernel is not running"
            );

            let nbformat::v4::Cell::Code { outputs, .. } = cell.to_nbformat_cell(cx) else {
                panic!("expected a code cell");
            };
            match outputs.as_slice() {
                [nbformat::v4::Output::Error(error)] => {
                    assert_eq!(error.ename, "Kernel Error");
                    let traceback = error.traceback.join("\n");
                    assert!(
                        traceback.contains("the kernel failed to launch"),
                        "error output should explain why the cell could not run, got: {traceback}"
                    );
                }
                other => panic!("expected a single error output, got: {other:?}"),
            }
        });
    }

    /// Opening a notebook as a single file (its own worktree) leaves the
    /// worktree-relative path empty, so only the absolute path carries the
    /// `.ipynb` extension. `try_open` must still recognize it as a notebook.
    #[gpui::test]
    async fn test_open_single_file_notebook(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "single.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let project =
            Project::test(fs.clone(), [path!("/notebooks/single.ipynb").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| {
            let worktree = project.worktrees(cx).next().unwrap();
            let worktree = worktree.read(cx);
            assert!(
                worktree.is_single_file(),
                "opening a bare .ipynb should create a single-file worktree"
            );
            ProjectPath {
                worktree_id: worktree.id(),
                path: worktree.root_entry().unwrap().path.clone(),
            }
        });

        assert!(
            project_path.path.extension().is_none(),
            "single-file worktree relative path should have no extension"
        );

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("single-file .ipynb should open as a notebook")
            })
            .await
            .expect("notebook should parse");

        notebook_item.read_with(cx, |item, _| {
            assert_eq!(item.notebook.cells.len(), 1);
        });
    }

    /// Notebooks must be saved through the project rather than through the
    /// client's own filesystem, otherwise a remote notebook's path is resolved
    /// against the local machine and the save fails.
    #[gpui::test]
    async fn test_save_goes_through_the_project(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("test.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        // Held across the save: a save that bypasses the project writes the file
        // behind this buffer's back, leaving it stale.
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer(project_path.clone(), cx)
            })
            .await
            .expect("notebook buffer should open");

        // Rendering the notebook animates the kernel status icon, which makes
        // `run_until_parked` spin forever; only the editor entity is needed here.
        let cx = cx.add_empty_window();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let cell_editor = notebook_editor.read_with(cx, |notebook_editor, cx| {
            let cell_id = notebook_editor
                .cell_order
                .first()
                .expect("notebook has one cell");
            let Some(Cell::Code(cell)) = notebook_editor.cell_map.get(cell_id) else {
                panic!("expected a code cell");
            };
            cell.read(cx).editor().clone()
        });
        cell_editor.update_in(cx, |cell_editor, window, cx| {
            cell_editor.set_text("print('goodbye')", window, cx);
        });

        notebook_editor
            .update_in(cx, |notebook_editor, window, cx| {
                notebook_editor.save(SaveOptions::default(), project.clone(), window, cx)
            })
            .await
            .expect("saving the notebook should succeed");

        let saved = String::from_utf8(
            fs.read_file_sync(path!("/notebooks/test.ipynb"))
                .expect("notebook should still exist"),
        )
        .expect("notebook should be valid UTF-8");
        assert!(
            saved.contains("print('goodbye')"),
            "the edited cell should be written to the notebook, got: {saved}"
        );

        buffer.read_with(cx, |buffer, _| {
            assert_eq!(
                buffer.text(),
                saved,
                "the project's buffer should hold the saved notebook"
            );
            assert!(!buffer.is_dirty(), "saving should leave the buffer clean");
        });
    }

    const NOTEBOOK_WITH_TWO_CODE_CELLS: &str = r#"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-one",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["print('one')"]
            },
            {
                "cell_type": "code",
                "id": "cell-two",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["print('two')"]
            }
        ]
    }"#;

    /// "Add to Agent Thread" resolves the focused editor through the active
    /// item's `act_as::<Editor>`. For notebooks that must be the selected cell's
    /// editor, not the notebook item itself.
    #[gpui::test]
    async fn test_selected_cell_editor_is_resolved_for_agent_context(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "two.ipynb": NOTEBOOK_WITH_TWO_CODE_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("two.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        // Rendering the notebook animates the kernel status icon, which makes
        // `run_until_parked` spin forever; only the editor entity is needed here.
        let cx = cx.add_empty_window();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let cell_editors = notebook_editor.read_with(cx, |notebook_editor, cx| {
            notebook_editor
                .cell_order
                .iter()
                .map(|cell_id| match notebook_editor.cell_map.get(cell_id) {
                    Some(Cell::Code(cell)) => cell.read(cx).editor().clone(),
                    _ => panic!("expected code cells"),
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(cell_editors.len(), 2);

        for (index, expected) in cell_editors.iter().enumerate() {
            notebook_editor.update(cx, |notebook_editor, _| {
                notebook_editor.selected_cell_index = index;
            });
            let resolved = cx.update(|_window, app| {
                let item: &dyn ItemHandle = &notebook_editor;
                item.act_as::<Editor>(app)
            });
            assert_eq!(
                resolved
                    .expect("notebook must expose its selected cell editor")
                    .entity_id(),
                expected.entity_id(),
                "selecting cell {index} must resolve its editor for agent context"
            );
        }
    }

    #[gpui::test]
    async fn test_focused_cell_editor_takes_priority_over_selected(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "two.ipynb": NOTEBOOK_WITH_TWO_CODE_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("two.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let cell_editors = notebook_editor.read_with(cx, |notebook_editor, cx| {
            notebook_editor
                .cell_order
                .iter()
                .map(|cell_id| match notebook_editor.cell_map.get(cell_id) {
                    Some(Cell::Code(cell)) => cell.read(cx).editor().clone(),
                    _ => panic!("expected code cells"),
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(cell_editors.len(), 2);

        notebook_editor.update(cx, |notebook_editor, cx| {
            let second_cell = notebook_editor.cell_order[1].clone();
            notebook_editor.select_cell_by_id(&second_cell, cx);
            notebook_editor.selected_cell_index = 0;
        });

        let resolved = cx.update(|_window, app| {
            let item: &dyn ItemHandle = &notebook_editor;
            item.act_as::<Editor>(app)
        });
        assert_eq!(
            resolved
                .expect("notebook must expose its focused cell editor")
                .entity_id(),
            cell_editors[1].entity_id(),
            "the focused cell must win over the selected cell for agent context"
        );
    }

    /// The language server only receives an interpreter when the kernel names an
    /// absolute binary. A venv-backed `PythonEnv` and an absolute Jupyter
    /// kernelspec do; the generic `python3` fallback does not.
    #[test]
    fn empty_file_starts_a_new_python_notebook_with_one_empty_cell() {
        for content in ["", "  \n\t"] {
            let notebook = parse_notebook_content(content).unwrap();
            assert_eq!(notebook.cells.len(), 1);
            assert!(matches!(
                &notebook.cells[0],
                nbformat::v4::Cell::Code { source, outputs, .. }
                    if source.is_empty() && outputs.is_empty()
            ));
            assert_eq!(
                notebook
                    .metadata
                    .language_info
                    .as_ref()
                    .map(|info| info.name.as_str()),
                Some("python")
            );
            assert_eq!(
                notebook
                    .metadata
                    .kernelspec
                    .as_ref()
                    .map(|spec| spec.name.as_str()),
                Some("python3")
            );
        }
    }

    #[test]
    fn python_interpreter_for_spec_extracts_absolute_paths_only() {
        let venv_interpreter = path!("/project/.venv/bin/python");
        let python_environment = KernelSpecification::PythonEnv(PythonEnvKernelSpecification {
            name: "Venv".to_string(),
            path: PathBuf::from(venv_interpreter),
            kernelspec: JupyterKernelspec {
                argv: vec![venv_interpreter.to_string()],
                display_name: "Venv".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
            has_ipykernel: true,
            environment_kind: Some("venv".to_string()),
        });
        assert_eq!(
            python_interpreter_for_spec(&python_environment),
            Some(PathBuf::from(venv_interpreter))
        );

        let bare_python = KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from("python3"),
            kernelspec: JupyterKernelspec {
                argv: vec!["python3".to_string()],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        });
        assert_eq!(python_interpreter_for_spec(&bare_python), None);
    }

    fn python_environment_specification(
        interpreter: &str,
        environment_kind: &str,
        has_ipykernel: bool,
    ) -> KernelSpecification {
        let interpreter = PathBuf::from(interpreter);
        let executable = interpreter.to_string_lossy().into_owned();
        KernelSpecification::PythonEnv(PythonEnvKernelSpecification {
            name: format!("Python ({environment_kind})"),
            path: interpreter,
            kernelspec: JupyterKernelspec {
                argv: vec![executable.clone()],
                display_name: executable,
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
            has_ipykernel,
            environment_kind: Some(environment_kind.to_string()),
        })
    }

    fn generic_python_jupyter_specification() -> KernelSpecification {
        KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from("python3"),
            kernelspec: JupyterKernelspec {
                argv: vec!["python3".to_string()],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        })
    }

    /// A `.venv` next to the notebook must win over a Conda base kernel that
    /// already has ipykernel.
    #[test]
    fn notebook_folder_venv_wins_over_conda_base_with_ipykernel() {
        let worktree_root = Path::new(path!("/project"));
        let notebook_path = Path::new(path!("/project/notebook.ipynb"));
        let conda_base =
            python_environment_specification(path!("/opt/conda/bin/python"), "Conda", true);
        let notebook_venv =
            python_environment_specification(path!("/project/.venv/bin/python"), "venv", true);
        let discovered = vec![conda_base.clone(), notebook_venv.clone()];

        let adoption = choose_kernel_for_notebook(
            notebook_path,
            worktree_root,
            Some(&conda_base),
            None,
            None,
            &discovered,
        );

        assert!(matches!(
            adoption,
            KernelAdoption::Switch(ref selected) if selected == &notebook_venv
        ));
    }

    /// A `.venv` in a parent directory is found, and the one closest to the
    /// notebook wins when several exist.
    #[test]
    fn venv_in_parent_directory_is_found_and_nearest_venv_wins() {
        let worktree_root = Path::new(path!("/project"));
        let notebook_path = Path::new(path!("/project/sub/notebook.ipynb"));
        let parent_venv =
            python_environment_specification(path!("/project/.venv/bin/python"), "venv", true);
        let closest_venv =
            python_environment_specification(path!("/project/sub/.venv/bin/python"), "venv", true);

        let adoption = choose_kernel_for_notebook(
            notebook_path,
            worktree_root,
            None,
            None,
            None,
            std::slice::from_ref(&parent_venv),
        );
        assert!(matches!(
            adoption,
            KernelAdoption::Switch(ref selected) if selected == &parent_venv
        ));

        let discovered = vec![parent_venv.clone(), closest_venv.clone()];
        let adoption = choose_kernel_for_notebook(
            notebook_path,
            worktree_root,
            Some(&parent_venv),
            None,
            None,
            &discovered,
        );
        assert!(matches!(
            adoption,
            KernelAdoption::Switch(ref selected) if selected == &closest_venv
        ));
    }

    /// A kernel chosen by the user is never replaced by a discovered `.venv`.
    #[test]
    fn explicit_selection_is_not_overridden_by_nearest_venv() {
        let worktree_root = Path::new(path!("/project"));
        let notebook_path = Path::new(path!("/project/notebook.ipynb"));
        let selected =
            python_environment_specification(path!("/opt/conda/bin/python"), "Conda", true);
        let notebook_venv =
            python_environment_specification(path!("/project/.venv/bin/python"), "venv", true);
        let discovered = vec![selected.clone(), notebook_venv];

        let adoption = choose_kernel_for_notebook(
            notebook_path,
            worktree_root,
            Some(&selected),
            Some(&selected),
            None,
            &discovered,
        );

        assert!(matches!(adoption, KernelAdoption::Keep));
    }

    /// Without any `.venv`, the previous recommendation/fallback selection is
    /// unchanged.
    #[test]
    fn without_venv_the_previous_fallback_is_kept() {
        let worktree_root = Path::new(path!("/project"));
        let notebook_path = Path::new(path!("/project/notebook.ipynb"));
        let conda_base =
            python_environment_specification(path!("/opt/conda/bin/python"), "Conda", true);
        let generic = generic_python_jupyter_specification();
        let discovered = vec![conda_base.clone()];
        let recommended_path = conda_base.path();

        let adoption = choose_kernel_for_notebook(
            notebook_path,
            worktree_root,
            Some(&generic),
            None,
            Some(recommended_path.as_ref()),
            &discovered,
        );
        assert!(matches!(
            adoption,
            KernelAdoption::Switch(ref selected) if selected == &conda_base
        ));

        let adoption = choose_kernel_for_notebook(
            notebook_path,
            worktree_root,
            Some(&conda_base),
            None,
            Some(recommended_path.as_ref()),
            &discovered,
        );
        assert!(matches!(adoption, KernelAdoption::Keep));

        let adoption =
            choose_kernel_for_notebook(notebook_path, worktree_root, None, None, None, &[]);
        assert!(matches!(adoption, KernelAdoption::Keep));
    }

    /// A single-file worktree's root is the notebook itself, so the kernel must
    /// run in the notebook's folder rather than trying to `cd` into the file.
    #[test]
    fn kernel_working_directory_uses_notebook_folder_for_single_file_worktree() {
        let notebook_path = Path::new(path!("/notebooks/test.ipynb"));

        assert_eq!(
            kernel_working_directory(Some(notebook_path), Some(notebook_path), true),
            PathBuf::from(path!("/notebooks"))
        );
        assert_eq!(
            kernel_working_directory(None, Some(notebook_path), true),
            PathBuf::from(path!("/notebooks"))
        );
        assert_eq!(
            kernel_working_directory(None, Some(Path::new(path!("/notebooks"))), false),
            PathBuf::from(path!("/notebooks"))
        );
    }

    /// A `.venv` in the notebook's folder is outside a single-file worktree, so
    /// the parent-directory scan has to find it on disk.
    #[gpui::test]
    async fn single_file_worktree_finds_venv_next_to_notebook(cx: &mut TestAppContext) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({
                "test.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL,
                ".venv": { "bin": { "python": "" } }
            }),
        )
        .await;

        let notebook_path = Path::new(path!("/notebooks/test.ipynb"));
        assert_eq!(
            nearest_virtual_environment_on_disk(fs.as_ref(), notebook_path).await,
            Some(PathBuf::from(path!("/notebooks/.venv/bin/python")))
        );

        let specification =
            python_environment_specification(path!("/notebooks/.venv/bin/python"), "venv", true);
        let adoption = choose_kernel_for_notebook(
            notebook_path,
            notebook_path,
            None,
            None,
            None,
            std::slice::from_ref(&specification),
        );
        assert!(matches!(
            adoption,
            KernelAdoption::Switch(ref selected) if selected == &specification
        ));
    }

    /// Walking up the notebook's parents finds a `.venv` above the notebook,
    /// and the closest one wins when several exist.
    #[gpui::test]
    async fn nearest_parent_venv_wins_over_farther_one(cx: &mut TestAppContext) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/project"),
            json!({
                ".venv": { "bin": { "python": "" } },
                "sub": {
                    ".venv": { "bin": { "python": "" } },
                    "notebook.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL
                }
            }),
        )
        .await;

        let notebook_path = Path::new(path!("/project/sub/notebook.ipynb"));
        assert_eq!(
            nearest_virtual_environment_on_disk(fs.as_ref(), notebook_path).await,
            Some(PathBuf::from(path!("/project/sub/.venv/bin/python")))
        );
    }

    /// Without a `.venv` anywhere the scan finds nothing and the previous
    /// fallback selection still applies.
    #[gpui::test]
    async fn no_venv_keeps_previous_kernel_behavior(cx: &mut TestAppContext) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/project"),
            json!({ "notebook.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let notebook_path = Path::new(path!("/project/notebook.ipynb"));
        assert_eq!(
            nearest_virtual_environment_on_disk(fs.as_ref(), notebook_path).await,
            None
        );
        assert!(matches!(
            choose_kernel_for_notebook(notebook_path, notebook_path, None, None, None, &[]),
            KernelAdoption::Keep
        ));
    }

    /// A discovered `.venv` without ipykernel surfaces an actionable error
    /// instead of silently switching to a broken kernel.
    #[test]
    fn venv_without_ipykernel_reports_missing_dependency() {
        let worktree_root = Path::new(path!("/project"));
        let notebook_path = Path::new(path!("/project/notebook.ipynb"));
        let notebook_venv =
            python_environment_specification(path!("/project/.venv/bin/python"), "venv", false);
        let discovered = vec![notebook_venv];

        let adoption =
            choose_kernel_for_notebook(notebook_path, worktree_root, None, None, None, &discovered);

        assert!(matches!(
            adoption,
            KernelAdoption::MissingIpykernel(ref environment)
                if environment.as_path() == Path::new(path!("/project/.venv"))
        ));
    }

    /// The agent tools must edit the live cell buffers (not a stale copy on
    /// disk) and leave the cell registry and structure coherent.
    #[gpui::test]
    async fn test_agent_live_cell_operations(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "agent.ipynb": NOTEBOOK_WITH_TWO_CODE_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("agent.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        cx.executor().allow_parking();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let absolute_path = notebook_editor.read_with(cx, |editor, _| {
            editor
                .notebook_path
                .clone()
                .expect("notebook should have an absolute path")
        });

        let edit = cx
            .cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::EditCell {
                        cell_number: 1,
                        new_source: "print('changed')\n".to_string(),
                        line_range: None,
                    },
                )
            })
            .expect("an open notebook must handle live edits")
            .expect("editing a valid cell should succeed");
        assert_eq!(edit.snapshot.cells[0].source, "print('changed')\n");

        let insert = cx
            .cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::InsertCell {
                        position: 2,
                        cell_type: NotebookCellType::Markdown,
                        source: "Inserted".to_string(),
                    },
                )
            })
            .expect("an open notebook must handle live inserts")
            .expect("inserting at a valid position should succeed");
        assert_eq!(insert.snapshot.cells.len(), 3);
        assert_eq!(
            insert.snapshot.cells[1].cell_type,
            NotebookCellType::Markdown
        );

        let moved = cx
            .cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::MoveCell {
                        from_cell: 1,
                        to_cell: 3,
                    },
                )
            })
            .expect("an open notebook must handle live moves")
            .expect("moving to a valid position should succeed");
        assert_eq!(moved.snapshot.cells[2].source, "print('changed')\n");

        let deleted = cx
            .cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::DeleteCell { cell_number: 2 },
                )
            })
            .expect("an open notebook must handle live deletes")
            .expect("deleting a valid cell should succeed");
        assert_eq!(deleted.snapshot.cells.len(), 2);

        notebook_editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 2);
            assert_eq!(
                editor.registered_cell_buffers.len(),
                editor.cell_order.len(),
                "the cell registry must follow structural edits"
            );
            assert!(
                editor
                    .agent_snapshot(cx)
                    .cells
                    .iter()
                    .all(|cell| !cell.source.is_empty()),
                "every remaining cell should keep its source"
            );
        });
    }

    const NOTEBOOK_WITH_MIXED_CELLS: &str = r#"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-one",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["print('one')"]
            },
            {
                "cell_type": "markdown",
                "id": "cell-two",
                "metadata": {},
                "source": ["two"]
            },
            {
                "cell_type": "raw",
                "id": "cell-three",
                "metadata": {},
                "source": ["three"]
            }
        ]
    }"#;

    fn cell_number_of(editor: &NotebookEditor, cell_id: &str, cx: &App) -> Option<usize> {
        editor
            .cell_order
            .iter()
            .find(|id| id.as_str() == cell_id)
            .and_then(|id| editor.cell_map.get(id))
            .and_then(|cell| match cell {
                Cell::Code(cell) => cell.read(cx).cell_number(),
                Cell::Markdown(cell) => cell.read(cx).cell_number(),
                Cell::Raw(cell) => cell.read(cx).cell_number(),
            })
    }

    /// Cell numbers are positions, so every structural edit must renumber the
    /// remaining cells rather than leaving stale numbers behind.
    #[gpui::test]
    async fn test_cell_numbers_follow_structural_changes(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "numbers.ipynb": NOTEBOOK_WITH_MIXED_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("numbers.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let absolute_path = notebook_editor.read_with(cx, |editor, _| {
            editor
                .notebook_path
                .clone()
                .expect("notebook should have an absolute path")
        });

        notebook_editor.read_with(cx, |editor, cx| {
            assert_eq!(cell_number_of(editor, "cell-one", cx), Some(1));
            assert_eq!(cell_number_of(editor, "cell-two", cx), Some(2));
            assert_eq!(cell_number_of(editor, "cell-three", cx), Some(3));
        });

        cx.cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::InsertCell {
                        position: 2,
                        cell_type: NotebookCellType::Markdown,
                        source: "inserted".to_string(),
                    },
                )
            })
            .expect("an open notebook must handle live inserts")
            .expect("inserting at a valid position should succeed");

        notebook_editor.read_with(cx, |editor, cx| {
            assert_eq!(cell_number_of(editor, "cell-one", cx), Some(1));
            assert_eq!(cell_number_of(editor, "cell-two", cx), Some(3));
            assert_eq!(cell_number_of(editor, "cell-three", cx), Some(4));
            let numbers: Vec<Option<usize>> = editor
                .cell_order
                .iter()
                .map(|cell_id| {
                    editor.cell_map.get(cell_id).and_then(|cell| match cell {
                        Cell::Code(cell) => cell.read(cx).cell_number(),
                        Cell::Markdown(cell) => cell.read(cx).cell_number(),
                        Cell::Raw(cell) => cell.read(cx).cell_number(),
                    })
                })
                .collect();
            assert_eq!(numbers, vec![Some(1), Some(2), Some(3), Some(4)]);
        });

        cx.cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::MoveCell {
                        from_cell: 1,
                        to_cell: 3,
                    },
                )
            })
            .expect("an open notebook must handle live moves")
            .expect("moving to a valid position should succeed");

        notebook_editor.read_with(cx, |editor, cx| {
            assert_eq!(cell_number_of(editor, "cell-one", cx), Some(3));
            let numbers: Vec<Option<usize>> = editor
                .cell_order
                .iter()
                .map(|cell_id| {
                    editor.cell_map.get(cell_id).and_then(|cell| match cell {
                        Cell::Code(cell) => cell.read(cx).cell_number(),
                        Cell::Markdown(cell) => cell.read(cx).cell_number(),
                        Cell::Raw(cell) => cell.read(cx).cell_number(),
                    })
                })
                .collect();
            assert_eq!(numbers, vec![Some(1), Some(2), Some(3), Some(4)]);
        });

        cx.cx
            .update(|cx| {
                notebook_agent_tools::run_live(
                    cx,
                    &absolute_path,
                    NotebookLiveCommand::DeleteCell { cell_number: 2 },
                )
            })
            .expect("an open notebook must handle live deletes")
            .expect("deleting a valid cell should succeed");

        notebook_editor.read_with(cx, |editor, cx| {
            assert_eq!(cell_number_of(editor, "cell-one", cx), Some(2));
            assert_eq!(cell_number_of(editor, "cell-three", cx), Some(3));
        });
    }

    /// Save As moves the live registration to the new path: the old path must
    /// stop answering the agent and the new one must take over.
    #[gpui::test]
    async fn test_agent_live_registration_follows_save_as(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "agent.ipynb": NOTEBOOK_WITH_TWO_CODE_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });
        let old_project_path = ProjectPath {
            worktree_id,
            path: rel_path("agent.ipynb").into(),
        };

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &old_project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        cx.executor().allow_parking();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let old_path = notebook_editor.read_with(cx, |editor, _| {
            editor
                .notebook_path
                .clone()
                .expect("notebook should have an absolute path")
        });
        assert!(
            cx.cx
                .update(|cx| notebook_agent_tools::run_live(
                    cx,
                    &old_path,
                    NotebookLiveCommand::Read
                ))
                .is_some(),
            "the opened notebook must answer on its original path"
        );

        let new_project_path = ProjectPath {
            worktree_id,
            path: rel_path("renamed.ipynb").into(),
        };
        let save = notebook_editor.update(cx, |editor, cx| {
            editor.save_impl(
                SaveDestination::NewPath(new_project_path),
                project.clone(),
                cx,
            )
        });
        save.await.expect("Save As should succeed");

        let new_path = notebook_editor.read_with(cx, |editor, _| {
            editor
                .notebook_path
                .clone()
                .expect("the notebook should track the new path")
        });
        assert_ne!(new_path, old_path);

        assert!(
            cx.cx
                .update(|cx| notebook_agent_tools::run_live(
                    cx,
                    &old_path,
                    NotebookLiveCommand::Read
                ))
                .is_none(),
            "the old path must stop answering after Save As"
        );
        assert!(
            cx.cx
                .update(|cx| notebook_agent_tools::run_live(
                    cx,
                    &new_path,
                    NotebookLiveCommand::Read
                ))
                .is_some(),
            "the new path must answer after Save As"
        );
    }

    /// An external tool rewriting the `.ipynb` on disk must be picked up by the
    /// open notebook, which reloads it because it has no unsaved local edits.
    #[gpui::test]
    async fn test_external_change_reloads_the_notebook(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("test.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        cx.executor().allow_parking();
        let notebook_editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        assert_eq!(
            notebook_editor.read_with(cx, |editor, _| editor.cell_order.len()),
            1
        );

        // An external tool rewrites the notebook with an extra cell.
        fs.write(
            path!("/notebooks/test.ipynb").as_ref(),
            NOTEBOOK_WITH_TWO_CODE_CELLS.as_bytes(),
        )
        .await
        .expect("external write should succeed");
        cx.executor().run_until_parked();

        assert_eq!(
            notebook_editor.read_with(cx, |editor, _| editor.cell_order.len()),
            2,
            "the notebook should reload the cell written by the external tool"
        );
    }

    /// A reload triggered by an external edit must keep the user's viewport,
    /// selection, focus, and cursor on the same cell instead of snapping back
    /// to the top of the notebook.
    #[gpui::test]
    async fn test_external_change_preserves_scroll_selection_and_focus(
        cx: &mut TestAppContext,
    ) {
        let cells = numbered_cells(30);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let (anchor_cell_id, focused_editor) = editor.update_in(cx, |editor, window, cx| {
            let anchor_cell_id = editor.cell_order[19].clone();
            editor.select_cell_by_id(&anchor_cell_id, cx);
            editor.cell_list.scroll_to(gpui::ListOffset {
                item_ix: 19,
                offset_in_item: gpui::px(40.),
            });
            let Some(Cell::Code(code_cell)) = editor.cell_map.get(&anchor_cell_id).cloned() else {
                panic!("cell 19 should be a code cell");
            };
            let cell_editor = code_cell.read(cx).editor().clone();
            window.focus(&cell_editor.focus_handle(cx), cx);
            (anchor_cell_id, cell_editor)
        });
        let cursor = "print('cell 19')".len();
        set_cursor(&focused_editor, cursor, cx);

        // Change cell 25 and insert a new cell before cell 10, both externally.
        let mut updated_cells = cells.clone();
        updated_cells[24].1 = "print('cell 24 changed')".to_string();
        updated_cells.insert(9, ("inserted-cell".to_string(), "print('inserted')".to_string()));
        let updated_content = notebook_with_cells(&updated_cells);

        let project = editor.read_with(cx, |editor, _| editor.project.clone());
        let fs = project.read_with(cx, |project, _| project.fs().clone());
        fs.write(
            path!("/notebooks/ops.ipynb").as_ref(),
            updated_content.as_bytes(),
        )
        .await
        .expect("external write should succeed");
        cx.executor().run_until_parked();

        let current_editor = editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 31);
            let scroll_top = editor.cell_list.logical_scroll_top();
            assert_eq!(
                scroll_top.item_ix, 20,
                "the inserted cell should shift the viewport to the cell's new index"
            );
            assert_eq!(scroll_top.offset_in_item, gpui::px(40.));
            assert_eq!(editor.cell_order[scroll_top.item_ix], anchor_cell_id);
            assert_eq!(editor.selected_cell_index, 20);
            assert_eq!(editor.cell_order[editor.selected_cell_index], anchor_cell_id);
            assert_eq!(editor.focused_cell_id.as_ref(), Some(&anchor_cell_id));
            cell_editor(editor, 20, cx)
        });
        assert_eq!(
            current_editor.entity_id(),
            focused_editor.entity_id(),
            "an unchanged cell should keep the same editor entity"
        );
        assert_eq!(cursor_offset(&current_editor, cx), cursor);
        cx.update(|window, cx| {
            let focus_handle = focused_editor.read(cx).focus_handle(cx);
            assert!(
                focus_handle.is_focused(window),
                "the cell editor should keep focus after the reload"
            );
        });
    }

    /// When the externally edited notebook deletes the selected cell, the
    /// nearest surviving neighbour becomes selected.
    #[gpui::test]
    async fn test_external_change_selects_neighbour_when_selected_cell_is_deleted(
        cx: &mut TestAppContext,
    ) {
        let cells = numbered_cells(5);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let deleted_cell_id = editor.update(cx, |editor, cx| {
            let deleted_cell_id = editor.cell_order[2].clone();
            editor.select_cell_by_id(&deleted_cell_id, cx);
            deleted_cell_id
        });

        let mut updated_cells = cells.clone();
        updated_cells.remove(2);
        let updated_content = notebook_with_cells(&updated_cells);

        let project = editor.read_with(cx, |editor, _| editor.project.clone());
        let fs = project.read_with(cx, |project, _| project.fs().clone());
        fs.write(
            path!("/notebooks/ops.ipynb").as_ref(),
            updated_content.as_bytes(),
        )
        .await
        .expect("external write should succeed");
        cx.executor().run_until_parked();

        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 4);
            assert!(!editor.cell_map.contains_key(&deleted_cell_id));
            let selected_cell_id = &editor.cell_order[editor.selected_cell_index];
            assert_ne!(selected_cell_id, &deleted_cell_id);
            let selected_cell = editor
                .cell_map
                .get(selected_cell_id)
                .expect("the selected neighbour should exist");
            assert_eq!(
                selected_cell.current_source(cx),
                "print('cell 1')",
                "the deleted cell's neighbour should be selected"
            );
        });
    }

    /// Unchanged cells must keep their existing entity so cursor, focus, and
    /// rendered outputs survive a reload; a cell whose source changed is
    /// updated in place instead of being rebuilt.
    #[gpui::test]
    async fn test_external_change_reuses_unchanged_cell_entities(cx: &mut TestAppContext) {
        let cells = numbered_cells(3);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let before = editor.update(cx, |editor, cx| {
            (0..3)
                .map(|index| cell_editor(editor, index, cx).entity_id())
                .collect::<Vec<_>>()
        });

        let mut updated_cells = cells.clone();
        updated_cells[1].1 = "print('cell 1 changed')".to_string();
        let updated_content = notebook_with_cells(&updated_cells);

        let project = editor.read_with(cx, |editor, _| editor.project.clone());
        let fs = project.read_with(cx, |project, _| project.fs().clone());
        fs.write(
            path!("/notebooks/ops.ipynb").as_ref(),
            updated_content.as_bytes(),
        )
        .await
        .expect("external write should succeed");
        cx.executor().run_until_parked();

        editor.read_with(cx, |editor, cx| {
            assert_eq!(cell_editor(editor, 0, cx).entity_id(), before[0]);
            assert_eq!(cell_editor(editor, 2, cx).entity_id(), before[2]);
            assert_eq!(
                cell_editor(editor, 1, cx).entity_id(),
                before[1],
                "the edited cell should be updated in place"
            );
            assert_eq!(cell_sources(editor, cx)[1], "print('cell 1 changed')");
        });
    }

    async fn write_external_notebook(
        editor: &Entity<NotebookEditor>,
        content: String,
        cx: &mut VisualTestContext,
    ) {
        let project = editor.read_with(cx, |editor, _| editor.project.clone());
        let fs = project.read_with(cx, |project, _| project.fs().clone());
        fs.write(path!("/notebooks/ops.ipynb").as_ref(), content.as_bytes())
            .await
            .expect("external write should succeed");
        cx.executor().run_until_parked();
    }

    fn notebook_with_stream_output(text: &str) -> String {
        json!({
            "metadata": {"language_info": {"name": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{
                "cell_type": "code",
                "id": "cell-output",
                "metadata": {},
                "execution_count": 1,
                "outputs": [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [text],
                }],
                "source": ["print('hello')"],
            }],
        })
        .to_string()
    }

    fn notebook_with_cells_without_ids(cells: &[(String, String)]) -> String {
        let cells: Vec<serde_json::Value> = cells
            .iter()
            .map(|(_, source)| {
                json!({
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": null,
                    "outputs": [],
                    "source": [source],
                })
            })
            .collect();
        json!({
            "metadata": {"language_info": {"name": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": cells,
        })
        .to_string()
    }

    /// Changing a single cell out of thirty must update only that cell's
    /// entity and leave every other entity, plus the viewport and selection,
    /// untouched.
    #[gpui::test]
    async fn test_external_change_updates_only_the_modified_cell(cx: &mut TestAppContext) {
        let cells = numbered_cells(30);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let before = editor.read_with(cx, |editor, cx| {
            (0..30)
                .map(|index| cell_editor(editor, index, cx).entity_id())
                .collect::<Vec<_>>()
        });
        let anchor_cell_id = editor.update_in(cx, |editor, window, cx| {
            let anchor_cell_id = editor.cell_order[9].clone();
            editor.select_cell_by_id(&anchor_cell_id, cx);
            editor.cell_list.scroll_to(gpui::ListOffset {
                item_ix: 9,
                offset_in_item: gpui::px(24.),
            });
            let Some(Cell::Code(code_cell)) = editor.cell_map.get(&anchor_cell_id).cloned() else {
                panic!("cell 9 should be a code cell");
            };
            window.focus(&code_cell.read(cx).editor().focus_handle(cx), cx);
            anchor_cell_id
        });

        let mut updated_cells = cells.clone();
        updated_cells[24].1 = "print('cell 24 changed')".to_string();
        write_external_notebook(&editor, notebook_with_cells(&updated_cells), cx).await;

        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 30);
            for index in 0..30 {
                assert_eq!(
                    cell_editor(editor, index, cx).entity_id(),
                    before[index],
                    "cell {index} should keep its entity"
                );
            }
            assert_eq!(cell_sources(editor, cx)[24], "print('cell 24 changed')");
            let scroll_top = editor.cell_list.logical_scroll_top();
            assert_eq!(scroll_top.item_ix, 9);
            assert_eq!(scroll_top.offset_in_item, gpui::px(24.));
            assert_eq!(editor.selected_cell_index, 9);
            assert_eq!(editor.cell_order[9], anchor_cell_id);
            assert_eq!(editor.focused_cell_id.as_ref(), Some(&anchor_cell_id));
        });
    }

    /// Output-only changes must refresh the outputs without recreating the
    /// cell, since the source and editor state did not change.
    #[gpui::test]
    async fn test_external_change_updates_outputs_in_place(cx: &mut TestAppContext) {
        let (editor, cx) =
            open_notebook_editor(cx, notebook_with_stream_output("old output\n")).await;
        cx.executor().allow_parking();

        let before = editor.read_with(cx, |editor, cx| cell_editor(editor, 0, cx).entity_id());
        write_external_notebook(&editor, notebook_with_stream_output("updated output\n"), cx).await;

        editor.read_with(cx, |editor, cx| {
            assert_eq!(cell_editor(editor, 0, cx).entity_id(), before);
            let cell_id = editor.cell_order[0].clone();
            let Some(Cell::Code(code_cell)) = editor.cell_map.get(&cell_id) else {
                panic!("cell should be a code cell");
            };
            let code_cell = code_cell.read(cx);
            assert_eq!(
                code_cell.output_summary().as_deref(),
                Some("outputs: 1 (stream)")
            );
            let outputs = match code_cell.to_nbformat_cell(cx) {
                nbformat::v4::Cell::Code { outputs, .. } => outputs,
                _ => panic!("cell should still be a code cell"),
            };
            let outputs = serde_json::to_string(&outputs).expect("outputs should serialize");
            assert!(outputs.contains("updated output"));
        });
    }

    /// A source change on the focused cell keeps focus and adjusts the cursor,
    /// because the edit is applied to the existing editor buffer.
    #[gpui::test]
    async fn test_external_change_keeps_focus_and_adjusts_cursor(cx: &mut TestAppContext) {
        let cells = numbered_cells(3);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let (focused_editor, cursor) = editor.update_in(cx, |editor, window, cx| {
            let cell_id = editor.cell_order[1].clone();
            editor.select_cell_by_id(&cell_id, cx);
            let Some(Cell::Code(code_cell)) = editor.cell_map.get(&cell_id).cloned() else {
                panic!("cell 1 should be a code cell");
            };
            let focused_editor = code_cell.read(cx).editor().clone();
            window.focus(&focused_editor.focus_handle(cx), cx);
            (focused_editor, "print('cell 1')".len())
        });
        set_cursor(&focused_editor, cursor, cx);

        let prefix = "# edited externally\n";
        let mut updated_cells = cells.clone();
        updated_cells[1].1 = format!("{prefix}print('cell 1')");
        write_external_notebook(&editor, notebook_with_cells(&updated_cells), cx).await;

        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.focused_cell_id.as_ref(), Some(&editor.cell_order[1]));
            assert_eq!(
                cell_editor(editor, 1, cx).entity_id(),
                focused_editor.entity_id()
            );
        });
        assert_eq!(
            cursor_offset(&focused_editor, cx),
            cursor + prefix.len(),
            "the cursor should move with the inserted prefix"
        );
        cx.update(|window, cx| {
            assert!(
                focused_editor.read(cx).focus_handle(cx).is_focused(window),
                "the edited cell should keep focus"
            );
        });
    }

    /// Inserting and removing cells updates the list incrementally: surviving
    /// cells keep their entities and the scroll anchor follows the cell.
    #[gpui::test]
    async fn test_external_change_insert_delete_preserves_entities_and_anchor(
        cx: &mut TestAppContext,
    ) {
        let cells = numbered_cells(30);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let before = editor.read_with(cx, |editor, cx| {
            (0..30)
                .map(|index| {
                    (
                        editor.cell_order[index].clone(),
                        cell_editor(editor, index, cx).entity_id(),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>()
        });
        let anchor_cell_id = editor.update_in(cx, |editor, _window, cx| {
            let anchor_cell_id = editor.cell_order[15].clone();
            editor.select_cell_by_id(&anchor_cell_id, cx);
            editor.cell_list.scroll_to(gpui::ListOffset {
                item_ix: 15,
                offset_in_item: gpui::px(12.),
            });
            anchor_cell_id
        });

        let mut updated_cells = cells.clone();
        updated_cells.insert(
            9,
            ("inserted-cell".to_string(), "print('inserted')".to_string()),
        );
        let removed_cell_id = updated_cells.remove(22).0;
        write_external_notebook(&editor, notebook_with_cells(&updated_cells), cx).await;

        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 30);
            assert!(
                !editor
                    .cell_map
                    .contains_key(&CellId::new(&removed_cell_id).expect("valid id"))
            );
            for (cell_id, entity_id) in &before {
                if cell_id.as_str() == removed_cell_id.as_str() {
                    continue;
                }
                let index = editor
                    .cell_order
                    .iter()
                    .position(|candidate| candidate == cell_id)
                    .expect("surviving cell should still be in the list");
                assert_eq!(cell_editor(editor, index, cx).entity_id(), *entity_id);
            }
            let scroll_top = editor.cell_list.logical_scroll_top();
            assert_eq!(scroll_top.item_ix, 16);
            assert_eq!(scroll_top.offset_in_item, gpui::px(12.));
            assert_eq!(editor.cell_order[scroll_top.item_ix], anchor_cell_id);
            assert_eq!(editor.selected_cell_index, 16);
            assert_eq!(cell_sources(editor, cx)[9], "print('inserted')");
        });
    }

    /// Cells written without ids are aligned by content, so existing entities
    /// survive an external rewrite that drops the ids.
    #[gpui::test]
    async fn test_external_change_falls_back_to_content_alignment_without_ids(
        cx: &mut TestAppContext,
    ) {
        let cells = numbered_cells(3);
        let (editor, cx) = open_notebook_editor(cx, notebook_with_cells(&cells)).await;
        cx.executor().allow_parking();

        let before = editor.read_with(cx, |editor, cx| {
            (0..3)
                .map(|index| {
                    (
                        editor.cell_order[index].clone(),
                        cell_editor(editor, index, cx).entity_id(),
                    )
                })
                .collect::<Vec<_>>()
        });

        let mut updated_cells = cells.clone();
        updated_cells.insert(
            1,
            (
                "ignored".to_string(),
                "print('inserted without id')".to_string(),
            ),
        );
        write_external_notebook(&editor, notebook_with_cells_without_ids(&updated_cells), cx).await;

        editor.read_with(cx, |editor, cx| {
            assert_eq!(
                cell_sources(editor, cx),
                vec![
                    "print('cell 0')",
                    "print('inserted without id')",
                    "print('cell 1')",
                    "print('cell 2')",
                ]
            );
            for (cell_id, entity_id) in &before {
                let index = editor
                    .cell_order
                    .iter()
                    .position(|candidate| candidate == cell_id)
                    .expect("content alignment should keep the existing id");
                assert_eq!(cell_editor(editor, index, cx).entity_id(), *entity_id);
            }
        });
    }

    const NOTEBOOK_WITH_CODE_OUTPUT: &str = r#"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-one",
                "metadata": {},
                "execution_count": 7,
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["hello\n"]
                    }
                ],
                "source": ["print('hello')"]
            }
        ]
    }"#;

    async fn open_notebook_editor(
        cx: &mut TestAppContext,
        content: impl Into<String>,
    ) -> (Entity<NotebookEditor>, &mut VisualTestContext) {
        open_notebook_editor_with_settings(cx, content, |_| {}).await
    }

    async fn open_notebook_editor_with_settings(
        cx: &mut TestAppContext,
        content: impl Into<String>,
        configure_settings: impl FnOnce(&mut SettingsContent),
    ) -> (Entity<NotebookEditor>, &mut VisualTestContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, configure_settings);
            });
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/notebooks"), json!({ "ops.ipynb": content.into() }))
            .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project
                .worktrees(cx)
                .next()
                .expect("worktree")
                .read(cx)
                .id(),
            path: rel_path("ops.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });
        (editor, cx)
    }

    fn cell_sources(editor: &NotebookEditor, cx: &App) -> Vec<String> {
        editor
            .cell_order
            .iter()
            .map(|cell_id| {
                editor
                    .cell_map
                    .get(cell_id)
                    .expect("cell should exist")
                    .current_source(cx)
            })
            .collect()
    }

    fn cell_editor(editor: &NotebookEditor, index: usize, cx: &App) -> Entity<editor::Editor> {
        let cell_id = editor.cell_order.get(index).expect("cell index").clone();
        editor
            .cell_map
            .get(&cell_id)
            .expect("cell should exist")
            .editor(cx)
            .expect("cell should have an editor")
            .clone()
    }

    fn set_cursor(cell_editor: &Entity<editor::Editor>, offset: usize, cx: &mut VisualTestContext) {
        cell_editor.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(offset)..MultiBufferOffset(offset)])
            });
        });
    }

    fn cursor_offset(cell_editor: &Entity<editor::Editor>, cx: &mut VisualTestContext) -> usize {
        cell_editor.update_in(cx, |editor, _window, cx| {
            editor
                .selections
                .newest::<MultiBufferOffset>(&editor.display_snapshot(cx))
                .head()
                .0
        })
    }

    fn notebook_with_cells(cells: &[(String, String)]) -> String {
        let cells: Vec<serde_json::Value> = cells
            .iter()
            .map(|(id, source)| {
                json!({
                    "cell_type": "code",
                    "id": id,
                    "metadata": {},
                    "execution_count": null,
                    "outputs": [],
                    "source": [source],
                })
            })
            .collect();
        json!({
            "metadata": {"language_info": {"name": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": cells,
        })
        .to_string()
    }

    fn numbered_cells(count: usize) -> Vec<(String, String)> {
        (0..count)
            .map(|index| (format!("cell-{index}"), format!("print('cell {index}')")))
            .collect()
    }

    #[gpui::test]
    async fn cell_operations_test_split_cell_at_cursor_and_undo(cx: &mut TestAppContext) {
        let (editor, cx) = open_notebook_editor(
            cx,
            r#"{
                "metadata": {"language_info": {"name": "python"}},
                "nbformat": 4,
                "nbformat_minor": 5,
                "cells": [{
                    "cell_type": "code",
                    "id": "cell-one",
                    "metadata": {},
                    "execution_count": null,
                    "outputs": [],
                    "source": ["first\n", "second\n"]
                }]
            }"#,
        )
        .await;

        let first_editor = editor.read_with(cx, |editor, cx| cell_editor(editor, 0, cx));
        set_cursor(&first_editor, "first\n".len(), cx);

        editor.update_in(cx, |editor, window, cx| {
            editor.split_cell(&SplitCell, window, cx)
        });

        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 2);
            assert_eq!(cell_sources(editor, cx), vec!["first\n", "second\n"]);
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.undo_structure(&Undo, window, cx)
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 1);
            assert_eq!(cell_sources(editor, cx), vec!["first\nsecond\n"]);
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.redo_structure(&Redo, window, cx)
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 2);
            assert_eq!(cell_sources(editor, cx), vec!["first\n", "second\n"]);
        });
    }

    #[gpui::test]
    async fn cell_operations_test_merge_cells_only_with_same_type_and_undo(
        cx: &mut TestAppContext,
    ) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_TWO_CODE_CELLS).await;

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 0;
            editor.merge_cell(false, window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 1);
            assert_eq!(
                cell_sources(editor, cx),
                vec!["print('one')\nprint('two')\n"]
            );
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.undo_structure(&Undo, window, cx)
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 2);
            assert_eq!(
                cell_sources(editor, cx),
                vec!["print('one')", "print('two')"]
            );
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 1;
            editor.merge_cell(true, window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.cell_order.len(), 1);
            assert_eq!(
                cell_sources(editor, cx),
                vec!["print('one')\nprint('two')\n"]
            );
        });
    }

    #[gpui::test]
    async fn cell_operations_test_merge_refuses_different_cell_types(cx: &mut TestAppContext) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_MIXED_CELLS).await;

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 0;
            editor.merge_cell(false, window, cx);
        });
        editor.read_with(cx, |editor, _| {
            assert_eq!(editor.cell_order.len(), 3);
        });
    }

    #[gpui::test]
    async fn cell_operations_test_convert_cell_preserves_text_and_resets_code_state(
        cx: &mut TestAppContext,
    ) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_CODE_OUTPUT).await;

        editor.update_in(cx, |editor, window, cx| {
            editor.convert_cell_to_markdown(window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            let cell_id = editor.cell_order.first().expect("cell").clone();
            let cell = editor.cell_map.get(&cell_id).expect("cell");
            assert!(matches!(cell, Cell::Markdown(_)));
            assert!(cell.current_source(cx).contains("print('hello')"));
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.convert_cell_to_code(window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            let cell_id = editor.cell_order.first().expect("cell").clone();
            let cell = editor.cell_map.get(&cell_id).expect("cell");
            let Cell::Code(code_cell) = cell else {
                panic!("converting back should produce a code cell");
            };
            let code_cell = code_cell.read(cx);
            assert!(code_cell.current_source(cx).contains("print('hello')"));
            let nbformat::v4::Cell::Code {
                execution_count,
                outputs,
                ..
            } = code_cell.to_nbformat_cell(cx)
            else {
                panic!("expected a code cell");
            };
            assert_eq!(execution_count, None);
            assert!(outputs.is_empty());
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.undo_structure(&Undo, window, cx)
        });
        editor.read_with(cx, |editor, _| {
            let cell_id = editor.cell_order.first().expect("cell").clone();
            assert!(matches!(
                editor.cell_map.get(&cell_id),
                Some(Cell::Markdown(_))
            ));
        });
    }

    #[gpui::test]
    async fn cell_operations_test_run_above_below_and_advance(cx: &mut TestAppContext) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_TWO_CODE_CELLS).await;

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 1;
            editor.run_above(&RunAbove, window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert!(cell_has_outputs(editor, 0, cx));
            assert!(!cell_has_outputs(editor, 1, cx));
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.clear_outputs(window, cx);
            editor.selected_cell_index = 0;
            editor.run_below(&RunBelow, window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert!(!cell_has_outputs(editor, 0, cx));
            assert!(cell_has_outputs(editor, 1, cx));
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = editor.cell_order.len() - 1;
            editor.run_and_advance(&RunAndAdvance, window, cx);
        });
        editor.read_with(cx, |editor, _| {
            assert_eq!(editor.cell_order.len(), 3);
            assert_eq!(editor.selected_cell_index, 2);
        });
    }

    fn cell_has_outputs(editor: &NotebookEditor, index: usize, cx: &App) -> bool {
        let cell_id = editor.cell_order.get(index).expect("cell index");
        match editor.cell_map.get(cell_id) {
            Some(Cell::Code(code_cell)) => code_cell.read(cx).has_outputs(),
            _ => false,
        }
    }

    #[gpui::test]
    async fn cell_operations_test_cell_outputs_clear_and_collapse_metadata(
        cx: &mut TestAppContext,
    ) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_CODE_OUTPUT).await;

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 0;
            editor.toggle_selected_cell_output(&ToggleCellOutput, window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            let cell_id = editor.cell_order.first().expect("cell").clone();
            let Cell::Code(code_cell) = editor.cell_map.get(&cell_id).expect("cell") else {
                panic!("expected a code cell");
            };
            let code_cell = code_cell.read(cx);
            assert!(code_cell.outputs_collapsed());
            let nbformat::v4::Cell::Code { metadata, .. } = code_cell.to_nbformat_cell(cx) else {
                panic!("expected a code cell");
            };
            assert_eq!(metadata.collapsed, Some(true));
            assert_eq!(
                metadata.jupyter.and_then(|jupyter| jupyter.outputs_hidden),
                Some(true)
            );
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.clear_selected_cell_outputs(&ClearCellOutputs, window, cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert!(!cell_has_outputs(editor, 0, cx));
        });
    }

    #[gpui::test]
    async fn cell_operations_test_notebook_undo_redo_for_insert_delete_and_move(
        cx: &mut TestAppContext,
    ) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_TWO_CODE_CELLS).await;

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 0;
            editor.add_code_block(window, cx);
        });
        editor.read_with(cx, |editor, _| assert_eq!(editor.cell_order.len(), 3));

        editor.update_in(cx, |editor, window, cx| {
            editor.undo_structure(&Undo, window, cx)
        });
        editor.read_with(cx, |editor, _| assert_eq!(editor.cell_order.len(), 2));

        editor.update_in(cx, |editor, window, cx| {
            editor.redo_structure(&Redo, window, cx)
        });
        editor.read_with(cx, |editor, _| assert_eq!(editor.cell_order.len(), 3));

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 0;
            let first = editor.cell_order[0].clone();
            let second = editor.cell_order[1].clone();
            editor.move_cell_down(window, cx);
            assert_eq!(editor.cell_order[0], second);
            assert_eq!(editor.cell_order[1], first);
        });
        editor.update_in(cx, |editor, window, cx| {
            editor.undo_structure(&Undo, window, cx)
        });
        editor.read_with(cx, |editor, _| {
            assert_eq!(editor.cell_order.len(), 3);
            assert_eq!(editor.selected_cell_index, 0);
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.selected_cell_index = 0;
            editor.delete_cell(window, cx);
        });
        editor.read_with(cx, |editor, _| assert_eq!(editor.cell_order.len(), 2));

        editor.update_in(cx, |editor, window, cx| {
            editor.undo_structure(&Undo, window, cx)
        });
        editor.read_with(cx, |editor, _| assert_eq!(editor.cell_order.len(), 3));
    }

    #[derive(Clone, Debug)]
    struct FakeRunningKernel {
        request_tx: futures::channel::mpsc::Sender<JupyterMessage>,
        stdin_tx: futures::channel::mpsc::Sender<JupyterMessage>,
        working_directory: PathBuf,
        execution_state: runtimelib::ExecutionState,
        kernel_info: Option<runtimelib::KernelInfoReply>,
    }

    impl crate::kernels::RunningKernel for FakeRunningKernel {
        fn request_tx(&self) -> futures::channel::mpsc::Sender<JupyterMessage> {
            self.request_tx.clone()
        }

        fn stdin_tx(&self) -> futures::channel::mpsc::Sender<JupyterMessage> {
            self.stdin_tx.clone()
        }

        fn working_directory(&self) -> &PathBuf {
            &self.working_directory
        }

        fn execution_state(&self) -> &runtimelib::ExecutionState {
            &self.execution_state
        }

        fn set_execution_state(&mut self, state: runtimelib::ExecutionState) {
            self.execution_state = state;
        }

        fn kernel_info(&self) -> Option<&runtimelib::KernelInfoReply> {
            self.kernel_info.as_ref()
        }

        fn set_kernel_info(&mut self, info: runtimelib::KernelInfoReply) {
            self.kernel_info = Some(info);
        }

        fn force_shutdown(
            &mut self,
            _window: &mut Window,
            _cx: &mut App,
        ) -> Task<anyhow::Result<()>> {
            Task::ready(Ok(()))
        }

        fn kill(&mut self) {}
    }

    async fn open_notebook_editor_with_fake_kernel<'a>(
        cx: &'a mut TestAppContext,
        content: &'static str,
        configure_settings: impl FnOnce(&mut SettingsContent),
    ) -> (
        Entity<NotebookEditor>,
        &'a mut VisualTestContext,
        futures::channel::mpsc::Receiver<JupyterMessage>,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, configure_settings);
            });
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/notebooks"), json!({ "test.ipynb": content }))
            .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("worktree")
                .read(cx)
                .id()
        });
        let missing_interpreter = path!("/nonexistent/python3");
        let broken_spec = KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from(missing_interpreter),
            kernelspec: JupyterKernelspec {
                argv: vec![
                    missing_interpreter.to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        });
        cx.update(|cx| {
            ReplStore::global(cx).update(cx, |store, cx| {
                store.set_active_kernelspec(worktree_id, broken_spec, cx);
            })
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("test.ipynb").into(),
                    },
                    cx,
                )
                .expect("notebook should open")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        cx.executor().allow_parking();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let pending_kernel = editor.read_with(cx, |editor, _| match &editor.kernel {
            Kernel::StartingKernel(task) => task.clone(),
            _ => panic!("kernel should be starting"),
        });
        pending_kernel.await;

        let (request_tx, request_rx) = futures::channel::mpsc::channel(8);
        let (stdin_tx, _stdin_rx) = futures::channel::mpsc::channel(4);
        editor.update(cx, |editor, _| {
            editor.kernel = Kernel::RunningKernel(Box::new(FakeRunningKernel {
                request_tx,
                stdin_tx,
                working_directory: PathBuf::from("/notebooks"),
                execution_state: runtimelib::ExecutionState::Idle,
                kernel_info: None,
            }));
        });

        (editor, cx, request_rx)
    }

    #[gpui::test]
    async fn test_variables_refresh_with_fake_kernel(cx: &mut TestAppContext) {
        use futures::StreamExt as _;

        let (editor, cx, mut request_rx) =
            open_notebook_editor_with_fake_kernel(cx, NOTEBOOK_WITH_ONE_CODE_CELL, |_| {}).await;

        editor.update(cx, |editor, cx| editor.refresh_variables(cx));

        let request = request_rx.next().await.expect("variables request");
        let payload = match &request.content {
            JupyterMessageContent::ExecuteRequest(request) => {
                assert!(request.silent);
                assert!(!request.store_history);
                assert!(request.code.contains("__zed_variables_payload"));
                assert_eq!(
                    request
                        .user_expressions
                        .as_ref()
                        .and_then(|expressions| expressions.get("payload"))
                        .map(String::as_str),
                    Some("__zed_variables_json__")
                );
                r#"{"variables":[{"name":"df","type":"pandas.core.frame.DataFrame","shape":"2 x 1","value":"   a","expandable":true}]}"#
            }
            other => panic!("expected an execute request, got {other:?}"),
        };

        let mut expressions = std::collections::HashMap::new();
        expressions.insert(
            "payload".to_string(),
            ExpressionResult::Ok {
                data: jupyter_protocol::Media {
                    content: vec![MediaType::Plain(payload.to_string())],
                },
                metadata: Default::default(),
            },
        );
        let reply = ExecuteReply {
            status: ReplyStatus::Ok,
            user_expressions: Some(expressions),
            ..Default::default()
        };
        let mut reply_message =
            JupyterMessage::new(JupyterMessageContent::ExecuteReply(reply), None);
        reply_message.parent_header = Some(request.header);

        editor.update_in(cx, |editor, window, cx| {
            editor.route(&reply_message, window, cx);
        });

        editor.read_with(cx, |editor, cx| {
            let variables = editor.variables_panel.read(cx).variables();
            assert_eq!(variables.len(), 1);
            assert_eq!(variables[0].name, "df");
            assert!(variables[0].expandable);
        });
    }

    #[gpui::test]
    async fn notebook_variables_panel_initial_state_follows_settings(cx: &mut TestAppContext) {
        let cases = [
            (
                settings::NotebookVariablesPanelContent::Right,
                true,
                VariablesPanelPosition::Right,
            ),
            (
                settings::NotebookVariablesPanelContent::Bottom,
                true,
                VariablesPanelPosition::Bottom,
            ),
            (
                settings::NotebookVariablesPanelContent::Hidden,
                false,
                VariablesPanelPosition::Right,
            ),
        ];

        for (configured, visible, position) in cases {
            let (editor, cx) = open_notebook_editor_with_settings(
                cx,
                NOTEBOOK_WITH_ONE_CODE_CELL,
                move |settings| {
                    settings
                        .editor
                        .jupyter
                        .get_or_insert_default()
                        .notebook_variables_panel = Some(configured);
                },
            )
            .await;

            editor.read_with(cx, |editor, cx| {
                let panel = editor.variables_panel.read(cx);
                assert_eq!(panel.is_visible(), visible);
                assert_eq!(panel.position(), position);
            });
        }
    }

    #[gpui::test]
    async fn notebook_variables_panel_position_toggles(cx: &mut TestAppContext) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_ONE_CODE_CELL).await;

        editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.variables_panel.read(cx).position(),
                VariablesPanelPosition::Right
            );
        });

        editor.update(cx, |editor, cx| {
            editor.toggle_variables_panel_position(cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.variables_panel.read(cx).position(),
                VariablesPanelPosition::Bottom
            );
        });

        editor.update(cx, |editor, cx| {
            editor.toggle_variables_panel_position(cx);
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.variables_panel.read(cx).position(),
                VariablesPanelPosition::Right
            );
        });
    }

    #[gpui::test]
    async fn notebook_variables_panel_sizes_are_clamped(cx: &mut TestAppContext) {
        let (editor, cx) = open_notebook_editor(cx, NOTEBOOK_WITH_ONE_CODE_CELL).await;

        editor.update(cx, |editor, _| {
            assert_eq!(editor.effective_variables_right_width(px(1000.)), px(280.));

            editor.variables_right_width = Some(px(10.));
            assert_eq!(
                editor.effective_variables_right_width(px(1000.)),
                px(clamp_variables_right_width(10.0, 1000.0))
            );

            editor.variables_right_width = Some(px(5000.));
            assert_eq!(
                editor.effective_variables_right_width(px(1000.)),
                px(clamp_variables_right_width(5000.0, 1000.0))
            );

            editor.variables_bottom_height = Some(px(10.));
            assert_eq!(
                editor.effective_variables_bottom_height(px(1000.)),
                px(variables_bottom_min_height())
            );

            editor.variables_bottom_height = Some(px(5000.));
            assert_eq!(
                editor.effective_variables_bottom_height(px(1000.)),
                px(clamp_variables_bottom_height(5000.0, 1000.0))
            );
        });
    }

    #[gpui::test]
    async fn notebook_variables_panel_hidden_skips_introspection(cx: &mut TestAppContext) {
        use futures::StreamExt as _;

        let (editor, cx, mut request_rx) =
            open_notebook_editor_with_fake_kernel(cx, NOTEBOOK_WITH_ONE_CODE_CELL, |settings| {
                settings
                    .editor
                    .jupyter
                    .get_or_insert_default()
                    .notebook_variables_panel =
                    Some(settings::NotebookVariablesPanelContent::Hidden);
            })
            .await;

        editor.update(cx, |editor, cx| {
            assert!(!editor.variables_panel.read(cx).is_visible());
            editor.refresh_variables(cx);
        });
        match request_rx.try_recv() {
            Err(futures::channel::mpsc::TryRecvError::Empty) => {}
            Err(error) => panic!("the request channel should stay open: {error:?}"),
            Ok(message) => panic!("the hidden panel sent a request: {message:?}"),
        }

        editor.update(cx, |editor, cx| editor.toggle_variables_panel(cx));
        editor.read_with(cx, |editor, cx| {
            assert!(editor.variables_panel.read(cx).is_visible());
        });

        let request = request_rx
            .next()
            .await
            .expect("showing the variables panel should refresh them");
        match &request.content {
            JupyterMessageContent::ExecuteRequest(request) => assert!(request.silent),
            other => panic!("expected an execute request, got {other:?}"),
        }

        editor.update(cx, |editor, cx| editor.toggle_variables_panel(cx));
        editor.update(cx, |editor, cx| {
            assert!(!editor.variables_panel.read(cx).is_visible());
            editor.refresh_variables(cx);
        });
        match request_rx.try_recv() {
            Err(futures::channel::mpsc::TryRecvError::Empty) => {}
            Err(error) => panic!("the request channel should stay open: {error:?}"),
            Ok(message) => panic!("the hidden panel sent a request: {message:?}"),
        }
    }

    const NOTEBOOK_WITH_THREE_CODE_CELLS: &str = r#"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-a",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["import numpy as np"]
            },
            {
                "cell_type": "code",
                "id": "cell-b",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["def attention(q, k, v):\n    return q"]
            },
            {
                "cell_type": "code",
                "id": "cell-c",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["for row in range(3):\n    print(row)"]
            }
        ]
    }"#;

    fn route_child(
        editor: &Entity<NotebookEditor>,
        parent: &JupyterMessage,
        content: JupyterMessageContent,
        cx: &mut VisualTestContext,
    ) {
        let mut message = JupyterMessage::new(content, None);
        message.parent_header = Some(parent.header.clone());
        editor.update_in(cx, |editor, window, cx| {
            editor.route(&message, window, cx);
        });
    }

    fn code_cell_counts(editor: &NotebookEditor, cx: &App) -> Vec<Option<i32>> {
        editor
            .cell_order
            .iter()
            .map(|cell_id| match editor.cell_map.get(cell_id) {
                Some(Cell::Code(cell)) => cell.read(cx).execution_count(),
                _ => None,
            })
            .collect()
    }

    #[gpui::test]
    async fn test_run_all_routes_each_cells_messages(cx: &mut TestAppContext) {
        use futures::StreamExt as _;

        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "runall.ipynb": NOTEBOOK_WITH_THREE_CODE_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("worktree")
                .read(cx)
                .id()
        });
        let missing_interpreter = path!("/nonexistent/python3");
        let broken_spec = KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from(missing_interpreter),
            kernelspec: JupyterKernelspec {
                argv: vec![
                    missing_interpreter.to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        });
        cx.update(|cx| {
            ReplStore::global(cx).update(cx, |store, cx| {
                store.set_active_kernelspec(worktree_id, broken_spec, cx);
            })
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("runall.ipynb").into(),
                    },
                    cx,
                )
                .expect("notebook should open")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        cx.executor().allow_parking();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        let pending_kernel = editor.read_with(cx, |editor, _| match &editor.kernel {
            Kernel::StartingKernel(task) => task.clone(),
            _ => panic!("kernel should be starting"),
        });
        pending_kernel.await;

        let (request_tx, mut request_rx) = futures::channel::mpsc::channel(16);
        let (stdin_tx, _stdin_rx) = futures::channel::mpsc::channel(16);
        editor.update(cx, |editor, _| {
            editor.kernel = Kernel::RunningKernel(Box::new(FakeRunningKernel {
                request_tx,
                stdin_tx,
                working_directory: PathBuf::from("/notebooks"),
                execution_state: runtimelib::ExecutionState::Idle,
                kernel_info: None,
            }));
        });

        editor.update_in(cx, |editor, window, cx| {
            editor.run_cells(window, cx);
        });

        let mut requests = Vec::new();
        for _ in 0..3 {
            let request = request_rx.next().await.expect("execute request");
            match &request.content {
                JupyterMessageContent::ExecuteRequest(request) => assert!(!request.silent),
                other => panic!("expected an execute request, got {other:?}"),
            }
            requests.push(request);
        }

        // Shell replies and iopub outputs travel on independent sockets, so the
        // client can deliver a reply before the outputs published for the same
        // request. The last cell reorders its messages this way to cover that.
        for (index, request) in requests.iter().enumerate() {
            let count = index + 1;
            let reply = JupyterMessageContent::ExecuteReply(ExecuteReply {
                status: ReplyStatus::Ok,
                execution_count: runtimelib::ExecutionCount::new(count),
                ..Default::default()
            });
            let input = JupyterMessageContent::ExecuteInput(runtimelib::ExecuteInput {
                code: String::new(),
                execution_count: runtimelib::ExecutionCount::new(count),
            });
            let output = match index {
                0 => Some(JupyterMessageContent::ExecuteResult(
                    runtimelib::ExecuteResult {
                        execution_count: runtimelib::ExecutionCount::new(count),
                        data: jupyter_protocol::Media::new(vec![MediaType::Plain(
                            "(3, 3)".to_string(),
                        )]),
                        metadata: Default::default(),
                        transient: None,
                    },
                )),
                2 => Some(JupyterMessageContent::StreamContent(
                    runtimelib::StreamContent::stdout("0\n1\n2\n"),
                )),
                _ => None,
            };

            route_child(
                &editor,
                request,
                JupyterMessageContent::Status(runtimelib::Status::busy()),
                cx,
            );
            if index == 2 {
                route_child(&editor, request, reply, cx);
                route_child(&editor, request, input, cx);
                if let Some(output) = output {
                    route_child(&editor, request, output, cx);
                }
            } else {
                route_child(&editor, request, input, cx);
                if let Some(output) = output {
                    route_child(&editor, request, output, cx);
                }
                route_child(&editor, request, reply, cx);
            }
            route_child(
                &editor,
                request,
                JupyterMessageContent::Status(runtimelib::Status::idle()),
                cx,
            );
        }

        editor.read_with(cx, |editor, cx| {
            assert_eq!(
                code_cell_counts(editor, cx),
                vec![Some(1), Some(2), Some(3)],
                "each cell must keep the execution count from its own execute_reply"
            );
            for index in 0..3 {
                let cell_id = &editor.cell_order[index];
                let Cell::Code(cell) = editor.cell_map.get(cell_id).expect("code cell") else {
                    panic!("expected a code cell");
                };
                let cell = cell.read(cx);
                assert!(
                    cell.execution_duration().is_some(),
                    "cell {index} should have finished executing"
                );
            }

            let first = match editor.cell_map.get(&editor.cell_order[0]).expect("cell") {
                Cell::Code(cell) => cell.read(cx).to_nbformat_cell(cx),
                _ => panic!("expected a code cell"),
            };
            let nbformat::v4::Cell::Code { outputs, .. } = first else {
                panic!("expected a code cell");
            };
            assert_eq!(
                outputs.len(),
                1,
                "cell A should keep its execute_result, got {outputs:?}"
            );

            let third = match editor.cell_map.get(&editor.cell_order[2]).expect("cell") {
                Cell::Code(cell) => cell.read(cx).to_nbformat_cell(cx),
                _ => panic!("expected a code cell"),
            };
            let nbformat::v4::Cell::Code { outputs, .. } = third else {
                panic!("expected a code cell");
            };
            assert_eq!(
                outputs.len(),
                1,
                "cell C should keep its stream output, got {outputs:?}"
            );
        });
    }

    #[gpui::test]
    async fn test_code_cells_receive_the_hosting_workspace(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let project_path = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project.worktrees(cx).next().unwrap().read(cx).id(),
            path: rel_path("test.ipynb").into(),
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(&project, &project_path, cx)
                    .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let notebook_editor = window
            .update(cx, |_multi_workspace, window, cx| {
                cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
            })
            .expect("the window should still be open");

        window
            .update(cx, |multi_workspace, window, cx| {
                let workspace = multi_workspace.workspace().clone();
                workspace.update(cx, |workspace, cx| {
                    workspace.add_item_to_active_pane(
                        Box::new(notebook_editor.clone()),
                        None,
                        true,
                        window,
                        cx,
                    );
                });
            })
            .expect("the window should still be open");

        notebook_editor.read_with(cx, |notebook_editor, cx| {
            let mut checked_cells = 0;
            for cell in notebook_editor.cell_map.values() {
                let Cell::Code(cell) = cell else {
                    continue;
                };
                cell.read(cx).editor().read_with(cx, |editor, _| {
                    assert!(
                        editor.workspace().is_some(),
                        "a code cell editor must know its workspace so it can open references and apply code actions"
                    );
                });
                checked_cells += 1;
            }
            assert_eq!(checked_cells, 1, "the notebook has one code cell");
        });
    }
}
