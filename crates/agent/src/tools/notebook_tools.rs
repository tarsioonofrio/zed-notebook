use super::tool_permissions::authorize_file_edit;
use crate::{AgentTool, Thread, ToolCallEventStream, ToolInput};
use agent_client_protocol::v2::schema as acp;
use fs::Fs;
use gpui::{App, Entity, SharedString, Task, WeakEntity};
use notebook_agent_tools::{LineRange, NotebookCellType, NotebookLiveCommand};
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};

// Cell mutations share the first mutation tool's permission entry so that a
// single "Notebook Editing" rule covers the whole notebook surface.
const NOTEBOOK_PERMISSION_TOOL: &str = "notebook_edit_cell";

fn authorize_notebook_edit(
    thread: &WeakEntity<Thread>,
    path: &str,
    event_stream: &ToolCallEventStream,
    cx: &mut App,
) -> Task<anyhow::Result<()>> {
    authorize_file_edit(
        NOTEBOOK_PERMISSION_TOOL,
        Path::new(path),
        thread,
        event_stream,
        cx,
    )
}

fn resolve_notebook_path(
    project: &Entity<Project>,
    path: &str,
    cx: &App,
) -> Result<(project::ProjectPath, std::path::PathBuf), String> {
    let project_path = project
        .read(cx)
        .find_project_path(path, cx)
        .ok_or_else(|| format!("Couldn't find {path} in this project."))?;
    if project_path.path.extension() != Some(notebook_agent_tools::NOTEBOOK_EXTENSION) {
        return Err(format!(
            "{path} is not a Jupyter notebook (.ipynb). Use the file tools for other files."
        ));
    }
    let absolute_path = project
        .read(cx)
        .absolute_path(&project_path, cx)
        .ok_or_else(|| format!("Couldn't resolve the absolute path of {path}."))?;
    Ok((project_path, absolute_path))
}

async fn read_notebook_json(
    fs: &Arc<dyn Fs>,
    absolute_path: &Path,
) -> Result<serde_json::Value, String> {
    let content = fs
        .load(absolute_path)
        .await
        .map_err(|error| format!("Failed to read {}: {error}", absolute_path.display()))?;
    serde_json::from_str(&content).map_err(|error| {
        format!(
            "{} is not valid notebook JSON: {error}",
            absolute_path.display()
        )
    })
}

async fn write_notebook_json(
    fs: &Arc<dyn Fs>,
    absolute_path: &Path,
    notebook: &serde_json::Value,
) -> Result<(), String> {
    let serialized = notebook_agent_tools::to_notebook_json(notebook)?;
    fs.write(absolute_path, serialized.as_bytes())
        .await
        .map_err(|error| format!("Failed to write {}: {error}", absolute_path.display()))
}

/// Reads a Jupyter notebook and lists its cells with 1-based positions, types,
/// ids, sources and a summary of each code cell's outputs.
///
/// Use this instead of `read_file` for `.ipynb` files so cells are presented in
/// order and the positions match the 1-based cell numbers used by the notebook
/// editing tools. Call it again after mutating cells to confirm the result.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NotebookReadToolInput {
    /// The path of the notebook to read, relative to a project root or absolute.
    pub path: String,
}

pub struct NotebookReadTool {
    project: Entity<Project>,
}

impl NotebookReadTool {
    pub fn new(project: Entity<Project>) -> Self {
        Self { project }
    }
}

impl AgentTool for NotebookReadTool {
    type Input = NotebookReadToolInput;
    type Output = String;

    const NAME: &'static str = "notebook_read";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!("Read notebook `{}`", input.path).into(),
            Err(_) => "Read notebook".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let (_project_path, absolute_path) =
                cx.update(|cx| resolve_notebook_path(&project, &input.path, cx))?;

            if let Some(result) = cx.update(|cx| {
                notebook_agent_tools::run_live(cx, &absolute_path, NotebookLiveCommand::Read)
            }) {
                let result = result?;
                return Ok(notebook_agent_tools::format_read_report(
                    &absolute_path,
                    &result.snapshot,
                ));
            }

            let fs = project.read_with(cx, |project, _cx| project.fs().clone());
            let notebook = read_notebook_json(&fs, &absolute_path).await?;
            let snapshot = notebook_agent_tools::read_cells(&notebook)?;
            Ok(notebook_agent_tools::format_read_report(
                &absolute_path,
                &snapshot,
            ))
        })
    }
}

/// Replaces the source of an existing cell in a Jupyter notebook.
///
/// Cell numbers are 1-based. Use `notebook_read` first to inspect the current
/// cells. To change only part of a cell, pass a 1-based inclusive line range
/// with both `start_line` and `end_line`; otherwise the whole source is
/// replaced. Metadata, outputs and every other cell are preserved.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NotebookEditCellToolInput {
    /// The path of the notebook, relative to a project root or absolute.
    pub path: String,
    /// The 1-based position of the cell to edit.
    pub cell_number: usize,
    /// The replacement source. When a line range is given this replaces only
    /// those lines.
    pub new_source: String,
    /// The first line to replace (1-based, inclusive). Requires `end_line`.
    pub start_line: Option<u32>,
    /// The last line to replace (1-based, inclusive). Requires `start_line`.
    pub end_line: Option<u32>,
}

pub struct NotebookEditCellTool {
    project: Entity<Project>,
    thread: WeakEntity<Thread>,
}

impl NotebookEditCellTool {
    pub fn new(project: Entity<Project>, thread: WeakEntity<Thread>) -> Self {
        Self { project, thread }
    }
}

impl AgentTool for NotebookEditCellTool {
    type Input = NotebookEditCellToolInput;
    type Output = String;

    const NAME: &'static str = "notebook_edit_cell";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Edit
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!("Edit cell {} in `{}`", input.cell_number, input.path).into(),
            Err(_) => "Edit notebook cell".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let line_range = parse_line_range(input.start_line, input.end_line)?;
            let (_, absolute_path) =
                cx.update(|cx| resolve_notebook_path(&project, &input.path, cx))?;

            let authorize =
                cx.update(|cx| authorize_notebook_edit(&thread, &input.path, &event_stream, cx));
            authorize.await.map_err(|error| error.to_string())?;

            let command = NotebookLiveCommand::EditCell {
                cell_number: input.cell_number,
                new_source: input.new_source.clone(),
                line_range,
            };
            if let Some(result) =
                cx.update(|cx| notebook_agent_tools::run_live(cx, &absolute_path, command.clone()))
            {
                return result.map(|result| result.message);
            }

            let fs = project.read_with(cx, |project, _cx| project.fs().clone());
            let mut notebook = read_notebook_json(&fs, &absolute_path).await?;
            let message = notebook_agent_tools::edit_cell(
                &mut notebook,
                input.cell_number,
                &input.new_source,
                line_range,
            )?;
            write_notebook_json(&fs, &absolute_path, &notebook).await?;
            Ok(message)
        })
    }
}

/// Inserts a new cell into a Jupyter notebook.
///
/// `position` is 1-based: the new cell becomes the cell at that position, so
/// `position = 1` inserts at the top and `position = cell_count + 1` appends.
/// Use `notebook_read` to see the current cell numbering.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NotebookInsertCellToolInput {
    /// The path of the notebook, relative to a project root or absolute.
    pub path: String,
    /// The 1-based position where the new cell should be inserted.
    pub position: usize,
    /// The kind of cell to create.
    pub cell_type: NotebookCellTypeInput,
    /// The initial source of the new cell.
    pub source: String,
}

/// The cell kind accepted by `notebook_insert_cell`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NotebookCellTypeInput {
    Code,
    Markdown,
}

impl From<NotebookCellTypeInput> for NotebookCellType {
    fn from(cell_type: NotebookCellTypeInput) -> Self {
        match cell_type {
            NotebookCellTypeInput::Code => NotebookCellType::Code,
            NotebookCellTypeInput::Markdown => NotebookCellType::Markdown,
        }
    }
}

pub struct NotebookInsertCellTool {
    project: Entity<Project>,
    thread: WeakEntity<Thread>,
}

impl NotebookInsertCellTool {
    pub fn new(project: Entity<Project>, thread: WeakEntity<Thread>) -> Self {
        Self { project, thread }
    }
}

impl AgentTool for NotebookInsertCellTool {
    type Input = NotebookInsertCellToolInput;
    type Output = String;

    const NAME: &'static str = "notebook_insert_cell";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Edit
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!(
                "Insert {} cell at position {} in `{}`",
                match input.cell_type {
                    NotebookCellTypeInput::Code => "code",
                    NotebookCellTypeInput::Markdown => "markdown",
                },
                input.position,
                input.path
            )
            .into(),
            Err(_) => "Insert notebook cell".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let (_, absolute_path) =
                cx.update(|cx| resolve_notebook_path(&project, &input.path, cx))?;

            let authorize =
                cx.update(|cx| authorize_notebook_edit(&thread, &input.path, &event_stream, cx));
            authorize.await.map_err(|error| error.to_string())?;

            let cell_type: NotebookCellType = input.cell_type.into();
            let command = NotebookLiveCommand::InsertCell {
                position: input.position,
                cell_type,
                source: input.source.clone(),
            };
            if let Some(result) =
                cx.update(|cx| notebook_agent_tools::run_live(cx, &absolute_path, command.clone()))
            {
                return result.map(|result| result.message);
            }

            let fs = project.read_with(cx, |project, _cx| project.fs().clone());
            let mut notebook = read_notebook_json(&fs, &absolute_path).await?;
            let message = notebook_agent_tools::insert_cell(
                &mut notebook,
                input.position,
                cell_type,
                &input.source,
            )?;
            write_notebook_json(&fs, &absolute_path, &notebook).await?;
            Ok(message)
        })
    }
}

/// Deletes the cell at the given 1-based position from a Jupyter notebook.
///
/// Use `notebook_read` first to confirm the cell number. The remaining cells
/// keep their content, ids, metadata and outputs.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NotebookDeleteCellToolInput {
    /// The path of the notebook, relative to a project root or absolute.
    pub path: String,
    /// The 1-based position of the cell to delete.
    pub cell_number: usize,
}

pub struct NotebookDeleteCellTool {
    project: Entity<Project>,
    thread: WeakEntity<Thread>,
}

impl NotebookDeleteCellTool {
    pub fn new(project: Entity<Project>, thread: WeakEntity<Thread>) -> Self {
        Self { project, thread }
    }
}

impl AgentTool for NotebookDeleteCellTool {
    type Input = NotebookDeleteCellToolInput;
    type Output = String;

    const NAME: &'static str = "notebook_delete_cell";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Delete
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!("Delete cell {} from `{}`", input.cell_number, input.path).into(),
            Err(_) => "Delete notebook cell".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let (_, absolute_path) =
                cx.update(|cx| resolve_notebook_path(&project, &input.path, cx))?;

            let authorize =
                cx.update(|cx| authorize_notebook_edit(&thread, &input.path, &event_stream, cx));
            authorize.await.map_err(|error| error.to_string())?;

            let command = NotebookLiveCommand::DeleteCell {
                cell_number: input.cell_number,
            };
            if let Some(result) =
                cx.update(|cx| notebook_agent_tools::run_live(cx, &absolute_path, command.clone()))
            {
                return result.map(|result| result.message);
            }

            let fs = project.read_with(cx, |project, _cx| project.fs().clone());
            let mut notebook = read_notebook_json(&fs, &absolute_path).await?;
            let message = notebook_agent_tools::delete_cell(&mut notebook, input.cell_number)?;
            write_notebook_json(&fs, &absolute_path, &notebook).await?;
            Ok(message)
        })
    }
}

/// Moves a cell to a different 1-based position in a Jupyter notebook,
/// preserving its source, id, metadata and outputs.
///
/// `to_cell` is the position the cell should end up at after the move.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct NotebookMoveCellToolInput {
    /// The path of the notebook, relative to a project root or absolute.
    pub path: String,
    /// The 1-based position of the cell to move.
    pub from_cell: usize,
    /// The 1-based position where the cell should end up.
    pub to_cell: usize,
}

pub struct NotebookMoveCellTool {
    project: Entity<Project>,
    thread: WeakEntity<Thread>,
}

impl NotebookMoveCellTool {
    pub fn new(project: Entity<Project>, thread: WeakEntity<Thread>) -> Self {
        Self { project, thread }
    }
}

impl AgentTool for NotebookMoveCellTool {
    type Input = NotebookMoveCellToolInput;
    type Output = String;

    const NAME: &'static str = "notebook_move_cell";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Move
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!(
                "Move cell {} to position {} in `{}`",
                input.from_cell, input.to_cell, input.path
            )
            .into(),
            Err(_) => "Move notebook cell".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let (_, absolute_path) =
                cx.update(|cx| resolve_notebook_path(&project, &input.path, cx))?;

            let authorize =
                cx.update(|cx| authorize_notebook_edit(&thread, &input.path, &event_stream, cx));
            authorize.await.map_err(|error| error.to_string())?;

            let command = NotebookLiveCommand::MoveCell {
                from_cell: input.from_cell,
                to_cell: input.to_cell,
            };
            if let Some(result) =
                cx.update(|cx| notebook_agent_tools::run_live(cx, &absolute_path, command.clone()))
            {
                return result.map(|result| result.message);
            }

            let fs = project.read_with(cx, |project, _cx| project.fs().clone());
            let mut notebook = read_notebook_json(&fs, &absolute_path).await?;
            let message =
                notebook_agent_tools::move_cell(&mut notebook, input.from_cell, input.to_cell)?;
            write_notebook_json(&fs, &absolute_path, &notebook).await?;
            Ok(message)
        })
    }
}

fn parse_line_range(
    start_line: Option<u32>,
    end_line: Option<u32>,
) -> Result<Option<LineRange>, String> {
    match (start_line, end_line) {
        (Some(start), Some(end)) => LineRange::new(start, end).map(Some),
        (None, None) => Ok(None),
        _ => Err("Provide both start_line and end_line, or neither of them.".to_string()),
    }
}
