mod native_kernel;
use std::{fmt::Debug, future::Future, path::PathBuf, sync::Arc};

use futures::{channel::mpsc, future::Shared};
use gpui::{App, Entity, Task, Window};
use language::LanguageName;
use log;
pub use native_kernel::*;

mod remote_kernels;
use project::{Project, ProjectPath, Toolchains, WorktreeId};
use remote::RemoteConnectionOptions;
pub use remote_kernels::*;

mod ssh_kernel;
pub use ssh_kernel::*;

mod wsl_kernel;
pub use wsl_kernel::*;

use std::collections::HashMap;

use anyhow::Result;
use futures::{FutureExt, StreamExt};
use gpui::{AppContext, AsyncWindowContext, Context};
use jupyter_protocol::{JupyterKernelspec, JupyterMessageContent};
use runtimelib::{
    ClientControlConnection, ClientIoPubConnection, ClientShellConnection, ClientStdinConnection,
    ExecutionState, JupyterMessage, KernelInfoReply,
};
use ui::{Icon, IconName, SharedString};
use util::rel_path::RelPath;

pub(crate) const VENV_DIR_NAMES: &[&str] = &[".venv", "venv", ".env", "env"];

// Build a POSIX shell script that attempts to find and exec the best Python binary to run with the given arguments.
pub(crate) fn build_python_exec_shell_script(
    python_args: &str,
    cd_command: &str,
    env_command: &str,
) -> String {
    let venv_dirs = VENV_DIR_NAMES.join(" ");
    format!(
        "set -e; \
         {cd_command}\
         {env_command}\
         for venv_dir in {venv_dirs}; do \
           if [ -f \"$venv_dir/pyvenv.cfg\" ] || [ -f \"$venv_dir/bin/activate\" ]; then \
             if [ -x \"$venv_dir/bin/python\" ]; then \
               exec \"$venv_dir/bin/python\" {python_args}; \
             elif [ -x \"$venv_dir/bin/python3\" ]; then \
               exec \"$venv_dir/bin/python3\" {python_args}; \
             fi; \
           fi; \
         done; \
         if command -v python3 >/dev/null 2>&1; then \
           exec python3 {python_args}; \
         elif command -v python >/dev/null 2>&1; then \
           exec python {python_args}; \
         else \
           echo 'Error: Python not found in virtual environment or PATH' >&2; \
           exit 127; \
         fi"
    )
}

/// Build a POSIX shell script that outputs the best Python binary.
#[cfg(target_os = "windows")]
pub(crate) fn build_python_discovery_shell_script() -> String {
    let venv_dirs = VENV_DIR_NAMES.join(" ");
    format!(
        "for venv_dir in {venv_dirs}; do \
           if [ -f \"$venv_dir/pyvenv.cfg\" ] || [ -f \"$venv_dir/bin/activate\" ]; then \
             if [ -x \"$venv_dir/bin/python\" ]; then \
               echo \"$venv_dir/bin/python\"; exit 0; \
             elif [ -x \"$venv_dir/bin/python3\" ]; then \
               echo \"$venv_dir/bin/python3\"; exit 0; \
             fi; \
           fi; \
         done; \
         if command -v python3 >/dev/null 2>&1; then \
           echo python3; exit 0; \
         elif command -v python >/dev/null 2>&1; then \
           echo python; exit 0; \
         fi; \
         exit 1"
    )
}

pub fn start_kernel_tasks<S: KernelSession + 'static>(
    session: Entity<S>,
    iopub_socket: ClientIoPubConnection,
    shell_socket: ClientShellConnection,
    control_socket: ClientControlConnection,
    stdin_socket: ClientStdinConnection,
    cx: &mut AsyncWindowContext,
) -> (
    futures::channel::mpsc::Sender<JupyterMessage>,
    futures::channel::mpsc::Sender<JupyterMessage>,
) {
    let (mut shell_send, shell_recv) = shell_socket.split();
    let (mut control_send, control_recv) = control_socket.split();
    let (mut stdin_send, stdin_recv) = stdin_socket.split();

    let (request_tx, mut request_rx) = futures::channel::mpsc::channel::<JupyterMessage>(100);
    let (stdin_tx, mut stdin_rx) = futures::channel::mpsc::channel::<JupyterMessage>(100);

    let recv_task = cx.spawn({
        let session = session.clone();
        let mut iopub = iopub_socket;
        let mut shell = shell_recv;
        let mut control = control_recv;
        let mut stdin = stdin_recv;

        async move |cx| -> anyhow::Result<()> {
            loop {
                let (channel, result) = futures::select! {
                    msg = iopub.read().fuse() => ("iopub", msg),
                    msg = shell.read().fuse() => ("shell", msg),
                    msg = control.read().fuse() => ("control", msg),
                    msg = stdin.read().fuse() => ("stdin", msg),
                };
                match result {
                    Ok(message) => {
                        session
                            .update_in(cx, |session, window, cx| {
                                session.route(&message, window, cx);
                            })
                            .ok();
                    }
                    Err(
                        ref err @ (runtimelib::RuntimeError::ParseError { .. }
                        | runtimelib::RuntimeError::SerdeError(_)),
                    ) => {
                        let error_detail = format!("Kernel issue on {channel} channel\n\n{err}");
                        log::warn!("kernel: {error_detail}");
                        session
                            .update_in(cx, |session, _window, cx| {
                                session.kernel_errored(error_detail, cx);
                                cx.notify();
                            })
                            .ok();
                    }
                    Err(err) => {
                        log::warn!("kernel: error reading from {channel}: {err:?}");
                        anyhow::bail!("{channel} recv: {err}");
                    }
                }
            }
        }
    });

    let routing_task = cx.background_spawn(async move {
        while let Some(message) = request_rx.next().await {
            match message.content {
                JupyterMessageContent::DebugRequest(_)
                | JupyterMessageContent::InterruptRequest(_)
                | JupyterMessageContent::ShutdownRequest(_) => {
                    control_send.send(message).await?;
                }
                _ => {
                    shell_send.send(message).await?;
                }
            }
        }
        anyhow::Ok(())
    });

    let stdin_routing_task = cx.background_spawn(async move {
        while let Some(message) = stdin_rx.next().await {
            stdin_send.send(message).await?;
        }
        anyhow::Ok(())
    });

    cx.spawn({
        async move |cx| {
            async fn with_name(
                name: &'static str,
                task: Task<Result<()>>,
            ) -> (&'static str, Result<()>) {
                (name, task.await)
            }

            let mut tasks = futures::stream::FuturesUnordered::new();
            tasks.push(with_name("recv task", recv_task));
            tasks.push(with_name("routing task", routing_task));
            tasks.push(with_name("stdin routing task", stdin_routing_task));

            while let Some((name, result)) = tasks.next().await {
                if let Err(err) = result {
                    session.update(cx, |session, cx| {
                        session.kernel_errored(format!("handling failed for {name}: {err}"), cx);
                        cx.notify();
                    });
                }
            }
        }
    })
    .detach();

    (request_tx, stdin_tx)
}

pub trait KernelSession: Sized {
    fn route(&mut self, message: &JupyterMessage, window: &mut Window, cx: &mut Context<Self>);
    fn kernel_errored(&mut self, error_message: String, cx: &mut Context<Self>);
}

#[derive(Debug, Clone)]
pub struct PythonEnvKernelSpecification {
    pub name: String,
    pub path: PathBuf,
    pub kernelspec: JupyterKernelspec,
    pub has_ipykernel: bool,
    /// Display label for the environment type: "venv", "Conda", "Pyenv", etc.
    pub environment_kind: Option<String>,
}

impl PartialEq for PythonEnvKernelSpecification {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.path == other.path
    }
}

impl Eq for PythonEnvKernelSpecification {}

impl PythonEnvKernelSpecification {
    pub fn as_local_spec(&self) -> LocalKernelSpecification {
        LocalKernelSpecification {
            name: self.name.clone(),
            path: self.path.clone(),
            kernelspec: self.kernelspec.clone(),
        }
    }

    pub fn is_uv(&self) -> bool {
        matches!(
            self.environment_kind.as_deref(),
            Some("uv" | "uv (Workspace)")
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelSpecification {
    JupyterServer(RemoteKernelSpecification),
    Jupyter(LocalKernelSpecification),
    PythonEnv(PythonEnvKernelSpecification),
    SshRemote(SshRemoteKernelSpecification),
    WslRemote(WslKernelSpecification),
}

#[derive(Debug, Clone)]
pub struct SshRemoteKernelSpecification {
    pub name: String,
    pub path: SharedString,
    pub kernelspec: JupyterKernelspec,
}

#[derive(Debug, Clone)]
pub struct WslKernelSpecification {
    pub name: String,
    pub kernelspec: JupyterKernelspec,
    pub distro: String,
}

impl PartialEq for SshRemoteKernelSpecification {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.kernelspec.argv == other.kernelspec.argv
            && self.path == other.path
            && self.kernelspec.display_name == other.kernelspec.display_name
            && self.kernelspec.language == other.kernelspec.language
            && self.kernelspec.interrupt_mode == other.kernelspec.interrupt_mode
            && self.kernelspec.env == other.kernelspec.env
            && self.kernelspec.metadata == other.kernelspec.metadata
    }
}

impl Eq for SshRemoteKernelSpecification {}

impl PartialEq for WslKernelSpecification {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.kernelspec.argv == other.kernelspec.argv
            && self.kernelspec.display_name == other.kernelspec.display_name
            && self.kernelspec.language == other.kernelspec.language
            && self.kernelspec.interrupt_mode == other.kernelspec.interrupt_mode
            && self.kernelspec.env == other.kernelspec.env
            && self.kernelspec.metadata == other.kernelspec.metadata
            && self.distro == other.distro
    }
}

impl Eq for WslKernelSpecification {}

impl KernelSpecification {
    pub fn name(&self) -> SharedString {
        match self {
            Self::Jupyter(spec) => spec.name.clone().into(),
            Self::PythonEnv(spec) => spec.name.clone().into(),
            Self::JupyterServer(spec) => spec.name.clone().into(),
            Self::SshRemote(spec) => spec.name.clone().into(),
            Self::WslRemote(spec) => spec.kernelspec.display_name.clone().into(),
        }
    }

    pub fn type_name(&self) -> SharedString {
        match self {
            Self::Jupyter(_) => "Jupyter".into(),
            Self::PythonEnv(spec) => SharedString::from(
                spec.environment_kind
                    .clone()
                    .unwrap_or_else(|| "Python Environment".to_string()),
            ),
            Self::JupyterServer(_) => "Jupyter Server".into(),
            Self::SshRemote(_) => "SSH Remote".into(),
            Self::WslRemote(_) => "WSL Remote".into(),
        }
    }

    pub fn path(&self) -> SharedString {
        SharedString::from(match self {
            Self::Jupyter(spec) => spec.path.to_string_lossy().into_owned(),
            Self::PythonEnv(spec) => spec.path.to_string_lossy().into_owned(),
            Self::JupyterServer(spec) => spec.url.to_string(),
            Self::SshRemote(spec) => spec.path.to_string(),
            Self::WslRemote(spec) => spec.distro.clone(),
        })
    }

    pub fn language(&self) -> SharedString {
        SharedString::from(match self {
            Self::Jupyter(spec) => spec.kernelspec.language.clone(),
            Self::PythonEnv(spec) => spec.kernelspec.language.clone(),
            Self::JupyterServer(spec) => spec.kernelspec.language.clone(),
            Self::SshRemote(spec) => spec.kernelspec.language.clone(),
            Self::WslRemote(spec) => spec.kernelspec.language.clone(),
        })
    }

    pub fn has_ipykernel(&self) -> bool {
        match self {
            Self::Jupyter(_) | Self::JupyterServer(_) | Self::SshRemote(_) | Self::WslRemote(_) => {
                true
            }
            Self::PythonEnv(spec) => spec.has_ipykernel,
        }
    }

    pub fn environment_kind_label(&self) -> Option<SharedString> {
        match self {
            Self::PythonEnv(spec) => spec
                .environment_kind
                .as_ref()
                .map(|kind| SharedString::from(kind.clone())),
            Self::Jupyter(_) => Some("Jupyter".into()),
            Self::JupyterServer(_) => Some("Jupyter Server".into()),
            Self::SshRemote(_) => Some("SSH Remote".into()),
            Self::WslRemote(_) => Some("WSL Remote".into()),
        }
    }

    pub fn icon(&self, cx: &App) -> Icon {
        let lang_name = match self {
            Self::Jupyter(spec) => spec.kernelspec.language.clone(),
            Self::PythonEnv(spec) => spec.kernelspec.language.clone(),
            Self::JupyterServer(spec) => spec.kernelspec.language.clone(),
            Self::SshRemote(spec) => spec.kernelspec.language.clone(),
            Self::WslRemote(spec) => spec.kernelspec.language.clone(),
        };

        file_icons::FileIcons::get(cx)
            .get_icon_for_type(&lang_name.to_lowercase(), cx)
            .map(Icon::from_path)
            .unwrap_or(Icon::new(IconName::ReplNeutral))
    }
}

fn extract_environment_kind(toolchain_json: &serde_json::Value) -> Option<String> {
    let kind_str = toolchain_json.get("kind")?.as_str()?;
    let label = match kind_str {
        "Conda" => "Conda",
        "Pixi" => "pixi",
        "Homebrew" => "Homebrew",
        "Pyenv" => "global (Pyenv)",
        "GlobalPaths" => "global",
        "PyenvVirtualEnv" => "Pyenv",
        "Pipenv" => "Pipenv",
        "Poetry" => "Poetry",
        "MacPythonOrg" => "global (Python.org)",
        "MacCommandLineTools" => "global (Command Line Tools for Xcode)",
        "LinuxGlobal" => "global",
        "MacXCode" => "global (Xcode)",
        "Venv" => "venv",
        "VirtualEnv" => "virtualenv",
        "VirtualEnvWrapper" => "virtualenvwrapper",
        "WindowsStore" => "global (Windows Store)",
        "WindowsRegistry" => "global (Windows Registry)",
        "Uv" => "uv",
        "UvWorkspace" => "uv (Workspace)",
        _ => kind_str,
    };
    Some(label.to_string())
}

/// Builds the kernel specification for a Python interpreter discovered on
/// disk, following the same conventions as toolchain-backed environments
/// (argv, display name, and the `PATH`/`VIRTUAL_ENV` variables the kernel
/// inherits).
pub(crate) fn python_environment_kernel_specification(
    name: String,
    python_path: PathBuf,
    environment_kind: Option<String>,
    has_ipykernel: bool,
) -> PythonEnvKernelSpecification {
    let mut env = HashMap::new();
    if let Some(python_bin_dir) = python_path.parent() {
        if let Some(path_var) = std::env::var_os("PATH") {
            let mut paths = std::env::split_paths(&path_var).collect::<Vec<_>>();
            paths.insert(0, python_bin_dir.to_path_buf());
            if let Ok(new_path) = std::env::join_paths(paths) {
                env.insert("PATH".to_string(), new_path.to_string_lossy().to_string());
            }
        }

        if let Some(venv_root) = python_bin_dir.parent() {
            env.insert(
                "VIRTUAL_ENV".to_string(),
                venv_root.to_string_lossy().to_string(),
            );
        }
    }

    let python_path_string = python_path.to_string_lossy().into_owned();
    let kernelspec = JupyterKernelspec {
        argv: vec![
            python_path_string,
            "-m".to_string(),
            "ipykernel_launcher".to_string(),
            "-f".to_string(),
            "{connection_file}".to_string(),
        ],
        display_name: name.clone(),
        language: "python".to_string(),
        interrupt_mode: None,
        metadata: None,
        env: Some(env),
    };

    PythonEnvKernelSpecification {
        name,
        path: python_path,
        kernelspec,
        has_ipykernel,
        environment_kind,
    }
}

/// Probes a virtual environment interpreter for `ipykernel` and builds its
/// kernel specification. The caller decides which interpreter is closest to
/// the notebook; this only makes the specification usable by the kernel picker.
pub(crate) async fn probe_python_environment_interpreter(
    interpreter: PathBuf,
    environment_kind: Option<String>,
) -> PythonEnvKernelSpecification {
    let interpreter_string = interpreter.to_string_lossy().into_owned();
    // The version is printed before the import so it is still reported when
    // `ipykernel` is missing and the command exits with an error.
    let probe = util::command::new_command(&interpreter_string)
        .args([
            "-c",
            "import sys; print('.'.join(map(str, sys.version_info[:3]))); import ipykernel",
        ])
        .output()
        .await
        .ok();
    let has_ipykernel = probe
        .as_ref()
        .is_some_and(|output| output.status.success());
    let version = probe.and_then(|output| {
        String::from_utf8(output.stdout)
            .ok()?
            .lines()
            .next()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
    });

    let environment_label = match (version, environment_kind.as_deref()) {
        (Some(version), Some(kind)) => format!("Python {version} ({kind})"),
        (Some(version), None) => format!("Python {version}"),
        (None, Some(kind)) => format!("Python ({kind})"),
        (None, None) => "Python".to_string(),
    };
    python_environment_kernel_specification(
        environment_label,
        interpreter,
        environment_kind,
        has_ipykernel,
    )
}

/// How long the remote server gets to list Python environments before the
/// directly probed ones are used alone. The listing can stall indefinitely on
/// hosts where the server's shell environment capture never finishes.
const REMOTE_TOOLCHAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

const REMOTE_PYTHON_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

fn remote_python_probe_script(notebook_directory: &str) -> String {
    let directory = shlex::try_quote(notebook_directory)
        .map(|quoted| quoted.into_owned())
        .unwrap_or_else(|_| "'.'".to_string());
    format!(
        r#"probe() {{
  python="$1"
  [ -x "$python" ] || return 0
  version=$("$python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null) || return 0
  if "$python" -c 'import ipykernel' >/dev/null 2>&1; then ipykernel=1; else ipykernel=0; fi
  printf 'ZED_PYTHON\t%s\t%s\t%s\t%s\n' "$python" "$version" "$ipykernel" "$2"
}}
directory={directory}
while :; do
  for name in .venv venv; do probe "$directory/$name/bin/python" "$name"; done
  [ "$directory" = "/" ] && break
  directory=$(dirname "$directory")
done
for name in python3 python; do
  found=$(command -v "$name" 2>/dev/null) && probe "$found" system
done
[ -n "$CONDA_PREFIX" ] && probe "$CONDA_PREFIX/bin/python" conda
base=$(conda info --base 2>/dev/null) && roots="$base"
for root in "$HOME/mambaforge" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/micromamba" /opt/conda; do
  roots="$roots
$root"
done
printf '%s\n' "$roots" | awk 'NF && !seen[$0]++' | while IFS= read -r root; do
  probe "$root/bin/python" conda
  for environment in "$root"/envs/*/; do probe "${{environment}}bin/python" "conda:$(basename "$environment")"; done
done
for version in "$HOME"/.pyenv/versions/*/; do probe "${{version}}bin/python" "pyenv:$(basename "$version")"; done
exit 0
"#
    )
}

fn parse_remote_python_probe_output(output: &str) -> Vec<SshRemoteKernelSpecification> {
    let mut seen = collections::HashSet::default();
    let mut specifications = Vec::new();
    for line in output.lines() {
        let mut fields = line.split('\t');
        if fields.next() != Some("ZED_PYTHON") {
            continue;
        }
        let (Some(path), Some(version), Some(has_ipykernel), Some(kind)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if has_ipykernel != "1" || !seen.insert(path.to_string()) {
            continue;
        }
        let display_name = format!("Python {version} ({kind})");
        specifications.push(SshRemoteKernelSpecification {
            name: format!("Remote {display_name}"),
            path: SharedString::from(path.to_string()),
            kernelspec: JupyterKernelspec {
                argv: vec![
                    path.to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name,
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        });
    }
    specifications
}

/// Finds Python environments that have `ipykernel` on the remote host by
/// running a probe script there, nearest virtual environment first. This does
/// not depend on the remote server's toolchain listing.
fn probe_remote_python_environments(
    project: &Entity<Project>,
    notebook_directory: PathBuf,
    cx: &App,
) -> Task<Vec<SshRemoteKernelSpecification>> {
    let Some(remote_client) = project.read(cx).remote_client() else {
        return Task::ready(Vec::new());
    };
    let script = remote_python_probe_script(&notebook_directory.to_string_lossy());
    let command = remote_client.read(cx).build_command(
        Some("bash".to_string()),
        &["-lc".to_string(), script],
        &HashMap::default(),
        None,
        None,
        remote::Interactive::No,
    );
    let command = match command {
        Ok(command) => command,
        Err(error) => {
            log::warn!("remote python probe: could not build the ssh command: {error:#}");
            return Task::ready(Vec::new());
        }
    };
    let timer = cx.background_executor().timer(REMOTE_PYTHON_PROBE_TIMEOUT);
    cx.background_spawn(async move {
        let mut command_builder = util::command::new_command(&command.program);
        command_builder.args(&command.args).envs(&command.env);
        let output = smol::future::or(
            async { command_builder.output().await.map_err(anyhow::Error::from) },
            async {
                timer.await;
                Err(anyhow::anyhow!("timed out"))
            },
        )
        .await;
        match output {
            Ok(output) => {
                let specifications =
                    parse_remote_python_probe_output(&String::from_utf8_lossy(&output.stdout));
                log::info!(
                    "remote python probe found {} environments with ipykernel",
                    specifications.len()
                );
                specifications
            }
            Err(error) => {
                log::warn!("remote python probe failed: {error:#}");
                Vec::new()
            }
        }
    })
}

pub fn python_env_kernel_specifications(
    project: &Entity<Project>,
    worktree_id: WorktreeId,
    relative_path: Arc<RelPath>,
    cx: &mut App,
) -> impl Future<Output = Result<Vec<KernelSpecification>>> + use<> {
    let python_language = LanguageName::new_static("Python");
    let is_remote = project.read(cx).is_remote();
    let wsl_distro = project
        .read(cx)
        .remote_connection_options(cx)
        .and_then(|opts| {
            if let RemoteConnectionOptions::Wsl(wsl) = opts {
                Some(wsl.distro_name.clone())
            } else {
                None
            }
        });

    let toolchains = project.read(cx).available_toolchains(
        ProjectPath {
            worktree_id,
            path: relative_path.clone(),
        },
        python_language,
        cx,
    );
    #[allow(unused)]
    let worktree_root_path: Option<std::sync::Arc<std::path::Path>> = project
        .read(cx)
        .worktree_for_id(worktree_id, cx)
        .map(|w| w.read(cx).abs_path());

    let remote_probe = if is_remote && wsl_distro.is_none() {
        let is_single_file = project
            .read(cx)
            .worktree_for_id(worktree_id, cx)
            .is_some_and(|worktree| worktree.read(cx).is_single_file());
        worktree_root_path.as_ref().map(|root| {
            let notebook_directory = if is_single_file {
                root.parent().unwrap_or(root.as_ref()).to_path_buf()
            } else {
                relative_path
                    .parent()
                    .map(|parent| root.join(parent.as_std_path()))
                    .unwrap_or_else(|| root.to_path_buf())
            };
            probe_remote_python_environments(project, notebook_directory, cx)
        })
    } else {
        None
    };

    let background_executor = cx.background_executor().clone();

    async move {
        let probed_remote_specifications = match remote_probe {
            Some(probe) => probe.await,
            None => Vec::new(),
        };

        let toolchains = if is_remote {
            let timeout = background_executor.timer(REMOTE_TOOLCHAIN_TIMEOUT);
            smol::future::or(toolchains, async {
                timeout.await;
                log::warn!(
                    "listing remote python environments timed out; using the probed ones"
                );
                None
            })
            .await
        } else {
            toolchains.await
        };

        let (toolchains, user_toolchains) = if let Some(Toolchains {
            toolchains,
            root_path: _,
            user_toolchains,
        }) = toolchains
        {
            (toolchains, user_toolchains)
        } else {
            return Ok(probed_remote_specifications
                .into_iter()
                .map(KernelSpecification::SshRemote)
                .collect());
        };

        let kernelspecs = user_toolchains
            .into_values()
            .flatten()
            .chain(toolchains.toolchains)
            .map(|toolchain| {
                let wsl_distro = wsl_distro.clone();
                background_executor.spawn(async move {
                    // For remote projects, we assume python is available assuming toolchain is reported.
                    // We can skip the `ipykernel` check or run it remotely.
                    // For MVP, lets trust the toolchain existence or do the check if it's cheap.
                    // `new_smol_command` runs locally. We need to run remotely if `is_remote`.

                    if is_remote {
                        let default_kernelspec = JupyterKernelspec {
                            argv: vec![
                                toolchain.path.to_string(),
                                "-m".to_string(),
                                "ipykernel_launcher".to_string(),
                                "-f".to_string(),
                                "{connection_file}".to_string(),
                            ],
                            display_name: toolchain.name.to_string(),
                            language: "python".to_string(),
                            interrupt_mode: None,
                            metadata: None,
                            env: None,
                        };

                        if let Some(distro) = wsl_distro {
                            log::debug!(
                                "python_env_kernel_specifications: returning WslRemote for toolchain {}",
                                toolchain.name
                            );
                            return Some(KernelSpecification::WslRemote(WslKernelSpecification {
                                name: toolchain.name.to_string(),
                                kernelspec: default_kernelspec,
                                distro,
                            }));
                        }

                        log::debug!(
                            "python_env_kernel_specifications: returning SshRemote for toolchain {}",
                            toolchain.name
                        );
                        return Some(KernelSpecification::SshRemote(
                            SshRemoteKernelSpecification {
                                name: format!("Remote {}", toolchain.name),
                                path: toolchain.path.clone(),
                                kernelspec: default_kernelspec,
                            },
                        ));
                    }

                    let python_path = toolchain.path.to_string();
                    let environment_kind = extract_environment_kind(&toolchain.as_json);

                    let has_ipykernel = util::command::new_command(&python_path)
                        .args(&["-c", "import ipykernel"])
                        .output()
                        .await
                        .map(|output| output.status.success())
                        .unwrap_or(false);

                    let specification = python_environment_kernel_specification(
                        toolchain.name.to_string(),
                        PathBuf::from(&python_path),
                        environment_kind,
                        has_ipykernel,
                    );

                    log::info!("Preparing Python kernel for toolchain: {}", toolchain.name);
                    log::info!("Python path: {}", python_path);
                    if let Some(path) = specification
                        .kernelspec
                        .env
                        .as_ref()
                        .and_then(|env| env.get("PATH"))
                    {
                        log::info!("Kernel PATH: {}", path);
                    } else {
                        log::info!("Kernel PATH not set in env");
                    }
                    if let Some(venv) = specification
                        .kernelspec
                        .env
                        .as_ref()
                        .and_then(|env| env.get("VIRTUAL_ENV"))
                    {
                        log::info!("Kernel VIRTUAL_ENV: {}", venv);
                    }

                    Some(KernelSpecification::PythonEnv(specification))
                })
            });

        #[allow(unused_mut)]
        let mut kernel_specs: Vec<KernelSpecification> = futures::stream::iter(kernelspecs)
            .buffer_unordered(4)
            .filter_map(|x| async move { x })
            .collect::<Vec<_>>()
            .await;

        if probed_remote_specifications.is_empty() {
            // `buffer_unordered` completes in whatever order the interpreter probes
            // finish. Sort so kernel selection does not depend on that timing.
            kernel_specs.sort_by_key(|specification| {
                (
                    specification.path().to_string(),
                    specification.name().to_string(),
                )
            });
        } else {
            // The probed environments come nearest first, which is the order
            // kernel selection relies on.
            let probed_paths = probed_remote_specifications
                .iter()
                .map(|specification| specification.path.to_string())
                .collect::<collections::HashSet<_>>();
            kernel_specs.retain(|specification| match specification {
                KernelSpecification::SshRemote(remote) => {
                    !probed_paths.contains(remote.path.as_ref())
                }
                _ => true,
            });
            let mut combined = probed_remote_specifications
                .into_iter()
                .map(KernelSpecification::SshRemote)
                .collect::<Vec<_>>();
            combined.extend(kernel_specs);
            kernel_specs = combined;
        }

        #[cfg(target_os = "windows")]
        if kernel_specs.is_empty() && !is_remote {
            if let Some(root_path) = worktree_root_path {
                let root_path_str: std::borrow::Cow<str> = root_path.to_string_lossy();
                let (distro, internal_path) =
                    if let Some(path_without_prefix) = root_path_str.strip_prefix(r"\\wsl$\") {
                        if let Some((distro, path)) = path_without_prefix.split_once('\\') {
                            let replaced_path: String = path.replace('\\', "/");
                            (Some(distro), Some(format!("/{}", replaced_path)))
                        } else {
                            (Some(path_without_prefix), Some("/".to_string()))
                        }
                    } else if let Some(path_without_prefix) =
                        root_path_str.strip_prefix(r"\\wsl.localhost\")
                    {
                        if let Some((distro, path)) = path_without_prefix.split_once('\\') {
                            let replaced_path: String = path.replace('\\', "/");
                            (Some(distro), Some(format!("/{}", replaced_path)))
                        } else {
                            (Some(path_without_prefix), Some("/".to_string()))
                        }
                    } else {
                        (None, None)
                    };

                if let (Some(distro), Some(internal_path)) = (distro, internal_path) {
                    let discovery_script = build_python_discovery_shell_script();
                    let script = format!(
                        "cd {} && {}",
                        shlex::try_quote(&internal_path)
                            .unwrap_or(std::borrow::Cow::Borrowed(&internal_path)),
                        discovery_script
                    );
                    let output = util::command::new_command("wsl")
                        .arg("-d")
                        .arg(distro)
                        .arg("bash")
                        .arg("-l")
                        .arg("-c")
                        .arg(&script)
                        .output()
                        .await;

                    if let Ok(output) = output {
                        if output.status.success() {
                            let python_cmd =
                                String::from_utf8_lossy(&output.stdout).trim().to_string();
                            let (python_path, display_suffix) = if python_cmd.contains('/') {
                                let venv_name = python_cmd.split('/').next().unwrap_or("venv");
                                (
                                    format!("{}/{}", internal_path, python_cmd),
                                    format!("({})", venv_name),
                                )
                            } else {
                                (python_cmd, "(System)".to_string())
                            };

                            let display_name = format!("WSL: {} {}", distro, display_suffix);
                            let default_kernelspec = JupyterKernelspec {
                                argv: vec![
                                    python_path,
                                    "-m".to_string(),
                                    "ipykernel_launcher".to_string(),
                                    "-f".to_string(),
                                    "{connection_file}".to_string(),
                                ],
                                display_name: display_name.clone(),
                                language: "python".to_string(),
                                interrupt_mode: None,
                                metadata: None,
                                env: None,
                            };

                            kernel_specs.push(KernelSpecification::WslRemote(
                                WslKernelSpecification {
                                    name: display_name,
                                    kernelspec: default_kernelspec,
                                    distro: distro.to_string(),
                                },
                            ));
                        }
                    }
                }
            }
        }

        anyhow::Ok(kernel_specs)
    }
}

pub trait RunningKernel: Send + Debug {
    fn request_tx(&self) -> mpsc::Sender<JupyterMessage>;
    fn stdin_tx(&self) -> mpsc::Sender<JupyterMessage>;
    fn working_directory(&self) -> &PathBuf;
    fn execution_state(&self) -> &ExecutionState;
    fn set_execution_state(&mut self, state: ExecutionState);
    fn kernel_info(&self) -> Option<&KernelInfoReply>;
    fn set_kernel_info(&mut self, info: KernelInfoReply);
    fn force_shutdown(&mut self, window: &mut Window, cx: &mut App) -> Task<anyhow::Result<()>>;
    fn kill(&mut self);
}

#[derive(Debug, Clone)]
pub enum KernelStatus {
    Idle,
    Busy,
    Starting,
    Error,
    ShuttingDown,
    Shutdown,
    Restarting,
}

impl KernelStatus {
    pub fn is_connected(&self) -> bool {
        matches!(self, KernelStatus::Idle | KernelStatus::Busy)
    }
}

impl ToString for KernelStatus {
    fn to_string(&self) -> String {
        match self {
            KernelStatus::Idle => "Idle".to_string(),
            KernelStatus::Busy => "Busy".to_string(),
            KernelStatus::Starting => "Starting".to_string(),
            KernelStatus::Error => "Error".to_string(),
            KernelStatus::ShuttingDown => "Shutting Down".to_string(),
            KernelStatus::Shutdown => "Shutdown".to_string(),
            KernelStatus::Restarting => "Restarting".to_string(),
        }
    }
}

#[derive(Debug)]
pub enum Kernel {
    RunningKernel(Box<dyn RunningKernel>),
    StartingKernel(Shared<Task<()>>),
    ErroredLaunch(String),
    ShuttingDown,
    Shutdown,
    Restarting,
}

impl From<&Kernel> for KernelStatus {
    fn from(kernel: &Kernel) -> Self {
        match kernel {
            Kernel::RunningKernel(kernel) => match kernel.execution_state() {
                ExecutionState::Idle => KernelStatus::Idle,
                ExecutionState::Busy => KernelStatus::Busy,
                ExecutionState::Unknown => KernelStatus::Error,
                ExecutionState::Starting => KernelStatus::Starting,
                ExecutionState::Restarting => KernelStatus::Restarting,
                ExecutionState::Terminating => KernelStatus::ShuttingDown,
                ExecutionState::AutoRestarting => KernelStatus::Restarting,
                ExecutionState::Dead => KernelStatus::Error,
                ExecutionState::Other(_) => KernelStatus::Error,
            },
            Kernel::StartingKernel(_) => KernelStatus::Starting,
            Kernel::ErroredLaunch(_) => KernelStatus::Error,
            Kernel::ShuttingDown => KernelStatus::ShuttingDown,
            Kernel::Shutdown => KernelStatus::Shutdown,
            Kernel::Restarting => KernelStatus::Restarting,
        }
    }
}

impl Kernel {
    pub fn status(&self) -> KernelStatus {
        self.into()
    }

    pub fn set_execution_state(&mut self, status: &ExecutionState) {
        if let Kernel::RunningKernel(running_kernel) = self {
            running_kernel.set_execution_state(status.clone());
        }
    }

    pub fn set_kernel_info(&mut self, kernel_info: &KernelInfoReply) {
        if let Kernel::RunningKernel(running_kernel) = self {
            running_kernel.set_kernel_info(kernel_info.clone());
        }
    }

    pub fn is_shutting_down(&self) -> bool {
        match self {
            Kernel::Restarting | Kernel::ShuttingDown => true,
            Kernel::RunningKernel(_)
            | Kernel::StartingKernel(_)
            | Kernel::ErroredLaunch(_)
            | Kernel::Shutdown => false,
        }
    }
}

#[cfg(test)]
mod remote_probe_tests {
    use super::*;

    #[test]
    fn probe_output_keeps_only_environments_with_ipykernel_without_duplicates() {
        let output = "banner from the login shell\n\
            ZED_PYTHON\t/work/.venv/bin/python\t3.12.1\t1\t.venv\n\
            ZED_PYTHON\t/usr/bin/python3\t3.6.8\t0\tsystem\n\
            ZED_PYTHON\t/work/.venv/bin/python\t3.12.1\t1\tvenv\n\
            ZED_PYTHON\t/opt/conda/bin/python\t3.10.13\t1\tconda\n\
            ZED_PYTHON\tbroken line\n";

        let specifications = parse_remote_python_probe_output(output);

        let names = specifications
            .iter()
            .map(|specification| specification.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["Remote Python 3.12.1 (.venv)", "Remote Python 3.10.13 (conda)"]
        );
        assert_eq!(
            specifications[0].kernelspec.argv,
            [
                "/work/.venv/bin/python",
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}"
            ]
        );
    }

    #[test]
    fn probe_script_quotes_the_notebook_directory() {
        let script = remote_python_probe_script("/work/my notebooks");
        assert!(script.contains("directory='/work/my notebooks'"));
    }
}
