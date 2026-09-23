use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    io::AsyncReadExt,
    process::Command,
    time::{timeout, Duration},
};

const DEFAULT_COMMAND_TIMEOUT_MS: u64 = 60_000;
const MAX_COMMAND_TIMEOUT_MS: u64 = 600_000;
const MAX_CAPTURED_OUTPUT_BYTES: usize = 64 * 1024;
pub const COMMAND_OUTPUT_EVENT: &str = "tools://command-output";

pub type CommandOutputCallback =
    Arc<dyn Fn(CommandOutputEvent) -> Result<(), String> + Send + Sync>;
type ProcessOutputCallback = dyn Fn(&str, &str) -> Result<(), String> + Send + Sync;

#[derive(Clone)]
pub struct ToolContext {
    pub workspace: PathBuf,
    pub request_id: String,
    pub command_id: String,
    pub command_output: Option<CommandOutputCallback>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CommandOutputEvent {
    pub request_id: String,
    pub command_id: String,
    pub command: String,
    pub phase: String,
    pub stream: Option<String>,
    pub chunk: Option<String>,
    pub status: Option<String>,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    AutoApprove,
    RequiresConfirmation,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub risk_level: RiskLevel,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolSchema {
    pub fn ollama_definition(&self) -> ToolDefinition {
        ToolDefinition {
            kind: "function".to_owned(),
            function: FunctionDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    pub function: ToolFunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolFunctionCall {
    pub name: String,
    pub arguments: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;

    async fn preview(
        &self,
        _arguments: Value,
        _context: ToolContext,
    ) -> Result<Option<Value>, String> {
        Ok(None)
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String>;
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn with_examples() -> Self {
        let mut registry = Self::default();
        registry
            .register(Arc::new(GetCurrentTime))
            .expect("the built-in time tool has a valid name");
        registry
            .register(Arc::new(ListWorkspaceFiles))
            .expect("the built-in workspace listing tool has a valid name");
        registry
            .register(Arc::new(ReadFile))
            .expect("the built-in file reading tool has a valid name");
        registry
            .register(Arc::new(CreateFile))
            .expect("the built-in file creation tool has a valid name");
        registry
            .register(Arc::new(EditFile))
            .expect("the built-in file editing tool has a valid name");
        registry
            .register(Arc::new(RunCommand))
            .expect("the built-in command tool has a valid name");
        registry
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), String> {
        let schema = tool.schema();
        if schema.name.trim().is_empty() {
            return Err("Tool name cannot be empty".to_owned());
        }
        if self.tools.contains_key(&schema.name) {
            return Err(format!("Tool '{}' is already registered", schema.name));
        }
        self.tools.insert(schema.name, tool);
        Ok(())
    }

    pub fn schemas(&self) -> Vec<ToolSchema> {
        let mut schemas = self
            .tools
            .values()
            .map(|tool| tool.schema())
            .collect::<Vec<_>>();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.schemas()
            .iter()
            .map(ToolSchema::ollama_definition)
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }
}

struct GetCurrentTime;

#[async_trait]
impl Tool for GetCurrentTime {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "get_current_time".to_owned(),
            description: "Get the current UTC time as milliseconds since the Unix epoch."
                .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, _arguments: Value, _context: ToolContext) -> Result<Value, String> {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("System clock is before the Unix epoch: {error}"))?
            .as_millis();
        Ok(json!({ "unix_time_ms": millis }))
    }
}

struct ListWorkspaceFiles;

#[async_trait]
impl Tool for ListWorkspaceFiles {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "list_workspace_files".to_owned(),
            description: "List the names and entry types in the workspace root directory."
                .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, _arguments: Value, context: ToolContext) -> Result<Value, String> {
        let entries = fs::read_dir(&context.workspace)
            .map_err(|error| format!("Could not list workspace: {error}"))?;
        let mut files = entries
            .map(|entry| {
                let entry =
                    entry.map_err(|error| format!("Could not read workspace entry: {error}"))?;
                let file_type = entry
                    .file_type()
                    .map_err(|error| format!("Could not inspect workspace entry: {error}"))?;
                Ok(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "type": if file_type.is_dir() { "directory" } else { "file" }
                }))
            })
            .collect::<Result<Vec<_>, String>>()?;
        files.sort_by(|left, right| left["name"].as_str().cmp(&right["name"].as_str()));
        Ok(json!({ "entries": files }))
    }
}

struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "read_file".to_owned(),
            description: "Read a text file using a path relative to the active workspace."
                .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative file path." }
                },
                "required": ["path"]
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let relative_path = argument_string(&arguments, "path")?;
        let path = resolve_workspace_path(&context.workspace, relative_path, true)?;
        let content = fs::read_to_string(&path)
            .map_err(|error| format!("Could not read {relative_path}: {error}"))?;
        Ok(json!({ "path": relative_path, "content": content }))
    }
}

struct CreateFile;

#[async_trait]
impl Tool for CreateFile {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "create_file".to_owned(),
            description: "Propose creating a workspace file with the supplied content. Existing files are only replaced when overwrite is true.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative file path." },
                    "content": { "type": "string", "description": "Complete file content." },
                    "overwrite": { "type": "boolean", "description": "Replace an existing file; defaults to false." }
                },
                "required": ["path", "content"]
            }),
            risk_level: RiskLevel::RequiresConfirmation,
        }
    }

    async fn preview(
        &self,
        arguments: Value,
        context: ToolContext,
    ) -> Result<Option<Value>, String> {
        let (relative_path, before, after) = create_file_change(&arguments, &context.workspace)?;
        Ok(Some(json!({
            "operation": "create",
            "path": relative_path,
            "before": before,
            "after": after
        })))
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let (relative_path, before, content) = create_file_change(&arguments, &context.workspace)?;
        ensure_approved_snapshot(&arguments, &before)?;
        let overwrite = argument_bool(&arguments, "overwrite", false)?;
        let path =
            resolve_workspace_path(&context.workspace, &relative_path.to_string_lossy(), false)?;
        if overwrite {
            fs::write(&path, content)
                .map_err(|error| format!("Could not write {}: {error}", relative_path.display()))?;
        } else {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| {
                    format!(
                        "Could not create {} (it may already exist): {error}",
                        relative_path.display()
                    )
                })?;
            file.write_all(content.as_bytes())
                .map_err(|error| format!("Could not write {}: {error}", relative_path.display()))?;
        }
        Ok(json!({ "path": relative_path, "created": true }))
    }
}

struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "edit_file".to_owned(),
            description:
                "Propose an exact, unique old_str to new_str replacement in a workspace file."
                    .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative file path." },
                    "old_str": { "type": "string", "description": "Exact text to replace; it must occur exactly once." },
                    "new_str": { "type": "string", "description": "Replacement text." }
                },
                "required": ["path", "old_str", "new_str"]
            }),
            risk_level: RiskLevel::RequiresConfirmation,
        }
    }

    async fn preview(
        &self,
        arguments: Value,
        context: ToolContext,
    ) -> Result<Option<Value>, String> {
        let (relative_path, before, after) = edit_file_change(&arguments, &context.workspace)?;
        Ok(Some(json!({
            "operation": "edit",
            "path": relative_path,
            "before": before,
            "after": after
        })))
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let (relative_path, before, after) = edit_file_change(&arguments, &context.workspace)?;
        ensure_approved_snapshot(&arguments, &before)?;
        let path =
            resolve_workspace_path(&context.workspace, &relative_path.to_string_lossy(), true)?;
        fs::write(&path, after)
            .map_err(|error| format!("Could not write {}: {error}", relative_path.display()))?;
        Ok(json!({ "path": relative_path, "edited": true }))
    }
}

fn ensure_approved_snapshot(arguments: &Value, current_before: &str) -> Result<(), String> {
    let expected_before = arguments
        .get("_approved_before")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing approved file preview; request approval again".to_owned())?;
    if expected_before != current_before {
        return Err(
            "File changed while awaiting approval; review the updated change and retry".to_owned(),
        );
    }
    Ok(())
}

fn argument_string<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Tool argument '{key}' must be a string"))
}

fn argument_bool(arguments: &Value, key: &str, default: bool) -> Result<bool, String> {
    match arguments.get(key) {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("Tool argument '{key}' must be a boolean")),
    }
}

fn resolve_workspace_path(
    workspace: &Path,
    relative_path: &str,
    require_existing: bool,
) -> Result<PathBuf, String> {
    let relative_path = Path::new(relative_path);
    if relative_path.as_os_str().is_empty() || relative_path.is_absolute() {
        return Err("File path must be a non-empty path relative to the workspace".to_owned());
    }

    let root = fs::canonicalize(workspace)
        .map_err(|error| format!("Could not resolve workspace folder: {error}"))?;
    if !root.is_dir() {
        return Err("Workspace path is not a directory".to_owned());
    }
    let candidate = root.join(relative_path);
    if candidate.exists() {
        let resolved = fs::canonicalize(&candidate)
            .map_err(|error| format!("Could not resolve file path: {error}"))?;
        if !resolved.starts_with(&root) {
            return Err("File path escapes the workspace folder".to_owned());
        }
        if !resolved.is_file() {
            return Err("File path must refer to a file".to_owned());
        }
        return Ok(resolved);
    }
    if require_existing {
        return Err(format!("File does not exist: {}", relative_path.display()));
    }

    let file_name = candidate
        .file_name()
        .ok_or_else(|| "File path must include a file name".to_owned())?;
    let parent = candidate
        .parent()
        .ok_or_else(|| "File path must have a workspace folder".to_owned())?;
    let resolved_parent = fs::canonicalize(parent)
        .map_err(|error| format!("Could not resolve parent folder: {error}"))?;
    if !resolved_parent.starts_with(&root) {
        return Err("File path escapes the workspace folder".to_owned());
    }
    Ok(resolved_parent.join(file_name))
}

fn create_file_change(
    arguments: &Value,
    workspace: &Path,
) -> Result<(PathBuf, String, String), String> {
    let relative_path = PathBuf::from(argument_string(arguments, "path")?);
    let content = argument_string(arguments, "content")?.to_owned();
    let overwrite = argument_bool(arguments, "overwrite", false)?;
    let path = resolve_workspace_path(workspace, &relative_path.to_string_lossy(), false)?;
    let before = if path.exists() {
        if !overwrite {
            return Err(format!(
                "File already exists: {}. Set overwrite to true to replace it.",
                relative_path.display()
            ));
        }
        fs::read_to_string(&path)
            .map_err(|error| format!("Could not read {}: {error}", relative_path.display()))?
    } else {
        String::new()
    };
    Ok((relative_path, before, content))
}

fn edit_file_change(
    arguments: &Value,
    workspace: &Path,
) -> Result<(PathBuf, String, String), String> {
    let relative_path = PathBuf::from(argument_string(arguments, "path")?);
    let old_str = argument_string(arguments, "old_str")?;
    let new_str = argument_string(arguments, "new_str")?;
    if old_str.is_empty() {
        return Err("old_str cannot be empty; provide specific context to replace".to_owned());
    }
    let path = resolve_workspace_path(workspace, &relative_path.to_string_lossy(), true)?;
    let before = fs::read_to_string(&path)
        .map_err(|error| format!("Could not read {}: {error}", relative_path.display()))?;
    let start = before
        .find(old_str)
        .ok_or_else(|| "old_str was not found in the file".to_owned())?;
    let occurrence_count = before
        .as_bytes()
        .windows(old_str.len())
        .filter(|candidate| *candidate == old_str.as_bytes())
        .count();
    if occurrence_count > 1 {
        return Err(
            "old_str occurs multiple times; provide more context to identify one match".to_owned(),
        );
    }
    let end = start + old_str.len();
    let after = format!("{}{}{}", &before[..start], new_str, &before[end..]);
    Ok((relative_path, before, after))
}

struct RunCommand;

#[derive(Debug, PartialEq, Eq)]
struct CommandResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

#[async_trait]
impl Tool for RunCommand {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "run_command".to_owned(),
            description: "Run a shell command in the active workspace directory. Commands are not sandboxed; broader sandboxing is tracked separately in issue #12.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds (default 60000, maximum 600000)."
                    }
                },
                "required": ["command"]
            }),
            risk_level: RiskLevel::RequiresConfirmation,
        }
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let command = arguments
            .get("command")
            .and_then(Value::as_str)
            .filter(|command| !command.trim().is_empty())
            .ok_or_else(|| "A non-empty command is required".to_owned())?
            .to_owned();
        let timeout_ms = arguments
            .get("timeout_ms")
            .map(|value| {
                value
                    .as_u64()
                    .filter(|timeout| (1..=MAX_COMMAND_TIMEOUT_MS).contains(timeout))
                    .ok_or_else(|| {
                        format!("timeout_ms must be between 1 and {MAX_COMMAND_TIMEOUT_MS}")
                    })
            })
            .transpose()?
            .unwrap_or(DEFAULT_COMMAND_TIMEOUT_MS);

        let workspace = context
            .workspace
            .canonicalize()
            .map_err(|error| format!("Could not access workspace directory: {error}"))?;
        if !workspace.is_dir() {
            return Err("Workspace path is not a directory".to_owned());
        }

        emit_command_event(
            &context,
            CommandOutputEvent {
                request_id: context.request_id.clone(),
                command_id: context.command_id.clone(),
                command: command.clone(),
                phase: "started".to_owned(),
                stream: None,
                chunk: None,
                status: Some("running".to_owned()),
                exit_code: None,
                error: None,
            },
        )?;

        let output_context = context.clone();
        let output_command = command.clone();
        let result = run_command(
            &command,
            &workspace,
            Duration::from_millis(timeout_ms),
            Arc::new(move |stream, chunk| {
                emit_command_event(
                    &output_context,
                    CommandOutputEvent {
                        request_id: output_context.request_id.clone(),
                        command_id: output_context.command_id.clone(),
                        command: output_command.clone(),
                        phase: "output".to_owned(),
                        stream: Some(stream.to_owned()),
                        chunk: Some(chunk.to_owned()),
                        status: None,
                        exit_code: None,
                        error: None,
                    },
                )
            }),
        )
        .await;

        match result {
            Ok(result) => {
                let status = if result.exit_code == 0 {
                    "succeeded"
                } else {
                    "failed"
                };
                emit_command_event(
                    &context,
                    CommandOutputEvent {
                        request_id: context.request_id.clone(),
                        command_id: context.command_id.clone(),
                        command: command.clone(),
                        phase: "completed".to_owned(),
                        stream: None,
                        chunk: None,
                        status: Some(status.to_owned()),
                        exit_code: Some(result.exit_code),
                        error: None,
                    },
                )?;
                Ok(json!({
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code
                }))
            }
            Err(error) => {
                let timed_out = error.starts_with("Command timed out");
                let status = if timed_out { "timed_out" } else { "failed" };
                emit_command_event(
                    &context,
                    CommandOutputEvent {
                        request_id: context.request_id.clone(),
                        command_id: context.command_id.clone(),
                        command,
                        phase: "completed".to_owned(),
                        stream: None,
                        chunk: None,
                        status: Some(status.to_owned()),
                        exit_code: None,
                        error: Some(error.clone()),
                    },
                )?;
                Err(error)
            }
        }
    }
}

fn emit_command_event(context: &ToolContext, event: CommandOutputEvent) -> Result<(), String> {
    if let Some(callback) = &context.command_output {
        callback(event)?;
    }
    Ok(())
}

async fn run_command(
    command: &str,
    workspace: &Path,
    timeout_duration: Duration,
    on_output: Arc<ProcessOutputCallback>,
) -> Result<CommandResult, String> {
    let mut child = shell_command(command, workspace);
    child.kill_on_drop(true);
    let mut child = child
        .spawn()
        .map_err(|error| format!("Could not start command: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not capture command stdout".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Could not capture command stderr".to_owned())?;
    let stdout_task = tokio::spawn(read_output(stdout, "stdout", on_output.clone()));
    let stderr_task = tokio::spawn(read_output(stderr, "stderr", on_output));

    let wait_result = timeout(timeout_duration, child.wait()).await;
    let timed_out = wait_result.is_err();
    if timed_out {
        terminate_process_tree(&mut child).await.map_err(|error| {
            format!(
                "Command timed out after {} ms; {error}",
                timeout_duration.as_millis()
            )
        })?;
    }
    let status = match wait_result {
        Ok(result) => result.map_err(|error| format!("Could not wait for command: {error}"))?,
        Err(_) => child
            .wait()
            .await
            .map_err(|error| format!("Could not wait for timed-out command: {error}"))?,
    };

    let stdout = stdout_task
        .await
        .map_err(|error| format!("Could not collect command stdout: {error}"))??;
    let stderr = stderr_task
        .await
        .map_err(|error| format!("Could not collect command stderr: {error}"))??;
    if timed_out {
        return Err(format!(
            "Command timed out after {} ms and was terminated",
            timeout_duration.as_millis()
        ));
    }
    if !status.success() && status.code().is_none() {
        return Err("Command was terminated without an exit code".to_owned());
    }

    Ok(CommandResult {
        stdout: captured_output_text(stdout),
        stderr: captured_output_text(stderr),
        exit_code: status.code().unwrap_or(-1),
    })
}

#[cfg(windows)]
fn shell_command(command: &str, workspace: &Path) -> Command {
    let mut process = Command::new("cmd.exe");
    process.arg("/C").arg(command);
    process
        .current_dir(workspace)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    process
}

#[cfg(not(windows))]
fn shell_command(command: &str, workspace: &Path) -> Command {
    let mut process = Command::new("sh");
    process.arg("-c").arg(command);
    process.process_group(0);
    process
        .current_dir(workspace)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    process
}

#[cfg(windows)]
async fn terminate_process_tree(child: &mut tokio::process::Child) -> Result<(), String> {
    let process_id = child
        .id()
        .ok_or_else(|| "Timed-out command no longer has a process ID".to_owned())?;
    let status = Command::new("taskkill")
        .args(["/PID", &process_id.to_string(), "/T", "/F"])
        .status()
        .await
        .map_err(|error| format!("Could not terminate timed-out command tree: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Could not terminate timed-out command tree (taskkill exited with {status})"
        ))
    }
}

#[cfg(unix)]
async fn terminate_process_tree(child: &mut tokio::process::Child) -> Result<(), String> {
    let process_id = child
        .id()
        .ok_or_else(|| "Timed-out command no longer has a process ID".to_owned())?;
    let status = Command::new("kill")
        .args(["-KILL", "--", &format!("-{process_id}")])
        .status()
        .await
        .map_err(|error| format!("Could not terminate timed-out command tree: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Could not terminate timed-out command tree (kill exited with {status})"
        ))
    }
}

#[cfg(not(any(unix, windows)))]
async fn terminate_process_tree(child: &mut tokio::process::Child) -> Result<(), String> {
    child
        .kill()
        .await
        .map_err(|error| format!("Could not terminate timed-out command: {error}"))
}

async fn read_output<R>(
    mut reader: R,
    stream: &'static str,
    on_output: Arc<ProcessOutputCallback>,
) -> Result<Vec<u8>, String>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut captured = Vec::new();
    let mut truncated = false;
    let mut callback_error = None;
    let mut buffer = [0_u8; 4096];
    loop {
        let read = reader
            .read(&mut buffer)
            .await
            .map_err(|error| format!("Could not read command {stream}: {error}"))?;
        if read == 0 {
            break;
        }
        if callback_error.is_none() {
            if let Err(error) = on_output(stream, &String::from_utf8_lossy(&buffer[..read])) {
                callback_error = Some(error);
            }
        }
        append_limited(&mut captured, &mut truncated, &buffer[..read]);
    }
    if let Some(error) = callback_error {
        return Err(error);
    }
    if truncated {
        captured.extend_from_slice(b"\n...[output truncated]");
    }
    Ok(captured)
}

fn append_limited(output: &mut Vec<u8>, truncated: &mut bool, chunk: &[u8]) {
    let available = MAX_CAPTURED_OUTPUT_BYTES.saturating_sub(output.len());
    let captured = available.min(chunk.len());
    output.extend_from_slice(&chunk[..captured]);
    *truncated |= captured < chunk.len();
}

fn captured_output_text(output: Vec<u8>) -> String {
    String::from_utf8_lossy(&output).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{
        append_limited, edit_file_change, read_output, resolve_workspace_path, run_command,
        RiskLevel, Tool, ToolContext, ToolRegistry, ToolSchema, MAX_CAPTURED_OUTPUT_BYTES,
    };
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::Arc;
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{Duration, SystemTime, UNIX_EPOCH},
    };
    use tokio::{io::AsyncWriteExt, time::timeout};

    struct TestTool(&'static str);

    #[async_trait]
    impl Tool for TestTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.0.to_owned(),
                description: "test".to_owned(),
                parameters: json!({ "type": "object" }),
                risk_level: RiskLevel::AutoApprove,
            }
        }

        async fn call(&self, _arguments: Value, _context: ToolContext) -> Result<Value, String> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn registers_tools_and_rejects_duplicate_names() {
        let mut registry = ToolRegistry::with_examples();
        assert!(registry.register(Arc::new(TestTool("custom"))).is_ok());
        assert!(registry.register(Arc::new(TestTool("custom"))).is_err());
        assert_eq!(registry.schemas().len(), 7);
        let command = registry
            .get("run_command")
            .expect("command tool registered");
        assert_eq!(command.schema().risk_level, RiskLevel::RequiresConfirmation);
    }

    fn temporary_workspace() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("legion-tools-{id}"));
        fs::create_dir_all(&path).expect("temporary workspace is created");
        path
    }

    #[test]
    fn rejects_paths_that_escape_workspace() {
        let root = temporary_workspace();
        let workspace = root.join("workspace");
        fs::create_dir(&workspace).expect("workspace is created");
        fs::write(root.join("outside.txt"), "outside").expect("outside file is written");
        let outside_file = PathBuf::from("..").join("outside.txt");
        let outside_new_file = PathBuf::from("..").join("new.txt");
        assert!(
            resolve_workspace_path(&workspace, &outside_file.to_string_lossy(), true)
                .unwrap_err()
                .contains("escapes")
        );
        assert!(
            resolve_workspace_path(&workspace, &outside_new_file.to_string_lossy(), false)
                .unwrap_err()
                .contains("escapes")
        );
        fs::remove_dir_all(root).expect("temporary workspace is removed");
    }

    #[test]
    fn edit_requires_one_non_empty_exact_match() {
        let workspace = temporary_workspace();
        fs::write(workspace.join("example.txt"), "one target, then target").expect("file written");

        let missing = edit_file_change(
            &json!({ "path": "example.txt", "old_str": "missing", "new_str": "new" }),
            &workspace,
        );
        assert!(missing.unwrap_err().contains("not found"));

        let ambiguous = edit_file_change(
            &json!({ "path": "example.txt", "old_str": "target", "new_str": "new" }),
            &workspace,
        );
        assert!(ambiguous.unwrap_err().contains("multiple times"));

        let empty = edit_file_change(
            &json!({ "path": "example.txt", "old_str": "", "new_str": "new" }),
            &workspace,
        );
        assert!(empty.unwrap_err().contains("cannot be empty"));

        let unique = edit_file_change(
            &json!({ "path": "example.txt", "old_str": "one target", "new_str": "a target" }),
            &workspace,
        )
        .expect("unique replacement succeeds");
        assert_eq!(unique.2, "a target, then target");

        fs::write(workspace.join("example.txt"), "aaa").expect("overlap case is written");
        let overlapping = edit_file_change(
            &json!({ "path": "example.txt", "old_str": "aa", "new_str": "b" }),
            &workspace,
        );
        assert!(overlapping.unwrap_err().contains("multiple times"));
        fs::remove_dir_all(workspace).expect("temporary workspace is removed");
    }

    #[test]
    fn truncates_captured_output_at_the_limit() {
        let mut output = Vec::new();
        let mut truncated = false;
        append_limited(
            &mut output,
            &mut truncated,
            &vec![b'x'; MAX_CAPTURED_OUTPUT_BYTES + 10],
        );
        assert_eq!(output.len(), MAX_CAPTURED_OUTPUT_BYTES);
        assert!(truncated);
    }

    #[tokio::test]
    async fn captures_and_truncates_streamed_output() {
        let (mut writer, reader) = tokio::io::duplex(1024);
        let reader_task = tokio::spawn(read_output(
            reader,
            "stdout",
            Arc::new(|_: &str, _: &str| Ok(())),
        ));
        let input = vec![b'x'; MAX_CAPTURED_OUTPUT_BYTES + 10];
        writer.write_all(&input).await.expect("write test output");
        drop(writer);

        let output = reader_task
            .await
            .expect("reader task completes")
            .expect("output is captured");
        assert!(output.starts_with(&vec![b'x'; MAX_CAPTURED_OUTPUT_BYTES]));
        assert!(output.ends_with(b"\n...[output truncated]"));
    }

    #[tokio::test]
    async fn captures_stdout_stderr_and_exit_code() {
        let workspace = std::env::current_dir().expect("workspace exists");
        let command = if cfg!(windows) {
            "echo stdout & echo stderr 1>&2 & exit /b 7"
        } else {
            "printf stdout; printf stderr >&2; exit 7"
        };
        let result = run_command(
            command,
            &workspace,
            Duration::from_secs(5),
            Arc::new(|_: &str, _: &str| Ok(())),
        )
        .await
        .expect("command completes");
        assert!(result.stdout.contains("stdout"));
        assert!(result.stderr.contains("stderr"));
        assert_eq!(result.exit_code, 7);
    }

    #[tokio::test]
    async fn terminates_a_command_when_its_timeout_expires() {
        let workspace = std::env::current_dir().expect("workspace exists");
        let command = if cfg!(windows) {
            "ping -n 5 127.0.0.1 > nul"
        } else {
            "sleep 5"
        };
        let result = timeout(
            Duration::from_secs(3),
            run_command(
                command,
                Path::new(&workspace),
                Duration::from_millis(100),
                Arc::new(|_: &str, _: &str| Ok(())),
            ),
        )
        .await
        .expect("command runner enforces timeout");
        assert!(result
            .expect_err("command times out")
            .contains("timed out after 100 ms"));
    }
}
