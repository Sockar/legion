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

#[derive(Clone, Debug)]
pub struct ToolContext {
    pub workspace: PathBuf,
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

#[cfg(test)]
mod tests {
    use super::{
        edit_file_change, resolve_workspace_path, RiskLevel, Tool, ToolContext, ToolRegistry,
        ToolSchema,
    };
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::Arc;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

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
        assert_eq!(registry.schemas().len(), 6);
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
        assert!(resolve_workspace_path(&workspace, "..\\outside.txt", true)
            .unwrap_err()
            .contains("escapes"));
        assert!(resolve_workspace_path(&workspace, "../new.txt", false)
            .unwrap_err()
            .contains("escapes"));
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
}
