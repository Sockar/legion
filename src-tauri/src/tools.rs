use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
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

#[cfg(test)]
mod tests {
    use super::{RiskLevel, Tool, ToolContext, ToolRegistry, ToolSchema};
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::Arc;

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
        assert_eq!(registry.schemas().len(), 3);
    }
}
