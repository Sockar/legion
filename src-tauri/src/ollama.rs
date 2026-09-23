use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::tools::{
    CommandOutputCallback, RiskLevel, Tool, ToolCall, ToolContext, ToolDefinition, ToolRegistry,
    ToolSchema, COMMAND_OUTPUT_EVENT,
};

const DEFAULT_OLLAMA_ENDPOINT: &str = "http://localhost:11434";
const PULL_PROGRESS_EVENT: &str = "ollama://pull-progress";
const CHAT_CHUNK_EVENT: &str = "ollama://chat-chunk";
const TOOL_APPROVAL_EVENT: &str = "tools://approval-request";
const MAX_TOOL_ITERATIONS: usize = 8;

pub type ProgressCallback = Arc<dyn Fn(PullProgress) -> Result<(), String> + Send + Sync>;
pub type ChatCallback = Arc<dyn Fn(ChatChunk) -> Result<(), String> + Send + Sync>;

#[derive(Clone)]
pub struct BackendState {
    backend: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    pending_approvals: PendingApprovals,
    chat_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl Default for BackendState {
    fn default() -> Self {
        Self {
            backend: Arc::new(OllamaBackend::new(DEFAULT_OLLAMA_ENDPOINT)),
            tools: Arc::new(ToolRegistry::with_examples()),
            pending_approvals: PendingApprovals::default(),
            chat_cancellations: Arc::default(),
        }
    }
}

impl BackendState {
    pub fn with_backend(backend: Arc<dyn LlmBackend>) -> Self {
        Self::with_backend_and_tools(backend, ToolRegistry::with_examples())
    }

    pub fn with_backend_and_tools(backend: Arc<dyn LlmBackend>, tools: ToolRegistry) -> Self {
        Self {
            backend,
            tools: Arc::new(tools),
            pending_approvals: PendingApprovals::default(),
            chat_cancellations: Arc::default(),
        }
    }

    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) -> Result<(), String> {
        Arc::get_mut(&mut self.tools)
            .ok_or_else(|| "Tools cannot be registered after backend state is shared".to_owned())?
            .register(tool)
    }
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn status(&self) -> ServerStatus;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, String>;
    async fn pull_model(&self, model: String, on_progress: ProgressCallback) -> Result<(), String>;
    async fn chat(
        &self,
        request: ChatRequest,
        tools: Vec<ToolDefinition>,
        cancellation: CancellationToken,
        request_id: String,
        on_chunk: ChatCallback,
    ) -> Result<ChatMessage, String>;
}

struct OllamaBackend {
    endpoint: String,
    client: Client,
}

impl OllamaBackend {
    fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            client: Client::new(),
        }
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}{path}", self.endpoint)
    }
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn status(&self) -> ServerStatus {
        let endpoint = self.endpoint.clone();
        match self.client.get(self.api_url("/api/tags")).send().await {
            Ok(response) if response.status().is_success() => ServerStatus {
                connected: true,
                endpoint,
                message: "Connected to Ollama".to_owned(),
            },
            Ok(response) => ServerStatus {
                connected: false,
                endpoint,
                message: format!("Ollama returned HTTP {}", response.status()),
            },
            Err(error) => ServerStatus {
                connected: false,
                endpoint,
                message: format!("Ollama is not running or unreachable: {error}"),
            },
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
        let response = self
            .client
            .get(self.api_url("/api/tags"))
            .send()
            .await
            .map_err(|error| format!("Could not reach Ollama: {error}"))?;
        ensure_success(response)
            .await?
            .json::<TagsResponse>()
            .await
            .map(|response| response.models)
            .map_err(|error| format!("Could not parse Ollama model list: {error}"))
    }

    async fn pull_model(&self, model: String, on_progress: ProgressCallback) -> Result<(), String> {
        let response = self
            .client
            .post(self.api_url("/api/pull"))
            .json(&PullRequest { name: model })
            .send()
            .await
            .map_err(|error| format!("Could not reach Ollama: {error}"))?;
        let response = ensure_success(response).await?;

        read_ndjson(response, |line| {
            let item: OllamaPullProgress = parse_ndjson_line(line)?;
            on_progress(PullProgress {
                request_id: String::new(),
                status: item.status,
                digest: item.digest,
                total: item.total,
                completed: item.completed,
            })?;
            Ok(())
        })
        .await
    }

    async fn chat(
        &self,
        request: ChatRequest,
        tools: Vec<ToolDefinition>,
        cancellation: CancellationToken,
        request_id: String,
        on_chunk: ChatCallback,
    ) -> Result<ChatMessage, String> {
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err("Chat cancelled".to_owned()),
            response = self
                .client
                .post(self.api_url("/api/chat"))
                .json(&ChatApiRequest {
                    model: request.model,
                    messages: request.messages,
                    tools,
                    stream: true,
                })
                .send() => response.map_err(|error| format!("Could not reach Ollama: {error}"))?,
        };
        let response = ensure_success(response).await?;

        let mut message = ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            tool_calls: None,
            tool_name: None,
        };
        let mut received_message = false;
        read_ndjson_cancellable(response, cancellation.clone(), |line| {
            let item: ChatApiResponse = parse_ndjson_line(line)?;
            if let Some(part) = item.message {
                received_message = true;
                if !part.content.is_empty() {
                    message.content.push_str(&part.content);
                    on_chunk(ChatChunk {
                        request_id: request_id.clone(),
                        content: part.content,
                        done: false,
                    })?;
                }
                message.role = part.role;
                if part.tool_calls.is_some() {
                    message.tool_calls = part.tool_calls;
                }
                if part.tool_name.is_some() {
                    message.tool_name = part.tool_name;
                }
            }
            Ok(())
        })
        .await?;
        if !received_message && !cancellation.is_cancelled() {
            return Err("Ollama returned a chat response without a message".to_owned());
        }
        Ok(message)
    }
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub workspace_path: Option<PathBuf>,
}

#[derive(Serialize)]
struct PullRequest {
    name: String,
}

#[derive(Serialize)]
struct ChatApiRequest {
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaPullProgress {
    status: String,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChatApiResponse {
    #[serde(default)]
    message: Option<ChatMessage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerStatus {
    pub connected: bool,
    pub endpoint: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PullProgress {
    pub request_id: String,
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatChunk {
    pub request_id: String,
    pub content: String,
    pub done: bool,
}

#[derive(Clone, Default)]
struct PendingApprovals(Arc<Mutex<HashMap<String, oneshot::Sender<bool>>>>);

#[derive(Clone, Debug, Serialize)]
pub struct ToolApprovalRequest {
    pub request_id: String,
    pub approval_id: String,
    pub tool: ToolSchema,
    pub arguments: Value,
    pub preview: Option<Value>,
}

#[async_trait]
trait ApprovalHandler: Send + Sync {
    async fn approve(
        &self,
        approval: ToolApprovalRequest,
        cancellation: CancellationToken,
    ) -> Result<bool, String>;
}

struct TauriApprovalHandler {
    app: AppHandle,
    pending: PendingApprovals,
}

#[async_trait]
impl ApprovalHandler for TauriApprovalHandler {
    async fn approve(
        &self,
        approval: ToolApprovalRequest,
        cancellation: CancellationToken,
    ) -> Result<bool, String> {
        let (sender, receiver) = oneshot::channel();
        self.pending
            .0
            .lock()
            .map_err(|error| format!("Could not manage pending tool approvals: {error}"))?
            .insert(approval.approval_id.clone(), sender);
        if let Err(error) = self.app.emit(TOOL_APPROVAL_EVENT, &approval) {
            self.pending
                .0
                .lock()
                .map_err(|lock_error| format!("Could not clean up tool approval: {lock_error}"))?
                .remove(&approval.approval_id);
            return Err(format!("Could not request tool approval: {error}"));
        }

        let decision = tokio::select! {
            _ = cancellation.cancelled() => None,
            decision = receiver => Some(
                decision.map_err(|_| "Tool approval response was dropped".to_owned())?
            ),
        };
        self.pending
            .0
            .lock()
            .map_err(|error| format!("Could not clean up tool approval: {error}"))?
            .remove(&approval.approval_id);
        Ok(decision.unwrap_or(false))
    }
}

async fn run_tool_loop(
    backend: &dyn LlmBackend,
    tools: &ToolRegistry,
    mut request: ChatRequest,
    request_id: &str,
    cancellation: CancellationToken,
    callbacks: ToolLoopCallbacks<'_>,
) -> Result<(), String> {
    let workspace = match &request.workspace_path {
        Some(workspace) => workspace.clone(),
        None => std::env::current_dir()
            .map_err(|error| format!("Could not determine workspace directory: {error}"))?,
    };
    for iteration in 0..=MAX_TOOL_ITERATIONS {
        if cancellation.is_cancelled() {
            return Ok(());
        }
        let message = backend
            .chat(
                request.clone(),
                tools.definitions(),
                cancellation.clone(),
                request_id.to_owned(),
                callbacks.on_chunk.clone(),
            )
            .await?;
        if cancellation.is_cancelled() {
            return Ok(());
        }

        let tool_calls = message.tool_calls.clone().unwrap_or_default();
        request.messages.push(message.clone());
        if tool_calls.is_empty() {
            (callbacks.on_chunk)(ChatChunk {
                request_id: request_id.to_owned(),
                content: String::new(),
                done: true,
            })?;
            return Ok(());
        }

        if iteration == MAX_TOOL_ITERATIONS {
            (callbacks.on_chunk)(ChatChunk {
                request_id: request_id.to_owned(),
                content: format!("Stopped at the tool iteration limit of {MAX_TOOL_ITERATIONS}."),
                done: true,
            })?;
            return Ok(());
        }

        for (call_index, call) in tool_calls.into_iter().enumerate() {
            if cancellation.is_cancelled() {
                return Ok(());
            }
            let result = if let Some(tool) = tools.get(&call.function.name) {
                let schema = tool.schema();
                let context = ToolContext {
                    workspace: workspace.clone(),
                    request_id: request_id.to_owned(),
                    command_id: format!("{request_id}-{iteration}-{call_index}"),
                    command_output: Some(callbacks.command_output.clone()),
                };
                let preview = if schema.risk_level == RiskLevel::RequiresConfirmation {
                    match tokio::select! {
                        _ = cancellation.cancelled() => return Ok(()),
                        result = tool.preview(call.function.arguments.clone(), context.clone()) => result,
                    } {
                        Ok(preview) => preview,
                        Err(error) => {
                            let content = serde_json::to_string(&json!({ "error": error }))
                                .map_err(|encode_error| {
                                    format!("Could not encode tool result: {encode_error}")
                                })?;
                            request.messages.push(ChatMessage {
                                role: "tool".to_owned(),
                                content,
                                tool_calls: None,
                                tool_name: Some(call.function.name),
                            });
                            continue;
                        }
                    }
                } else {
                    None
                };
                let approved = if schema.risk_level == RiskLevel::RequiresConfirmation {
                    callbacks
                        .approval_handler
                        .approve(
                            ToolApprovalRequest {
                                request_id: request_id.to_owned(),
                                approval_id: format!("{request_id}-{iteration}-{call_index}"),
                                tool: schema.clone(),
                                arguments: call.function.arguments.clone(),
                                preview: preview.clone(),
                            },
                            cancellation.clone(),
                        )
                        .await?
                } else {
                    true
                };
                if !approved {
                    json!({ "error": "User denied permission to run this tool." })
                } else {
                    let mut approved_arguments = call.function.arguments;
                    if let Some(before) = preview.as_ref().and_then(|preview| preview.get("before"))
                    {
                        if let Some(arguments) = approved_arguments.as_object_mut() {
                            arguments.insert("_approved_before".to_owned(), before.clone());
                        }
                    }
                    tokio::select! {
                        _ = cancellation.cancelled() => return Ok(()),
                        result = tool.call(
                            approved_arguments,
                            context.clone(),
                        ) => {
                            result.unwrap_or_else(|error| json!({ "error": error }))
                        }
                    }
                }
            } else {
                json!({ "error": format!("Tool '{}' is not registered.", call.function.name) })
            };

            let content = serde_json::to_string(&result)
                .map_err(|error| format!("Could not encode tool result: {error}"))?;
            request.messages.push(ChatMessage {
                role: "tool".to_owned(),
                content,
                tool_calls: None,
                tool_name: Some(call.function.name),
            });
        }
    }

    unreachable!("tool loop returns on the final allowed iteration")
}

struct ToolLoopCallbacks<'a> {
    approval_handler: &'a dyn ApprovalHandler,
    command_output: CommandOutputCallback,
    on_chunk: &'a ChatCallback,
}

#[tauri::command]
pub async fn ollama_status(backend: State<'_, BackendState>) -> Result<ServerStatus, String> {
    Ok(backend.backend.status().await)
}

#[tauri::command]
pub async fn ollama_list_models(
    backend: State<'_, BackendState>,
) -> Result<Vec<ModelInfo>, String> {
    backend.backend.list_models().await
}

#[tauri::command]
pub async fn ollama_pull_model(
    app: AppHandle,
    backend: State<'_, BackendState>,
    model: String,
    request_id: String,
) -> Result<(), String> {
    let progress_app = app.clone();
    let progress_request_id = request_id.clone();
    let on_progress = Arc::new(move |mut progress: PullProgress| {
        progress.request_id.clone_from(&progress_request_id);
        progress_app
            .emit(PULL_PROGRESS_EVENT, progress)
            .map_err(|error| format!("Could not emit model pull progress: {error}"))
    });
    backend.backend.pull_model(model, on_progress).await
}

#[tauri::command]
pub async fn ollama_chat(
    app: AppHandle,
    backend: State<'_, BackendState>,
    request: ChatRequest,
    request_id: String,
) -> Result<(), String> {
    let cancellation = CancellationToken::new();
    {
        let mut active_chats = backend
            .chat_cancellations
            .lock()
            .map_err(|error| format!("Could not manage active chats: {error}"))?;
        if active_chats.contains_key(&request_id) {
            return Err(format!("Chat request {request_id} is already active"));
        }
        active_chats.insert(request_id.clone(), cancellation.clone());
    }
    let event_request_id = request_id.clone();
    let event_app = app.clone();
    let on_chunk: ChatCallback = Arc::new(move |mut chunk: ChatChunk| {
        chunk.request_id.clone_from(&event_request_id);
        event_app
            .emit(CHAT_CHUNK_EVENT, chunk)
            .map_err(|error| format!("Could not emit chat response: {error}"))
    });
    let approval_handler = TauriApprovalHandler {
        app: app.clone(),
        pending: backend.pending_approvals.clone(),
    };
    let output_app = app.clone();
    let command_output: CommandOutputCallback = Arc::new(move |event| {
        output_app
            .emit(COMMAND_OUTPUT_EVENT, event)
            .map_err(|error| format!("Could not emit command output: {error}"))
    });
    let result = run_tool_loop(
        backend.backend.as_ref(),
        &backend.tools,
        request,
        &request_id,
        cancellation,
        ToolLoopCallbacks {
            approval_handler: &approval_handler,
            command_output,
            on_chunk: &on_chunk,
        },
    )
    .await;
    backend
        .chat_cancellations
        .lock()
        .map_err(|error| format!("Could not clean up active chats: {error}"))?
        .remove(&request_id);
    result
}

#[tauri::command]
pub async fn respond_tool_approval(
    backend: State<'_, BackendState>,
    approval_id: String,
    approved: bool,
) -> Result<(), String> {
    let sender = backend
        .pending_approvals
        .0
        .lock()
        .map_err(|error| format!("Could not access pending tool approvals: {error}"))?
        .remove(&approval_id)
        .ok_or_else(|| format!("No pending approval found for {approval_id}"))?;
    sender
        .send(approved)
        .map_err(|_| format!("Tool approval request {approval_id} is no longer active"))
}

#[tauri::command]
pub async fn ollama_cancel_chat(
    backend: State<'_, BackendState>,
    request_id: String,
) -> Result<(), String> {
    if let Some(cancellation) = backend
        .chat_cancellations
        .lock()
        .map_err(|error| format!("Could not access active chats: {error}"))?
        .get(&request_id)
    {
        cancellation.cancel();
    }
    Ok(())
}

async fn ensure_success(response: Response) -> Result<Response, String> {
    let status = response.status();
    if status.is_success() {
        Ok(response)
    } else {
        let body = response.text().await.unwrap_or_default();
        Err(format!("Ollama returned HTTP {status}: {body}"))
    }
}

async fn read_ndjson(
    response: Response,
    mut on_line: impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut stream = response.bytes_stream();
    let mut buffer = NdjsonBuffer::default();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("Ollama stream failed: {error}"))?;
        for line in buffer.push(&chunk) {
            on_line(&line)?;
        }
    }
    if let Some(line) = buffer.finish() {
        on_line(&line)?;
    }
    Ok(())
}

async fn read_ndjson_cancellable(
    response: Response,
    cancellation: CancellationToken,
    mut on_line: impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut stream = response.bytes_stream();
    let mut buffer = NdjsonBuffer::default();
    loop {
        let chunk = tokio::select! {
            _ = cancellation.cancelled() => return Ok(()),
            chunk = stream.next() => chunk,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|error| format!("Ollama stream failed: {error}"))?;
        for line in buffer.push(&chunk) {
            on_line(&line)?;
        }
    }
    if let Some(line) = buffer.finish() {
        on_line(&line)?;
    }
    Ok(())
}

fn parse_ndjson_line<T: for<'de> Deserialize<'de>>(line: &str) -> Result<T, String> {
    serde_json::from_str(line).map_err(|error| format!("Invalid Ollama stream response: {error}"))
}

#[derive(Default)]
struct NdjsonBuffer {
    bytes: Vec<u8>,
}

impl NdjsonBuffer {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.bytes.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(newline) = self.bytes.iter().position(|byte| *byte == b'\n') {
            let line = self.bytes.drain(..=newline).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            if !line.is_empty() {
                lines.push(line.to_owned());
            }
        }
        lines
    }

    fn finish(self) -> Option<String> {
        let line = String::from_utf8_lossy(&self.bytes);
        let line = line.trim();
        (!line.is_empty()).then(|| line.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_ndjson_line, run_tool_loop, ApprovalHandler, ChatApiResponse, ChatCallback,
        ChatChunk, ChatMessage, ChatRequest, LlmBackend, ModelInfo, NdjsonBuffer, ServerStatus,
        ToolApprovalRequest, ToolLoopCallbacks, MAX_TOOL_ITERATIONS,
    };
    use crate::tools::{
        CommandOutputCallback, RiskLevel, Tool, ToolCall, ToolContext, ToolDefinition,
        ToolFunctionCall, ToolRegistry, ToolSchema,
    };
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::{
        collections::VecDeque,
        path::PathBuf,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };
    use tokio_util::sync::CancellationToken;

    struct MockBackend {
        responses: Mutex<VecDeque<ChatMessage>>,
        requests: Mutex<Vec<ChatRequest>>,
    }

    impl MockBackend {
        fn new(responses: Vec<ChatMessage>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LlmBackend for MockBackend {
        async fn status(&self) -> ServerStatus {
            ServerStatus {
                connected: true,
                endpoint: String::new(),
                message: String::new(),
            }
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
            Ok(Vec::new())
        }

        async fn pull_model(
            &self,
            _model: String,
            _on_progress: super::ProgressCallback,
        ) -> Result<(), String> {
            Ok(())
        }

        async fn chat(
            &self,
            request: ChatRequest,
            _tools: Vec<ToolDefinition>,
            _cancellation: CancellationToken,
            request_id: String,
            on_chunk: ChatCallback,
        ) -> Result<ChatMessage, String> {
            self.requests.lock().unwrap().push(request);
            let message = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| "No mock response left".to_owned())?;
            if !message.content.is_empty() {
                on_chunk(ChatChunk {
                    request_id,
                    content: message.content.clone(),
                    done: false,
                })?;
            }
            Ok(message)
        }
    }

    struct FixedApproval(bool, AtomicUsize);

    #[async_trait]
    impl ApprovalHandler for FixedApproval {
        async fn approve(
            &self,
            _approval: ToolApprovalRequest,
            _cancellation: CancellationToken,
        ) -> Result<bool, String> {
            self.1.fetch_add(1, Ordering::SeqCst);
            Ok(self.0)
        }
    }

    struct TestTool {
        risk_level: RiskLevel,
        calls: AtomicUsize,
        fails: bool,
        workspaces: Mutex<Vec<PathBuf>>,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "test_tool".to_owned(),
                description: "A test tool.".to_owned(),
                parameters: json!({ "type": "object" }),
                risk_level: self.risk_level.clone(),
            }
        }

        async fn call(&self, _arguments: Value, context: ToolContext) -> Result<Value, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.workspaces.lock().unwrap().push(context.workspace);
            if self.fails {
                Err("expected test failure".to_owned())
            } else {
                Ok(json!({ "ok": true }))
            }
        }
    }

    fn tool_call_response() -> ChatMessage {
        ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                function: ToolFunctionCall {
                    name: "test_tool".to_owned(),
                    arguments: json!({}),
                },
            }]),
            tool_name: None,
        }
    }

    fn final_response() -> ChatMessage {
        ChatMessage {
            role: "assistant".to_owned(),
            content: "done".to_owned(),
            tool_calls: None,
            tool_name: None,
        }
    }

    fn chat_request() -> ChatRequest {
        ChatRequest {
            model: "mock".to_owned(),
            workspace_path: Some(std::path::PathBuf::from("test-workspace")),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "run it".to_owned(),
                tool_calls: None,
                tool_name: None,
            }],
        }
    }

    fn callback() -> (ChatCallback, Arc<Mutex<Vec<ChatChunk>>>) {
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let captured = chunks.clone();
        (
            Arc::new(move |chunk| {
                captured.lock().unwrap().push(chunk);
                Ok(())
            }),
            chunks,
        )
    }

    fn command_output_callback() -> CommandOutputCallback {
        Arc::new(|_| Ok(()))
    }

    #[tokio::test]
    async fn reports_tool_failures_to_model_and_reaches_final_answer() {
        let backend = MockBackend::new(vec![tool_call_response(), final_response()]);
        let tool = Arc::new(TestTool {
            risk_level: RiskLevel::AutoApprove,
            calls: AtomicUsize::new(0),
            fails: true,
            workspaces: Mutex::new(Vec::new()),
        });
        let mut tools = ToolRegistry::default();
        tools.register(tool.clone()).expect("tool registers");
        let approval = FixedApproval(false, AtomicUsize::new(0));
        let (on_chunk, chunks) = callback();

        run_tool_loop(
            &backend,
            &tools,
            chat_request(),
            "request",
            CancellationToken::new(),
            ToolLoopCallbacks {
                approval_handler: &approval,
                command_output: command_output_callback(),
                on_chunk: &on_chunk,
            },
        )
        .await
        .expect("tool error is handled");

        assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            tool.workspaces.lock().unwrap().as_slice(),
            &[std::path::PathBuf::from("test-workspace")]
        );
        assert_eq!(approval.1.load(Ordering::SeqCst), 0);
        let requests = backend.requests.lock().unwrap();
        let tool_result = &requests[1]
            .messages
            .last()
            .expect("tool result exists")
            .content;
        assert_eq!(
            serde_json::from_str::<Value>(tool_result).expect("tool result is JSON")["error"],
            "expected test failure"
        );
        assert_eq!(
            chunks
                .lock()
                .unwrap()
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<String>(),
            "done"
        );
    }

    #[tokio::test]
    async fn waits_for_confirmation_and_skips_a_denied_tool() {
        let backend = MockBackend::new(vec![tool_call_response(), final_response()]);
        let tool = Arc::new(TestTool {
            risk_level: RiskLevel::RequiresConfirmation,
            calls: AtomicUsize::new(0),
            fails: false,
            workspaces: Mutex::new(Vec::new()),
        });
        let mut tools = ToolRegistry::default();
        tools.register(tool.clone()).expect("tool registers");
        let approval = FixedApproval(false, AtomicUsize::new(0));
        let (on_chunk, _) = callback();

        run_tool_loop(
            &backend,
            &tools,
            chat_request(),
            "request",
            CancellationToken::new(),
            ToolLoopCallbacks {
                approval_handler: &approval,
                command_output: command_output_callback(),
                on_chunk: &on_chunk,
            },
        )
        .await
        .expect("denial is reported to model");

        assert_eq!(approval.1.load(Ordering::SeqCst), 1);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
        let requests = backend.requests.lock().unwrap();
        let tool_result = &requests[1]
            .messages
            .last()
            .expect("denial result exists")
            .content;
        assert_eq!(
            serde_json::from_str::<Value>(tool_result).expect("tool result is JSON")["error"],
            "User denied permission to run this tool."
        );
    }

    #[tokio::test]
    async fn stops_tool_execution_at_the_iteration_limit() {
        let backend = MockBackend::new(vec![tool_call_response(); MAX_TOOL_ITERATIONS + 1]);
        let tool = Arc::new(TestTool {
            risk_level: RiskLevel::AutoApprove,
            calls: AtomicUsize::new(0),
            fails: false,
            workspaces: Mutex::new(Vec::new()),
        });
        let mut tools = ToolRegistry::default();
        tools.register(tool.clone()).expect("tool registers");
        let approval = FixedApproval(false, AtomicUsize::new(0));
        let (on_chunk, chunks) = callback();

        run_tool_loop(
            &backend,
            &tools,
            chat_request(),
            "request",
            CancellationToken::new(),
            ToolLoopCallbacks {
                approval_handler: &approval,
                command_output: command_output_callback(),
                on_chunk: &on_chunk,
            },
        )
        .await
        .expect("iteration limit is a user-facing stop");

        assert_eq!(tool.calls.load(Ordering::SeqCst), MAX_TOOL_ITERATIONS);
        assert_eq!(
            backend.requests.lock().unwrap().len(),
            MAX_TOOL_ITERATIONS + 1
        );
        assert!(chunks
            .lock()
            .unwrap()
            .last()
            .expect("limit message is emitted")
            .content
            .contains("iteration limit"));
    }

    #[test]
    fn splits_ndjson_across_arbitrary_chunks() {
        let mut buffer = NdjsonBuffer::default();
        assert!(buffer.push(br#"{"name":"llama"#).is_empty());
        assert_eq!(
            buffer.push(b".3\"}\n{\"name\":\"qwen\"}\n"),
            vec![r#"{"name":"llama.3"}"#, r#"{"name":"qwen"}"#]
        );
    }

    #[test]
    fn parses_model_tags_with_optional_details() {
        let model: ModelInfo =
            serde_json::from_str(r#"{"name":"llama3.2:latest","size":123,"digest":"abc"}"#)
                .expect("model parses");
        assert_eq!(model.name, "llama3.2:latest");
        assert_eq!(model.size, 123);
        assert!(model.details.is_none());
    }

    #[test]
    fn parses_chat_stream_message_with_tool_calls() {
        let chunk: ChatApiResponse = parse_ndjson_line(
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_current_time","arguments":{}}}]}}"#,
        )
        .expect("chat chunk parses");
        let message = chunk.message.expect("message exists");
        assert_eq!(
            message.tool_calls.expect("tool call exists")[0]
                .function
                .name,
            "get_current_time"
        );
    }
}
