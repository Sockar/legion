use std::{
    collections::HashMap,
    future::Future,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::persistence::{PersistenceState, ToolAuditRecord};
use crate::tools::{
    ApprovalDecision, CommandOutputCallback, OutOfWorkspaceApprover, RiskLevel, Tool, ToolCall,
    ToolContext, ToolDefinition, ToolRegistry, ToolSchema, COMMAND_OUTPUT_EVENT,
};

const DEFAULT_OLLAMA_ENDPOINT: &str = "http://localhost:11434";
const PULL_PROGRESS_EVENT: &str = "ollama://pull-progress";
const CHAT_CHUNK_EVENT: &str = "ollama://chat-chunk";
const CHAT_THINKING_EVENT: &str = "ollama://chat-thinking";
const TOOL_CALL_EVENT: &str = "tools://tool-call";
const TOOL_APPROVAL_EVENT: &str = "tools://approval-request";
const MAX_TOOL_ITERATIONS: usize = 8;

pub type ProgressCallback = Arc<dyn Fn(PullProgress) -> Result<(), String> + Send + Sync>;
pub type ChatCallback = Arc<dyn Fn(ChatChunk) -> Result<(), String> + Send + Sync>;
pub type ThinkingCallback = Arc<dyn Fn(ChatThinkingChunk) -> Result<(), String> + Send + Sync>;
pub type ToolCallCallback = Arc<dyn Fn(ToolCallEvent) -> Result<(), String> + Send + Sync>;

fn emit_pull_progress(
    mut progress: PullProgress,
    request_id: &str,
    emit: impl FnOnce(&'static str, PullProgress) -> Result<(), String>,
) -> Result<(), String> {
    progress.request_id = request_id.to_owned();
    emit(PULL_PROGRESS_EVENT, progress)
        .map_err(|error| format!("Could not emit model pull progress: {error}"))
}

fn emit_chat_thinking(
    mut chunk: ChatThinkingChunk,
    request_id: &str,
    emit: impl FnOnce(&'static str, ChatThinkingChunk) -> Result<(), String>,
) -> Result<(), String> {
    chunk.request_id = request_id.to_owned();
    emit(CHAT_THINKING_EVENT, chunk)
        .map_err(|error| format!("Could not emit chat reasoning: {error}"))
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolCallEvent {
    pub id: String,
    pub request_id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub arguments_summary: String,
    pub result: Value,
    pub status: String,
    pub approval_status: String,
    pub created_at: String,
}

#[derive(Clone)]
pub struct BackendState {
    backend: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    pending_approvals: PendingApprovals,
    chat_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    download_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl Default for BackendState {
    fn default() -> Self {
        Self {
            backend: Arc::new(OllamaBackend::new(DEFAULT_OLLAMA_ENDPOINT)),
            tools: Arc::new(ToolRegistry::with_examples()),
            pending_approvals: PendingApprovals::default(),
            chat_cancellations: Arc::default(),
            download_cancellations: Arc::default(),
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
            download_cancellations: Arc::default(),
        }
    }

    pub(crate) fn register_download_cancellation(
        &self,
        request_id: &str,
    ) -> Result<CancellationToken, String> {
        let mut downloads = self
            .download_cancellations
            .lock()
            .map_err(|error| format!("Could not manage active downloads: {error}"))?;
        if downloads.contains_key(request_id) {
            return Err(format!("Download request {request_id} is already active"));
        }
        let cancellation = CancellationToken::new();
        downloads.insert(request_id.to_owned(), cancellation.clone());
        Ok(cancellation)
    }

    pub(crate) fn cancel_download(&self, request_id: &str) -> Result<(), String> {
        if let Some(cancellation) = self
            .download_cancellations
            .lock()
            .map_err(|error| format!("Could not access active downloads: {error}"))?
            .get(request_id)
        {
            cancellation.cancel();
        }
        Ok(())
    }

    fn remove_download_cancellation(&self, request_id: &str) -> Result<(), String> {
        self.download_cancellations
            .lock()
            .map_err(|error| format!("Could not clean up active downloads: {error}"))?
            .remove(request_id);
        Ok(())
    }

    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) -> Result<(), String> {
        Arc::get_mut(&mut self.tools)
            .ok_or_else(|| "Tools cannot be registered after backend state is shared".to_owned())?
            .register(tool)
    }
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn status(&self, endpoint: &str) -> ServerStatus;
    async fn list_models(&self, endpoint: &str) -> Result<Vec<ModelInfo>, String>;
    async fn pull_model(
        &self,
        model: String,
        endpoint: &str,
        cancellation: CancellationToken,
        on_progress: ProgressCallback,
    ) -> Result<(), String>;
    async fn chat(
        &self,
        request: ChatRequest,
        tools: Vec<ToolDefinition>,
        cancellation: CancellationToken,
        request_id: String,
        on_chunk: ChatCallback,
        on_thinking: ThinkingCallback,
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

    fn api_url(&self, endpoint: &str, path: &str) -> Result<String, String> {
        Ok(format!("{}{path}", normalize_endpoint(endpoint)?))
    }
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn status(&self, endpoint: &str) -> ServerStatus {
        let endpoint = match normalize_endpoint(endpoint) {
            Ok(endpoint) => endpoint,
            Err(message) => {
                return ServerStatus {
                    connected: false,
                    endpoint: endpoint.to_owned(),
                    message,
                };
            }
        };
        let url = match self.api_url(&endpoint, "/api/tags") {
            Ok(url) => url,
            Err(message) => {
                return ServerStatus {
                    connected: false,
                    endpoint,
                    message,
                };
            }
        };
        match self.client.get(url).send().await {
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

    async fn list_models(&self, endpoint: &str) -> Result<Vec<ModelInfo>, String> {
        let response = self
            .client
            .get(self.api_url(endpoint, "/api/tags")?)
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

    async fn pull_model(
        &self,
        model: String,
        endpoint: &str,
        cancellation: CancellationToken,
        on_progress: ProgressCallback,
    ) -> Result<(), String> {
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err("Model pull cancelled".to_owned()),
            response = self
                .client
                .post(self.api_url(endpoint, "/api/pull")?)
                .json(&PullRequest {
                    name: model.clone(),
                })
                .send() => response.map_err(|error| format!("Could not reach Ollama: {error}"))?,
        };
        let response = ensure_success(response).await?;

        let mut aggregator = PullProgressAggregator::default();
        let model_name = model;
        read_ndjson_cancellable(response, cancellation.clone(), |line| {
            let item: OllamaPullProgress = parse_ndjson_line(line)?;
            on_progress(aggregator.update(&model_name, item))?;
            Ok(())
        })
        .await?;
        if cancellation.is_cancelled() {
            return Err("Model pull cancelled".to_owned());
        }
        Ok(())
    }

    async fn chat(
        &self,
        request: ChatRequest,
        tools: Vec<ToolDefinition>,
        cancellation: CancellationToken,
        request_id: String,
        on_chunk: ChatCallback,
        on_thinking: ThinkingCallback,
    ) -> Result<ChatMessage, String> {
        request.options.validate()?;
        let endpoint = if request.endpoint.is_empty() {
            self.endpoint.as_str()
        } else {
            request.endpoint.as_str()
        };
        let api_request = ChatApiRequest {
            model: request.model.clone(),
            messages: request.messages.clone(),
            tools,
            options: request.options.clone(),
            think: request.enable_reasoning.then_some(true),
            stream: true,
        };
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err("Chat cancelled".to_owned()),
            response = self
                .client
                .post(self.api_url(endpoint, "/api/chat")?)
                .json(&api_request)
                .send() => response.map_err(|error| format!("Could not reach Ollama: {error}"))?,
        };
        let response = match ensure_success(response).await {
            Ok(response) => response,
            Err(error) if api_request.think.is_some() && is_thinking_unsupported_error(&error) => {
                let mut fallback_request = api_request;
                fallback_request.think = None;
                let response = tokio::select! {
                    _ = cancellation.cancelled() => return Err("Chat cancelled".to_owned()),
                    response = self
                        .client
                        .post(self.api_url(endpoint, "/api/chat")?)
                        .json(&fallback_request)
                        .send() => response.map_err(|error| format!("Could not reach Ollama: {error}"))?,
                };
                ensure_success(response).await?
            }
            Err(error) => return Err(error),
        };

        let mut message = ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            tool_calls: None,
            tool_name: None,
            thinking: String::new(),
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
                if !part.thinking.is_empty() {
                    message.thinking.push_str(&part.thinking);
                    on_thinking(ChatThinkingChunk {
                        request_id: request_id.clone(),
                        thinking: part.thinking,
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
    #[serde(default, skip_serializing)]
    pub thinking: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub workspace_path: Option<PathBuf>,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub options: ChatOptions,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub strict_mode: bool,
    #[serde(default)]
    pub enable_reasoning: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatOptions {
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_num_ctx")]
    pub num_ctx: u32,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            num_ctx: default_num_ctx(),
        }
    }
}

impl ChatOptions {
    fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err("Temperature must be between 0 and 2.".to_owned());
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err("Top-p must be between 0 and 1.".to_owned());
        }
        if !(256..=131_072).contains(&self.num_ctx) {
            return Err("Context length must be between 256 and 131072.".to_owned());
        }
        Ok(())
    }
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.9
}

fn default_num_ctx() -> u32 {
    4096
}

fn normalize_endpoint(endpoint: &str) -> Result<String, String> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    let parsed =
        reqwest::Url::parse(endpoint).map_err(|error| format!("Invalid Ollama URL: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host().is_none() {
        return Err("Ollama URL must be an HTTP or HTTPS URL with a host.".to_owned());
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err("Ollama URL cannot include a query or fragment.".to_owned());
    }
    Ok(endpoint.to_owned())
}

#[derive(Serialize)]
struct PullRequest {
    name: String,
}

#[derive(Clone, Serialize)]
struct ChatApiRequest {
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    options: ChatOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
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

#[derive(Default)]
struct PullProgressAggregator {
    layers: HashMap<String, (u64, u64)>,
}

impl PullProgressAggregator {
    fn update(&mut self, model: &str, item: OllamaPullProgress) -> PullProgress {
        if let (Some(digest), Some(total)) = (&item.digest, item.total) {
            let completed = item.completed.unwrap_or_default().min(total);
            self.layers.insert(digest.clone(), (total, completed));
        }

        let (total, mut completed) = if self.layers.is_empty() {
            (item.total, item.completed)
        } else {
            (
                Some(
                    self.layers
                        .values()
                        .map(|(total, _)| total)
                        .copied()
                        .fold(0_u64, u64::saturating_add),
                ),
                Some(
                    self.layers
                        .values()
                        .map(|(_, completed)| completed)
                        .copied()
                        .fold(0_u64, u64::saturating_add),
                ),
            )
        };
        let succeeded = item.status == "success";
        if succeeded {
            completed = total.or(completed);
        }
        let percentage = if succeeded {
            Some(100)
        } else {
            total
                .filter(|total| *total > 0)
                .zip(completed)
                .map(|(total, completed)| {
                    ((completed.min(total) as f64 / total as f64) * 100.0).round() as u8
                })
        };

        PullProgress {
            request_id: String::new(),
            name: model.to_owned(),
            status: item.status,
            digest: item.digest,
            total,
            completed,
            percentage,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatApiResponse {
    #[serde(default)]
    message: Option<ChatMessage>,
}

fn is_thinking_unsupported_error(error: &str) -> bool {
    error.to_ascii_lowercase().contains("think")
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
    pub name: String,
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
    pub percentage: Option<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatChunk {
    pub request_id: String,
    pub content: String,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatThinkingChunk {
    pub request_id: String,
    pub thinking: String,
}

#[derive(Clone, Default)]
struct PendingApprovals(Arc<Mutex<HashMap<String, oneshot::Sender<ApprovalDecision>>>>);

#[derive(Clone, Debug, Serialize)]
pub struct ToolApprovalRequest {
    pub request_id: String,
    pub approval_id: String,
    pub tool: ToolSchema,
    pub arguments: Value,
    pub preview: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_path: Option<String>,
}

#[async_trait]
trait ApprovalHandler: Send + Sync {
    async fn approve(
        &self,
        approval: ToolApprovalRequest,
        cancellation: CancellationToken,
    ) -> Result<ApprovalDecision, String>;
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
    ) -> Result<ApprovalDecision, String> {
        request_approval(&self.app, &self.pending, approval, cancellation).await
    }
}

async fn request_approval(
    app: &AppHandle,
    pending: &PendingApprovals,
    approval: ToolApprovalRequest,
    cancellation: CancellationToken,
) -> Result<ApprovalDecision, String> {
    let (sender, receiver) = oneshot::channel();
    pending
        .0
        .lock()
        .map_err(|error| format!("Could not manage pending tool approvals: {error}"))?
        .insert(approval.approval_id.clone(), sender);
    if let Err(error) = app.emit(TOOL_APPROVAL_EVENT, &approval) {
        pending
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
    pending
        .0
        .lock()
        .map_err(|error| format!("Could not clean up tool approval: {error}"))?
        .remove(&approval.approval_id);
    Ok(decision.unwrap_or(ApprovalDecision::Deny))
}

struct TauriWorkspaceAccess {
    app: AppHandle,
    pending: PendingApprovals,
    persistence: PersistenceState,
    cancellation: CancellationToken,
    request_id: String,
    session_id: String,
    approval_id: String,
    tool: ToolSchema,
    arguments: Value,
}

struct WorkspaceAccessProvider {
    app: AppHandle,
    pending: PendingApprovals,
    persistence: PersistenceState,
}

impl WorkspaceAccessProvider {
    fn for_tool(
        &self,
        request_id: &str,
        session_id: &str,
        approval_id: &str,
        tool: ToolSchema,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Arc<dyn OutOfWorkspaceApprover> {
        Arc::new(TauriWorkspaceAccess {
            app: self.app.clone(),
            pending: self.pending.clone(),
            persistence: self.persistence.clone(),
            cancellation,
            request_id: request_id.to_owned(),
            session_id: session_id.to_owned(),
            approval_id: approval_id.to_owned(),
            tool,
            arguments,
        })
    }
}

#[async_trait]
impl OutOfWorkspaceApprover for TauriWorkspaceAccess {
    async fn approve_path(&self, path: &Path) -> Result<ApprovalDecision, String> {
        if self
            .persistence
            .0
            .lock()
            .map_err(|error| format!("Could not check trusted paths: {error}"))?
            .is_path_trusted(path)?
        {
            self.audit_access(path, "trusted_path")?;
            return Ok(ApprovalDecision::AllowAlways);
        }

        let decision = request_approval(
            &self.app,
            &self.pending,
            ToolApprovalRequest {
                request_id: self.request_id.clone(),
                approval_id: self.approval_id.clone(),
                tool: self.tool.clone(),
                arguments: self.arguments.clone(),
                preview: None,
                approval_type: Some("out_of_workspace_access".to_owned()),
                requested_path: Some(path.to_string_lossy().into_owned()),
            },
            self.cancellation.clone(),
        )
        .await?;
        match decision {
            ApprovalDecision::AllowOnce => self.audit_access(path, "allow_once")?,
            ApprovalDecision::AllowAlways => {
                let trusted_directory = if path.is_dir() {
                    path
                } else {
                    path.parent()
                        .ok_or_else(|| "Requested path has no parent directory".to_owned())?
                };
                self.persistence
                    .0
                    .lock()
                    .map_err(|error| format!("Could not store trusted path: {error}"))?
                    .grant_trusted_path(trusted_directory)?;
                self.audit_access(path, "allow_always")?;
            }
            ApprovalDecision::Deny => {}
        }
        Ok(decision)
    }
}

impl TauriWorkspaceAccess {
    fn audit_access(&self, path: &Path, approval_status: &str) -> Result<(), String> {
        self.persistence
            .0
            .lock()
            .map_err(|error| format!("Could not access SQLite audit log: {error}"))?
            .append_tool_audit_log(&ToolAuditRecord {
                id: format!("{}-path-access", self.approval_id),
                session_id: self.session_id.clone(),
                tool_name: self.tool.name.clone(),
                arguments_summary: serde_json::to_string(&json!({
                    "requested_path": path
                }))
                .map_err(|error| format!("Could not summarize approved path: {error}"))?,
                approval_status: approval_status.to_owned(),
                execution_status: "authorized".to_owned(),
                created_at: unix_timestamp(),
            })
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
                callbacks.on_thinking.clone(),
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
            let call_arguments = call.function.arguments.clone();
            let mut approval_status = "approved";
            let result = if let Some(tool) = tools.get(&call.function.name) {
                let schema = tool.schema();
                let call_id = format!("{request_id}-{iteration}-{call_index}");
                let path_approver = callbacks.workspace_access.map(|provider| {
                    provider.for_tool(
                        request_id,
                        &request.session_id,
                        &format!("{call_id}-path"),
                        schema.clone(),
                        call_arguments.clone(),
                        cancellation.clone(),
                    )
                });
                let context = ToolContext::new(
                    workspace.clone(),
                    request_id.to_owned(),
                    call_id.clone(),
                    Some(callbacks.command_output.clone()),
                    path_approver,
                );
                let blocked_reason = if call.function.name == "run_command" {
                    call_arguments
                        .get("command")
                        .and_then(Value::as_str)
                        .map(crate::security::blocked_command_reason)
                        .transpose()?
                        .flatten()
                } else {
                    None
                };
                if let Some(reason) = blocked_reason {
                    approval_status = "rejected";
                    json!({ "error": format!("Command rejected by the dangerous-command blocklist: {reason}") })
                } else {
                    let requires_confirmation =
                        request.strict_mode || schema.risk_level == RiskLevel::RequiresConfirmation;
                    let mut rejection_result = None;
                    let preview = if requires_confirmation {
                        match tokio::select! {
                            _ = cancellation.cancelled() => return Ok(()),
                            result = tool.preview(call_arguments.clone(), context.clone()) => result,
                        } {
                            Ok(preview) => preview,
                            Err(error) => {
                                approval_status = "rejected";
                                rejection_result = Some(json!({ "error": error }));
                                None
                            }
                        }
                    } else {
                        None
                    };
                    if requires_confirmation && rejection_result.is_none() {
                        let approved = callbacks
                            .approval_handler
                            .approve(
                                ToolApprovalRequest {
                                    request_id: request_id.to_owned(),
                                    approval_id: format!("{request_id}-{iteration}-{call_index}"),
                                    tool: schema.clone(),
                                    arguments: call_arguments.clone(),
                                    preview: preview.clone(),
                                    approval_type: None,
                                    requested_path: None,
                                },
                                cancellation.clone(),
                            )
                            .await;
                        let decision = match approved {
                            Ok(decision) => decision,
                            Err(error) => {
                                approval_status = "rejected";
                                rejection_result = Some(json!({ "error": error }));
                                ApprovalDecision::Deny
                            }
                        };
                        if decision == ApprovalDecision::Deny && rejection_result.is_none() {
                            approval_status = "rejected";
                            rejection_result = Some(
                                json!({ "error": "User denied permission to run this tool." }),
                            );
                        }
                    }
                    match rejection_result {
                        Some(result) => result,
                        None => {
                            let mut approved_arguments = call_arguments.clone();
                            if let Some(before) =
                                preview.as_ref().and_then(|preview| preview.get("before"))
                            {
                                if let Some(arguments) = approved_arguments.as_object_mut() {
                                    arguments.insert("_approved_before".to_owned(), before.clone());
                                }
                            }
                            tokio::select! {
                                _ = cancellation.cancelled() => return Ok(()),
                                result = tool.call(approved_arguments, context.clone()) => {
                                    result.unwrap_or_else(|error| json!({ "error": error }))
                                }
                            }
                        }
                    }
                }
            } else {
                approval_status = "rejected";
                json!({ "error": format!("Tool '{}' is not registered.", call.function.name) })
            };

            let execution_status = execution_status(&result, approval_status);
            (callbacks.on_tool_call)(ToolCallEvent {
                id: format!("{request_id}-{iteration}-{call_index}"),
                request_id: request_id.to_owned(),
                tool_name: call.function.name.clone(),
                arguments: call_arguments,
                arguments_summary: summarize_tool_arguments(&call.function.arguments)?,
                status: if approval_status == "rejected" {
                    "rejected".to_owned()
                } else {
                    execution_status.to_owned()
                },
                approval_status: approval_status.to_owned(),
                result: result.clone(),
                created_at: unix_timestamp(),
            })?;
            let content = serde_json::to_string(&result)
                .map_err(|error| format!("Could not encode tool result: {error}"))?;
            request.messages.push(ChatMessage {
                role: "tool".to_owned(),
                content,
                tool_calls: None,
                tool_name: Some(call.function.name),
                thinking: String::new(),
            });
        }
    }

    unreachable!("tool loop returns on the final allowed iteration")
}

struct ToolLoopCallbacks<'a> {
    approval_handler: &'a dyn ApprovalHandler,
    workspace_access: Option<&'a WorkspaceAccessProvider>,
    command_output: CommandOutputCallback,
    on_chunk: ChatCallback,
    on_thinking: ThinkingCallback,
    on_tool_call: &'a ToolCallCallback,
}

fn unix_timestamp() -> String {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or_else(
        |_| "0".to_owned(),
        |duration| duration.as_millis().to_string(),
    )
}

fn execution_status(result: &Value, approval_status: &str) -> &'static str {
    if approval_status == "rejected" {
        "not_run"
    } else if result.get("error").is_some()
        || result
            .get("exit_code")
            .and_then(Value::as_i64)
            .is_some_and(|exit_code| exit_code != 0)
    {
        "failed"
    } else {
        "succeeded"
    }
}

fn summarize_tool_arguments(arguments: &Value) -> Result<String, String> {
    const MAX_SUMMARY_CHARACTERS: usize = 2_000;
    let encoded = serde_json::to_string(arguments)
        .map_err(|error| format!("Could not summarize tool arguments: {error}"))?;
    let mut characters = encoded.chars();
    let summary = characters
        .by_ref()
        .take(MAX_SUMMARY_CHARACTERS)
        .collect::<String>();
    if characters.next().is_some() {
        Ok(format!("{summary}...[truncated]"))
    } else {
        Ok(summary)
    }
}

#[tauri::command]
pub async fn ollama_status(
    backend: State<'_, BackendState>,
    endpoint: String,
) -> Result<ServerStatus, String> {
    Ok(backend.backend.status(&endpoint).await)
}

#[tauri::command]
pub async fn ollama_list_models(
    backend: State<'_, BackendState>,
    endpoint: String,
) -> Result<Vec<ModelInfo>, String> {
    backend.backend.list_models(&endpoint).await
}

#[tauri::command]
pub async fn ollama_pull_model(
    app: AppHandle,
    backend: State<'_, BackendState>,
    model: String,
    endpoint: String,
    request_id: String,
) -> Result<(), String> {
    let progress_app = app.clone();
    let progress_request_id = request_id.clone();
    let on_progress = Arc::new(move |progress: PullProgress| {
        emit_pull_progress(progress, &progress_request_id, |event, progress| {
            progress_app
                .emit(event, progress)
                .map_err(|error| error.to_string())
        })
    });
    let llm_backend = backend.backend.clone();
    with_download_cancellation(&backend, &request_id, |cancellation| async move {
        llm_backend
            .pull_model(model, &endpoint, cancellation, on_progress)
            .await
    })
    .await
}

#[tauri::command]
pub async fn ollama_cancel_pull(
    backend: State<'_, BackendState>,
    request_id: String,
) -> Result<(), String> {
    backend.cancel_download(&request_id)
}

pub(crate) async fn with_download_cancellation<T, F, Fut>(
    backend: &BackendState,
    request_id: &str,
    operation: F,
) -> Result<T, String>
where
    F: FnOnce(CancellationToken) -> Fut,
    Fut: Future<Output = Result<T, String>>,
{
    let cancellation = backend.register_download_cancellation(request_id)?;
    let result = operation(cancellation).await;
    backend.remove_download_cancellation(request_id)?;
    result
}

#[tauri::command]
pub async fn ollama_chat(
    app: AppHandle,
    backend: State<'_, BackendState>,
    persistence: State<'_, PersistenceState>,
    mut request: ChatRequest,
    request_id: String,
) -> Result<(), String> {
    request.options.validate()?;
    request.endpoint = normalize_endpoint(if request.endpoint.is_empty() {
        DEFAULT_OLLAMA_ENDPOINT
    } else {
        &request.endpoint
    })?;
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
    let thinking_request_id = request_id.clone();
    let thinking_app = app.clone();
    let on_thinking: ThinkingCallback = Arc::new(move |chunk| {
        emit_chat_thinking(chunk, &thinking_request_id, |event, chunk| {
            thinking_app
                .emit(event, chunk)
                .map_err(|error| error.to_string())
        })
    });
    let approval_handler = TauriApprovalHandler {
        app: app.clone(),
        pending: backend.pending_approvals.clone(),
    };
    let workspace_access = WorkspaceAccessProvider {
        app: app.clone(),
        pending: backend.pending_approvals.clone(),
        persistence: PersistenceState(persistence.0.clone()),
    };
    let output_app = app.clone();
    let command_output: CommandOutputCallback = Arc::new(move |event| {
        output_app
            .emit(COMMAND_OUTPUT_EVENT, event)
            .map_err(|error| format!("Could not emit command output: {error}"))
    });
    let audit_session_id = request.session_id.clone();
    let audit_database = persistence.0.clone();
    let tool_call_app = app.clone();
    let on_tool_call: ToolCallCallback = Arc::new(move |event| {
        audit_database
            .lock()
            .map_err(|error| format!("Could not access SQLite audit log: {error}"))?
            .append_tool_audit_log(&ToolAuditRecord {
                id: event.id.clone(),
                session_id: audit_session_id.clone(),
                tool_name: event.tool_name.clone(),
                arguments_summary: event.arguments_summary.clone(),
                approval_status: event.approval_status.clone(),
                execution_status: execution_status(&event.result, &event.approval_status)
                    .to_owned(),
                created_at: event.created_at.clone(),
            })?;
        tool_call_app
            .emit(TOOL_CALL_EVENT, event)
            .map_err(|error| format!("Could not emit tool call history: {error}"))
    });
    let result = run_tool_loop(
        backend.backend.as_ref(),
        &backend.tools,
        request,
        &request_id,
        cancellation.clone(),
        ToolLoopCallbacks {
            approval_handler: &approval_handler,
            workspace_access: Some(&workspace_access),
            command_output,
            on_chunk,
            on_thinking,
            on_tool_call: &on_tool_call,
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
    decision: ApprovalDecision,
) -> Result<(), String> {
    let sender = backend
        .pending_approvals
        .0
        .lock()
        .map_err(|error| format!("Could not access pending tool approvals: {error}"))?
        .remove(&approval_id)
        .ok_or_else(|| format!("No pending approval found for {approval_id}"))?;
    sender
        .send(decision)
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
        emit_chat_thinking, emit_pull_progress, normalize_endpoint, parse_ndjson_line,
        run_tool_loop, with_download_cancellation, ApprovalHandler, BackendState, ChatApiRequest,
        ChatApiResponse, ChatCallback, ChatChunk, ChatMessage, ChatOptions, ChatRequest,
        ChatThinkingChunk, LlmBackend, ModelInfo, NdjsonBuffer, OllamaBackend, OllamaPullProgress,
        PullProgress, PullProgressAggregator, ServerStatus, ThinkingCallback, ToolApprovalRequest,
        ToolCallCallback, ToolLoopCallbacks, MAX_TOOL_ITERATIONS,
    };
    use crate::tools::{
        ApprovalDecision, CommandOutputCallback, RiskLevel, Tool, ToolCall, ToolContext,
        ToolDefinition, ToolFunctionCall, ToolRegistry, ToolSchema,
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_util::sync::CancellationToken;

    #[test]
    fn aggregates_pull_progress_across_layers_and_marks_success() {
        let mut aggregator = PullProgressAggregator::default();
        let first = aggregator.update(
            "llama3.1:8b",
            OllamaPullProgress {
                status: "pulling".to_owned(),
                digest: Some("layer-a".to_owned()),
                total: Some(100),
                completed: Some(50),
            },
        );
        assert_eq!(first.total, Some(100));
        assert_eq!(first.completed, Some(50));
        assert_eq!(first.percentage, Some(50));
        assert_eq!(first.name, "llama3.1:8b");

        let second = aggregator.update(
            "llama3.1:8b",
            OllamaPullProgress {
                status: "pulling".to_owned(),
                digest: Some("layer-b".to_owned()),
                total: Some(300),
                completed: Some(100),
            },
        );
        assert_eq!(second.total, Some(400));
        assert_eq!(second.completed, Some(150));
        assert_eq!(second.percentage, Some(38));

        let complete = aggregator.update(
            "llama3.1:8b",
            OllamaPullProgress {
                status: "success".to_owned(),
                digest: None,
                total: None,
                completed: None,
            },
        );
        assert_eq!(complete.completed, Some(400));
        assert_eq!(complete.percentage, Some(100));
    }

    #[test]
    fn emits_aggregated_model_progress_with_request_id() {
        let mut emitted = None;
        emit_pull_progress(
            PullProgress {
                request_id: String::new(),
                name: "llama3.1:8b".to_owned(),
                status: "pulling".to_owned(),
                digest: Some("layer-a".to_owned()),
                total: Some(100),
                completed: Some(25),
                percentage: Some(25),
            },
            "request-1",
            |event, progress| {
                emitted = Some((event, progress));
                Ok(())
            },
        )
        .expect("progress event is emitted");

        let (event, progress) = emitted.expect("emitter receives the event");
        assert_eq!(event, super::PULL_PROGRESS_EVENT);
        assert_eq!(progress.request_id, "request-1");
        assert_eq!(progress.name, "llama3.1:8b");
        assert_eq!(progress.percentage, Some(25));
    }

    #[test]
    fn emits_chat_thinking_with_the_request_id() {
        let mut emitted = None;
        emit_chat_thinking(
            ChatThinkingChunk {
                request_id: String::new(),
                thinking: "reasoning delta".to_owned(),
            },
            "request-1",
            |event, chunk| {
                emitted = Some((event, chunk));
                Ok(())
            },
        )
        .expect("thinking event is emitted");

        let (event, chunk) = emitted.expect("emitter receives the event");
        assert_eq!(event, super::CHAT_THINKING_EVENT);
        assert_eq!(chunk.request_id, "request-1");
        assert_eq!(chunk.thinking, "reasoning delta");
    }

    #[tokio::test]
    async fn streams_thinking_separately_from_chat_content() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local HTTP listener starts");
        let address = listener
            .local_addr()
            .expect("listener address is available");
        let response_body = concat!(
            "{\"message\":{\"role\":\"assistant\",\"thinking\":\"Let me consider. \",\"content\":\"\"}}\n",
            "{\"message\":{\"role\":\"assistant\",\"thinking\":\"Now answer.\",\"content\":\"42\"}}\n",
            "{\"done\":true}\n"
        );
        let server_body = response_body.to_owned();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("request is accepted");
            let request = read_http_request(&mut stream).await;
            let request: Value =
                serde_json::from_slice(request_body(&request)).expect("request is JSON");
            assert_eq!(request["think"], true);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                server_body.len(),
                server_body
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("response is written");
        });

        let backend = OllamaBackend::new(format!("http://{address}"));
        let mut request = chat_request();
        request.endpoint = format!("http://{address}");
        request.enable_reasoning = true;
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let thinking_chunks = Arc::new(Mutex::new(Vec::new()));
        let captured_chunks = chunks.clone();
        let captured_thinking = thinking_chunks.clone();
        let message = backend
            .chat(
                request,
                Vec::new(),
                CancellationToken::new(),
                "request-1".to_owned(),
                Arc::new(move |chunk| {
                    captured_chunks.lock().unwrap().push(chunk);
                    Ok(())
                }),
                Arc::new(move |chunk| {
                    captured_thinking.lock().unwrap().push(chunk);
                    Ok(())
                }),
            )
            .await
            .expect("chat response is parsed");
        server.await.expect("server completes");

        assert_eq!(message.content, "42");
        assert_eq!(message.thinking, "Let me consider. Now answer.");
        assert_eq!(
            chunks
                .lock()
                .unwrap()
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<String>(),
            "42"
        );
        let thinking_chunks = thinking_chunks.lock().unwrap();
        assert_eq!(thinking_chunks.len(), 2);
        assert!(thinking_chunks
            .iter()
            .all(|chunk| chunk.request_id == "request-1"));
        assert_eq!(
            thinking_chunks
                .iter()
                .map(|chunk| chunk.thinking.as_str())
                .collect::<String>(),
            "Let me consider. Now answer."
        );
    }

    #[tokio::test]
    async fn retries_without_thinking_when_the_server_rejects_think() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local HTTP listener starts");
        let address = listener
            .local_addr()
            .expect("listener address is available");
        let server = tokio::spawn(async move {
            let (mut first_stream, _) = listener.accept().await.expect("first request is accepted");
            let first_request = read_http_request(&mut first_stream).await;
            let first_request: Value =
                serde_json::from_slice(request_body(&first_request)).expect("request is JSON");
            assert_eq!(first_request["think"], true);
            let body = r#"{"error":"model does not support thinking"}"#;
            let response = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            first_stream
                .write_all(response.as_bytes())
                .await
                .expect("unsupported-thinking response is written");
            drop(first_stream);

            let (mut second_stream, _) = listener
                .accept()
                .await
                .expect("fallback request is accepted");
            let fallback_request = read_http_request(&mut second_stream).await;
            let fallback_request: Value = serde_json::from_slice(request_body(&fallback_request))
                .expect("fallback request is JSON");
            assert!(fallback_request.get("think").is_none());
            let body = "{\"message\":{\"role\":\"assistant\",\"content\":\"answer\"}}\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            second_stream
                .write_all(response.as_bytes())
                .await
                .expect("fallback chat response is written");
        });

        let backend = OllamaBackend::new(format!("http://{address}"));
        let mut request = chat_request();
        request.endpoint = format!("http://{address}");
        request.enable_reasoning = true;
        let message = backend
            .chat(
                request,
                Vec::new(),
                CancellationToken::new(),
                "request-1".to_owned(),
                Arc::new(|_| Ok(())),
                Arc::new(|_| Ok(())),
            )
            .await
            .expect("unsupported reasoning falls back to regular chat");
        server.await.expect("server completes");
        assert_eq!(message.content, "answer");
        assert!(message.thinking.is_empty());
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        loop {
            let mut buffer = [0_u8; 4096];
            let bytes_read = stream.read(&mut buffer).await.expect("request is read");
            if bytes_read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..bytes_read]);
            if let Some(headers_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..headers_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("valid content length"))
                    })
                    .unwrap_or_default();
                if request.len() >= headers_end + 4 + content_length {
                    break;
                }
            }
        }
        request
    }

    fn request_body(request: &[u8]) -> &[u8] {
        let body_start = request
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .expect("request has headers")
            + 4;
        &request[body_start..]
    }

    #[tokio::test]
    async fn fetches_installed_models_from_the_ollama_tags_endpoint() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local HTTP listener starts");
        let address = listener
            .local_addr()
            .expect("listener address is available");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("request is accepted");
            let mut request = [0_u8; 1024];
            let bytes_read = stream.read(&mut request).await.expect("request is read");
            let request = String::from_utf8_lossy(&request[..bytes_read]);
            assert!(request.starts_with("GET /api/tags HTTP/1.1"));
            let body = r#"{"models":[{"name":"qwen2.5:7b","size":123,"digest":"abc"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("response is written");
        });

        let backend = OllamaBackend::new(format!("http://{address}"));
        let models = backend
            .list_models(&format!("http://{address}"))
            .await
            .expect("model list is returned");
        server.await.expect("local server completes");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "qwen2.5:7b");
        assert_eq!(models[0].size, 123);
    }

    #[tokio::test]
    async fn cancels_model_pull_mid_stream_and_cleans_up_its_registry_entry() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local HTTP listener starts");
        let address = listener
            .local_addr()
            .expect("listener address is available");
        let (chunk_sent, chunk_received) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("request is accepted");
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.expect("request is read");
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n",
                )
                .await
                .expect("stream response headers are written");
            let progress =
                b"{\"status\":\"pulling\",\"digest\":\"layer\",\"total\":10,\"completed\":1}\n";
            let chunk = format!("{:X}\r\n", progress.len());
            stream
                .write_all(chunk.as_bytes())
                .await
                .expect("chunk header is written");
            stream
                .write_all(progress)
                .await
                .expect("partial progress is written");
            stream
                .write_all(b"\r\n")
                .await
                .expect("chunk terminator is written");
            let _ = chunk_sent.send(());
            let mut byte = [0_u8; 1];
            let _ = stream.read(&mut byte).await;
        });

        let backend = Arc::new(OllamaBackend::new(format!("http://{address}")));
        let state = BackendState::with_backend(backend.clone());
        let request_state = state.clone();
        let endpoint = format!("http://{address}");
        let pull = tokio::spawn(async move {
            with_download_cancellation(&request_state, "pull-1", |cancellation| async move {
                backend
                    .pull_model(
                        "llama3.1:8b".to_owned(),
                        &endpoint,
                        cancellation,
                        Arc::new(|_| Ok(())),
                    )
                    .await
            })
            .await
        });

        chunk_received
            .await
            .expect("server sends an in-progress model layer");
        assert!(state
            .download_cancellations
            .lock()
            .expect("download registry is accessible")
            .contains_key("pull-1"));
        state
            .cancel_download("pull-1")
            .expect("model pull cancellation is requested");
        let error = pull
            .await
            .expect("model pull task completes")
            .expect_err("cancelled model pull returns an error");
        assert_eq!(error, "Model pull cancelled");
        assert!(!state
            .download_cancellations
            .lock()
            .expect("download registry is accessible")
            .contains_key("pull-1"));
        server.await.expect("server observes the cancelled request");
    }

    #[test]
    fn validates_ollama_endpoint_and_sampling_options() {
        assert_eq!(
            normalize_endpoint(" https://ollama.example/ "),
            Ok("https://ollama.example".to_owned())
        );
        assert!(normalize_endpoint("file:///tmp/ollama").is_err());
        assert!(normalize_endpoint("http://localhost:11434/?token=x").is_err());

        let options = ChatOptions::default();
        assert!(options.validate().is_ok());
        assert!(ChatOptions {
            temperature: 2.1,
            ..options.clone()
        }
        .validate()
        .is_err());
        assert!(ChatOptions {
            top_p: -0.1,
            ..options
        }
        .validate()
        .is_err());
    }

    #[test]
    fn serializes_sampling_options_for_ollama_chat() {
        let request = ChatApiRequest {
            model: "test-model".to_owned(),
            messages: Vec::new(),
            tools: Vec::new(),
            options: ChatOptions {
                temperature: 0.4,
                top_p: 0.8,
                num_ctx: 8192,
            },
            think: Some(true),
            stream: true,
        };
        let value = serde_json::to_value(request).expect("request serializes");
        assert_eq!(value["options"]["temperature"], 0.4);
        assert_eq!(value["options"]["top_p"], 0.8);
        assert_eq!(value["options"]["num_ctx"], 8192);
        assert_eq!(value["think"], true);
        assert_eq!(
            serde_json::to_value(ChatMessage {
                role: "assistant".to_owned(),
                content: "answer".to_owned(),
                tool_calls: None,
                tool_name: None,
                thinking: "private reasoning".to_owned(),
            })
            .expect("message serializes")["thinking"],
            Value::Null
        );
    }

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
        async fn status(&self, _endpoint: &str) -> ServerStatus {
            ServerStatus {
                connected: true,
                endpoint: String::new(),
                message: String::new(),
            }
        }

        async fn list_models(&self, _endpoint: &str) -> Result<Vec<ModelInfo>, String> {
            Ok(Vec::new())
        }

        async fn pull_model(
            &self,
            _model: String,
            _endpoint: &str,
            _cancellation: CancellationToken,
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
            on_thinking: ThinkingCallback,
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
                    request_id: request_id.clone(),
                    content: message.content.clone(),
                    done: false,
                })?;
            }
            if !message.thinking.is_empty() {
                on_thinking(ChatThinkingChunk {
                    request_id,
                    thinking: message.thinking.clone(),
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
        ) -> Result<ApprovalDecision, String> {
            self.1.fetch_add(1, Ordering::SeqCst);
            Ok(if self.0 {
                ApprovalDecision::AllowOnce
            } else {
                ApprovalDecision::Deny
            })
        }
    }

    struct TestTool {
        name: &'static str,
        risk_level: RiskLevel,
        calls: AtomicUsize,
        fails: bool,
        workspaces: Mutex<Vec<PathBuf>>,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.to_owned(),
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
        tool_call_response_with("test_tool", json!({}))
    }

    fn tool_call_response_with(name: &str, arguments: Value) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                function: ToolFunctionCall {
                    name: name.to_owned(),
                    arguments,
                },
            }]),
            tool_name: None,
            thinking: String::new(),
        }
    }

    fn final_response() -> ChatMessage {
        ChatMessage {
            role: "assistant".to_owned(),
            content: "done".to_owned(),
            tool_calls: None,
            tool_name: None,
            thinking: String::new(),
        }
    }

    fn chat_request() -> ChatRequest {
        ChatRequest {
            model: "mock".to_owned(),
            workspace_path: Some(std::path::PathBuf::from("test-workspace")),
            endpoint: super::DEFAULT_OLLAMA_ENDPOINT.to_owned(),
            options: super::ChatOptions::default(),
            session_id: "session".to_owned(),
            strict_mode: false,
            enable_reasoning: false,
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: "run it".to_owned(),
                tool_calls: None,
                tool_name: None,
                thinking: String::new(),
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

    fn tool_call_callback() -> ToolCallCallback {
        Arc::new(|_| Ok(()))
    }

    #[tokio::test]
    async fn reports_tool_failures_to_model_and_reaches_final_answer() {
        let backend = MockBackend::new(vec![tool_call_response(), final_response()]);
        let tool = Arc::new(TestTool {
            name: "test_tool",
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
                workspace_access: None,
                command_output: command_output_callback(),
                on_chunk: on_chunk.clone(),
                on_thinking: Arc::new(|_| Ok(())),
                on_tool_call: &tool_call_callback(),
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
            name: "test_tool",
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
                workspace_access: None,
                command_output: command_output_callback(),
                on_chunk: on_chunk.clone(),
                on_thinking: Arc::new(|_| Ok(())),
                on_tool_call: &tool_call_callback(),
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
    async fn blocklisted_commands_are_rejected_before_approval() {
        let backend = MockBackend::new(vec![
            tool_call_response_with("run_command", json!({ "command": "rm -rf /" })),
            final_response(),
        ]);
        let tool = Arc::new(TestTool {
            name: "run_command",
            risk_level: RiskLevel::RequiresConfirmation,
            calls: AtomicUsize::new(0),
            fails: false,
            workspaces: Mutex::new(Vec::new()),
        });
        let mut tools = ToolRegistry::default();
        tools.register(tool.clone()).expect("tool registers");
        let approval = FixedApproval(true, AtomicUsize::new(0));
        let (on_chunk, _) = callback();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = events.clone();
        let on_tool_call: ToolCallCallback = Arc::new(move |event| {
            captured_events.lock().unwrap().push(event);
            Ok(())
        });

        run_tool_loop(
            &backend,
            &tools,
            chat_request(),
            "request",
            CancellationToken::new(),
            ToolLoopCallbacks {
                approval_handler: &approval,
                workspace_access: None,
                command_output: command_output_callback(),
                on_chunk: on_chunk.clone(),
                on_thinking: Arc::new(|_| Ok(())),
                on_tool_call: &on_tool_call,
            },
        )
        .await
        .expect("blocklisted command is returned as a tool error");

        assert_eq!(approval.1.load(Ordering::SeqCst), 0);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
        let event = events
            .lock()
            .unwrap()
            .first()
            .cloned()
            .expect("audit event");
        assert_eq!(event.approval_status, "rejected");
        assert_eq!(event.status, "rejected");
        assert!(event.result["error"]
            .as_str()
            .expect("error is a string")
            .contains("blocklist"));
    }

    #[tokio::test]
    async fn strict_mode_requires_approval_for_auto_approved_tools() {
        let backend = MockBackend::new(vec![tool_call_response(), final_response()]);
        let tool = Arc::new(TestTool {
            name: "test_tool",
            risk_level: RiskLevel::AutoApprove,
            calls: AtomicUsize::new(0),
            fails: false,
            workspaces: Mutex::new(Vec::new()),
        });
        let mut tools = ToolRegistry::default();
        tools.register(tool.clone()).expect("tool registers");
        let approval = FixedApproval(true, AtomicUsize::new(0));
        let (on_chunk, _) = callback();
        let mut request = chat_request();
        request.strict_mode = true;

        run_tool_loop(
            &backend,
            &tools,
            request,
            "request",
            CancellationToken::new(),
            ToolLoopCallbacks {
                approval_handler: &approval,
                workspace_access: None,
                command_output: command_output_callback(),
                on_chunk: on_chunk.clone(),
                on_thinking: Arc::new(|_| Ok(())),
                on_tool_call: &tool_call_callback(),
            },
        )
        .await
        .expect("strict mode call completes");

        assert_eq!(approval.1.load(Ordering::SeqCst), 1);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stops_tool_execution_at_the_iteration_limit() {
        let backend = MockBackend::new(vec![tool_call_response(); MAX_TOOL_ITERATIONS + 1]);
        let tool = Arc::new(TestTool {
            name: "test_tool",
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
                workspace_access: None,
                command_output: command_output_callback(),
                on_chunk: on_chunk.clone(),
                on_thinking: Arc::new(|_| Ok(())),
                on_tool_call: &tool_call_callback(),
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
