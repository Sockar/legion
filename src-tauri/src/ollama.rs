use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tokio_util::sync::CancellationToken;

const DEFAULT_OLLAMA_ENDPOINT: &str = "http://localhost:11434";
const PULL_PROGRESS_EVENT: &str = "ollama://pull-progress";
const CHAT_CHUNK_EVENT: &str = "ollama://chat-chunk";

pub type ProgressCallback = Arc<dyn Fn(PullProgress) -> Result<(), String> + Send + Sync>;
pub type ChatCallback = Arc<dyn Fn(ChatChunk) -> Result<(), String> + Send + Sync>;

#[derive(Clone)]
pub struct BackendState {
    backend: Arc<dyn LlmBackend>,
    chat_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl Default for BackendState {
    fn default() -> Self {
        Self::with_backend(Arc::new(OllamaBackend::new(DEFAULT_OLLAMA_ENDPOINT)))
    }
}

impl BackendState {
    pub fn with_backend(backend: Arc<dyn LlmBackend>) -> Self {
        Self {
            backend,
            chat_cancellations: Arc::default(),
        }
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
        cancellation: CancellationToken,
        on_chunk: ChatCallback,
    ) -> Result<(), String>;
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
        cancellation: CancellationToken,
        on_chunk: ChatCallback,
    ) -> Result<(), String> {
        let response = self
            .client
            .post(self.api_url("/api/chat"))
            .json(&ChatApiRequest {
                model: request.model,
                messages: request.messages,
                stream: true,
            })
            .send()
            .await
            .map_err(|error| format!("Could not reach Ollama: {error}"))?;
        let response = ensure_success(response).await?;

        read_ndjson_cancellable(response, cancellation, |line| {
            let item: ChatApiResponse = parse_ndjson_line(line)?;
            on_chunk(ChatChunk {
                request_id: String::new(),
                content: item
                    .message
                    .map_or_else(String::new, |message| message.content),
                done: item.done,
            })?;
            Ok(())
        })
        .await
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
struct PullRequest {
    name: String,
}

#[derive(Serialize)]
struct ChatApiRequest {
    model: String,
    messages: Vec<ChatMessage>,
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
    #[serde(default)]
    done: bool,
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
    let on_chunk = Arc::new(move |mut chunk: ChatChunk| {
        chunk.request_id.clone_from(&event_request_id);
        app.emit(CHAT_CHUNK_EVENT, chunk)
            .map_err(|error| format!("Could not emit chat response: {error}"))
    });
    let result = backend.backend.chat(request, cancellation, on_chunk).await;
    backend
        .chat_cancellations
        .lock()
        .map_err(|error| format!("Could not clean up active chats: {error}"))?
        .remove(&request_id);
    result
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
    use super::{parse_ndjson_line, ChatApiResponse, ModelInfo, NdjsonBuffer};

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
    fn parses_chat_stream_chunk() {
        let chunk: ChatApiResponse =
            parse_ndjson_line(r#"{"message":{"role":"assistant","content":"hello"},"done":false}"#)
                .expect("chat chunk parses");
        assert_eq!(chunk.message.expect("message exists").content, "hello");
        assert!(!chunk.done);
    }
}
