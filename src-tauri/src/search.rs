use std::{
    cmp::Ordering,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    tools::{RiskLevel, Tool, ToolContext, ToolSchema},
    workspace::{relative_workspace_path, resolve_workspace_path},
};

const DEFAULT_OLLAMA_ENDPOINT: &str = "http://localhost:11434";
const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";
const INDEX_DIRECTORY: &str = ".legion";
const INDEX_FILE: &str = "semantic-index.json";
const MAX_FILES_SCANNED: usize = 20_000;
const MAX_INDEX_FILES: usize = 5_000;
const MAX_FILE_BYTES: u64 = 1_000_000;
const MAX_RESULTS: usize = 100;
const MAX_GREP_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_CONTEXT_LINES: usize = 5;
const MAX_LINE_CHARACTERS: usize = 1_000;
const MAX_GLOB_OUTPUT_BYTES: usize = 32 * 1024;
const CHUNK_LINE_LIMIT: usize = 80;
const CHUNK_CHARACTER_LIMIT: usize = 6_000;
const EMBEDDING_BATCH_SIZE: usize = 32;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct ContextLine {
    line_number: usize,
    content: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct GrepMatch {
    path: String,
    line_number: usize,
    line: String,
    context: Vec<ContextLine>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct FileChunk {
    path: String,
    start_line: usize,
    end_line: usize,
    content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EmbeddedChunk {
    path: String,
    start_line: usize,
    end_line: usize,
    content: String,
    embedding: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SemanticIndex {
    version: u32,
    model: String,
    chunks: Vec<EmbeddedChunk>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

pub struct GrepSearch;
pub struct GlobSearch;
pub struct IndexWorkspace;
pub struct SemanticSearch;

#[async_trait]
impl Tool for GrepSearch {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "grep_search".to_owned(),
            description: "Search workspace file contents using a literal string or regular expression. Returns matching paths, line numbers, and bounded surrounding context.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Text or regular expression to find." },
                    "is_regex": { "type": "boolean", "description": "Treat pattern as a regular expression; defaults to false." },
                    "glob": { "type": "string", "description": "Optional file glob, such as **/*.rs." },
                    "context_lines": { "type": "integer", "description": "Surrounding lines per match, from 0 to 5; defaults to 2." },
                    "max_results": { "type": "integer", "description": "Maximum matches to return; capped at 100." }
                },
                "required": ["pattern"]
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let pattern = required_string(&arguments, "pattern")?;
        if pattern.is_empty() {
            return Err("Search pattern cannot be empty".to_owned());
        }
        let is_regex = optional_bool(&arguments, "is_regex", false)?;
        let expression = if is_regex {
            Regex::new(pattern).map_err(|error| format!("Invalid search regex: {error}"))?
        } else {
            Regex::new(&regex::escape(pattern)).expect("escaped literal is a valid regex")
        };
        let matcher = optional_glob(&arguments)?;
        let context_lines = optional_usize(&arguments, "context_lines", 2)?.min(MAX_CONTEXT_LINES);
        let max_results = optional_usize(&arguments, "max_results", 50)?.clamp(1, MAX_RESULTS);
        let root = resolve_workspace_path(&context.workspace, Path::new("."), true)?;
        let paths = collect_files(&root, matcher.as_ref(), MAX_FILES_SCANNED)?;
        let mut matches = Vec::new();
        let mut skipped_large_files = 0;
        let mut truncated = false;

        'files: for path in paths {
            let metadata = fs::metadata(&path)
                .map_err(|error| format!("Could not inspect {}: {error}", path.display()))?;
            if metadata.len() > MAX_FILE_BYTES {
                skipped_large_files += 1;
                continue;
            }
            let content = match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => continue,
                Err(error) => return Err(format!("Could not read {}: {error}", path.display())),
            };
            let lines = content.lines().collect::<Vec<_>>();
            for (index, line) in lines.iter().enumerate() {
                if expression.is_match(line) {
                    if matches.len() == max_results {
                        truncated = true;
                        break 'files;
                    }
                    let start = index.saturating_sub(context_lines);
                    let end = (index + context_lines + 1).min(lines.len());
                    matches.push(GrepMatch {
                        path: relative_path(&root, &path)?,
                        line_number: index + 1,
                        line: bounded_line(line),
                        context: (start..end)
                            .filter(|line_index| *line_index != index)
                            .map(|line_index| ContextLine {
                                line_number: line_index + 1,
                                content: bounded_line(lines[line_index]),
                            })
                            .collect(),
                    });
                }
            }
        }

        Ok(format_grep_results(
            matches,
            truncated,
            max_results,
            skipped_large_files,
        ))
    }
}

#[async_trait]
impl Tool for GlobSearch {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "glob_search".to_owned(),
            description: "Find workspace files whose relative paths match a glob pattern. Hidden and generated directories are excluded, and results are bounded.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Workspace-relative glob, for example src/**/*.rs." },
                    "max_results": { "type": "integer", "description": "Maximum paths to return; capped at 100." }
                },
                "required": ["pattern"]
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let pattern = required_string(&arguments, "pattern")?;
        let matcher = Glob::new(pattern)
            .map_err(|error| format!("Invalid file glob: {error}"))?
            .compile_matcher();
        let max_results = optional_usize(&arguments, "max_results", 100)?.clamp(1, MAX_RESULTS);
        let root = resolve_workspace_path(&context.workspace, Path::new("."), true)?;
        let paths = collect_files(&root, Some(&matcher), MAX_FILES_SCANNED)?;
        let mut results = Vec::new();
        let mut truncated = false;
        for path in paths {
            if results.len() == max_results {
                truncated = true;
                break;
            }
            results.push(relative_path(&root, &path)?);
        }
        Ok(format_glob_results(results, truncated, max_results))
    }
}

#[async_trait]
impl Tool for IndexWorkspace {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "index_workspace".to_owned(),
            description: "Build or refresh the local semantic code-search index on demand using Ollama embeddings. Returns an unavailable status if Ollama or the embedding model cannot be reached.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Optional Ollama embedding model; defaults to LEGION_EMBEDDING_MODEL or nomic-embed-text." }
                },
                "required": []
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let model = embedding_model(&arguments)?;
        let endpoint = ollama_endpoint();
        let root = resolve_workspace_path(&context.workspace, Path::new("."), true)?;
        let paths = collect_files(&root, None, MAX_INDEX_FILES)?;
        let mut chunks = Vec::new();
        let mut skipped_large_files = 0;
        for path in paths {
            let metadata = fs::metadata(&path)
                .map_err(|error| format!("Could not inspect {}: {error}", path.display()))?;
            if metadata.len() > MAX_FILE_BYTES {
                skipped_large_files += 1;
                continue;
            }
            let content = match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => continue,
                Err(error) => return Err(format!("Could not read {}: {error}", path.display())),
            };
            let path = relative_path(&root, &path)?;
            chunks.extend(chunk_file(&path, &content));
            if chunks.len() > MAX_FILES_SCANNED {
                return Err(format!(
                    "Workspace has more than {MAX_FILES_SCANNED} indexable chunks; narrow the workspace before indexing"
                ));
            }
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|error| format!("Could not configure Ollama client: {error}"))?;
        let mut embedded = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
            let texts = batch
                .iter()
                .map(|chunk| chunk.content.clone())
                .collect::<Vec<_>>();
            let Some(vectors) = embed_texts(&client, &endpoint, &model, &texts).await? else {
                return Ok(json!({
                    "status": "unavailable",
                    "message": format!("Ollama or embedding model '{model}' is unavailable; no index was written."),
                    "model": model
                }));
            };
            for (chunk, embedding) in batch.iter().zip(vectors) {
                embedded.push(EmbeddedChunk {
                    path: chunk.path.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    content: chunk.content.clone(),
                    embedding,
                });
            }
        }

        let index = SemanticIndex {
            version: 1,
            model: model.clone(),
            chunks: embedded,
        };
        save_index(&root, &index)?;
        Ok(json!({
            "status": "indexed",
            "model": model,
            "chunks": index.chunks.len(),
            "skipped_large_files": skipped_large_files,
            "index_path": format!("{INDEX_DIRECTORY}/{INDEX_FILE}")
        }))
    }
}

#[async_trait]
impl Tool for SemanticSearch {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "semantic_search".to_owned(),
            description: "Find semantically relevant code chunks using the local Ollama embeddings index. Call index_workspace to build or refresh the index first.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Concept or question to search for." },
                    "top_n": { "type": "integer", "description": "Number of chunks to return; capped at 10." },
                    "model": { "type": "string", "description": "Optional Ollama embedding model; must match the indexed model." }
                },
                "required": ["query"]
            }),
            risk_level: RiskLevel::AutoApprove,
        }
    }

    async fn call(&self, arguments: Value, context: ToolContext) -> Result<Value, String> {
        let query = required_string(&arguments, "query")?;
        if query.trim().is_empty() {
            return Err("Semantic query cannot be empty".to_owned());
        }
        let top_n = optional_usize(&arguments, "top_n", 5)?.clamp(1, 10);
        let model = embedding_model(&arguments)?;
        let root = resolve_workspace_path(&context.workspace, Path::new("."), true)?;
        let index_path = index_path(&root, false)?;
        if !index_path.exists() {
            return Ok(json!({
                "status": "not_indexed",
                "message": "No semantic index exists. Call index_workspace to build one."
            }));
        }
        let index_content = fs::read_to_string(&index_path)
            .map_err(|error| format!("Could not read semantic index: {error}"))?;
        let index: SemanticIndex = serde_json::from_str(&index_content)
            .map_err(|error| format!("Could not parse semantic index: {error}"))?;
        if index.version != 1 {
            return Err(format!(
                "Unsupported semantic index version: {}",
                index.version
            ));
        }
        if index.model != model {
            return Ok(json!({
                "status": "reindex_required",
                "message": format!("The index uses '{}', but this search uses '{model}'. Call index_workspace with the desired model.", index.model)
            }));
        }
        if index.chunks.is_empty() {
            return Ok(json!({ "status": "empty_index", "results": [] }));
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|error| format!("Could not configure Ollama client: {error}"))?;
        let Some(mut vectors) =
            embed_texts(&client, &ollama_endpoint(), &model, &[query.to_owned()]).await?
        else {
            return Ok(json!({
                "status": "unavailable",
                "message": format!("Ollama or embedding model '{model}' is unavailable.")
            }));
        };
        let query_vector = vectors
            .pop()
            .ok_or_else(|| "Ollama returned no query embedding".to_owned())?;
        let ranked = rank_chunks(&query_vector, &index.chunks, top_n);
        let mut results = Vec::new();
        for (chunk_index, score) in ranked {
            let chunk = &index.chunks[chunk_index];
            let result = json!({
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": score,
                "content": chunk.content
            });
            let mut candidate = results.clone();
            candidate.push(result.clone());
            if serde_json::to_vec(&json!({ "status": "ok", "results": candidate }))
                .map_err(|error| format!("Could not encode semantic results: {error}"))?
                .len()
                > MAX_GREP_OUTPUT_BYTES
            {
                break;
            }
            results.push(result);
        }
        Ok(json!({ "status": "ok", "results": results }))
    }
}

fn required_string<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Tool argument '{key}' must be a string"))
}

fn optional_bool(arguments: &Value, key: &str, default: bool) -> Result<bool, String> {
    match arguments.get(key) {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("Tool argument '{key}' must be a boolean")),
    }
}

fn optional_usize(arguments: &Value, key: &str, default: usize) -> Result<usize, String> {
    match arguments.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| format!("Tool argument '{key}' must be a non-negative integer")),
    }
}

fn optional_glob(arguments: &Value) -> Result<Option<GlobMatcher>, String> {
    arguments
        .get("glob")
        .map(|value| {
            let pattern = value
                .as_str()
                .ok_or_else(|| "Tool argument 'glob' must be a string".to_owned())?;
            Glob::new(pattern)
                .map(|glob| Some(glob.compile_matcher()))
                .map_err(|error| format!("Invalid file glob: {error}"))
        })
        .transpose()
        .map(Option::flatten)
}

fn collect_files(
    root: &Path,
    matcher: Option<&GlobMatcher>,
    max_files: usize,
) -> Result<Vec<PathBuf>, String> {
    let mut pending = vec![root.to_owned()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("Could not list {}: {error}", directory.display()))?;
        let mut entries = entries
            .map(|entry| entry.map_err(|error| format!("Could not read workspace entry: {error}")))
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("Could not inspect workspace entry: {error}"))?;
            if file_type.is_dir() {
                if !is_ignored_directory(&name) {
                    pending.push(entry.path());
                }
            } else if file_type.is_file() {
                let relative = relative_workspace_path(root, &entry.path())?;
                let path = root.join(&relative);
                let relative = relative.to_string_lossy().replace('\\', "/");
                let matches = match matcher {
                    Some(matcher) => matcher.is_match(&relative),
                    None => true,
                };
                if matches {
                    files.push(path);
                    if files.len() > max_files {
                        return Err(format!(
                            "Workspace has more than {max_files} searchable files; narrow the workspace before searching"
                        ));
                    }
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_ignored_directory(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "dist" | "build" | "vendor"
        )
}

fn relative_path(root: &Path, path: &Path) -> Result<String, String> {
    let relative = relative_workspace_path(root, path)?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn bounded_line(line: &str) -> String {
    let mut result = line.chars().take(MAX_LINE_CHARACTERS).collect::<String>();
    if line.chars().count() > MAX_LINE_CHARACTERS {
        result.push_str("...[line truncated]");
    }
    result
}

fn format_grep_results(
    matches: Vec<GrepMatch>,
    truncated: bool,
    max_results: usize,
    skipped_large_files: usize,
) -> Value {
    let mut output = Vec::new();
    let mut output_truncated = truncated;
    for item in matches.into_iter().take(max_results) {
        output.push(item);
        let candidate = json!({
            "matches": output,
            "truncated": output_truncated,
            "max_results": max_results,
            "skipped_large_files": skipped_large_files
        });
        if serde_json::to_vec(&candidate)
            .map(|encoded| encoded.len() > MAX_GREP_OUTPUT_BYTES)
            .unwrap_or(true)
        {
            output.pop();
            output_truncated = true;
            break;
        }
    }
    json!({
        "matches": output,
        "truncated": output_truncated,
        "max_results": max_results,
        "skipped_large_files": skipped_large_files
    })
}

fn format_glob_results(paths: Vec<String>, truncated: bool, max_results: usize) -> Value {
    let mut output = Vec::new();
    let mut output_truncated = truncated;
    for path in paths.into_iter().take(max_results) {
        output.push(path);
        let candidate = json!({
            "paths": output,
            "truncated": output_truncated,
            "max_results": max_results
        });
        if serde_json::to_vec(&candidate)
            .map(|encoded| encoded.len() > MAX_GLOB_OUTPUT_BYTES)
            .unwrap_or(true)
        {
            output.pop();
            output_truncated = true;
            break;
        }
    }
    json!({
        "paths": output,
        "truncated": output_truncated,
        "max_results": max_results
    })
}

fn chunk_file(path: &str, content: &str) -> Vec<FileChunk> {
    let mut chunks = Vec::new();
    let mut lines = Vec::new();
    let mut characters = 0_usize;
    let mut start_line = 1;
    for (index, line) in content.lines().enumerate() {
        let line = line.chars().take(CHUNK_CHARACTER_LIMIT).collect::<String>();
        let line_size = line.chars().count();
        if !lines.is_empty()
            && (lines.len() >= CHUNK_LINE_LIMIT
                || characters.saturating_add(line_size) > CHUNK_CHARACTER_LIMIT)
        {
            chunks.push(FileChunk {
                path: path.to_owned(),
                start_line,
                end_line: index,
                content: lines.join("\n"),
            });
            lines.clear();
            characters = 0;
            start_line = index + 1;
        }
        characters += line_size;
        lines.push(line);
    }
    if !lines.is_empty() {
        chunks.push(FileChunk {
            path: path.to_owned(),
            start_line,
            end_line: content.lines().count(),
            content: lines.join("\n"),
        });
    }
    chunks
}

fn embedding_model(arguments: &Value) -> Result<String, String> {
    let model = match arguments.get("model") {
        Some(value) => value
            .as_str()
            .ok_or_else(|| "Tool argument 'model' must be a string".to_owned())?
            .to_owned(),
        None => std::env::var("LEGION_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL.to_owned()),
    };
    if model.trim().is_empty() {
        return Err("Embedding model name cannot be empty".to_owned());
    }
    Ok(model)
}

fn ollama_endpoint() -> String {
    std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| DEFAULT_OLLAMA_ENDPOINT.to_owned())
        .trim_end_matches('/')
        .to_owned()
}

async fn embed_texts(
    client: &Client,
    endpoint: &str,
    model: &str,
    texts: &[String],
) -> Result<Option<Vec<Vec<f32>>>, String> {
    let response = match client
        .post(format!("{endpoint}/api/embed"))
        .json(&json!({ "model": model, "input": texts }))
        .send()
        .await
    {
        Ok(response) => response,
        Err(_) => return Ok(None),
    };
    if !response.status().is_success() {
        return Ok(None);
    }
    let response = response
        .json::<EmbeddingResponse>()
        .await
        .map_err(|error| format!("Could not parse Ollama embedding response: {error}"))?;
    if response.embeddings.len() != texts.len()
        || response.embeddings.iter().any(|embedding| {
            embedding.is_empty() || embedding.iter().any(|value| !value.is_finite())
        })
    {
        return Err("Ollama returned invalid or incomplete embeddings".to_owned());
    }
    Ok(Some(response.embeddings))
}

fn save_index(root: &Path, index: &SemanticIndex) -> Result<(), String> {
    let path = index_path(root, true)?;
    let temporary_relative_path =
        Path::new(INDEX_DIRECTORY).join(format!("{INDEX_FILE}.{}.tmp", std::process::id()));
    let temporary_path = resolve_workspace_path(root, &temporary_relative_path, false)?;
    let serialized = serde_json::to_vec(index)
        .map_err(|error| format!("Could not encode semantic index: {error}"))?;
    let mut temporary_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .map_err(|error| format!("Could not create temporary semantic index: {error}"))?;
    temporary_file
        .write_all(&serialized)
        .map_err(|error| format!("Could not write temporary semantic index: {error}"))?;
    drop(temporary_file);
    fs::rename(&temporary_path, &path)
        .map_err(|error| format!("Could not replace semantic index: {error}"))
}

fn index_path(root: &Path, create: bool) -> Result<PathBuf, String> {
    let directory = resolve_workspace_path(root, Path::new(INDEX_DIRECTORY), false)?;
    if create && !directory.exists() {
        fs::create_dir(&directory)
            .map_err(|error| format!("Could not create semantic index folder: {error}"))?;
    }
    if !directory.exists() {
        return resolve_workspace_path(root, &Path::new(INDEX_DIRECTORY).join(INDEX_FILE), false);
    }
    let resolved_directory = resolve_workspace_path(root, Path::new(INDEX_DIRECTORY), true)?;
    if !resolved_directory.is_dir() {
        return Err("Semantic index folder is not a directory".to_owned());
    }
    let path = resolve_workspace_path(root, &Path::new(INDEX_DIRECTORY).join(INDEX_FILE), false)?;
    if path.exists() {
        let resolved_path =
            resolve_workspace_path(root, &Path::new(INDEX_DIRECTORY).join(INDEX_FILE), true)?;
        if !resolved_path.is_file() {
            return Err("Semantic index path is not a file".to_owned());
        }
        return Ok(resolved_path);
    }
    Ok(path)
}

fn rank_chunks(query: &[f32], chunks: &[EmbeddedChunk], top_n: usize) -> Vec<(usize, f32)> {
    let mut ranked = chunks
        .iter()
        .enumerate()
        .filter_map(|(index, chunk)| {
            cosine_similarity(query, &chunk.embedding).map(|score| (index, score))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_n);
    ranked
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    let score = dot / (left_norm * right_norm);
    (left_norm > 0.0 && right_norm > 0.0 && score.is_finite()).then_some(score)
}

#[cfg(test)]
mod tests {
    use super::{
        cosine_similarity, format_glob_results, format_grep_results, rank_chunks, ContextLine,
        EmbeddedChunk, GlobSearch, GrepMatch, GrepSearch, MAX_GLOB_OUTPUT_BYTES,
        MAX_GREP_OUTPUT_BYTES,
    };
    use crate::tools::{RiskLevel, Tool, ToolContext};
    use serde_json::{json, Value};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temporary_workspace() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("legion-search-{id}"));
        fs::create_dir_all(&path).expect("temporary workspace is created");
        path
    }

    fn context(workspace: PathBuf) -> ToolContext {
        ToolContext {
            workspace,
            request_id: "test-request".to_owned(),
            command_id: "test-command".to_owned(),
            command_output: None,
        }
    }

    fn chunk(path: &str, embedding: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk {
            path: path.to_owned(),
            start_line: 1,
            end_line: 3,
            content: "example".to_owned(),
            embedding,
        }
    }

    #[test]
    fn formats_grep_context_and_respects_result_limit() {
        let matches = vec![
            GrepMatch {
                path: "src/lib.rs".to_owned(),
                line_number: 4,
                line: "needle".to_owned(),
                context: vec![ContextLine {
                    line_number: 3,
                    content: "before".to_owned(),
                }],
            },
            GrepMatch {
                path: "src/main.rs".to_owned(),
                line_number: 7,
                line: "needle again".to_owned(),
                context: Vec::new(),
            },
        ];
        let result = format_grep_results(matches, true, 1, 0);
        assert_eq!(result["matches"].as_array().unwrap().len(), 1);
        assert_eq!(result["matches"][0]["line_number"], 4);
        assert_eq!(result["matches"][0]["context"][0]["content"], "before");
        assert_eq!(result["truncated"], true);
    }

    #[test]
    fn truncates_grep_output_by_serialized_size() {
        let matches = (0..100)
            .map(|index| GrepMatch {
                path: format!("src/{index}.rs"),
                line_number: index + 1,
                line: "x".repeat(1_000),
                context: Vec::new(),
            })
            .collect();
        let result = format_grep_results(matches, false, 100, 0);
        assert!(serde_json::to_vec(&result).unwrap().len() <= MAX_GREP_OUTPUT_BYTES);
        assert_eq!(result["truncated"], true);
    }

    #[test]
    fn formats_and_truncates_glob_results() {
        let result = format_glob_results(vec!["src/lib.rs".to_owned()], false, 1);
        assert_eq!(result["paths"][0], "src/lib.rs");
        assert_eq!(result["truncated"], false);

        let paths = (0..1_000)
            .map(|index| format!("src/{}.rs", "x".repeat(100) + &index.to_string()))
            .collect();
        let result = format_glob_results(paths, false, 1_000);
        assert!(serde_json::to_vec(&result).unwrap().len() <= MAX_GLOB_OUTPUT_BYTES);
        assert_eq!(result["truncated"], true);
    }

    #[test]
    fn semantic_results_rank_by_cosine_similarity() {
        let chunks = vec![
            chunk("orthogonal.rs", vec![0.0, 1.0]),
            chunk("best.rs", vec![1.0, 0.0]),
            chunk("partial.rs", vec![0.8, 0.6]),
            chunk("wrong-dimension.rs", vec![1.0]),
        ];
        let results = rank_chunks(&[1.0, 0.0], &chunks, 3);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 2);
        assert_eq!(results[2].0, 0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn output_format_is_json_value() {
        let output = format_glob_results(vec!["README.md".to_owned()], false, 10);
        assert!(matches!(output, Value::Object(_)));
    }

    #[tokio::test]
    async fn grep_and_glob_search_match_workspace_files_only() {
        let workspace = temporary_workspace();
        fs::create_dir_all(workspace.join("src/nested")).expect("source folders are created");
        fs::create_dir_all(workspace.join(".hidden")).expect("hidden folder is created");
        fs::write(
            workspace.join("src/nested/example.rs"),
            "first line\nneedle value\nlast line\n",
        )
        .expect("source file is written");
        fs::write(workspace.join(".hidden/secret.rs"), "needle secret\n")
            .expect("hidden file is written");
        let tool_context = context(workspace.clone());

        let grep = GrepSearch
            .call(
                json!({
                    "pattern": "need.*value",
                    "is_regex": true,
                    "glob": "src/**/*.rs",
                    "context_lines": 1
                }),
                tool_context.clone(),
            )
            .await
            .expect("grep search succeeds");
        assert_eq!(grep["matches"].as_array().unwrap().len(), 1);
        assert_eq!(grep["matches"][0]["path"], "src/nested/example.rs");
        assert_eq!(grep["matches"][0]["line_number"], 2);
        assert_eq!(grep["matches"][0]["context"].as_array().unwrap().len(), 2);

        let glob = GlobSearch
            .call(json!({ "pattern": "**/*.rs" }), tool_context)
            .await
            .expect("glob search succeeds");
        assert_eq!(glob["paths"].as_array().unwrap().len(), 1);
        assert_eq!(glob["paths"][0], "src/nested/example.rs");
        assert_eq!(GrepSearch.schema().risk_level, RiskLevel::AutoApprove);
        assert_eq!(GlobSearch.schema().risk_level, RiskLevel::AutoApprove);
        fs::remove_dir_all(workspace).expect("temporary workspace is removed");
    }
}
