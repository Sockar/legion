# Legion

Legion is a desktop coding assistant inspired by GitHub Copilot. It is being built
to work with local and open-source language models, including models served by
[Ollama](https://ollama.com/).

The app includes a chat interface with streaming Markdown responses and code
copying, powered by the local [Ollama](https://ollama.com/) HTTP API. The app
detects the server at `http://localhost:11434`, lists installed models, streams
model-pull progress, and streams chat responses.

Chat sessions are stored locally and restored when the app restarts. Each
session keeps its own conversation and model selection and is associated with a
workspace folder chosen through the native folder picker. Session persistence is
behind a repository interface so the storage can be replaced with SQLite later.

Chat requests include registered function schemas. The Rust tool registry
dispatches Ollama tool calls, returns results (including failures) to the model,
and continues for up to eight tool iterations. Tools declare `auto_approve` or
`requires_confirmation`; the latter prompts the user in the app before running.
The initial read-only examples are `get_current_time` and
`list_workspace_files`; workspace-scoped `read_file`, `create_file`, and
`edit_file` tools are also available. File tools resolve paths under the active
session's workspace and reject paths that escape it. Creates fail for existing
files unless `overwrite` is explicitly true. Edits use an exact `old_str` to
`new_str` replacement and reject missing or ambiguous matches. File mutations
require approval and show a highlighted diff in the chat before writing.
The high-risk `run_command` tool also requires confirmation and displays the
exact command before execution. It runs in the active workspace, streams
stdout/stderr to the terminal panel, and stops after 60 seconds by default
(configurable up to 10 minutes). Commands are not sandboxed; broader command
sandboxing and restrictions are tracked separately in issue #12.
New tools implement the Rust `Tool` trait and can be registered on
`BackendState` before it is shared.

The agent also has read-only `grep_search` (literal or regular-expression
matching with bounded context), `glob_search`, and `semantic_search` tools. To
build or refresh the local semantic index, the agent can call
`index_workspace` on demand. The index is stored under `.legion/` in the active
workspace. Semantic search uses Ollama's `/api/embed` endpoint and defaults to
`nomic-embed-text`; set `LEGION_EMBEDDING_MODEL` to choose another embedding
model, or pass a `model` argument to either semantic tool. Set `OLLAMA_HOST` to
change the Ollama endpoint. If Ollama or the embedding model is unavailable,
indexing and semantic search return a status message without interrupting chat.
Search skips hidden/generated directories such as `.git`, `node_modules`,
`target`, and `.legion`, and all discovered files are checked against the
workspace boundary.

## Development

### Prerequisites

- Node.js 22 or newer and npm
- Rust stable
- The platform dependencies required by [Tauri 2](https://v2.tauri.app/start/prerequisites/)

### Commands

```sh
npm ci
npm run tauri dev
```

Ollama is only required at runtime. If it is not installed or running, the app
builds normally and the status panel reports that it is unreachable.

Check the frontend and Rust code with:

```sh
npm run lint
npm run format:check
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
```
