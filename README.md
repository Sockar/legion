# Legion

Legion is a desktop coding assistant inspired by GitHub Copilot. It is being built
to work with local and open-source language models, including models served by
[Ollama](https://ollama.com/).

The app includes a chat interface with streaming Markdown responses and code
copying, powered by the local [Ollama](https://ollama.com/) HTTP API. The app
detects the server at `http://localhost:11434`, lists installed models, streams
model-pull progress, and streams chat responses. Settings includes popular
recommended models to download and an advanced manual model-name option.
Model and installer downloads show byte and percentage progress, with success
and error feedback.

Chat sessions are stored locally and restored when the app restarts. Each
session keeps its own conversation, model selection, sampling parameters, and
system prompt, and is associated with a workspace folder chosen through the
native folder picker. The Settings panel also stores the Ollama base URL and can
test the connection. These preferences share the session repository and persist
across restarts; older saved sessions are restored with the default settings.
Sessions, messages, tool-call history, and settings are stored in a versioned
SQLite database in the app's local data directory. Existing browser-stored
session data is imported on first launch after upgrading.

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
exact command before execution. A first-layer deny-list in
`src-tauri/src/security.rs` rejects selected destructive command patterns before
they can be approved. It runs in the active workspace, streams stdout/stderr to
the terminal panel, and stops after 60 seconds by default (configurable up to 10
minutes). The Settings panel includes a per-session strict mode that requires
approval for every tool call, including read-only tools. Tool calls and their
approval/execution outcomes are appended to the local SQLite audit log and
recent activity can be reviewed in Settings.

These controls are defense in depth, not a full OS-level sandbox. The command
deny-list is intentionally incomplete and can be bypassed by unlisted commands,
scripts, or command interpreters; approved shell commands can still access
files outside the workspace. Workspace path validation canonicalizes paths and
checks symlinks, but cannot eliminate filesystem race conditions. Use strict
mode when you want to review every tool invocation, and do not run Legion with
privileges you would not grant to the agent.
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

Ollama is only required at runtime. If it is not installed or running, Legion
reports that it is unreachable and, by default, asks before downloading and
installing the official Ollama release. Turn off **Ask to download and install
Ollama when it is unavailable** in Settings to disable these prompts.

### Ollama installation

The installer uses the latest assets published by
[Ollama on GitHub](https://github.com/ollama/ollama/releases): `OllamaSetup.exe`
on 64-bit Windows, `Ollama-darwin.zip` on macOS, and the official
`ollama-linux-{amd64,arm64}.tar.zst` archive on Linux. Downloads are made over
HTTPS, checked for an expected content type and size, and verified against the
SHA-256 digest in the official GitHub release API when it is published. If a
release does not provide a digest, Legion reports that verification was limited
to HTTPS and content-type/size checks. Installer sizes vary by release (roughly
200 MB on macOS and 1.5 GB on Windows or Linux); language models are separate
downloads.

Windows setup opens interactively because Ollama does not document a reliable
silent-install option. On macOS, the verified app bundle is installed in the
user's `Applications` folder. On Linux, Legion installs the verified official
archive under its app-data directory and starts its `ollama serve` process; it
does not invoke `sudo` or install a system service. This user-scoped Linux
approach avoids an implicit privilege escalation. If the install or server
startup fails, the app reports the error and offers the manual download page.

Check the frontend and Rust code with:

```sh
npm test
npm run lint
npm run build
npm run format:check
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
```

See [docs/RELEASING.md](docs/RELEASING.md) for release and updater setup.
