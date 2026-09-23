# Legion

Legion is a desktop coding assistant inspired by GitHub Copilot. It is being built
to work with local and open-source language models, including models served by
[Ollama](https://ollama.com/).

The app includes a chat interface with streaming Markdown responses and code
copying, powered by the local [Ollama](https://ollama.com/) HTTP API. The app
detects the server at `http://localhost:11434`, lists installed models, streams
model-pull progress, and streams chat responses.

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
