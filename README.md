# Legion

Legion is a desktop coding assistant inspired by GitHub Copilot. It is being built
to work with local and open-source language models, including models served by
[Ollama](https://ollama.com/).

The app includes a chat interface with streaming Markdown responses and code
copying. The chat currently uses an in-memory mock backend; model management and
the production assistant integration are still under development.

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

Check the frontend and Rust code with:

```sh
npm run lint
npm run format:check
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
```
