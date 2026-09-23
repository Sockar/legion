# Legion

Legion is a desktop coding assistant inspired by GitHub Copilot. It is being built
to work with local and open-source language models, including models served by
[Ollama](https://ollama.com/).

This repository currently contains the cross-platform desktop app scaffold. Chat,
model management, and coding-assistant features will be added separately.

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
