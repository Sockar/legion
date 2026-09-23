pub mod ollama;
pub mod tools;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(ollama::BackendState::default())
        .invoke_handler(tauri::generate_handler![
            ollama::ollama_status,
            ollama::ollama_list_models,
            ollama::ollama_pull_model,
            ollama::ollama_chat,
            ollama::ollama_cancel_chat,
            ollama::respond_tool_approval,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
