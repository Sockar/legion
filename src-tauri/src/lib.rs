pub mod ollama;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ollama::BackendState::default())
        .invoke_handler(tauri::generate_handler![
            ollama::ollama_status,
            ollama::ollama_list_models,
            ollama::ollama_pull_model,
            ollama::ollama_chat,
            ollama::ollama_cancel_chat,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
