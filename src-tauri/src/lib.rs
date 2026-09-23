pub mod ollama;
pub mod persistence;
pub mod search;
pub mod tools;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(ollama::BackendState::default())
        .setup(|app| {
            let persistence =
                persistence::initialize(app.handle()).map_err(std::io::Error::other)?;
            app.manage(persistence);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            persistence::load_session_state,
            persistence::save_session_state,
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
