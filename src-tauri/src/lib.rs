pub mod ollama;
pub mod ollama_installer;
pub mod persistence;
pub mod search;
pub mod security;
pub mod tools;
pub mod workspace;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
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
            ollama_installer::install_ollama,
            ollama_installer::ollama_cancel_install,
            ollama::ollama_list_models,
            ollama::ollama_pull_model,
            ollama::ollama_cancel_pull,
            ollama::ollama_chat,
            ollama::ollama_cancel_chat,
            ollama::respond_tool_approval,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
