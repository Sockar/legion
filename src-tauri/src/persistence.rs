use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Manager, State};

const MIGRATIONS: &[(i64, &str)] = &[
    (
        1,
        "CREATE TABLE sessions (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        workspace_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        archived_at TEXT
    );
    CREATE TABLE messages (
        id TEXT PRIMARY KEY NOT NULL,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        tool_call_data TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE session_settings (
        session_id TEXT PRIMARY KEY NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        temperature REAL NOT NULL,
        top_p REAL NOT NULL,
        num_ctx INTEGER NOT NULL,
        system_prompt TEXT NOT NULL
    );
    CREATE TABLE app_settings (
        key TEXT PRIMARY KEY NOT NULL,
        value TEXT NOT NULL
    );
    CREATE TABLE tool_call_history (
        id TEXT PRIMARY KEY NOT NULL,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        tool_name TEXT NOT NULL,
        arguments_json TEXT NOT NULL,
        result_json TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX messages_session_id_idx ON messages(session_id);
    CREATE INDEX tool_call_history_session_id_idx ON tool_call_history(session_id);",
    ),
    (
        2,
        "ALTER TABLE session_settings
        ADD COLUMN strict_mode INTEGER NOT NULL DEFAULT 0;
     CREATE TABLE tool_audit_log (
        id TEXT PRIMARY KEY NOT NULL,
        session_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        arguments_summary TEXT NOT NULL,
        approval_status TEXT NOT NULL,
        execution_status TEXT NOT NULL,
        created_at TEXT NOT NULL
     );
     CREATE INDEX tool_audit_log_session_id_idx
        ON tool_audit_log(session_id, created_at);",
    ),
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSettings {
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_num_ctx")]
    pub num_ctx: i64,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub strict_mode: bool,
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            num_ctx: default_num_ctx(),
            system_prompt: String::new(),
            strict_mode: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub tool_call_data: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallHistory {
    pub id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub result: Value,
    pub status: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolAuditRecord {
    pub id: String,
    pub session_id: String,
    pub tool_name: String,
    pub arguments_summary: String,
    pub approval_status: String,
    pub execution_status: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSession {
    pub id: String,
    pub name: String,
    pub workspace_path: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub settings: ChatSettings,
    #[serde(default)]
    pub archived_at: Option<String>,
    #[serde(default)]
    pub tool_call_history: Vec<ToolCallHistory>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistedSessionState {
    #[serde(default)]
    pub sessions: Vec<ChatSession>,
    #[serde(default)]
    pub active_session_id: Option<String>,
    #[serde(default = "default_ollama_endpoint")]
    pub ollama_endpoint: String,
    #[serde(default = "default_auto_install_ollama")]
    pub auto_install_ollama: bool,
    #[serde(default)]
    pub tool_audit_log: Vec<ToolAuditRecord>,
}

impl Default for PersistedSessionState {
    fn default() -> Self {
        Self {
            sessions: Vec::new(),
            active_session_id: None,
            ollama_endpoint: default_ollama_endpoint(),
            auto_install_ollama: default_auto_install_ollama(),
            tool_audit_log: Vec::new(),
        }
    }
}

pub struct Database {
    connection: Connection,
}

pub struct PersistenceState(pub Arc<Mutex<Database>>);

impl Database {
    pub fn open(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let mut connection = Connection::open(path)?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        migrate(&mut connection)?;
        Ok(Self { connection })
    }

    pub fn load(&mut self, legacy_json: Option<&str>) -> Result<PersistedSessionState, String> {
        let session_count: i64 = self
            .connection
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .map_err(storage_error)?;
        if session_count == 0 {
            if let Some(legacy_json) = legacy_json {
                let imported: PersistedSessionState = serde_json::from_str(legacy_json)
                    .map_err(|error| format!("Could not parse legacy session data: {error}"))?;
                validate_state(&imported)?;
                self.save(&imported)?;
            }
        }
        self.load_from_database()
    }

    pub fn save(&mut self, state: &PersistedSessionState) -> Result<(), String> {
        validate_state(state)?;
        let transaction = self.connection.transaction().map_err(storage_error)?;
        write_state(&transaction, state)?;
        transaction.commit().map_err(storage_error)
    }

    pub fn append_tool_audit_log(&mut self, record: &ToolAuditRecord) -> Result<(), String> {
        self.connection
            .execute(
                "INSERT INTO tool_audit_log
                 (id, session_id, tool_name, arguments_summary, approval_status,
                  execution_status, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    record.id,
                    record.session_id,
                    record.tool_name,
                    record.arguments_summary,
                    record.approval_status,
                    record.execution_status,
                    record.created_at,
                ],
            )
            .map_err(storage_error)?;
        Ok(())
    }

    fn load_from_database(&self) -> Result<PersistedSessionState, String> {
        let endpoint = self
            .connection
            .query_row(
                "SELECT value FROM app_settings WHERE key = 'ollama_endpoint'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(storage_error)?
            .unwrap_or_else(default_ollama_endpoint);
        let auto_install_ollama = self
            .connection
            .query_row(
                "SELECT value FROM app_settings WHERE key = 'auto_install_ollama'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(storage_error)?
            .map_or_else(
                || Ok(default_auto_install_ollama()),
                |value| {
                    value.parse::<bool>().map_err(|error| {
                        format!("SQLite persistence error: invalid auto-install setting: {error}")
                    })
                },
            )?;
        let active_session_id = self
            .connection
            .query_row(
                "SELECT value FROM app_settings WHERE key = 'active_session_id'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(storage_error)?;
        let mut sessions = Vec::new();
        let mut session_statement = self
            .connection
            .prepare(
                "SELECT id, name, workspace_path, created_at, updated_at, model, archived_at
             FROM sessions ORDER BY updated_at DESC, rowid DESC",
            )
            .map_err(storage_error)?;
        let rows = session_statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            })
            .map_err(storage_error)?;
        for row in rows {
            let (id, name, workspace_path, created_at, updated_at, model, archived_at) =
                row.map_err(storage_error)?;
            let settings = self
                .connection
                .query_row(
                    "SELECT temperature, top_p, num_ctx, system_prompt, strict_mode
                     FROM session_settings WHERE session_id = ?1",
                    [&id],
                    |row| {
                        Ok(ChatSettings {
                            temperature: row.get(0)?,
                            top_p: row.get(1)?,
                            num_ctx: row.get(2)?,
                            system_prompt: row.get(3)?,
                            strict_mode: row.get(4)?,
                        })
                    },
                )
                .optional()
                .map_err(storage_error)?
                .unwrap_or_default();
            let mut messages = Vec::new();
            let mut message_statement = self
                .connection
                .prepare(
                    "SELECT id, role, content, created_at, tool_call_data
                 FROM messages WHERE session_id = ?1 ORDER BY rowid",
                )
                .map_err(storage_error)?;
            let message_rows = message_statement
                .query_map([&id], |row| {
                    let tool_call_data = row
                        .get::<_, Option<String>>(4)?
                        .map(|encoded| serde_json::from_str(&encoded))
                        .transpose()
                        .map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                4,
                                rusqlite::types::Type::Text,
                                Box::new(error),
                            )
                        })?;
                    Ok(ChatMessage {
                        id: row.get(0)?,
                        role: row.get(1)?,
                        content: row.get(2)?,
                        created_at: row.get(3)?,
                        tool_call_data,
                    })
                })
                .map_err(storage_error)?;
            for message in message_rows {
                messages.push(message.map_err(storage_error)?);
            }
            let mut tool_call_history = Vec::new();
            let mut tool_statement = self
                .connection
                .prepare(
                    "SELECT id, tool_name, arguments_json, result_json, status, created_at
                 FROM tool_call_history WHERE session_id = ?1 ORDER BY rowid",
                )
                .map_err(storage_error)?;
            let tool_rows = tool_statement
                .query_map([&id], |row| {
                    let arguments = decode_json_column(row, 2)?;
                    let result = decode_json_column(row, 3)?;
                    Ok(ToolCallHistory {
                        id: row.get(0)?,
                        tool_name: row.get(1)?,
                        arguments,
                        result,
                        status: row.get(4)?,
                        created_at: row.get(5)?,
                    })
                })
                .map_err(storage_error)?;
            for tool_call in tool_rows {
                tool_call_history.push(tool_call.map_err(storage_error)?);
            }
            sessions.push(ChatSession {
                id,
                name,
                workspace_path,
                created_at,
                updated_at,
                messages,
                model,
                settings,
                archived_at,
                tool_call_history,
            });
        }

        let mut tool_audit_log = Vec::new();
        let mut audit_statement = self
            .connection
            .prepare(
                "SELECT id, session_id, tool_name, arguments_summary,
                        approval_status, execution_status, created_at
                 FROM tool_audit_log ORDER BY rowid",
            )
            .map_err(storage_error)?;
        let audit_rows = audit_statement
            .query_map([], |row| {
                Ok(ToolAuditRecord {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    tool_name: row.get(2)?,
                    arguments_summary: row.get(3)?,
                    approval_status: row.get(4)?,
                    execution_status: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })
            .map_err(storage_error)?;
        for record in audit_rows {
            tool_audit_log.push(record.map_err(storage_error)?);
        }

        Ok(PersistedSessionState {
            sessions,
            active_session_id,
            ollama_endpoint: endpoint,
            auto_install_ollama,
            tool_audit_log,
        })
    }
}

fn migrate(connection: &mut Connection) -> rusqlite::Result<()> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
         INSERT INTO schema_version (version)
         SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM schema_version);",
    )?;
    let mut current_version: i64 =
        connection.query_row("SELECT version FROM schema_version", [], |row| row.get(0))?;
    for (version, sql) in MIGRATIONS {
        if *version > current_version {
            let transaction = connection.transaction()?;
            transaction.execute_batch(sql)?;
            transaction.execute("UPDATE schema_version SET version = ?1", [version])?;
            transaction.commit()?;
            current_version = *version;
        }
    }
    Ok(())
}

fn write_state(transaction: &Transaction<'_>, state: &PersistedSessionState) -> Result<(), String> {
    transaction
        .execute_batch(
            "DELETE FROM tool_call_history;
             DELETE FROM messages;
             DELETE FROM session_settings;
             DELETE FROM sessions;
             DELETE FROM app_settings;",
        )
        .map_err(storage_error)?;
    for session in &state.sessions {
        transaction
            .execute(
                "INSERT INTO sessions
                 (id, name, workspace_path, created_at, updated_at, model, archived_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    session.id,
                    session.name,
                    session.workspace_path,
                    session.created_at,
                    session.updated_at,
                    session.model,
                    session.archived_at,
                ],
            )
            .map_err(storage_error)?;
        transaction
            .execute(
                "INSERT INTO session_settings
                 (session_id, temperature, top_p, num_ctx, system_prompt, strict_mode)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    session.id,
                    session.settings.temperature,
                    session.settings.top_p,
                    session.settings.num_ctx,
                    session.settings.system_prompt,
                    session.settings.strict_mode,
                ],
            )
            .map_err(storage_error)?;
        for message in &session.messages {
            let tool_call_data = message
                .tool_call_data
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(storage_error)?;
            transaction
                .execute(
                    "INSERT INTO messages
                     (id, session_id, role, content, tool_call_data, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        message.id,
                        session.id,
                        message.role,
                        message.content,
                        tool_call_data,
                        nonempty_timestamp(&message.created_at),
                    ],
                )
                .map_err(storage_error)?;
        }
        for tool_call in &session.tool_call_history {
            let arguments = serde_json::to_string(&tool_call.arguments).map_err(storage_error)?;
            let result = serde_json::to_string(&tool_call.result).map_err(storage_error)?;
            transaction
                .execute(
                    "INSERT INTO tool_call_history
                     (id, session_id, tool_name, arguments_json, result_json, status, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        tool_call.id,
                        session.id,
                        tool_call.tool_name,
                        arguments,
                        result,
                        tool_call.status,
                        tool_call.created_at,
                    ],
                )
                .map_err(storage_error)?;
        }
    }
    transaction
        .execute(
            "INSERT INTO app_settings (key, value) VALUES ('ollama_endpoint', ?1)",
            [&state.ollama_endpoint],
        )
        .map_err(storage_error)?;
    transaction
        .execute(
            "INSERT INTO app_settings (key, value) VALUES ('auto_install_ollama', ?1)",
            [state.auto_install_ollama.to_string()],
        )
        .map_err(storage_error)?;
    if let Some(active_session_id) = &state.active_session_id {
        transaction
            .execute(
                "INSERT INTO app_settings (key, value) VALUES ('active_session_id', ?1)",
                [active_session_id],
            )
            .map_err(storage_error)?;
    }
    Ok(())
}

fn validate_state(state: &PersistedSessionState) -> Result<(), String> {
    let mut session_ids = std::collections::HashSet::new();
    for session in &state.sessions {
        if session.id.is_empty() || !session_ids.insert(&session.id) {
            return Err("Saved session data contains duplicate or empty session IDs.".to_owned());
        }
        let settings = &session.settings;
        if !settings.temperature.is_finite()
            || !(0.0..=2.0).contains(&settings.temperature)
            || !settings.top_p.is_finite()
            || !(0.0..=1.0).contains(&settings.top_p)
            || !(256..=131_072).contains(&settings.num_ctx)
        {
            return Err(format!(
                "Saved settings for session '{}' are invalid.",
                session.id
            ));
        }
        let mut message_ids = std::collections::HashSet::new();
        for message in &session.messages {
            if message.id.is_empty()
                || !message_ids.insert(&message.id)
                || (message.role != "user" && message.role != "assistant")
            {
                return Err(format!(
                    "Saved messages for session '{}' are invalid.",
                    session.id
                ));
            }
        }
    }
    if state
        .active_session_id
        .as_ref()
        .is_some_and(|id| !session_ids.contains(id))
    {
        return Err("Saved session data contains an invalid active session reference.".to_owned());
    }
    Ok(())
}

fn decode_json_column(row: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<Value> {
    let encoded: String = row.get(index)?;
    serde_json::from_str(&encoded).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}

fn nonempty_timestamp(timestamp: &str) -> String {
    if timestamp.is_empty() {
        SystemTime::now().duration_since(UNIX_EPOCH).map_or_else(
            |_| "0".to_owned(),
            |duration| duration.as_millis().to_string(),
        )
    } else {
        timestamp.to_owned()
    }
}

fn storage_error(error: impl std::fmt::Display) -> String {
    format!("SQLite persistence error: {error}")
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.9
}

fn default_num_ctx() -> i64 {
    4096
}

fn default_ollama_endpoint() -> String {
    "http://localhost:11434".to_owned()
}

fn default_auto_install_ollama() -> bool {
    true
}

#[tauri::command]
pub fn load_session_state(
    legacy_json: Option<String>,
    state: State<'_, PersistenceState>,
) -> Result<PersistedSessionState, String> {
    let mut database = state
        .0
        .lock()
        .map_err(|error| format!("Could not access SQLite persistence: {error}"))?;
    database.load(legacy_json.as_deref())
}

#[tauri::command]
pub fn save_session_state(
    session_state: PersistedSessionState,
    state: State<'_, PersistenceState>,
) -> Result<(), String> {
    let mut database = state
        .0
        .lock()
        .map_err(|error| format!("Could not access SQLite persistence: {error}"))?;
    database.save(&session_state)
}

pub fn initialize(app: &AppHandle) -> Result<PersistenceState, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Could not locate the app data directory: {error}"))?;
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("Could not create the app data directory: {error}"))?;
    let database = Database::open(directory.join("legion.sqlite3"))
        .map_err(|error| format!("Could not open the session database: {error}"))?;
    Ok(PersistenceState(Arc::new(Mutex::new(database))))
}

#[cfg(test)]
mod tests {
    use super::{migrate, ChatMessage, ChatSession, ChatSettings, Database, PersistedSessionState};
    use rusqlite::Connection;
    use serde_json::json;

    fn sample_state() -> PersistedSessionState {
        PersistedSessionState {
            sessions: vec![ChatSession {
                id: "session-1".to_owned(),
                name: "project".to_owned(),
                workspace_path: "C:\\work\\project".to_owned(),
                created_at: "2026-09-23T12:00:00.000Z".to_owned(),
                updated_at: "2026-09-23T12:01:00.000Z".to_owned(),
                messages: vec![ChatMessage {
                    id: "message-1".to_owned(),
                    role: "user".to_owned(),
                    content: "hello".to_owned(),
                    created_at: "2026-09-23T12:00:30.000Z".to_owned(),
                    tool_call_data: Some(json!({"tool_calls": []})),
                }],
                model: "llama3.2".to_owned(),
                settings: ChatSettings {
                    temperature: 0.4,
                    top_p: 0.8,
                    num_ctx: 8192,
                    system_prompt: "Be concise".to_owned(),
                    strict_mode: true,
                },
                archived_at: None,
                tool_call_history: vec![super::ToolCallHistory {
                    id: "request-0-0".to_owned(),
                    tool_name: "read_file".to_owned(),
                    arguments: json!({"path": "README.md"}),
                    result: json!({"content": "hello"}),
                    status: "succeeded".to_owned(),
                    created_at: "2026-09-23T12:00:45.000Z".to_owned(),
                }],
            }],
            active_session_id: Some("session-1".to_owned()),
            ollama_endpoint: "http://localhost:11434".to_owned(),
            auto_install_ollama: false,
            tool_audit_log: Vec::new(),
        }
    }

    #[test]
    fn migrations_create_and_upgrade_schema_once() {
        let mut connection = Connection::open_in_memory().expect("in-memory db opens");
        migrate(&mut connection).expect("first migration succeeds");
        migrate(&mut connection).expect("re-running migrations succeeds");
        let version: i64 = connection
            .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
            .expect("schema version exists");
        let table_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'
                 AND name IN ('sessions', 'messages', 'session_settings',
                              'app_settings', 'tool_call_history', 'tool_audit_log')",
                [],
                |row| row.get(0),
            )
            .expect("schema tables can be counted");
        assert_eq!(version, 2);
        assert_eq!(table_count, 6);
    }

    #[test]
    fn repository_round_trips_and_replaces_sessions_messages_and_settings() {
        let mut database = Database {
            connection: Connection::open_in_memory().expect("in-memory db opens"),
        };
        database
            .connection
            .pragma_update(None, "foreign_keys", "ON")
            .expect("foreign keys enable");
        migrate(&mut database.connection).expect("schema migrates");
        let state = sample_state();
        database.save(&state).expect("state saves");
        assert_eq!(database.load(None).expect("state loads"), state);

        let mut updated = state;
        updated.sessions[0].name = "renamed project".to_owned();
        updated.sessions[0].messages[0].content = "updated message".to_owned();
        updated.sessions[0].settings.system_prompt = "Updated prompt".to_owned();
        database.save(&updated).expect("updated state saves");
        assert_eq!(database.load(None).expect("updated state loads"), updated);

        updated.sessions.clear();
        updated.active_session_id = None;
        database.save(&updated).expect("empty state saves");
        assert_eq!(database.load(None).expect("empty state loads"), updated);
    }

    #[test]
    fn imports_legacy_json_only_when_database_has_no_sessions() {
        let mut database = Database {
            connection: Connection::open_in_memory().expect("in-memory db opens"),
        };
        migrate(&mut database.connection).expect("schema migrates");
        let legacy_json = serde_json::to_string(&sample_state()).expect("legacy JSON encodes");
        let imported = database
            .load(Some(&legacy_json))
            .expect("legacy data imports");
        assert_eq!(imported, sample_state());

        let ignored_legacy =
            serde_json::to_string(&PersistedSessionState::default()).expect("legacy JSON encodes");
        assert_eq!(
            database
                .load(Some(&ignored_legacy))
                .expect("existing db loads"),
            imported
        );
    }

    #[test]
    fn imports_legacy_sessions_without_newer_message_or_tool_fields() {
        let mut database = Database {
            connection: Connection::open_in_memory().expect("in-memory db opens"),
        };
        migrate(&mut database.connection).expect("schema migrates");
        let legacy_json = json!({
            "sessions": [{
                "id": "legacy-session",
                "name": "legacy",
                "workspacePath": "C:\\legacy",
                "createdAt": "2026-09-23T12:00:00.000Z",
                "updatedAt": "2026-09-23T12:00:00.000Z",
                "messages": [{
                    "id": "legacy-message",
                    "role": "user",
                    "content": "old message"
                }],
                "model": "llama3.2",
                "settings": {
                    "temperature": 0.7,
                    "topP": 0.9,
                    "numCtx": 4096,
                    "systemPrompt": ""
                },
                "archivedAt": null
            }],
            "activeSessionId": "legacy-session",
            "ollamaEndpoint": "http://localhost:11434"
        })
        .to_string();

        let imported = database
            .load(Some(&legacy_json))
            .expect("old-format data imports");
        let session = &imported.sessions[0];
        assert_eq!(session.messages[0].content, "old message");
        assert!(!session.messages[0].created_at.is_empty());
        assert_eq!(session.tool_call_history, Vec::new());
        assert!(!session.settings.strict_mode);
        assert!(imported.auto_install_ollama);
    }

    #[test]
    fn appends_and_loads_tool_audit_records() {
        let mut database = Database {
            connection: Connection::open_in_memory().expect("in-memory db opens"),
        };
        migrate(&mut database.connection).expect("schema migrates");
        let record = super::ToolAuditRecord {
            id: "request-0-0".to_owned(),
            session_id: "session-1".to_owned(),
            tool_name: "run_command".to_owned(),
            arguments_summary: r#"{"command":"cargo test"}"#.to_owned(),
            approval_status: "approved".to_owned(),
            execution_status: "succeeded".to_owned(),
            created_at: "2026-09-23T12:00:45.000Z".to_owned(),
        };

        database
            .append_tool_audit_log(&record)
            .expect("audit record appends");
        assert_eq!(
            database.load(None).expect("audit log loads").tool_audit_log,
            vec![record]
        );
    }
}
