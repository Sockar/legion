import { invoke } from "@tauri-apps/api/core";
import { type PersistedSessionState } from "../types/session";

export interface SessionRepository {
  load(): Promise<PersistedSessionState>;
  save(state: PersistedSessionState): Promise<void>;
}

export const LEGACY_STORAGE_KEY = "legion.session-state.v1";
export const DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434";

export class SQLiteSessionRepository implements SessionRepository {
  private saveQueue: Promise<void> = Promise.resolve();

  async load(): Promise<PersistedSessionState> {
    const legacyJson = window.localStorage.getItem(LEGACY_STORAGE_KEY);
    const state = await invoke<PersistedSessionState>("load_session_state", {
      legacyJson,
    });
    if (legacyJson !== null) {
      window.localStorage.removeItem(LEGACY_STORAGE_KEY);
    }
    return state;
  }

  save(state: PersistedSessionState): Promise<void> {
    const save = this.saveQueue.then(() =>
      invoke<void>("save_session_state", { sessionState: state }),
    );
    this.saveQueue = save.catch(() => {});
    return save;
  }
}
