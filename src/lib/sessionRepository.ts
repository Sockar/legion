import {
  DEFAULT_CHAT_SETTINGS,
  type ChatSession,
  type ChatSettings,
  type PersistedSessionState,
} from "../types/session";

export interface SessionRepository {
  load(): PersistedSessionState;
  save(state: PersistedSessionState): void;
}

const STORAGE_KEY = "legion.session-state.v1";
export const DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isChatSettings(value: unknown): value is ChatSettings {
  return (
    isRecord(value) &&
    typeof value.temperature === "number" &&
    Number.isFinite(value.temperature) &&
    value.temperature >= 0 &&
    value.temperature <= 2 &&
    typeof value.topP === "number" &&
    Number.isFinite(value.topP) &&
    value.topP >= 0 &&
    value.topP <= 1 &&
    typeof value.numCtx === "number" &&
    Number.isInteger(value.numCtx) &&
    value.numCtx >= 256 &&
    value.numCtx <= 131072 &&
    typeof value.systemPrompt === "string"
  );
}

function isChatSession(value: unknown): value is ChatSession {
  if (!isRecord(value)) return false;

  return (
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    typeof value.workspacePath === "string" &&
    typeof value.createdAt === "string" &&
    typeof value.updatedAt === "string" &&
    typeof value.model === "string" &&
    (value.settings === undefined || isChatSettings(value.settings)) &&
    (typeof value.archivedAt === "string" || value.archivedAt === null) &&
    Array.isArray(value.messages) &&
    value.messages.every(
      (message) =>
        isRecord(message) &&
        typeof message.id === "string" &&
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string",
    )
  );
}

function parseSessionState(value: unknown): PersistedSessionState {
  if (
    !isRecord(value) ||
    !Array.isArray(value.sessions) ||
    !value.sessions.every(isChatSession) ||
    (typeof value.activeSessionId !== "string" &&
      value.activeSessionId !== null) ||
    (value.ollamaEndpoint !== undefined &&
      typeof value.ollamaEndpoint !== "string")
  ) {
    throw new Error("Saved session data has an invalid format.");
  }

  const sessions = value.sessions.map((session) => ({
    ...session,
    settings: session.settings ?? DEFAULT_CHAT_SETTINGS,
  }));
  const ids = new Set(sessions.map((session) => session.id));
  if (
    ids.size !== sessions.length ||
    (value.activeSessionId !== null && !ids.has(value.activeSessionId))
  ) {
    throw new Error("Saved session data contains invalid session references.");
  }

  return {
    sessions,
    activeSessionId: value.activeSessionId,
    ollamaEndpoint:
      typeof value.ollamaEndpoint === "string"
        ? value.ollamaEndpoint
        : DEFAULT_OLLAMA_ENDPOINT,
  };
}

export class LocalStorageSessionRepository implements SessionRepository {
  load(): PersistedSessionState {
    const savedState = window.localStorage.getItem(STORAGE_KEY);
    if (savedState === null) {
      return {
        sessions: [],
        activeSessionId: null,
        ollamaEndpoint: DEFAULT_OLLAMA_ENDPOINT,
      };
    }

    return parseSessionState(JSON.parse(savedState) as unknown);
  }

  save(state: PersistedSessionState): void {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }
}
