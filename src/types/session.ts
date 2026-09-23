import type { ChatMessage } from "./chat";

export interface ChatSettings {
  temperature: number;
  topP: number;
  numCtx: number;
  systemPrompt: string;
}

export const DEFAULT_CHAT_SETTINGS: ChatSettings = {
  temperature: 0.7,
  topP: 0.9,
  numCtx: 4096,
  systemPrompt: "",
};

export interface ChatSession {
  id: string;
  name: string;
  workspacePath: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  model: string;
  settings: ChatSettings;
  archivedAt: string | null;
  toolCallHistory: ToolCallHistory[];
}

export interface ToolCallHistory {
  id: string;
  toolName: string;
  arguments: Record<string, unknown>;
  result: unknown;
  status: string;
  createdAt: string;
}

export interface PersistedSessionState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  ollamaEndpoint: string;
}
