import type { ChatMessage } from "./chat";

export interface ChatSession {
  id: string;
  name: string;
  workspacePath: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  model: string;
  archivedAt: string | null;
}

export interface PersistedSessionState {
  sessions: ChatSession[];
  activeSessionId: string | null;
}
