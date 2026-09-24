export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  reasoning?: string;
  createdAt?: string;
  toolCallData?: unknown;
}
