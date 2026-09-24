import type { ChatMessage as OllamaChatMessage } from "./ollama";
import type { ChatMessage } from "../types/chat";

export const DEFAULT_SYSTEM_PROMPT = `You have file tools, including workspace listing, which access files within the active workspace folder by default. When the user's request requires access to a file outside that folder, you can and must still attempt the relevant tool call using the absolute path. Do not refuse in advance or claim that you lack access: Legion will ask the user whether to allow access once, always allow it, or deny it. If the user denies access, the tool will return a clear error; explain that result to the user.`;

export function buildChatMessages(
  history: ChatMessage[],
  systemPrompt: string,
): OllamaChatMessage[] {
  return [
    { role: "system", content: DEFAULT_SYSTEM_PROMPT },
    ...(systemPrompt.trim()
      ? [{ role: "system" as const, content: systemPrompt }]
      : []),
    ...history.map(({ role, content }) => ({ role, content })),
  ];
}
