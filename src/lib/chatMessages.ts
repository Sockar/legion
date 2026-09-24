import type { ChatMessage as OllamaChatMessage } from "./ollama";
import type { ChatMessage } from "../types/chat";
import type { TargetOS } from "../types/platform";

export const DEFAULT_SYSTEM_PROMPT = `You have file tools, including workspace listing, which access files within the active workspace folder by default. When the user's request requires access to a file outside that folder, you can and must still attempt the relevant tool call using the absolute path. Do not refuse in advance or claim that you lack access: Legion will ask the user whether to allow access once, always allow it, or deny it. If the user denies access, the tool will return a clear error; explain that result to the user.`;

function getTargetOSInstructions(targetOS: TargetOS): string {
  switch (targetOS) {
    case "windows":
      return "You are running on Windows. The run_command tool executes commands with cmd.exe. Use Windows command syntax and backslash paths such as C:\\Users\\... .";
    case "macos":
      return "You are running on macOS. The run_command tool executes commands with POSIX sh. Use POSIX command syntax and forward-slash paths such as /Users/... .";
    case "linux":
      return "You are running on Linux. The run_command tool executes commands with POSIX sh. Use POSIX command syntax and forward-slash paths such as /home/user/... .";
  }
}

export function buildChatMessages(
  history: ChatMessage[],
  systemPrompt: string,
  targetOS: TargetOS,
): OllamaChatMessage[] {
  return [
    {
      role: "system",
      content: `${DEFAULT_SYSTEM_PROMPT}\n\n${getTargetOSInstructions(targetOS)} Do not mix path or command conventions between operating systems.`,
    },
    ...(systemPrompt.trim()
      ? [{ role: "system" as const, content: systemPrompt }]
      : []),
    ...history.map(({ role, content }) => ({ role, content })),
  ];
}
