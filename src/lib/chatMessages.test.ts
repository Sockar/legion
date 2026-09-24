import { describe, expect, it } from "vitest";
import { buildChatMessages, DEFAULT_SYSTEM_PROMPT } from "./chatMessages";

describe("buildChatMessages", () => {
  it("always puts out-of-workspace access instructions ahead of the conversation", () => {
    const messages = buildChatMessages(
      [
        {
          id: "user-1",
          role: "user",
          content: "Read C:\\Users\\me\\Documents\\notes.txt",
        },
      ],
      "",
      "windows",
    );

    expect(messages).toEqual([
      {
        role: "system",
        content: `${DEFAULT_SYSTEM_PROMPT}\n\nYou are running on Windows. The run_command tool executes commands with cmd.exe. Use Windows command syntax and backslash paths such as C:\\Users\\... . Do not mix path or command conventions between operating systems.`,
      },
      { role: "user", content: "Read C:\\Users\\me\\Documents\\notes.txt" },
    ]);
    expect(messages[0].content).toContain("must still attempt");
    expect(messages[0].content).toContain("If the user denies access");
    expect(messages[0].content).toContain("cmd.exe");
    expect(messages[0].content).toContain("C:\\Users\\...");
  });

  it("keeps the optional user system prompt after the default instructions", () => {
    const messages = buildChatMessages(
      [{ id: "assistant-1", role: "assistant", content: "Previous answer" }],
      "  Keep answers concise.  ",
      "macos",
    );

    expect(messages).toEqual([
      {
        role: "system",
        content: `${DEFAULT_SYSTEM_PROMPT}\n\nYou are running on macOS. The run_command tool executes commands with POSIX sh. Use POSIX command syntax and forward-slash paths such as /Users/... . Do not mix path or command conventions between operating systems.`,
      },
      { role: "system", content: "  Keep answers concise.  " },
      { role: "assistant", content: "Previous answer" },
    ]);
  });

  it("uses Linux shell and path conventions when running on Linux", () => {
    const [systemMessage] = buildChatMessages([], "", "linux");

    expect(systemMessage.content).toContain("You are running on Linux");
    expect(systemMessage.content).toContain("POSIX sh");
    expect(systemMessage.content).toContain("/home/user/...");
    expect(systemMessage.content).toContain(
      "Do not mix path or command conventions between operating systems.",
    );
  });
});
