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
    );

    expect(messages).toEqual([
      { role: "system", content: DEFAULT_SYSTEM_PROMPT },
      { role: "user", content: "Read C:\\Users\\me\\Documents\\notes.txt" },
    ]);
    expect(messages[0].content).toContain("must still attempt");
    expect(messages[0].content).toContain("If the user denies access");
  });

  it("keeps the optional user system prompt after the default instructions", () => {
    const messages = buildChatMessages(
      [{ id: "assistant-1", role: "assistant", content: "Previous answer" }],
      "  Keep answers concise.  ",
    );

    expect(messages).toEqual([
      { role: "system", content: DEFAULT_SYSTEM_PROMPT },
      { role: "system", content: "  Keep answers concise.  " },
      { role: "assistant", content: "Previous answer" },
    ]);
  });
});
