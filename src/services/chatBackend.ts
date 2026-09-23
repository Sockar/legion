export interface ChatBackend {
  streamResponse(prompt: string, signal: AbortSignal): AsyncIterable<string>;
}

function waitForToken(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    let timeout = 0;
    const finish = () => {
      window.clearTimeout(timeout);
      signal.removeEventListener("abort", finish);
      resolve();
    };

    timeout = window.setTimeout(finish, 35);
    signal.addEventListener("abort", finish, { once: true });
    if (signal.aborted) finish();
  });
}

export class InMemoryChatBackend implements ChatBackend {
  private readonly prompts: string[] = [];

  async *streamResponse(
    prompt: string,
    signal: AbortSignal,
  ): AsyncGenerator<string> {
    this.prompts.push(prompt);
    const response = [
      `You said: ${prompt}`,
      "",
      "Here is a small TypeScript example:",
      "",
      "```ts",
      `const reply = ${JSON.stringify(prompt)};`,
      "console.log(reply);",
      "```",
      "",
      "This mock response streams one word at a time.",
    ].join("\n");

    for (const token of response.match(/\S+\s*/g) ?? []) {
      if (signal.aborted) {
        return;
      }

      await waitForToken(signal);

      if (signal.aborted) {
        return;
      }

      yield token;
    }
  }
}
