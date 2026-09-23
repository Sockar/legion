import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface ModelInfo {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details?: {
    format: string;
    family: string;
    parameter_size: string;
    quantization_level: string;
  } | null;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ServerStatus {
  connected: boolean;
  endpoint: string;
  message: string;
}

export interface PullProgress {
  request_id: string;
  status: string;
  digest?: string | null;
  total?: number | null;
  completed?: number | null;
}

export interface ChatChunk {
  request_id: string;
  content: string;
  done: boolean;
}

export async function getOllamaStatus(): Promise<ServerStatus> {
  return invoke<ServerStatus>("ollama_status");
}

export async function listOllamaModels(): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>("ollama_list_models");
}

export async function pullOllamaModel(
  model: string,
  onProgress: (progress: PullProgress) => void,
): Promise<void> {
  const requestId = crypto.randomUUID();
  const unlisten = await listen<PullProgress>(
    "ollama://pull-progress",
    ({ payload }) => {
      if (payload.request_id === requestId) onProgress(payload);
    },
  );

  try {
    await invoke("ollama_pull_model", { model, requestId });
  } finally {
    unlisten();
  }
}

export async function streamOllamaChat(
  model: string,
  messages: ChatMessage[],
  onChunk: (chunk: ChatChunk) => void,
  signal?: AbortSignal,
): Promise<void> {
  const requestId = crypto.randomUUID();
  const unlisten = await listen<ChatChunk>(
    "ollama://chat-chunk",
    ({ payload }) => {
      if (payload.request_id === requestId) onChunk(payload);
    },
  );
  let onAbort: (() => void) | undefined;

  try {
    if (signal?.aborted) throw new DOMException("Chat cancelled", "AbortError");

    const chatRequest = invoke<void>("ollama_chat", {
      request: { model, messages },
      requestId,
    });
    if (signal) {
      const cancelled = new Promise<never>((_, reject) => {
        onAbort = () => {
          void invoke("ollama_cancel_chat", { requestId })
            .then(() =>
              reject(new DOMException("Chat cancelled", "AbortError")),
            )
            .catch(reject);
        };
        signal.addEventListener("abort", onAbort, { once: true });
      });
      await Promise.race([chatRequest, cancelled]);
    } else {
      await chatRequest;
    }
  } finally {
    if (onAbort) signal?.removeEventListener("abort", onAbort);
    unlisten();
  }
}
