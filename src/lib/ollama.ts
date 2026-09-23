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
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: ToolCall[];
  tool_name?: string;
}

export interface JsonSchema {
  type: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  description?: string;
  [key: string]: unknown;
}

export interface ToolSchema {
  name: string;
  description: string;
  parameters: JsonSchema;
  risk_level: "auto_approve" | "requires_confirmation";
}

export interface ToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

export interface ToolApprovalRequest {
  request_id: string;
  approval_id: string;
  tool: ToolSchema;
  arguments: Record<string, unknown>;
  preview?: {
    operation: "create" | "edit";
    path: string;
    before: string;
    after: string;
  } | null;
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
  workspacePath: string,
  onChunk: (chunk: ChatChunk) => void,
  onApproval: (approval: ToolApprovalRequest) => void,
  signal?: AbortSignal,
): Promise<void> {
  const requestId = crypto.randomUUID();
  const unlisten = await listen<ChatChunk>(
    "ollama://chat-chunk",
    ({ payload }) => {
      if (payload.request_id === requestId) onChunk(payload);
    },
  );
  let unlistenApproval = () => {};
  let onAbort: (() => void) | undefined;

  try {
    unlistenApproval = await listen<ToolApprovalRequest>(
      "tools://approval-request",
      ({ payload }) => {
        if (payload.request_id === requestId) onApproval(payload);
      },
    );
    if (signal?.aborted) throw new DOMException("Chat cancelled", "AbortError");

    const chatRequest = invoke<void>("ollama_chat", {
      request: { model, messages, workspace_path: workspacePath },
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
    unlistenApproval();
  }
}

export async function respondToToolApproval(
  approvalId: string,
  approved: boolean,
): Promise<void> {
  await invoke("respond_tool_approval", { approvalId, approved });
}
