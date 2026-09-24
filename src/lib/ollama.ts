import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ChatSettings } from "../types/session";

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

export interface CommandOutputEvent {
  request_id: string;
  command_id: string;
  command: string;
  phase: "started" | "output" | "completed";
  stream: "stdout" | "stderr" | null;
  chunk: string | null;
  status: "running" | "succeeded" | "failed" | "timed_out" | null;
  exit_code: number | null;
  error: string | null;
}

export interface ToolCallEvent {
  id: string;
  request_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  arguments_summary: string;
  result: unknown;
  status: string;
  approval_status: string;
  created_at: string;
}

export interface ServerStatus {
  connected: boolean;
  endpoint: string;
  message: string;
}

export interface DownloadProgress {
  request_id: string;
  name: string;
  status: string;
  total?: number | null;
  completed?: number | null;
  percentage?: number | null;
}

export interface PullProgress {
  request_id: string;
  name: string;
  status: string;
  digest?: string | null;
  total?: number | null;
  completed?: number | null;
  percentage?: number | null;
}

export interface ChatChunk {
  request_id: string;
  content: string;
  done: boolean;
}

export interface ChatThinkingChunk {
  request_id: string;
  thinking: string;
}

export async function getOllamaStatus(endpoint: string): Promise<ServerStatus> {
  return invoke<ServerStatus>("ollama_status", { endpoint });
}

export async function installOllama(
  onProgress: (progress: DownloadProgress) => void,
  signal?: AbortSignal,
): Promise<string> {
  const requestId = crypto.randomUUID();
  const unlisten = await listen<DownloadProgress>(
    "ollama://install-progress",
    ({ payload }) => {
      if (payload.request_id === requestId) onProgress(payload);
    },
  );
  let onAbort: (() => void) | undefined;
  try {
    if (signal?.aborted) {
      throw new DOMException("Installation cancelled", "AbortError");
    }
    const installRequest = invoke<string>("install_ollama", { requestId });
    if (!signal) return await installRequest;
    const cancelled = new Promise<never>((_, reject) => {
      onAbort = () => {
        void invoke("ollama_cancel_install", { requestId })
          .then(() =>
            reject(new DOMException("Installation cancelled", "AbortError")),
          )
          .catch(reject);
      };
      signal.addEventListener("abort", onAbort, { once: true });
    });
    return await Promise.race([installRequest, cancelled]);
  } finally {
    if (onAbort) signal?.removeEventListener("abort", onAbort);
    unlisten();
  }
}

export async function listOllamaModels(endpoint: string): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>("ollama_list_models", { endpoint });
}

export async function pullOllamaModel(
  model: string,
  endpoint: string,
  onProgress: (progress: PullProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const requestId = crypto.randomUUID();
  const unlisten = await listen<PullProgress>(
    "ollama://pull-progress",
    ({ payload }) => {
      if (payload.request_id === requestId) onProgress(payload);
    },
  );

  let onAbort: (() => void) | undefined;
  try {
    if (signal?.aborted) {
      throw new DOMException("Model pull cancelled", "AbortError");
    }
    const pullRequest = invoke("ollama_pull_model", {
      model,
      endpoint,
      requestId,
    });
    if (!signal) {
      await pullRequest;
      return;
    }
    const cancelled = new Promise<never>((_, reject) => {
      onAbort = () => {
        void invoke("ollama_cancel_pull", { requestId })
          .then(() =>
            reject(new DOMException("Model pull cancelled", "AbortError")),
          )
          .catch(reject);
      };
      signal.addEventListener("abort", onAbort, { once: true });
    });
    await Promise.race([pullRequest, cancelled]);
  } finally {
    if (onAbort) signal?.removeEventListener("abort", onAbort);
    unlisten();
  }
}

export async function streamOllamaChat(
  sessionId: string,
  model: string,
  messages: ChatMessage[],
  workspacePath: string,
  endpoint: string,
  settings: ChatSettings,
  onChunk: (chunk: ChatChunk) => void,
  onThinking: (chunk: ChatThinkingChunk) => void,
  onApproval: (approval: ToolApprovalRequest) => void,
  onToolCall: (event: ToolCallEvent) => void,
  onCommandOutput: (event: CommandOutputEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const requestId = crypto.randomUUID();
  const unlisten = await listen<ChatChunk>(
    "ollama://chat-chunk",
    ({ payload }) => {
      if (payload.request_id === requestId) onChunk(payload);
    },
  );
  const unlistenThinking = await listen<ChatThinkingChunk>(
    "ollama://chat-thinking",
    ({ payload }) => {
      if (payload.request_id === requestId) onThinking(payload);
    },
  );
  let unlistenApproval = () => {};
  let unlistenToolCall = () => {};
  let unlistenCommandOutput = () => {};
  let onAbort: (() => void) | undefined;

  try {
    unlistenApproval = await listen<ToolApprovalRequest>(
      "tools://approval-request",
      ({ payload }) => {
        if (payload.request_id === requestId) onApproval(payload);
      },
    );
    unlistenToolCall = await listen<ToolCallEvent>(
      "tools://tool-call",
      ({ payload }) => {
        if (payload.request_id === requestId) onToolCall(payload);
      },
    );
    unlistenCommandOutput = await listen<CommandOutputEvent>(
      "tools://command-output",
      ({ payload }) => {
        if (payload.request_id === requestId) onCommandOutput(payload);
      },
    );
    if (signal?.aborted) throw new DOMException("Chat cancelled", "AbortError");

    const chatRequest = invoke<void>("ollama_chat", {
      request: {
        model,
        messages,
        workspace_path: workspacePath,
        endpoint,
        session_id: sessionId,
        strict_mode: settings.strictMode,
        enable_reasoning: settings.enableReasoning,
        options: {
          temperature: settings.temperature,
          top_p: settings.topP,
          num_ctx: settings.numCtx,
        },
      },
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
    unlistenThinking();
    unlistenApproval();
    unlistenToolCall();
    unlistenCommandOutput();
  }
}

export async function respondToToolApproval(
  approvalId: string,
  approved: boolean,
): Promise<void> {
  await invoke("respond_tool_approval", { approvalId, approved });
}
