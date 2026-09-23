import { useCallback, useEffect, useRef, useState } from "react";
import { ChatInput } from "./components/ChatInput";
import { MessageList } from "./components/MessageList";
import {
  getOllamaStatus,
  listOllamaModels,
  pullOllamaModel,
  streamOllamaChat,
  type ModelInfo,
  type PullProgress,
  type ServerStatus,
} from "./lib/ollama";
import type { ChatMessage } from "./types/chat";
import "./App.css";

let nextMessageId = 0;

function createMessageId() {
  nextMessageId += 1;
  return `message-${Date.now()}-${nextMessageId}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<ServerStatus>();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelName, setModelName] = useState("");
  const [pullModelName, setPullModelName] = useState("llama3.2");
  const [pullProgress, setPullProgress] = useState<PullProgress>();
  const [isPulling, setIsPulling] = useState(false);
  const [runtimeError, setRuntimeError] = useState("");
  const controllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getOllamaStatus());
    } catch (error) {
      setRuntimeError(errorMessage(error));
    }
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const availableModels = await listOllamaModels();
      setModels(availableModels);
      setModelName((current) =>
        availableModels.some((model) => model.name === current)
          ? current
          : (availableModels[0]?.name ?? ""),
      );
      setRuntimeError("");
    } catch (error) {
      setRuntimeError(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    void refreshModels();
  }, [refreshModels, refreshStatus]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const generateResponse = useCallback(
    async (assistantId: string, history: ChatMessage[]) => {
      if (controllerRef.current || !modelName) return;

      const controller = new AbortController();
      controllerRef.current = controller;
      setIsStreaming(true);
      setMessages((current) =>
        current.some((message) => message.id === assistantId)
          ? current.map((message) =>
              message.id === assistantId
                ? { ...message, content: "" }
                : message,
            )
          : [...current, { id: assistantId, role: "assistant", content: "" }],
      );

      try {
        await streamOllamaChat(
          modelName,
          history.map(({ role, content }) => ({ role, content })),
          ({ content }) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + content }
                  : message,
              ),
            );
          },
          controller.signal,
        );
      } catch (error) {
        if (!controller.signal.aborted) {
          const detail = errorMessage(error);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: `The response failed: ${detail}` }
                : message,
            ),
          );
        }
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
          setIsStreaming(false);
        }
      }
    },
    [modelName],
  );

  const handleSend = useCallback(
    (content: string) => {
      const userMessage: ChatMessage = {
        id: createMessageId(),
        role: "user",
        content,
      };
      const assistantId = createMessageId();
      setMessages((current) => [
        ...current,
        userMessage,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      void generateResponse(assistantId, [...messages, userMessage]);
    },
    [generateResponse, messages],
  );

  const handleStop = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const handleRegenerate = useCallback(
    (assistantId: string) => {
      const assistantIndex = messages.findIndex(
        (message) => message.id === assistantId,
      );
      const history = messages.slice(0, assistantIndex);
      const previousUserMessage = history
        .slice()
        .reverse()
        .find((message) => message.role === "user");

      if (previousUserMessage) {
        void generateResponse(assistantId, history);
      }
    },
    [generateResponse, messages],
  );

  const handlePullModel = async () => {
    setIsPulling(true);
    setPullProgress(undefined);
    setRuntimeError("");
    try {
      await pullOllamaModel(pullModelName.trim(), setPullProgress);
      await refreshModels();
    } catch (error) {
      setRuntimeError(errorMessage(error));
    } finally {
      setIsPulling(false);
    }
  };

  return (
    <main className="chat-app">
      <header className="chat-header">
        <div className="chat-header__brand">
          <span className="chat-header__mark" aria-hidden="true">
            L
          </span>
          <div>
            <h1>Legion</h1>
            <p>Local coding assistant</p>
          </div>
        </div>
        <div className="runtime-controls">
          <label className="visually-hidden" htmlFor="chat-model">
            Chat model
          </label>
          <select
            id="chat-model"
            value={modelName}
            onChange={(event) => setModelName(event.target.value)}
            disabled={!models.length || isStreaming}
          >
            {models.length ? (
              models.map((model) => (
                <option key={model.digest || model.name} value={model.name}>
                  {model.name}
                </option>
              ))
            ) : (
              <option value="">No installed models</option>
            )}
          </select>
          <label className="visually-hidden" htmlFor="pull-model">
            Model name to download
          </label>
          <input
            id="pull-model"
            value={pullModelName}
            onChange={(event) => setPullModelName(event.target.value)}
            placeholder="Model to pull"
            disabled={isPulling}
          />
          <button
            className="pull-button"
            type="button"
            onClick={() => void handlePullModel()}
            disabled={!status?.connected || !pullModelName.trim() || isPulling}
          >
            {isPulling ? (pullProgress?.status ?? "Pulling…") : "Pull"}
          </button>
        </div>
        <button
          className="connection-status"
          type="button"
          onClick={() => void refreshStatus()}
          title="Check Ollama connection"
        >
          <span
            className={`connection-status__dot${status?.connected ? " connection-status__dot--online" : ""}`}
            aria-hidden="true"
          />
          {status?.message ?? "Checking Ollama…"}
        </button>
      </header>

      {runtimeError && (
        <p className="runtime-error" role="alert">
          {runtimeError}
        </p>
      )}

      <section className="conversation" aria-label="Conversation">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state__mark" aria-hidden="true">
              L
            </div>
            <h2>What can I help you build?</h2>
            <p>Ask a question or describe a coding task to get started.</p>
            {pullProgress && (
              <p aria-live="polite">
                {pullProgress.status}
                {pullProgress.total && pullProgress.completed != null
                  ? ` — ${Math.round((pullProgress.completed / pullProgress.total) * 100)}%`
                  : ""}
              </p>
            )}
          </div>
        ) : (
          <MessageList
            messages={messages}
            isStreaming={isStreaming}
            onRegenerate={handleRegenerate}
          />
        )}
        <div ref={bottomRef} />
      </section>

      <footer className="composer">
        {isStreaming && (
          <button className="stop-button" type="button" onClick={handleStop}>
            Stop generating
          </button>
        )}
        <ChatInput
          onSend={handleSend}
          disabled={isStreaming || !status?.connected || !modelName}
        />
        <p className="composer__hint">
          {status?.connected
            ? `Using ${modelName || "no model selected"} · Enter to send · Shift+Enter for a new line`
            : "Start Ollama to send a message"}
        </p>
      </footer>
    </main>
  );
}

export default App;
