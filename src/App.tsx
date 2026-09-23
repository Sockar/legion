import { confirm, open } from "@tauri-apps/plugin-dialog";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChatInput } from "./components/ChatInput";
import { MessageList } from "./components/MessageList";
import { SessionSidebar } from "./components/SessionSidebar";
import {
  LocalStorageSessionRepository,
  type SessionRepository,
} from "./lib/sessionRepository";
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
import type { ChatSession, PersistedSessionState } from "./types/session";
import "./App.css";

let nextMessageId = 0;

const emptyMessages: ChatMessage[] = [];
const sessionRepository: SessionRepository =
  new LocalStorageSessionRepository();

function createMessageId() {
  nextMessageId += 1;
  return `message-${Date.now()}-${nextMessageId}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function getWorkspaceName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function updateSessionTimestamp(session: ChatSession): ChatSession {
  return { ...session, updatedAt: new Date().toISOString() };
}

function App() {
  const [sessionState, setSessionState] = useState<PersistedSessionState>({
    sessions: [],
    activeSessionId: null,
  });
  const [isSessionStateLoaded, setIsSessionStateLoaded] = useState(false);
  const [storageError, setStorageError] = useState("");
  const [runtimeError, setRuntimeError] = useState("");
  const [status, setStatus] = useState<ServerStatus>();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [pullModelName, setPullModelName] = useState("llama3.2");
  const [pullProgress, setPullProgress] = useState<PullProgress>();
  const [isPulling, setIsPulling] = useState(false);
  const [streamingSessionIds, setStreamingSessionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const controllersRef = useRef(new Map<string, AbortController>());
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeSession =
    sessionState.sessions.find(
      (session) => session.id === sessionState.activeSessionId,
    ) ?? null;
  const messages = activeSession?.messages ?? emptyMessages;
  const isStreaming = activeSession
    ? streamingSessionIds.has(activeSession.id)
    : false;

  useEffect(() => {
    try {
      const savedState = sessionRepository.load();
      const savedActiveSession = savedState.sessions.find(
        (session) =>
          session.id === savedState.activeSessionId &&
          session.archivedAt === null,
      );
      setSessionState({
        ...savedState,
        activeSessionId:
          savedActiveSession?.id ??
          savedState.sessions.find((session) => session.archivedAt === null)
            ?.id ??
          null,
      });
    } catch (error) {
      setStorageError(
        `Unable to restore saved sessions: ${errorMessage(error)}`,
      );
    } finally {
      setIsSessionStateLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!isSessionStateLoaded || storageError) return;

    try {
      sessionRepository.save(sessionState);
    } catch (error) {
      setStorageError(`Unable to save sessions: ${errorMessage(error)}`);
    }
  }, [isSessionStateLoaded, sessionState, storageError]);

  useEffect(
    () => () => {
      controllersRef.current.forEach((controller) => controller.abort());
    },
    [],
  );

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
      setSessionState((current) => ({
        ...current,
        sessions: current.sessions.map((session) =>
          session.model &&
          availableModels.some((model) => model.name === session.model)
            ? session
            : updateSessionTimestamp({
                ...session,
                model: availableModels[0]?.name ?? "",
              }),
        ),
      }));
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

  const updateSession = useCallback(
    (sessionId: string, update: (session: ChatSession) => ChatSession) => {
      setSessionState((current) => ({
        ...current,
        sessions: current.sessions.map((session) =>
          session.id === sessionId ? update(session) : session,
        ),
      }));
    },
    [],
  );

  const generateResponse = useCallback(
    async (
      sessionId: string,
      assistantId: string,
      history: ChatMessage[],
      model: string,
    ) => {
      if (controllersRef.current.has(sessionId) || !model) return;

      const controller = new AbortController();
      controllersRef.current.set(sessionId, controller);
      setStreamingSessionIds((current) => new Set(current).add(sessionId));
      updateSession(sessionId, (session) =>
        updateSessionTimestamp({
          ...session,
          messages: session.messages.map((message) =>
            message.id === assistantId ? { ...message, content: "" } : message,
          ),
        }),
      );

      try {
        await streamOllamaChat(
          model,
          history.map(({ role, content }) => ({ role, content })),
          ({ content }) => {
            updateSession(sessionId, (session) =>
              updateSessionTimestamp({
                ...session,
                messages: session.messages.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content + content }
                    : message,
                ),
              }),
            );
          },
          controller.signal,
        );
      } catch (error) {
        if (!controller.signal.aborted) {
          const detail = errorMessage(error);
          updateSession(sessionId, (session) =>
            updateSessionTimestamp({
              ...session,
              messages: session.messages.map((message) =>
                message.id === assistantId
                  ? { ...message, content: `The response failed: ${detail}` }
                  : message,
              ),
            }),
          );
        }
      } finally {
        if (controllersRef.current.get(sessionId) === controller) {
          controllersRef.current.delete(sessionId);
          setStreamingSessionIds((current) => {
            const next = new Set(current);
            next.delete(sessionId);
            return next;
          });
        }
      }
    },
    [updateSession],
  );

  const handleCreateSession = useCallback(async () => {
    setRuntimeError("");
    try {
      const selectedPath = await open({
        directory: true,
        multiple: false,
        title: "Choose a workspace folder",
      });
      if (typeof selectedPath !== "string" || !selectedPath) return;

      const now = new Date().toISOString();
      const session: ChatSession = {
        id: crypto.randomUUID(),
        name: getWorkspaceName(selectedPath),
        workspacePath: selectedPath,
        createdAt: now,
        updatedAt: now,
        messages: [],
        model: models[0]?.name ?? "",
        archivedAt: null,
      };
      setSessionState((current) => ({
        sessions: [session, ...current.sessions],
        activeSessionId: session.id,
      }));
    } catch (error) {
      setRuntimeError(
        `Unable to choose a workspace folder: ${errorMessage(error)}`,
      );
    }
  }, [models]);

  const handleSelectSession = useCallback((sessionId: string) => {
    setSessionState((current) => {
      const session = current.sessions.find((item) => item.id === sessionId);
      if (!session || session.archivedAt !== null) return current;
      return { ...current, activeSessionId: sessionId };
    });
  }, []);

  const handleRenameSession = useCallback(
    (sessionId: string, name: string) => {
      updateSession(sessionId, (session) =>
        updateSessionTimestamp({ ...session, name }),
      );
    },
    [updateSession],
  );

  const handleArchiveSession = useCallback((sessionId: string) => {
    setSessionState((current) => {
      const archivedAt = new Date().toISOString();
      const sessions = current.sessions.map((session) =>
        session.id === sessionId
          ? { ...session, archivedAt, updatedAt: archivedAt }
          : session,
      );
      const nextActive =
        current.activeSessionId === sessionId
          ? (sessions.find((session) => session.archivedAt === null)?.id ??
            null)
          : current.activeSessionId;
      return { sessions, activeSessionId: nextActive };
    });
  }, []);

  const handleRestoreSession = useCallback((sessionId: string) => {
    setSessionState((current) => {
      const sessions = current.sessions.map((session) =>
        session.id === sessionId
          ? updateSessionTimestamp({ ...session, archivedAt: null })
          : session,
      );
      return {
        sessions,
        activeSessionId: current.activeSessionId ?? sessionId,
      };
    });
  }, []);

  const handleDeleteSession = useCallback(
    async (sessionId: string) => {
      const session = sessionState.sessions.find(
        (item) => item.id === sessionId,
      );
      if (!session) return;

      try {
        const shouldDelete = await confirm(
          `Permanently delete "${session.name}" and its message history?`,
          { title: "Delete session", kind: "warning" },
        );
        if (!shouldDelete) return;

        controllersRef.current.get(sessionId)?.abort();
        setSessionState((current) => {
          const sessions = current.sessions.filter(
            (item) => item.id !== sessionId,
          );
          const nextActive =
            current.activeSessionId === sessionId
              ? (sessions.find((item) => item.archivedAt === null)?.id ?? null)
              : current.activeSessionId;
          return { sessions, activeSessionId: nextActive };
        });
      } catch (error) {
        setRuntimeError(`Unable to delete session: ${errorMessage(error)}`);
      }
    },
    [sessionState.sessions],
  );

  const handleSend = useCallback(
    (content: string) => {
      if (!activeSession) return;
      const userMessage: ChatMessage = {
        id: createMessageId(),
        role: "user",
        content,
      };
      const assistantId = createMessageId();
      const history = [...activeSession.messages, userMessage];
      updateSession(activeSession.id, (session) =>
        updateSessionTimestamp({
          ...session,
          messages: [
            ...session.messages,
            userMessage,
            { id: assistantId, role: "assistant", content: "" },
          ],
        }),
      );
      void generateResponse(
        activeSession.id,
        assistantId,
        history,
        activeSession.model,
      );
    },
    [activeSession, generateResponse, updateSession],
  );

  const handleStop = useCallback(() => {
    if (activeSession) {
      controllersRef.current.get(activeSession.id)?.abort();
    }
  }, [activeSession]);

  const handleRegenerate = useCallback(
    (assistantId: string) => {
      if (!activeSession) return;
      const assistantIndex = activeSession.messages.findIndex(
        (message) => message.id === assistantId,
      );
      const history = activeSession.messages.slice(0, assistantIndex);
      const previousUserMessage = history
        .slice()
        .reverse()
        .find((message) => message.role === "user");

      if (previousUserMessage) {
        void generateResponse(
          activeSession.id,
          assistantId,
          history,
          activeSession.model,
        );
      }
    },
    [activeSession, generateResponse],
  );

  const handleModelChange = useCallback(
    (model: string) => {
      if (!activeSession) return;
      updateSession(activeSession.id, (session) =>
        updateSessionTimestamp({ ...session, model }),
      );
    },
    [activeSession, updateSession],
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

  if (!isSessionStateLoaded) {
    return (
      <main className="chat-app chat-app--loading" aria-live="polite">
        {storageError || "Restoring sessions…"}
      </main>
    );
  }

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
            value={activeSession?.model ?? ""}
            onChange={(event) => handleModelChange(event.target.value)}
            disabled={!activeSession || !models.length || isStreaming}
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

      {storageError && (
        <p className="runtime-error" role="alert">
          {storageError}
        </p>
      )}
      {runtimeError && (
        <p className="runtime-error" role="alert">
          {runtimeError}
        </p>
      )}

      <div className="chat-body">
        <SessionSidebar
          activeSessionId={activeSession?.id ?? null}
          sessions={sessionState.sessions}
          onArchive={handleArchiveSession}
          onCreate={() => void handleCreateSession()}
          onDelete={(sessionId) => void handleDeleteSession(sessionId)}
          onRename={handleRenameSession}
          onRestore={handleRestoreSession}
          onSelect={handleSelectSession}
        />

        {activeSession ? (
          <section className="chat-workspace">
            <div className="workspace-heading">
              <div>
                <h2>{activeSession.name}</h2>
                <p title={activeSession.workspacePath}>
                  Workspace: {activeSession.workspacePath}
                </p>
              </div>
              {streamingSessionIds.has(activeSession.id) && (
                <span className="workspace-heading__streaming">
                  Generating response
                </span>
              )}
            </div>
            <section className="conversation" aria-label="Conversation">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state__mark" aria-hidden="true">
                    L
                  </div>
                  <h2>What can I help you build?</h2>
                  <p>
                    Ask a question or describe a coding task to get started.
                  </p>
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
                <button
                  className="stop-button"
                  type="button"
                  onClick={handleStop}
                >
                  Stop generating
                </button>
              )}
              <ChatInput
                onSend={handleSend}
                disabled={
                  isStreaming || !status?.connected || !activeSession.model
                }
              />
              <p className="composer__hint">
                {status?.connected
                  ? `Using ${activeSession.model || "no model selected"} · Enter to send · Shift+Enter for a new line`
                  : "Start Ollama to send a message"}
              </p>
            </footer>
          </section>
        ) : (
          <section className="workspace-empty">
            <div className="empty-state__mark" aria-hidden="true">
              L
            </div>
            <h2>Choose a workspace to start</h2>
            <p>Each session keeps its own conversation and model selection.</p>
            <button
              className="session-sidebar__new"
              onClick={() => void handleCreateSession()}
              type="button"
            >
              Choose workspace folder
            </button>
          </section>
        )}
      </div>
    </main>
  );
}

export default App;
