import { confirm, open } from "@tauri-apps/plugin-dialog";
import { useCallback, useEffect, useRef, useState } from "react";
import { FileChangeReview } from "./components/FileChangeReview";
import { ChatInput } from "./components/ChatInput";
import { MessageList } from "./components/MessageList";
import { SettingsPanel } from "./components/SettingsPanel";
import { SessionSidebar } from "./components/SessionSidebar";
import { TerminalPanel, type TerminalOutput } from "./components/TerminalPanel";
import {
  SQLiteSessionRepository,
  type SessionRepository,
} from "./lib/sessionRepository";
import {
  getOllamaStatus,
  listOllamaModels,
  pullOllamaModel,
  streamOllamaChat,
  respondToToolApproval,
  type ModelInfo,
  type CommandOutputEvent,
  type PullProgress,
  type ServerStatus,
  type ToolApprovalRequest,
} from "./lib/ollama";
import type { ChatMessage } from "./types/chat";
import {
  DEFAULT_CHAT_SETTINGS,
  type ChatSession,
  type ChatSettings,
  type PersistedSessionState,
} from "./types/session";
import { DEFAULT_OLLAMA_ENDPOINT } from "./lib/sessionRepository";
import "./App.css";

let nextMessageId = 0;

const emptyMessages: ChatMessage[] = [];
const sessionRepository: SessionRepository = new SQLiteSessionRepository();
const MAX_TERMINAL_OUTPUT_LENGTH = 100_000;

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

function appendTerminalOutput(current: string, chunk: string): string {
  const output = current + chunk;
  if (output.length <= MAX_TERMINAL_OUTPUT_LENGTH) return output;
  return `[Earlier output omitted]\n${output.slice(-MAX_TERMINAL_OUTPUT_LENGTH)}`;
}

function App() {
  const [sessionState, setSessionState] = useState<PersistedSessionState>({
    sessions: [],
    activeSessionId: null,
    ollamaEndpoint: DEFAULT_OLLAMA_ENDPOINT,
  });
  const [isSessionStateLoaded, setIsSessionStateLoaded] = useState(false);
  const [storageError, setStorageError] = useState("");
  const [runtimeError, setRuntimeError] = useState("");
  const [status, setStatus] = useState<ServerStatus>();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [pullModelName, setPullModelName] = useState("llama3.2");
  const [pullProgress, setPullProgress] = useState<PullProgress>();
  const [isPulling, setIsPulling] = useState(false);
  const [streamingSessionIds, setStreamingSessionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [toolApprovals, setToolApprovals] = useState<
    { approval: ToolApprovalRequest; sessionId: string }[]
  >([]);
  const [resolvingApprovalId, setResolvingApprovalId] = useState<string>();
  const [commandOutputs, setCommandOutputs] = useState<
    Record<string, TerminalOutput>
  >({});
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
  const currentToolApproval = toolApprovals[0]?.approval;
  const approvalCommand =
    currentToolApproval?.tool.name === "run_command" &&
    typeof currentToolApproval.arguments.command === "string"
      ? currentToolApproval.arguments.command
      : null;
  const inlineFileApproval =
    activeSession &&
    toolApprovals[0]?.sessionId === activeSession.id &&
    currentToolApproval?.preview
      ? currentToolApproval
      : undefined;
  const activeCommandOutput = activeSession
    ? (commandOutputs[activeSession.id] ?? null)
    : null;

  useEffect(() => {
    void (async () => {
      try {
        const savedState = await sessionRepository.load();
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
    })();
  }, []);

  useEffect(() => {
    if (!isSessionStateLoaded || storageError) return;

    void sessionRepository
      .save(sessionState)
      .catch((error: unknown) =>
        setStorageError(`Unable to save sessions: ${errorMessage(error)}`),
      );
  }, [isSessionStateLoaded, sessionState, storageError]);

  useEffect(
    () => () => {
      controllersRef.current.forEach((controller) => controller.abort());
    },
    [],
  );

  const refreshStatus = useCallback(
    async (endpoint = sessionState.ollamaEndpoint) => {
      try {
        setStatus(await getOllamaStatus(endpoint));
      } catch (error) {
        setRuntimeError(errorMessage(error));
      }
    },
    [sessionState.ollamaEndpoint],
  );

  const refreshModels = useCallback(
    async (endpoint = sessionState.ollamaEndpoint) => {
      try {
        const availableModels = await listOllamaModels(endpoint);
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
    },
    [sessionState.ollamaEndpoint],
  );

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

  const updateCommandOutput = useCallback(
    (sessionId: string, event: CommandOutputEvent) => {
      setCommandOutputs((current) => {
        const previous = current[sessionId];
        const output: TerminalOutput =
          event.phase === "started" || !previous
            ? {
                commandId: event.command_id,
                command: event.command,
                status: event.status ?? "running",
                exitCode: event.exit_code,
                error: event.error,
                stdout: "",
                stderr: "",
              }
            : { ...previous };

        if (event.phase === "output" && event.chunk) {
          const key = event.stream === "stderr" ? "stderr" : "stdout";
          output[key] = appendTerminalOutput(output[key], event.chunk);
        } else if (event.phase === "completed") {
          output.status = event.status ?? "failed";
          output.exitCode = event.exit_code;
          output.error = event.error;
        }

        return { ...current, [sessionId]: output };
      });
    },
    [],
  );

  const generateResponse = useCallback(
    async (
      sessionId: string,
      assistantId: string,
      history: ChatMessage[],
      model: string,
      workspacePath: string,
      settings: ChatSettings,
      endpoint: string,
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
          [
            ...(settings.systemPrompt.trim()
              ? [{ role: "system" as const, content: settings.systemPrompt }]
              : []),
            ...history.map(({ role, content }) => ({ role, content })),
          ],
          workspacePath,
          endpoint,
          settings,
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
          (approval) =>
            setToolApprovals((current) =>
              current.some(
                (pending) =>
                  pending.approval.approval_id === approval.approval_id,
              )
                ? current
                : [...current, { approval, sessionId }],
            ),
          (event) =>
            updateSession(sessionId, (session) =>
              updateSessionTimestamp({
                ...session,
                toolCallHistory: [
                  ...session.toolCallHistory,
                  {
                    id: event.id,
                    toolName: event.tool_name,
                    arguments: event.arguments,
                    result: event.result,
                    status: event.status,
                    createdAt: event.created_at,
                  },
                ],
              }),
            ),
          (event) => updateCommandOutput(sessionId, event),
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
        } else {
          setCommandOutputs((current) => {
            const output = current[sessionId];
            return output?.status === "running"
              ? {
                  ...current,
                  [sessionId]: {
                    ...output,
                    status: "cancelled",
                    error: "Command cancelled.",
                  },
                }
              : current;
          });
        }
      } finally {
        if (controllersRef.current.get(sessionId) === controller) {
          controllersRef.current.delete(sessionId);
          setStreamingSessionIds((current) => {
            const next = new Set(current);
            next.delete(sessionId);
            return next;
          });
          setToolApprovals((current) =>
            current.filter((pending) => pending.sessionId !== sessionId),
          );
        }
      }
    },
    [updateCommandOutput, updateSession],
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
        settings: DEFAULT_CHAT_SETTINGS,
        archivedAt: null,
        toolCallHistory: [],
      };
      setSessionState((current) => ({
        ...current,
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
      return { ...current, sessions, activeSessionId: nextActive };
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
        ...current,
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
          return { ...current, sessions, activeSessionId: nextActive };
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
        createdAt: new Date().toISOString(),
      };
      const assistantId = createMessageId();
      const history = [...activeSession.messages, userMessage];
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
      };
      updateSession(activeSession.id, (session) =>
        updateSessionTimestamp({
          ...session,
          messages: [...session.messages, userMessage, assistantMessage],
        }),
      );
      void generateResponse(
        activeSession.id,
        assistantId,
        history,
        activeSession.model,
        activeSession.workspacePath,
        activeSession.settings,
        sessionState.ollamaEndpoint,
      );
    },
    [
      activeSession,
      generateResponse,
      sessionState.ollamaEndpoint,
      updateSession,
    ],
  );

  const handleStop = useCallback(() => {
    if (activeSession) {
      controllersRef.current.get(activeSession.id)?.abort();
    }
  }, [activeSession]);

  const handleToolApproval = useCallback(
    async (approved: boolean) => {
      if (!currentToolApproval || resolvingApprovalId) return;
      setResolvingApprovalId(currentToolApproval.approval_id);
      try {
        await respondToToolApproval(currentToolApproval.approval_id, approved);
        setToolApprovals((current) =>
          current.filter(
            (pending) =>
              pending.approval.approval_id !== currentToolApproval.approval_id,
          ),
        );
      } catch (error) {
        setRuntimeError(errorMessage(error));
      } finally {
        setResolvingApprovalId(undefined);
      }
    },
    [currentToolApproval, resolvingApprovalId],
  );

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
          activeSession.workspacePath,
          activeSession.settings,
          sessionState.ollamaEndpoint,
        );
      }
    },
    [activeSession, generateResponse, sessionState.ollamaEndpoint],
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
      await pullOllamaModel(
        pullModelName.trim(),
        sessionState.ollamaEndpoint,
        setPullProgress,
      );
      await refreshModels();
    } catch (error) {
      setRuntimeError(errorMessage(error));
    } finally {
      setIsPulling(false);
    }
  };

  const handleSaveSettings = useCallback(
    (
      sessionId: string | null,
      model: string,
      settings: ChatSettings,
      endpoint: string,
    ) => {
      setSessionState((current) => ({
        ...current,
        ollamaEndpoint: endpoint,
        sessions: current.sessions.map((session) =>
          session.id === sessionId
            ? updateSessionTimestamp({ ...session, model, settings })
            : session,
        ),
      }));
      void refreshStatus(endpoint);
      void refreshModels(endpoint);
    },
    [refreshModels, refreshStatus],
  );

  const handleTestConnection = useCallback(
    async (endpoint: string) => {
      const nextStatus = await getOllamaStatus(endpoint);
      setStatus(nextStatus);
      if (nextStatus.connected) await refreshModels(endpoint);
      return nextStatus;
    },
    [refreshModels],
  );

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
        <button
          aria-label="Open settings"
          className="settings-button"
          onClick={() => setIsSettingsOpen(true)}
          title="Settings"
          type="button"
        >
          ⚙
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

      {isSettingsOpen && (
        <SettingsPanel
          key={activeSession?.id ?? "no-active-session"}
          activeSession={activeSession}
          endpoint={sessionState.ollamaEndpoint}
          models={models}
          onClose={() => setIsSettingsOpen(false)}
          onSave={handleSaveSettings}
          onTestConnection={handleTestConnection}
        />
      )}

      {currentToolApproval && !inlineFileApproval && (
        <div className="tool-approval-backdrop">
          <section
            className="tool-approval"
            role="dialog"
            aria-modal="true"
            aria-labelledby="tool-approval-title"
          >
            <h2 id="tool-approval-title">
              {currentToolApproval.preview
                ? "Review proposed file change"
                : "Allow tool execution?"}
            </h2>
            {currentToolApproval.preview ? (
              <FileChangeReview
                preview={currentToolApproval.preview}
                disabled={
                  resolvingApprovalId === currentToolApproval.approval_id
                }
                onAccept={() => void handleToolApproval(true)}
                onReject={() => void handleToolApproval(false)}
              />
            ) : (
              <>
                <p>
                  <strong>{currentToolApproval.tool.name}</strong>
                  {`: ${currentToolApproval.tool.description}`}
                </p>
                {approvalCommand !== null ? (
                  <>
                    <p className="tool-approval__command-label">
                      Exact command to run:
                    </p>
                    <pre>{approvalCommand}</pre>
                  </>
                ) : (
                  <pre>
                    {JSON.stringify(currentToolApproval.arguments, null, 2)}
                  </pre>
                )}
                <div className="tool-approval__actions">
                  <button
                    type="button"
                    disabled={
                      resolvingApprovalId === currentToolApproval.approval_id
                    }
                    onClick={() => void handleToolApproval(false)}
                  >
                    Deny
                  </button>
                  <button
                    className="tool-approval__allow"
                    type="button"
                    disabled={
                      resolvingApprovalId === currentToolApproval.approval_id
                    }
                    onClick={() => void handleToolApproval(true)}
                  >
                    Allow once
                  </button>
                </div>
              </>
            )}
          </section>
        </div>
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
            <TerminalPanel output={activeCommandOutput} />
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
              {inlineFileApproval?.preview && (
                <FileChangeReview
                  preview={inlineFileApproval.preview}
                  disabled={
                    resolvingApprovalId === inlineFileApproval.approval_id
                  }
                  onAccept={() => void handleToolApproval(true)}
                  onReject={() => void handleToolApproval(false)}
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
