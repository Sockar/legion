import { confirm, open } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { useCallback, useEffect, useRef, useState } from "react";
import { FileChangeReview } from "./components/FileChangeReview";
import { ChatInput } from "./components/ChatInput";
import { MessageList } from "./components/MessageList";
import {
  DownloadProgress,
  type DownloadFeedback,
} from "./components/DownloadProgress";
import { SettingsPanel } from "./components/SettingsPanel";
import { SessionSidebar } from "./components/SessionSidebar";
import { TerminalPanel, type TerminalOutput } from "./components/TerminalPanel";
import {
  SQLiteSessionRepository,
  type SessionRepository,
} from "./lib/sessionRepository";
import {
  getOllamaStatus,
  installOllama,
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

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function comparableEndpoint(endpoint: string): string {
  const trimmed = endpoint.trim();
  try {
    const parsed = new URL(trimmed);
    parsed.pathname = parsed.pathname.replace(/\/+$/, "");
    return parsed.toString().replace(/\/$/, "");
  } catch {
    return trimmed.replace(/\/+$/, "");
  }
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
    autoInstallOllama: true,
    toolAuditLog: [],
  });
  const [isSessionStateLoaded, setIsSessionStateLoaded] = useState(false);
  const [storageError, setStorageError] = useState("");
  const [runtimeError, setRuntimeError] = useState("");
  const [ollamaInstallFeedback, setOllamaInstallFeedback] = useState("");
  const [isInstallingOllama, setIsInstallingOllama] = useState(false);
  const [downloadFeedback, setDownloadFeedback] = useState<DownloadFeedback>();
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
  const activeDownloadControllerRef = useRef<AbortController | undefined>(
    undefined,
  );
  const startupInstallPromptShownRef = useRef(false);
  const installPromptInFlightRef = useRef(false);
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
        const nextStatus = await getOllamaStatus(endpoint);
        setStatus(nextStatus);
        if (nextStatus.connected) {
          setRuntimeError("");
          setOllamaInstallFeedback("");
        }
      } catch (error) {
        setStatus({
          connected: false,
          endpoint,
          message: `Ollama is unreachable: ${errorMessage(error)}`,
        });
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
          sessionId,
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
            setSessionState((current) => ({
              ...current,
              toolAuditLog: [
                ...current.toolAuditLog,
                {
                  id: event.id,
                  sessionId,
                  toolName: event.tool_name,
                  argumentsSummary: event.arguments_summary,
                  approvalStatus: event.approval_status,
                  executionStatus:
                    event.approval_status === "rejected"
                      ? "not_run"
                      : event.status,
                  createdAt: event.created_at,
                },
              ],
              sessions: current.sessions.map((session) =>
                session.id === sessionId
                  ? updateSessionTimestamp({
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
                    })
                  : session,
              ),
            })),
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

  const handlePullModel = useCallback(
    async (modelName: string) => {
      const name = modelName.trim();
      if (!name) return;
      const controller = new AbortController();
      activeDownloadControllerRef.current = controller;
      setIsPulling(true);
      setPullProgress(undefined);
      setRuntimeError("");
      setDownloadFeedback({
        phase: "downloading",
        progress: {
          request_id: "",
          name,
          status: "Starting model download",
          percentage: null,
        },
      });
      try {
        await pullOllamaModel(
          name,
          sessionState.ollamaEndpoint,
          (progress) => {
            setPullProgress(progress);
            setDownloadFeedback({ phase: "downloading", progress });
          },
          controller.signal,
        );
        await refreshModels();
        setDownloadFeedback((current) =>
          current
            ? {
                phase: "success",
                progress: {
                  ...current.progress,
                  status: "Complete",
                  percentage: 100,
                  completed:
                    current.progress.total ?? current.progress.completed,
                },
                message: `${name} is ready to use.`,
              }
            : undefined,
        );
      } catch (error) {
        const message = errorMessage(error);
        const cancelled = isAbortError(error);
        if (!cancelled) setRuntimeError(message);
        setDownloadFeedback((current) =>
          current
            ? {
                ...current,
                phase: cancelled ? "cancelled" : "error",
                message: cancelled ? "Model download cancelled." : message,
              }
            : undefined,
        );
        throw error;
      } finally {
        if (activeDownloadControllerRef.current === controller) {
          activeDownloadControllerRef.current = undefined;
        }
        setIsPulling(false);
      }
    },
    [refreshModels, sessionState.ollamaEndpoint],
  );

  const handleSaveSettings = useCallback(
    (
      sessionId: string | null,
      model: string,
      settings: ChatSettings,
      endpoint: string,
      autoInstallOllama: boolean,
    ) => {
      setSessionState((current) => ({
        ...current,
        ollamaEndpoint: endpoint,
        autoInstallOllama,
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

  const handleInstallOllama = useCallback(
    async (endpoint = sessionState.ollamaEndpoint) => {
      if (installPromptInFlightRef.current) return;
      installPromptInFlightRef.current = true;
      try {
        const shouldInstall = await confirm(
          [
            `Ollama could not be reached at ${endpoint}.`,
            "",
            "Legion can download and install Ollama from the official Ollama GitHub releases (https://github.com/ollama/ollama/releases; also available at https://ollama.com/download). The download is approximately 200 MB on macOS and 1.5 GB on Windows or Linux; the exact size varies by release. Language models are not included and require additional downloads.",
            "",
            "Windows opens the official installer for you to complete. macOS installs Ollama.app in your user Applications folder. Linux installs the official release archive into Legion's app data and does not request administrator access.",
            "",
            "Continue only if you consent to downloading and installing this software.",
          ].join("\n"),
          { title: "Download and install Ollama?", kind: "warning" },
        );
        if (!shouldInstall) {
          setOllamaInstallFeedback("Ollama installation was cancelled.");
          return;
        }

        setIsInstallingOllama(true);
        const controller = new AbortController();
        activeDownloadControllerRef.current = controller;
        setRuntimeError("");
        setOllamaInstallFeedback("Downloading and installing Ollama…");
        try {
          setDownloadFeedback({
            phase: "downloading",
            progress: {
              request_id: "",
              name: "Ollama installer",
              status: "Preparing download",
              percentage: null,
            },
          });
          let latestInstallProgress: DownloadFeedback["progress"] | undefined;
          const installMessage = await installOllama((progress) => {
            latestInstallProgress = progress;
            setDownloadFeedback({
              phase:
                progress.status === "Installing" ? "installing" : "downloading",
              progress,
            });
          }, controller.signal);
          setDownloadFeedback((current) =>
            current
              ? {
                  ...current,
                  phase: "success",
                  progress: { ...current.progress, status: "Complete" },
                  message: "Ollama installer downloaded and launched.",
                }
              : undefined,
          );
          setOllamaInstallFeedback(
            `${installMessage} Checking the connection…`,
          );

          let nextStatus: ServerStatus | undefined;
          for (let attempt = 0; attempt < 30; attempt += 1) {
            nextStatus = await getOllamaStatus(endpoint);
            setStatus(nextStatus);
            if (nextStatus.connected) break;
            await new Promise((resolve) => window.setTimeout(resolve, 1000));
          }
          if (nextStatus?.connected) {
            setOllamaInstallFeedback(
              `${installMessage} Connected to Ollama successfully.`,
            );
            setDownloadFeedback({
              phase: "success",
              progress: {
                ...(latestInstallProgress ?? {
                  request_id: "",
                  name: "Ollama installer",
                  status: "Complete",
                }),
                status: "Complete",
                percentage: 100,
              },
              message: "Ollama was installed and is ready.",
            });
            setRuntimeError("");
            await refreshModels(endpoint);
          } else {
            throw new Error(
              `Ollama was installed, but the configured endpoint could not be reached. ${nextStatus?.message ?? ""}`.trim(),
            );
          }
        } catch (error) {
          const cancelled = isAbortError(error);
          const message = cancelled
            ? "Ollama installation was cancelled."
            : `Could not install or start Ollama: ${errorMessage(error)}`;
          setOllamaInstallFeedback(message);
          if (cancelled) {
            setRuntimeError("");
          } else {
            setRuntimeError(message);
          }
          setDownloadFeedback((current) =>
            current
              ? {
                  ...current,
                  phase: cancelled ? "cancelled" : "error",
                  message,
                }
              : undefined,
          );
        } finally {
          if (activeDownloadControllerRef.current === controller) {
            activeDownloadControllerRef.current = undefined;
          }
          setIsInstallingOllama(false);
        }
      } catch (error) {
        const message = `Could not confirm Ollama installation: ${errorMessage(error)}`;
        setOllamaInstallFeedback(message);
        setRuntimeError(message);
      } finally {
        installPromptInFlightRef.current = false;
      }
    },
    [refreshModels, sessionState.ollamaEndpoint],
  );

  const handleAutoInstallOllamaChange = useCallback((enabled: boolean) => {
    setSessionState((current) => ({ ...current, autoInstallOllama: enabled }));
  }, []);

  useEffect(() => {
    if (
      !isSessionStateLoaded ||
      !sessionState.autoInstallOllama ||
      status?.connected !== false ||
      comparableEndpoint(status.endpoint) !==
        comparableEndpoint(sessionState.ollamaEndpoint) ||
      startupInstallPromptShownRef.current
    ) {
      return;
    }
    startupInstallPromptShownRef.current = true;
    void handleInstallOllama();
  }, [
    handleInstallOllama,
    isSessionStateLoaded,
    sessionState.autoInstallOllama,
    sessionState.ollamaEndpoint,
    status?.endpoint,
    status?.connected,
  ]);

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
            onClick={() =>
              void handlePullModel(pullModelName).catch((error: unknown) =>
                isAbortError(error)
                  ? undefined
                  : setRuntimeError(errorMessage(error)),
              )
            }
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

      {downloadFeedback && (
        <DownloadProgress
          feedback={downloadFeedback}
          onCancel={() => activeDownloadControllerRef.current?.abort()}
          onDismiss={() => setDownloadFeedback(undefined)}
        />
      )}

      {storageError && (
        <p className="runtime-error" role="alert">
          {storageError}
        </p>
      )}
      {runtimeError && runtimeError !== ollamaInstallFeedback && (
        <p className="runtime-error" role="alert">
          {runtimeError}
        </p>
      )}
      {ollamaInstallFeedback && (
        <div className="runtime-error" role="status">
          <p>{ollamaInstallFeedback}</p>
          {!status?.connected && (
            <button
              type="button"
              onClick={() =>
                void openUrl("https://ollama.com/download").catch((error) =>
                  setRuntimeError(
                    `Could not open Ollama's download page: ${errorMessage(error)}`,
                  ),
                )
              }
            >
              Open Ollama download page
            </button>
          )}
        </div>
      )}

      {isSettingsOpen && (
        <SettingsPanel
          key={activeSession?.id ?? "no-active-session"}
          activeSession={activeSession}
          endpoint={sessionState.ollamaEndpoint}
          autoInstallOllama={sessionState.autoInstallOllama}
          toolAuditLog={sessionState.toolAuditLog}
          models={models}
          onClose={() => setIsSettingsOpen(false)}
          onSave={handleSaveSettings}
          onTestConnection={handleTestConnection}
          onAutoInstallOllamaChange={handleAutoInstallOllamaChange}
          onInstallOllama={handleInstallOllama}
          onPullModel={handlePullModel}
          isPullingModel={isPulling}
          isInstallingOllama={isInstallingOllama}
          installFeedback={ollamaInstallFeedback}
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
                      {pullProgress.percentage != null
                        ? ` — ${pullProgress.percentage}%`
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
