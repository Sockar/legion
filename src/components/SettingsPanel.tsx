import { useState, type FormEvent } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { confirm } from "@tauri-apps/plugin-dialog";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import {
  DEFAULT_CHAT_SETTINGS,
  type ChatSession,
  type ChatSettings,
  type ToolAuditRecord,
} from "../types/session";
import type { ModelInfo, ServerStatus } from "../lib/ollama";

interface SettingsPanelProps {
  activeSession: ChatSession | null;
  endpoint: string;
  autoInstallOllama: boolean;
  toolAuditLog: ToolAuditRecord[];
  models: ModelInfo[];
  onClose: () => void;
  onSave: (
    sessionId: string | null,
    model: string,
    settings: ChatSettings,
    endpoint: string,
    autoInstallOllama: boolean,
  ) => void;
  onTestConnection: (endpoint: string) => Promise<ServerStatus>;
  onAutoInstallOllamaChange: (enabled: boolean) => void;
  onInstallOllama: (endpoint: string) => Promise<void>;
  isInstallingOllama: boolean;
  installFeedback: string;
}

export function SettingsPanel({
  activeSession,
  endpoint,
  autoInstallOllama,
  toolAuditLog,
  models,
  onClose,
  onSave,
  onTestConnection,
  onAutoInstallOllamaChange,
  onInstallOllama,
  isInstallingOllama,
  installFeedback,
}: SettingsPanelProps) {
  const [model, setModel] = useState(activeSession?.model ?? "");
  const [settings, setSettings] = useState<ChatSettings>(
    activeSession?.settings ?? DEFAULT_CHAT_SETTINGS,
  );
  const [endpointDraft, setEndpointDraft] = useState(endpoint);
  const [autoInstallEnabled, setAutoInstallEnabled] =
    useState(autoInstallOllama);
  const [testStatus, setTestStatus] = useState<ServerStatus>();
  const [testError, setTestError] = useState("");
  const [isTesting, setIsTesting] = useState(false);
  const [validationError, setValidationError] = useState("");
  const [isCheckingUpdates, setIsCheckingUpdates] = useState(false);
  const [updateStatus, setUpdateStatus] = useState("");
  const [updateError, setUpdateError] = useState("");

  const handleCheckForUpdates = async () => {
    setIsCheckingUpdates(true);
    setUpdateStatus("");
    setUpdateError("");
    try {
      const update = await check();
      if (!update) {
        setUpdateStatus("You're up to date.");
        return;
      }

      const notes = update.body ? `\n\n${update.body}` : "";
      const shouldInstall = await confirm(
        `Version ${update.version} is available. Install it now?${notes}`,
        { title: "Update available", kind: "info" },
      );
      if (!shouldInstall) {
        setUpdateStatus(`Version ${update.version} is available.`);
        return;
      }

      setUpdateStatus("Downloading and installing update…");
      await update.downloadAndInstall();
      await relaunch();
    } catch (error) {
      setUpdateError(
        `Unable to check for updates: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    } finally {
      setIsCheckingUpdates(false);
    }
  };

  const recentToolCalls = toolAuditLog
    .filter((record) => record.sessionId === activeSession?.id)
    .slice(-50)
    .reverse();

  const updateNumber = (
    key: "temperature" | "topP" | "numCtx",
    value: string,
  ) => {
    setSettings((current) => ({ ...current, [key]: Number(value) }));
  };

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestError("");
    setTestStatus(undefined);
    try {
      const result = await onTestConnection(endpointDraft.trim());
      setTestStatus(result);
      if (!result.connected && autoInstallEnabled) {
        await onInstallOllama(endpointDraft.trim());
        setTestStatus(await onTestConnection(endpointDraft.trim()));
      }
    } catch (error) {
      setTestError(error instanceof Error ? error.message : String(error));
      if (autoInstallEnabled) {
        await onInstallOllama(endpointDraft.trim());
        try {
          setTestStatus(await onTestConnection(endpointDraft.trim()));
        } catch (retryError) {
          setTestError(
            retryError instanceof Error
              ? retryError.message
              : String(retryError),
          );
        }
      }
    } finally {
      setIsTesting(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (
      !Number.isFinite(settings.temperature) ||
      settings.temperature < 0 ||
      settings.temperature > 2 ||
      !Number.isFinite(settings.topP) ||
      settings.topP < 0 ||
      settings.topP > 1 ||
      !Number.isInteger(settings.numCtx) ||
      settings.numCtx < 256 ||
      settings.numCtx > 131072
    ) {
      setValidationError(
        "Use values within the ranges shown for each setting.",
      );
      return;
    }
    if (!endpointDraft.trim()) {
      setValidationError("Enter an Ollama base URL.");
      return;
    }
    setValidationError("");
    onSave(
      activeSession?.id ?? null,
      model,
      settings,
      endpointDraft.trim(),
      autoInstallEnabled,
    );
    onClose();
  };

  return (
    <div
      className="settings-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        aria-labelledby="settings-title"
        aria-modal="true"
        className="settings-panel"
        role="dialog"
      >
        <form onSubmit={handleSubmit}>
          <header className="settings-panel__header">
            <div>
              <h2 id="settings-title">Settings</h2>
              <p>
                {activeSession
                  ? `For ${activeSession.name}`
                  : "Session options are available when a session is selected."}
              </p>
            </div>
            <button
              aria-label="Close settings"
              className="settings-panel__close"
              onClick={onClose}
              type="button"
            >
              ×
            </button>
          </header>

          <div className="settings-panel__content">
            <fieldset disabled={!activeSession}>
              <legend>Session</legend>
              <label>
                Active model
                <select
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                >
                  {!models.some((item) => item.name === model) && model && (
                    <option value={model}>{model} (not installed)</option>
                  )}
                  {models.map((item) => (
                    <option key={item.digest || item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                  {!models.length && (
                    <option value="">No installed models</option>
                  )}
                </select>
              </label>
              <div className="settings-panel__grid">
                <label>
                  Temperature <span>0–2</span>
                  <input
                    max="2"
                    min="0"
                    onChange={(event) =>
                      updateNumber("temperature", event.target.value)
                    }
                    step="0.1"
                    type="number"
                    value={settings.temperature}
                  />
                </label>
                <label>
                  Top-p <span>0–1</span>
                  <input
                    max="1"
                    min="0"
                    onChange={(event) =>
                      updateNumber("topP", event.target.value)
                    }
                    step="0.05"
                    type="number"
                    value={settings.topP}
                  />
                </label>
                <label>
                  Context length <span>256–131072</span>
                  <input
                    max="131072"
                    min="256"
                    onChange={(event) =>
                      updateNumber("numCtx", event.target.value)
                    }
                    step="1"
                    type="number"
                    value={settings.numCtx}
                  />
                </label>
              </div>
              <label>
                System prompt
                <textarea
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      systemPrompt: event.target.value,
                    }))
                  }
                  placeholder="Optional instructions for this session"
                  rows={5}
                  value={settings.systemPrompt}
                />
              </label>
              <label className="settings-panel__toggle">
                <input
                  checked={settings.strictMode}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      strictMode: event.target.checked,
                    }))
                  }
                  type="checkbox"
                />
                <span>
                  Strict mode — require approval for every tool call, including
                  read-only tools.
                </span>
              </label>
            </fieldset>

            <fieldset>
              <legend>Recent tool activity</legend>
              {!activeSession ? (
                <p>Select a session to review its audit log.</p>
              ) : recentToolCalls.length ? (
                <ol className="settings-panel__audit-log">
                  {recentToolCalls.map((record) => (
                    <li key={record.id}>
                      <details>
                        <summary>
                          {new Date(Number(record.createdAt)).toLocaleString()}{" "}
                          · {record.toolName} · {record.approvalStatus} ·{" "}
                          {record.executionStatus}
                        </summary>
                        <pre>{record.argumentsSummary}</pre>
                      </details>
                    </li>
                  ))}
                </ol>
              ) : (
                <p>No tool calls have been recorded for this session.</p>
              )}
            </fieldset>

            <fieldset>
              <legend>Ollama connection</legend>
              <label>
                Base URL
                <input
                  autoComplete="url"
                  onChange={(event) => setEndpointDraft(event.target.value)}
                  placeholder="http://localhost:11434"
                  type="url"
                  value={endpointDraft}
                />
              </label>
              <div className="settings-panel__connection">
                <button
                  disabled={!endpointDraft.trim() || isTesting}
                  onClick={() => void handleTestConnection()}
                  type="button"
                >
                  {isTesting ? "Testing…" : "Test connection"}
                </button>
                {testStatus && (
                  <span
                    className={
                      testStatus.connected
                        ? "settings-panel__connected"
                        : "settings-panel__error"
                    }
                    role="status"
                  >
                    {testStatus.message}
                  </span>
                )}
                {testError && (
                  <span className="settings-panel__error" role="alert">
                    {testError}
                  </span>
                )}
              </div>
              <label className="settings-panel__toggle">
                <input
                  checked={autoInstallEnabled}
                  onChange={(event) => {
                    setAutoInstallEnabled(event.target.checked);
                    onAutoInstallOllamaChange(event.target.checked);
                  }}
                  type="checkbox"
                />
                <span>
                  Ask to download and install Ollama when it is unavailable.
                </span>
              </label>
              {isInstallingOllama && (
                <p role="status">Downloading and installing Ollama…</p>
              )}
              {installFeedback && (
                <p
                  className={
                    testStatus?.connected
                      ? "settings-panel__connected"
                      : "settings-panel__error"
                  }
                  role="status"
                >
                  {installFeedback}
                </p>
              )}
            </fieldset>

            <fieldset>
              <legend>Application updates</legend>
              <div className="settings-panel__connection">
                <button
                  disabled={!isTauri() || isCheckingUpdates}
                  onClick={() => void handleCheckForUpdates()}
                  type="button"
                >
                  {isCheckingUpdates ? "Checking…" : "Check for updates"}
                </button>
                {!isTauri() && (
                  <span className="settings-panel__update-note">
                    Updates are available in the desktop app.
                  </span>
                )}
                {updateStatus && (
                  <span
                    className="settings-panel__connected"
                    role="status"
                    aria-live="polite"
                  >
                    {updateStatus}
                  </span>
                )}
                {updateError && (
                  <span className="settings-panel__error" role="alert">
                    {updateError}
                  </span>
                )}
              </div>
            </fieldset>
          </div>

          {validationError && (
            <p className="settings-panel__error" role="alert">
              {validationError}
            </p>
          )}
          <footer className="settings-panel__actions">
            <button onClick={onClose} type="button">
              Cancel
            </button>
            <button className="settings-panel__save" type="submit">
              Save settings
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}
