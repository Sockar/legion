import { useState, type FormEvent } from "react";
import {
  DEFAULT_CHAT_SETTINGS,
  type ChatSession,
  type ChatSettings,
} from "../types/session";
import type { ModelInfo, ServerStatus } from "../lib/ollama";

interface SettingsPanelProps {
  activeSession: ChatSession | null;
  endpoint: string;
  models: ModelInfo[];
  onClose: () => void;
  onSave: (
    sessionId: string | null,
    model: string,
    settings: ChatSettings,
    endpoint: string,
  ) => void;
  onTestConnection: (endpoint: string) => Promise<ServerStatus>;
}

export function SettingsPanel({
  activeSession,
  endpoint,
  models,
  onClose,
  onSave,
  onTestConnection,
}: SettingsPanelProps) {
  const [model, setModel] = useState(activeSession?.model ?? "");
  const [settings, setSettings] = useState<ChatSettings>(
    activeSession?.settings ?? DEFAULT_CHAT_SETTINGS,
  );
  const [endpointDraft, setEndpointDraft] = useState(endpoint);
  const [testStatus, setTestStatus] = useState<ServerStatus>();
  const [testError, setTestError] = useState("");
  const [isTesting, setIsTesting] = useState(false);
  const [validationError, setValidationError] = useState("");

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
      setTestStatus(await onTestConnection(endpointDraft.trim()));
    } catch (error) {
      setTestError(error instanceof Error ? error.message : String(error));
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
    onSave(activeSession?.id ?? null, model, settings, endpointDraft.trim());
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
