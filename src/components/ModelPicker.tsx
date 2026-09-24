import { useEffect, useState } from "react";
import type { ModelInfo } from "../lib/ollama";

export const RECOMMENDED_MODELS = [
  { name: "llama3.1:8b", label: "Llama 3.1 · 8B" },
  { name: "mistral:7b", label: "Mistral · 7B" },
  { name: "qwen2.5:7b", label: "Qwen 2.5 · 7B" },
  { name: "codellama:7b", label: "Code Llama · 7B" },
  { name: "nomic-embed-text", label: "Nomic Embed Text" },
  { name: "phi4:14b", label: "Phi-4 · 14B" },
  { name: "gemma3:4b", label: "Gemma 3 · 4B" },
] as const;

const MANUAL_MODEL = "__manual_model__";

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isManualModel(model: string, models: ModelInfo[]): boolean {
  return (
    !!model &&
    !models.some((item) => item.name === model) &&
    !RECOMMENDED_MODELS.some((item) => item.name === model)
  );
}

interface ModelPickerProps {
  model: string;
  models: ModelInfo[];
  disabled?: boolean;
  onChange: (model: string) => void;
  onPullModel: (model: string) => Promise<void>;
}

export function ModelPicker({
  model,
  models,
  disabled = false,
  onChange,
  onPullModel,
}: ModelPickerProps) {
  const [manualEntry, setManualEntry] = useState(() =>
    isManualModel(model, models),
  );
  const [pullError, setPullError] = useState("");

  useEffect(() => {
    if (models.some((item) => item.name === model)) setManualEntry(false);
  }, [model, models]);

  const handleSelection = (selected: string) => {
    setPullError("");
    if (selected === MANUAL_MODEL) {
      setManualEntry(true);
      if (!isManualModel(model, models)) onChange("");
      return;
    }

    setManualEntry(false);
    onChange(selected);
    if (!models.some((item) => item.name === selected)) {
      void onPullModel(selected).catch((error: unknown) => {
        onChange(model);
        setManualEntry(isManualModel(model, models));
        if (!isAbortError(error)) {
          setPullError(error instanceof Error ? error.message : String(error));
        }
      });
    }
  };

  return (
    <div className="model-picker">
      <label>
        Active model
        <select
          aria-label="Active model"
          disabled={disabled}
          onChange={(event) => handleSelection(event.target.value)}
          value={manualEntry ? MANUAL_MODEL : model}
        >
          <option disabled value="">
            Choose a model
          </option>
          <optgroup label="Installed models">
            {models.map((item) => (
              <option key={item.digest || item.name} value={item.name}>
                {item.name}
              </option>
            ))}
            {!models.length && (
              <option disabled value="__none__">
                No installed models
              </option>
            )}
          </optgroup>
          <optgroup label="Recommended · to download">
            {RECOMMENDED_MODELS.filter(
              (item) =>
                !models.some((installed) => installed.name === item.name),
            ).map((item) => (
              <option key={item.name} value={item.name}>
                {item.label} · to download
              </option>
            ))}
          </optgroup>
          <option value={MANUAL_MODEL}>Advanced: enter model name…</option>
        </select>
      </label>
      {manualEntry && (
        <>
          <label>
            Manual model name
            <input
              autoComplete="off"
              onChange={(event) => onChange(event.target.value)}
              placeholder="e.g. custom-model:latest"
              value={model}
            />
          </label>
          {!!model.trim() &&
            !models.some((item) => item.name === model.trim()) && (
              <button
                className="model-picker__download"
                disabled={disabled}
                onClick={() =>
                  void onPullModel(model.trim()).catch((error: unknown) => {
                    if (!isAbortError(error)) {
                      setPullError(
                        error instanceof Error ? error.message : String(error),
                      );
                    }
                  })
                }
                type="button"
              >
                Download {model.trim()}
              </button>
            )}
        </>
      )}
      {pullError && (
        <p className="settings-panel__error" role="alert">
          Could not download model: {pullError}
        </p>
      )}
    </div>
  );
}
