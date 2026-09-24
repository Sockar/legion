import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ModelInfo } from "../lib/ollama";
import { ModelPicker } from "./ModelPicker";

const installedModels: ModelInfo[] = [
  {
    name: "mistral:7b",
    model: "mistral:7b",
    modified_at: "",
    size: 0,
    digest: "mistral",
  },
];

describe("ModelPicker", () => {
  it("lists installed and recommended models and pulls an uninstalled selection", () => {
    const onChange = vi.fn();
    const onPullModel = vi.fn().mockResolvedValue(undefined);
    render(
      <ModelPicker
        model="mistral:7b"
        models={installedModels}
        onChange={onChange}
        onPullModel={onPullModel}
      />,
    );

    const picker = screen.getByRole("combobox", { name: "Active model" });
    expect(picker.textContent).toContain("mistral:7b");
    expect(picker.textContent).toContain("Llama 3.1 · 8B · to download");
    fireEvent.change(picker, { target: { value: "llama3.1:8b" } });

    expect(onChange).toHaveBeenCalledWith("llama3.1:8b");
    expect(onPullModel).toHaveBeenCalledWith("llama3.1:8b");
  });

  it("selects installed models without starting a download", () => {
    const onChange = vi.fn();
    const onPullModel = vi.fn().mockResolvedValue(undefined);
    render(
      <ModelPicker
        model=""
        models={installedModels}
        onChange={onChange}
        onPullModel={onPullModel}
      />,
    );

    fireEvent.change(screen.getByRole("combobox", { name: "Active model" }), {
      target: { value: "mistral:7b" },
    });

    expect(onChange).toHaveBeenCalledWith("mistral:7b");
    expect(onPullModel).not.toHaveBeenCalled();
  });

  it("offers a manual model name fallback", () => {
    const onPullModel = vi.fn().mockResolvedValue(undefined);
    function ControlledPicker() {
      const [model, setModel] = useState("");
      return (
        <ModelPicker
          model={model}
          models={installedModels}
          onChange={setModel}
          onPullModel={onPullModel}
        />
      );
    }
    render(<ControlledPicker />);

    fireEvent.change(screen.getByRole("combobox", { name: "Active model" }), {
      target: { value: "__manual_model__" },
    });
    const input = screen.getByLabelText<HTMLInputElement>("Manual model name");
    fireEvent.change(input, { target: { value: "custom:latest" } });

    expect(input.value).toBe("custom:latest");
    fireEvent.click(
      screen.getByRole("button", { name: "Download custom:latest" }),
    );
    expect(onPullModel).toHaveBeenCalledWith("custom:latest");
  });
});
