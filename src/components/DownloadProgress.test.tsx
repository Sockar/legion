import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DownloadProgress } from "./DownloadProgress";

describe("DownloadProgress", () => {
  it("renders the model name, byte counts, and percentage while downloading", () => {
    render(
      <DownloadProgress
        feedback={{
          phase: "downloading",
          progress: {
            request_id: "pull-1",
            name: "llama3.1:8b",
            status: "pulling model layers",
            total: 10 * 1024 * 1024,
            completed: 4 * 1024 * 1024,
            percentage: 40,
          },
        }}
      />,
    );

    expect(screen.getByText("llama3.1:8b")).toBeTruthy();
    expect(
      screen
        .getByLabelText("llama3.1:8b download progress")
        .getAttribute("value"),
    ).toBe("40");
    expect(screen.getByText(/40% · 4\.0 MB of 10\.0 MB/)).toBeTruthy();
  });

  it("renders successful and failed terminal states accessibly", () => {
    const onDismiss = vi.fn();
    const { rerender } = render(
      <DownloadProgress
        onDismiss={onDismiss}
        feedback={{
          phase: "success",
          progress: {
            request_id: "install-1",
            name: "OllamaSetup.exe",
            status: "Complete",
            percentage: 100,
          },
          message: "Ollama is ready.",
        }}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "Ollama is ready.",
    );
    expect(
      screen.queryByRole("button", { name: "Cancel download" }),
    ).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "Dismiss download status" }),
    );
    expect(onDismiss).toHaveBeenCalledOnce();

    rerender(
      <DownloadProgress
        onDismiss={onDismiss}
        feedback={{
          phase: "error",
          progress: {
            request_id: "install-1",
            name: "OllamaSetup.exe",
            status: "Failed",
          },
          message: "Network connection interrupted.",
        }}
      />,
    );
    expect(screen.getByRole("alert").textContent).toContain(
      "Network connection interrupted.",
    );
    expect(
      screen.queryByRole("button", { name: "Cancel download" }),
    ).toBeNull();
  });

  it("cancels an active download instead of dismissing it", () => {
    const onCancel = vi.fn();
    const onDismiss = vi.fn();
    render(
      <DownloadProgress
        feedback={{
          phase: "downloading",
          progress: {
            request_id: "pull-1",
            name: "llama3.1:8b",
            status: "pulling model layers",
          },
        }}
        onCancel={onCancel}
        onDismiss={onDismiss}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel download" }));

    expect(onCancel).toHaveBeenCalledOnce();
    expect(onDismiss).not.toHaveBeenCalled();
  });
});
