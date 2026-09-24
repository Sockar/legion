import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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
    const { rerender } = render(
      <DownloadProgress
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

    rerender(
      <DownloadProgress
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
  });
});
