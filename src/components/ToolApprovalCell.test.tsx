import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ToolApprovalRequest } from "../lib/ollama";
import { ToolApprovalCell } from "./ToolApprovalCell";

const genericApproval: ToolApprovalRequest = {
  request_id: "request-1",
  approval_id: "approval-1",
  tool: {
    name: "run_command",
    description: "Run a command in the workspace",
    parameters: { type: "object" },
    risk_level: "requires_confirmation",
  },
  arguments: { command: "npm test" },
};

describe("ToolApprovalCell", () => {
  it("renders a pending generic request inline without modal semantics", () => {
    const onDecision = vi.fn();
    const { container } = render(
      <ToolApprovalCell
        approval={genericApproval}
        disabled={false}
        onDecision={onDecision}
      />,
    );

    expect(
      screen.getByRole("group", { name: "Allow tool execution?" }),
    ).toBeTruthy();
    expect(screen.getByText("Exact command to run:")).toBeTruthy();
    expect(screen.getByText("npm test")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Allow once" })).toBeTruthy();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(container.querySelector("[aria-modal]")).toBeNull();
  });

  it.each([
    ["allow_once", "Approved"],
    ["deny", "Denied"],
  ] as const)(
    "shows the %s decision and disables actions",
    (decision, label) => {
      const { rerender } = render(
        <ToolApprovalCell
          approval={genericApproval}
          disabled={false}
          onDecision={vi.fn()}
        />,
      );

      rerender(
        <ToolApprovalCell
          approval={genericApproval}
          decision={decision}
          disabled={false}
          onDecision={vi.fn()}
        />,
      );

      expect(screen.getByRole("status").textContent).toBe(label);
      expect(
        screen
          .getAllByRole("button")
          .every((button) => (button as HTMLButtonElement).disabled),
      ).toBe(true);
    },
  );

  it("preserves an always-allowed decision for out-of-workspace access", () => {
    const approval: ToolApprovalRequest = {
      ...genericApproval,
      approval_type: "out_of_workspace_access",
      requested_path: "C:\\outside\\notes.txt",
    };

    render(
      <ToolApprovalCell
        approval={approval}
        decision="allow_always"
        disabled={false}
        onDecision={vi.fn()}
      />,
    );

    expect(screen.getByRole("status").textContent).toBe("Always allowed");
    expect(screen.getByText("C:\\outside\\notes.txt")).toBeTruthy();
  });

  it("shows the decision for a proposed file change", () => {
    const approval: ToolApprovalRequest = {
      ...genericApproval,
      preview: {
        operation: "edit",
        path: "src/example.ts",
        before: "const value = 1;",
        after: "const value = 2;",
      },
    };
    const onDecision = vi.fn();

    render(
      <ToolApprovalCell
        approval={approval}
        decision="deny"
        disabled={false}
        onDecision={onDecision}
      />,
    );

    expect(screen.getByRole("status").textContent).toBe("Denied");
    expect(screen.getByText("Edit src/example.ts")).toBeTruthy();
    expect(
      screen
        .getAllByRole("button")
        .every((button) => (button as HTMLButtonElement).disabled),
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(onDecision).not.toHaveBeenCalled();
  });

  it("routes file-change acceptance and rejection to the existing decisions", () => {
    const approval: ToolApprovalRequest = {
      ...genericApproval,
      preview: {
        operation: "create",
        path: "src/new.ts",
        before: "",
        after: "export {};\n",
      },
    };
    const onDecision = vi.fn();

    render(
      <ToolApprovalCell
        approval={approval}
        disabled={false}
        onDecision={onDecision}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Accept change" }));
    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(onDecision.mock.calls).toEqual([["allow_once"], ["deny"]]);
  });
});
