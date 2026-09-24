import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OutOfWorkspaceApproval } from "./OutOfWorkspaceApproval";

describe("OutOfWorkspaceApproval", () => {
  it("shows the requested path and offers once, always, and deny decisions", () => {
    const onDecision = vi.fn();
    const requestedPath = "C:\\Users\\example\\notes.txt";
    render(
      <OutOfWorkspaceApproval
        path={requestedPath}
        disabled={false}
        onDecision={onDecision}
      />,
    );

    expect(screen.getByText(requestedPath)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Allow once" }));
    fireEvent.click(screen.getByRole("button", { name: "Always allow" }));
    fireEvent.click(screen.getByRole("button", { name: "Deny" }));
    expect(onDecision.mock.calls).toEqual([
      ["allow_once"],
      ["allow_always"],
      ["deny"],
    ]);
  });

  it("disables every decision while the request is being resolved", () => {
    render(
      <OutOfWorkspaceApproval
        path="C:\\outside.txt"
        disabled
        onDecision={vi.fn()}
      />,
    );

    expect(screen.getAllByRole("button")).toHaveLength(3);
    expect(
      screen
        .getAllByRole("button")
        .every((button) => (button as HTMLButtonElement).disabled),
    ).toBe(true);
  });
});
