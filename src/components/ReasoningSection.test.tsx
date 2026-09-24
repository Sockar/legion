import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReasoningSection } from "./ReasoningSection";

describe("ReasoningSection", () => {
  it("renders the reasoning content while it is streaming", () => {
    render(
      <ReasoningSection
        hasFinalContent={false}
        isStreaming
        reasoning="Working through the problem"
      />,
    );

    expect(screen.getByText("Thinking")).toBeTruthy();
    expect(screen.getByText("Working through the problem")).toBeTruthy();
    expect(screen.getByRole("button").getAttribute("aria-expanded")).toBe(
      "true",
    );
  });

  it("collapses when final content starts streaming and can be toggled manually", () => {
    const { rerender } = render(
      <ReasoningSection
        hasFinalContent={false}
        isStreaming
        reasoning="Working through the problem"
      />,
    );

    rerender(
      <ReasoningSection
        hasFinalContent
        isStreaming
        reasoning="Working through the problem"
      />,
    );
    expect(screen.queryByText("Working through the problem")).toBeNull();
    expect(screen.getByRole("button").getAttribute("aria-expanded")).toBe(
      "false",
    );

    fireEvent.click(screen.getByRole("button", { name: /Thinking/ }));
    expect(screen.getByText("Working through the problem")).toBeTruthy();
    expect(screen.getByRole("button").getAttribute("aria-expanded")).toBe(
      "true",
    );
  });

  it("renders nothing when there is no reasoning", () => {
    const { container } = render(
      <ReasoningSection hasFinalContent={false} isStreaming reasoning="" />,
    );

    expect(container.firstChild).toBeNull();
  });
});
