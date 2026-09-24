import type { ApprovalDecision, ToolApprovalRequest } from "../lib/ollama";
import { FileChangeReview } from "./FileChangeReview";
import { OutOfWorkspaceApproval } from "./OutOfWorkspaceApproval";

interface ToolApprovalCellProps {
  approval: ToolApprovalRequest;
  decision?: ApprovalDecision;
  disabled: boolean;
  onDecision: (decision: ApprovalDecision) => void;
}

function getDecisionLabel(decision: ApprovalDecision): string {
  switch (decision) {
    case "allow_once":
      return "Approved";
    case "allow_always":
      return "Always allowed";
    case "deny":
      return "Denied";
  }
}

export function ToolApprovalCell({
  approval,
  decision,
  disabled,
  onDecision,
}: ToolApprovalCellProps) {
  const isOutOfWorkspace = approval.approval_type === "out_of_workspace_access";
  const title = isOutOfWorkspace
    ? "Allow access outside the workspace?"
    : approval.preview
      ? "Review proposed file change"
      : "Allow tool execution?";
  const actionsDisabled = disabled || decision !== undefined;

  return (
    <section className="tool-approval-cell" role="group" aria-label={title}>
      <div className="tool-approval-cell__heading">
        <h3>{title}</h3>
        {decision && (
          <span className="tool-approval-cell__decision" role="status">
            {getDecisionLabel(decision)}
          </span>
        )}
      </div>
      {isOutOfWorkspace ? (
        <OutOfWorkspaceApproval
          path={approval.requested_path ?? ""}
          disabled={actionsDisabled}
          onDecision={onDecision}
        />
      ) : approval.preview ? (
        <FileChangeReview
          preview={approval.preview}
          disabled={actionsDisabled}
          onAccept={() => onDecision("allow_once")}
          onReject={() => onDecision("deny")}
        />
      ) : (
        <>
          <p>
            <strong>{approval.tool.name}</strong>
            {`: ${approval.tool.description}`}
          </p>
          {approval.tool.name === "run_command" &&
          typeof approval.arguments.command === "string" ? (
            <>
              <p className="tool-approval__command-label">
                Exact command to run:
              </p>
              <pre>{approval.arguments.command}</pre>
            </>
          ) : (
            <pre>{JSON.stringify(approval.arguments, null, 2)}</pre>
          )}
          <div className="tool-approval__actions">
            <button
              type="button"
              disabled={actionsDisabled}
              onClick={() => onDecision("deny")}
            >
              Deny
            </button>
            <button
              className="tool-approval__allow"
              type="button"
              disabled={actionsDisabled}
              onClick={() => onDecision("allow_once")}
            >
              Allow once
            </button>
          </div>
        </>
      )}
    </section>
  );
}
