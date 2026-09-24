import type { ApprovalDecision } from "../lib/ollama";

interface OutOfWorkspaceApprovalProps {
  path: string;
  disabled: boolean;
  onDecision: (decision: ApprovalDecision) => void;
}

export function OutOfWorkspaceApproval({
  path,
  disabled,
  onDecision,
}: OutOfWorkspaceApprovalProps) {
  return (
    <>
      <p>Allow file access outside the active workspace?</p>
      <pre>{path}</pre>
      <div className="tool-approval__actions">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onDecision("deny")}
        >
          Deny
        </button>
        <button
          className="tool-approval__allow"
          type="button"
          disabled={disabled}
          onClick={() => onDecision("allow_once")}
        >
          Allow once
        </button>
        <button
          className="tool-approval__allow"
          type="button"
          disabled={disabled}
          onClick={() => onDecision("allow_always")}
        >
          Always allow
        </button>
      </div>
    </>
  );
}
