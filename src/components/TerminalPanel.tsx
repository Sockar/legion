export interface TerminalOutput {
  commandId: string;
  command: string;
  status: "running" | "succeeded" | "failed" | "timed_out" | "cancelled";
  exitCode: number | null;
  error: string | null;
  stdout: string;
  stderr: string;
}

interface TerminalPanelProps {
  output: TerminalOutput | null;
}

const statusLabels: Record<TerminalOutput["status"], string> = {
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  timed_out: "Timed out",
  cancelled: "Cancelled",
};

export function TerminalPanel({ output }: TerminalPanelProps) {
  return (
    <section className="terminal-panel" aria-label="Terminal output">
      <div className="terminal-panel__heading">
        <h2>Terminal</h2>
        {output && (
          <span
            className={`terminal-panel__status terminal-panel__status--${output.status}`}
            aria-live="polite"
          >
            {statusLabels[output.status]}
            {output.exitCode !== null ? ` · exit ${output.exitCode}` : ""}
          </span>
        )}
      </div>
      {output ? (
        <>
          <div className="terminal-panel__command">
            <span>Command</span>
            <code>{output.command}</code>
          </div>
          {output.error && (
            <p className="terminal-panel__error" role="status">
              {output.error}
            </p>
          )}
          <div className="terminal-panel__streams" aria-live="polite">
            <section>
              <h3>stdout</h3>
              <pre>
                {output.stdout ||
                  (output.status === "running" ? "Waiting for output…" : "")}
              </pre>
            </section>
            <section>
              <h3>stderr</h3>
              <pre>{output.stderr}</pre>
            </section>
          </div>
        </>
      ) : (
        <p className="terminal-panel__empty">
          Shell command output will appear here.
        </p>
      )}
    </section>
  );
}
