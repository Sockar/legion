import type { DownloadProgress as ProgressUpdate } from "../lib/ollama";

export interface DownloadFeedback {
  phase: "downloading" | "success" | "error";
  progress: ProgressUpdate;
  message?: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DownloadProgress({
  feedback,
  onDismiss,
}: {
  feedback: DownloadFeedback;
  onDismiss?: () => void;
}) {
  const { phase, progress, message } = feedback;
  const percentage = progress.percentage ?? null;
  const completed = progress.completed ?? null;
  const total = progress.total ?? null;

  return (
    <section
      aria-label="Download progress"
      aria-live="polite"
      className={`download-progress download-progress--${phase}`}
      role={phase === "error" ? "alert" : "status"}
    >
      <div className="download-progress__heading">
        <div>
          <strong>{progress.name}</strong>
          <span>
            {phase === "success"
              ? "Complete"
              : phase === "error"
                ? "Failed"
                : progress.status}
          </span>
        </div>
        {onDismiss && (
          <button
            aria-label="Dismiss download status"
            onClick={onDismiss}
            type="button"
          >
            ×
          </button>
        )}
      </div>
      {phase === "downloading" && (
        <>
          <progress
            aria-label={`${progress.name} download progress`}
            max={100}
            value={percentage ?? undefined}
          />
          <p>
            {percentage == null ? progress.status : `${percentage}%`}
            {completed != null && total != null
              ? ` · ${formatBytes(completed)} of ${formatBytes(total)}`
              : ""}
          </p>
        </>
      )}
      {(phase === "success" || phase === "error") && (
        <p>
          {message ??
            (phase === "success" ? "Download completed." : "Download failed.")}
        </p>
      )}
    </section>
  );
}
