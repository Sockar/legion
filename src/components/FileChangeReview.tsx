import hljs from "highlight.js";

export interface FileChangePreview {
  operation: "create" | "edit";
  path: string;
  before: string;
  after: string;
}

interface FileChangeReviewProps {
  preview: FileChangePreview;
  disabled: boolean;
  onAccept: () => void;
  onReject: () => void;
}

function createUnifiedDiff(preview: FileChangePreview): string {
  const before = preview.before.split(/\r?\n/);
  const after = preview.after.split(/\r?\n/);
  let prefixLength = 0;
  while (
    prefixLength < before.length &&
    prefixLength < after.length &&
    before[prefixLength] === after[prefixLength]
  ) {
    prefixLength += 1;
  }

  let suffixLength = 0;
  while (
    suffixLength < before.length - prefixLength &&
    suffixLength < after.length - prefixLength &&
    before[before.length - suffixLength - 1] ===
      after[after.length - suffixLength - 1]
  ) {
    suffixLength += 1;
  }

  const contextBeforeStart = Math.max(0, prefixLength - 3);
  const contextAfterLength = Math.min(3, suffixLength);
  const beforeChangedEnd = before.length - suffixLength;
  const afterChangedEnd = after.length - suffixLength;
  const oldChanged = before.slice(prefixLength, beforeChangedEnd);
  const newChanged = after.slice(prefixLength, afterChangedEnd);
  const contextBefore = before.slice(contextBeforeStart, prefixLength);
  const contextAfter = before.slice(
    before.length - suffixLength,
    before.length - suffixLength + contextAfterLength,
  );
  const oldLineCount =
    contextBefore.length + oldChanged.length + contextAfter.length;
  const newLineCount =
    contextBefore.length + newChanged.length + contextAfter.length;
  const oldStart = contextBeforeStart + 1;
  const newStart = contextBeforeStart + 1;

  return [
    `--- a/${preview.path}`,
    `+++ b/${preview.path}`,
    `@@ -${oldStart},${oldLineCount} +${newStart},${newLineCount} @@`,
    ...contextBefore.map((line) => ` ${line}`),
    ...oldChanged.map((line) => `-${line}`),
    ...newChanged.map((line) => `+${line}`),
    ...contextAfter.map((line) => ` ${line}`),
  ].join("\n");
}

export function FileChangeReview({
  preview,
  disabled,
  onAccept,
  onReject,
}: FileChangeReviewProps) {
  const diffHtml = hljs.highlight(createUnifiedDiff(preview), {
    language: "diff",
  }).value;

  return (
    <section className="file-change-review" aria-label="Proposed file change">
      <div className="file-change-review__heading">
        <div>
          <strong>
            {preview.operation === "create" ? "Create" : "Edit"} {preview.path}
          </strong>
          <span>Review the proposed change before it is written.</span>
        </div>
        <div className="file-change-review__actions">
          <button type="button" onClick={onReject} disabled={disabled}>
            Reject
          </button>
          <button
            className="file-change-review__accept"
            type="button"
            onClick={onAccept}
            disabled={disabled}
          >
            Accept change
          </button>
        </div>
      </div>
      <pre className="file-change-review__diff">
        <code dangerouslySetInnerHTML={{ __html: diffHtml }} />
      </pre>
    </section>
  );
}
