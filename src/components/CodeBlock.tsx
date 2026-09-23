import { Children, isValidElement, useState, type ReactNode } from "react";

interface CodeBlockProps {
  children: ReactNode;
}

function extractCode(children: ReactNode): string {
  return Children.toArray(children)
    .map((child) => {
      if (typeof child === "string" || typeof child === "number") {
        return String(child);
      }

      return isValidElement<{ children?: ReactNode }>(child)
        ? extractCode(child.props.children)
        : "";
    })
    .join("");
}

export function CodeBlock({ children }: CodeBlockProps) {
  const [copyStatus, setCopyStatus] = useState("");
  const code = extractCode(children);

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopyStatus("Copied");
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setCopyStatus(`Copy failed: ${detail}`);
    }
  };

  return (
    <div className="code-block">
      <div className="code-block__toolbar">
        <span>Code</span>
        <button type="button" onClick={copyCode} aria-live="polite">
          {copyStatus || "Copy code"}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  );
}
