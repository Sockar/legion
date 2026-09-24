import { useEffect, useState } from "react";

interface ReasoningSectionProps {
  hasFinalContent: boolean;
  isStreaming: boolean;
  reasoning: string;
}

export function ReasoningSection({
  hasFinalContent,
  isStreaming,
  reasoning,
}: ReasoningSectionProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    if (isStreaming && hasFinalContent) setIsExpanded(false);
  }, [hasFinalContent, isStreaming]);

  if (!reasoning) return null;

  return (
    <section className="reasoning-section">
      <button
        aria-expanded={isExpanded}
        className="reasoning-section__toggle"
        onClick={() => setIsExpanded((expanded) => !expanded)}
        type="button"
      >
        <span aria-hidden="true">*</span>
        <span>Thinking</span>
        <span aria-hidden="true">{isExpanded ? "v" : ">"}</span>
      </button>
      {isExpanded && (
        <div className="reasoning-section__content">{reasoning}</div>
      )}
    </section>
  );
}
