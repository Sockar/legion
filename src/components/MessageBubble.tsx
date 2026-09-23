import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import type { ChatMessage } from "../types/chat";
import { CodeBlock } from "./CodeBlock";

interface MessageBubbleProps {
  canRegenerate: boolean;
  isStreaming: boolean;
  message: ChatMessage;
  onRegenerate: (messageId: string) => void;
}

export function MessageBubble({
  canRegenerate,
  isStreaming,
  message,
  onRegenerate,
}: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <article className={`message message--${message.role}`}>
      <div className="message__avatar" aria-hidden="true">
        {isUser ? "Y" : "L"}
      </div>
      <div className="message__body">
        <div className="message__heading">
          <span>{isUser ? "You" : "Legion"}</span>
          {canRegenerate && (
            <button
              className="message__action"
              type="button"
              onClick={() => onRegenerate(message.id)}
            >
              Regenerate
            </button>
          )}
        </div>
        {isUser ? (
          <div className="message__text">{message.content}</div>
        ) : message.content ? (
          <div className="markdown-content">
            <ReactMarkdown
              rehypePlugins={[rehypeHighlight]}
              components={{
                pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ) : isStreaming ? (
          <div className="typing-indicator" aria-label="Legion is responding">
            <span />
            <span />
            <span />
          </div>
        ) : null}
      </div>
    </article>
  );
}
