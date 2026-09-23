import type { ChatMessage } from "../types/chat";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  isStreaming: boolean;
  messages: ChatMessage[];
  onRegenerate: (messageId: string) => void;
}

export function MessageList({
  isStreaming,
  messages,
  onRegenerate,
}: MessageListProps) {
  const lastAssistantId = [...messages]
    .reverse()
    .find((message) => message.role === "assistant")?.id;

  return (
    <div
      className="message-list"
      aria-live="polite"
      aria-relevant="additions text"
    >
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          isStreaming={isStreaming && message.id === lastAssistantId}
          canRegenerate={!isStreaming && message.id === lastAssistantId}
          onRegenerate={onRegenerate}
        />
      ))}
    </div>
  );
}
