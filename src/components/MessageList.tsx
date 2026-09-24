import { Fragment } from "react";
import type { ApprovalDecision, ToolApprovalRequest } from "../lib/ollama";
import type { ChatMessage } from "../types/chat";
import { MessageBubble } from "./MessageBubble";
import { ToolApprovalCell } from "./ToolApprovalCell";

interface MessageListProps {
  approvals: {
    approval: ToolApprovalRequest;
    decision?: ApprovalDecision;
    disabled: boolean;
    onDecision: (decision: ApprovalDecision) => void;
    assistantId: string;
  }[];
  isStreaming: boolean;
  messages: ChatMessage[];
  onRegenerate: (messageId: string) => void;
}

export function MessageList({
  approvals,
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
        <Fragment key={message.id}>
          <MessageBubble
            message={message}
            isStreaming={isStreaming && message.id === lastAssistantId}
            canRegenerate={!isStreaming && message.id === lastAssistantId}
            onRegenerate={onRegenerate}
          />
          {message.role === "assistant" &&
            approvals
              .filter((item) => item.assistantId === message.id)
              .map((item) => (
                <ToolApprovalCell
                  key={item.approval.approval_id}
                  approval={item.approval}
                  decision={item.decision}
                  disabled={item.disabled}
                  onDecision={item.onDecision}
                />
              ))}
        </Fragment>
      ))}
    </div>
  );
}
