import { useCallback, useEffect, useRef, useState } from "react";
import { ChatInput } from "./components/ChatInput";
import { MessageList } from "./components/MessageList";
import type { ChatMessage } from "./types/chat";
import { InMemoryChatBackend } from "./services/chatBackend";
import "./App.css";

const backend = new InMemoryChatBackend();
let nextMessageId = 0;

function createMessageId() {
  nextMessageId += 1;
  return `message-${Date.now()}-${nextMessageId}`;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const generateResponse = useCallback(
    async (prompt: string, assistantId: string, addUserMessage: boolean) => {
      if (controllerRef.current) return;

      const controller = new AbortController();
      controllerRef.current = controller;
      setIsStreaming(true);

      if (addUserMessage) {
        setMessages((current) => [
          ...current,
          { id: createMessageId(), role: "user", content: prompt },
          { id: assistantId, role: "assistant", content: "" },
        ]);
      } else {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, content: "" } : message,
          ),
        );
      }

      try {
        for await (const token of backend.streamResponse(
          prompt,
          controller.signal,
        )) {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + token }
                : message,
            ),
          );
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          const detail = error instanceof Error ? error.message : String(error);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: `The response failed: ${detail}`,
                  }
                : message,
            ),
          );
        }
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
          setIsStreaming(false);
        }
      }
    },
    [],
  );

  const handleSend = useCallback(
    (content: string) => {
      void generateResponse(content, createMessageId(), true);
    },
    [generateResponse],
  );

  const handleStop = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const handleRegenerate = useCallback(
    (assistantId: string) => {
      const assistantIndex = messages.findIndex(
        (message) => message.id === assistantId,
      );
      const previousUserMessage = messages
        .slice(0, assistantIndex)
        .reverse()
        .find((message) => message.role === "user");

      if (previousUserMessage) {
        void generateResponse(previousUserMessage.content, assistantId, false);
      }
    },
    [generateResponse, messages],
  );

  return (
    <main className="chat-app">
      <header className="chat-header">
        <div className="chat-header__brand">
          <span className="chat-header__mark" aria-hidden="true">
            L
          </span>
          <div>
            <h1>Legion</h1>
            <p>Local coding assistant</p>
          </div>
        </div>
        <span className="connection-status">
          <span className="connection-status__dot" aria-hidden="true" />
          Mock backend
        </span>
      </header>

      <section className="conversation" aria-label="Conversation">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state__mark" aria-hidden="true">
              L
            </div>
            <h2>What can I help you build?</h2>
            <p>Ask a question or describe a coding task to get started.</p>
          </div>
        ) : (
          <MessageList
            messages={messages}
            isStreaming={isStreaming}
            onRegenerate={handleRegenerate}
          />
        )}
        <div ref={bottomRef} />
      </section>

      <footer className="composer">
        {isStreaming && (
          <button className="stop-button" type="button" onClick={handleStop}>
            Stop generating
          </button>
        )}
        <ChatInput onSend={handleSend} disabled={isStreaming} />
        <p className="composer__hint">
          Enter to send · Shift+Enter for a new line
        </p>
      </footer>
    </main>
  );
}

export default App;
