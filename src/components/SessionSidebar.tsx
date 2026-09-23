import { useRef, useState, type KeyboardEvent } from "react";
import type { ChatSession } from "../types/session";

interface SessionSidebarProps {
  activeSessionId: string | null;
  sessions: ChatSession[];
  onArchive: (sessionId: string) => void;
  onCreate: () => void;
  onDelete: (sessionId: string) => void;
  onRename: (sessionId: string, name: string) => void;
  onRestore: (sessionId: string) => void;
  onSelect: (sessionId: string) => void;
}

function SessionRow({
  active,
  archived,
  session,
  onArchive,
  onDelete,
  onRename,
  onRestore,
  onSelect,
}: {
  active: boolean;
  archived: boolean;
  session: ChatSession;
  onArchive: () => void;
  onDelete: () => void;
  onRename: (name: string) => void;
  onRestore: () => void;
  onSelect: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(session.name);
  const cancelRenameRef = useRef(false);

  const finishRename = () => {
    const trimmedName = name.trim();
    if (trimmedName) onRename(trimmedName);
    else setName(session.name);
    setRenaming(false);
  };

  const handleRenameKey = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.blur();
    } else if (event.key === "Escape") {
      cancelRenameRef.current = true;
      setName(session.name);
      event.currentTarget.blur();
    }
  };

  return (
    <li className="session-row">
      {renaming ? (
        <input
          aria-label="Session name"
          autoFocus
          className="session-row__rename"
          value={name}
          onBlur={() => {
            if (cancelRenameRef.current) {
              cancelRenameRef.current = false;
              setRenaming(false);
              return;
            }
            finishRename();
          }}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={handleRenameKey}
        />
      ) : (
        <button
          aria-current={active ? "page" : undefined}
          className={`session-row__select${active ? " session-row__select--active" : ""}`}
          onClick={onSelect}
          type="button"
        >
          <span className="session-row__name">{session.name}</span>
          <span className="session-row__workspace">
            {session.workspacePath}
          </span>
        </button>
      )}
      <div className="session-row__actions">
        {!archived && (
          <>
            <button
              aria-label={`Rename ${session.name}`}
              onClick={() => {
                cancelRenameRef.current = false;
                setName(session.name);
                setRenaming(true);
              }}
              title="Rename session"
              type="button"
            >
              Rename
            </button>
            <button
              aria-label={`Archive ${session.name}`}
              onClick={onArchive}
              title="Archive session"
              type="button"
            >
              Archive
            </button>
          </>
        )}
        {archived && (
          <button
            aria-label={`Restore ${session.name}`}
            onClick={onRestore}
            title="Restore session"
            type="button"
          >
            Restore
          </button>
        )}
        <button
          aria-label={`Delete ${session.name}`}
          onClick={onDelete}
          title="Delete session"
          type="button"
        >
          Delete
        </button>
      </div>
    </li>
  );
}

export function SessionSidebar({
  activeSessionId,
  sessions,
  onArchive,
  onCreate,
  onDelete,
  onRename,
  onRestore,
  onSelect,
}: SessionSidebarProps) {
  const [showArchived, setShowArchived] = useState(false);
  const visibleSessions = sessions
    .filter((session) => (session.archivedAt !== null) === showArchived)
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return (
    <aside aria-label="Chat sessions" className="session-sidebar">
      <div className="session-sidebar__header">
        <h2>Sessions</h2>
        <button
          className="session-sidebar__new"
          onClick={onCreate}
          type="button"
        >
          + New
        </button>
      </div>
      <button
        aria-pressed={showArchived}
        className="session-sidebar__archived-toggle"
        onClick={() => setShowArchived((current) => !current)}
        type="button"
      >
        {showArchived ? "Show active sessions" : "Show archived sessions"}
      </button>
      {visibleSessions.length ? (
        <ul className="session-list">
          {visibleSessions.map((session) => (
            <SessionRow
              key={session.id}
              active={session.id === activeSessionId}
              archived={session.archivedAt !== null}
              session={session}
              onArchive={() => onArchive(session.id)}
              onDelete={() => onDelete(session.id)}
              onRename={(name) => onRename(session.id, name)}
              onRestore={() => onRestore(session.id)}
              onSelect={() => onSelect(session.id)}
            />
          ))}
        </ul>
      ) : (
        <p className="session-sidebar__empty">
          {showArchived ? "No archived sessions." : "No sessions yet."}
        </p>
      )}
    </aside>
  );
}
