"""Low-level SQLite persistence for conversation sessions."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from runtime import settings

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL DEFAULT 'telegram',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id
    ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    covers_until_message_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON conversation_summaries(session_id, id DESC);
"""


@dataclass(frozen=True)
class StoredMessage:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    channel: str
    created_at: str
    updated_at: str
    message_count: int
    summary: str


class SqliteStore:
    """SQLite connection manager and data-access primitives."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or settings.MEMORY_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    @property
    def db_path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _initialize(self) -> None:
        with self.transaction():
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.execute(
                """
                INSERT INTO schema_meta (key, value)
                VALUES ('version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(_SCHEMA_VERSION),),
            )

    def ensure_session(self, session_id: str, channel: str = "telegram") -> SessionRecord:
        now = _utc_now()
        with self.transaction():
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return _row_to_session(row)

            self._conn.execute(
                """
                INSERT INTO sessions (session_id, channel, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, channel, now, now),
            )
        return SessionRecord(
            session_id=session_id,
            channel=channel,
            created_at=now,
            updated_at=now,
            message_count=0,
            summary="",
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return _row_to_session(row) if row else None

    def is_session_empty(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is None or int(row["message_count"]) == 0

    def insert_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        channel: str = "telegram",
    ) -> StoredMessage:
        now = _utc_now()
        with self.transaction():
            self.ensure_session(session_id, channel=channel)
            cursor = self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            message_id = int(cursor.lastrowid)
            self._conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, message_count = message_count + 1
                WHERE session_id = ?
                """,
                (now, session_id),
            )
        return StoredMessage(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
        )

    def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        messages = [_row_to_message(row) for row in rows]
        messages.reverse()
        return messages

    def get_messages_before_id(
        self,
        session_id: str,
        before_id: int,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        query = """
            SELECT * FROM messages
            WHERE session_id = ? AND id < ?
            ORDER BY id ASC
        """
        params: list[Any] = [session_id, before_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_message(row) for row in rows]

    def count_messages(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def update_session_summary(self, session_id: str, summary: str) -> None:
        now = _utc_now()
        with self.transaction():
            self._conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, now, session_id),
            )

    def get_session_summary(self, session_id: str) -> str:
        row = self._conn.execute(
            "SELECT summary FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return str(row["summary"]) if row and row["summary"] else ""

    def append_summary_record(
        self,
        session_id: str,
        summary: str,
        covers_until_message_id: int,
    ) -> None:
        now = _utc_now()
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO conversation_summaries
                    (session_id, summary, covers_until_message_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, summary, covers_until_message_id, now),
            )
            self._conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, now, session_id),
            )

    def search_messages(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[StoredMessage]:
        pattern = f"%{query.strip()}%"
        rows = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND content LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, pattern, limit),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    def list_sessions(self, *, limit: int = 50, active_since: str | None = None) -> list[SessionRecord]:
        if active_since:
            rows = self._conn.execute(
                """
                SELECT * FROM sessions
                WHERE updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (active_since, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_session(row) for row in rows]

    def migrate_from_json(self, json_path: Path) -> int:
        """Import legacy sessions.json / dialogs.json into SQLite."""
        if not json_path.exists():
            return 0

        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read legacy session file: %s", json_path)
            return 0

        imported = 0
        now = _utc_now()
        with self.transaction():
            for session_id, turns in raw.items():
                sid = str(session_id)
                existing = self._conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (sid,),
                ).fetchone()
                if existing and self.count_messages(sid) > 0:
                    continue

                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, channel, created_at, updated_at, message_count)
                    VALUES (?, 'telegram', ?, ?, 0)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (sid, now, now),
                )
                count = 0
                for turn in turns:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if not content:
                        continue
                    self._conn.execute(
                        """
                        INSERT INTO messages (session_id, role, content, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (sid, role, content, now),
                    )
                    count += 1
                    imported += 1
                if count:
                    self._conn.execute(
                        """
                        UPDATE sessions
                        SET message_count = ?, updated_at = ?
                        WHERE session_id = ?
                        """,
                        (count, now, sid),
                    )
        if imported:
            logger.info("Migrated %d messages from %s", imported, json_path.name)
        return imported

    def clear_session(self, session_id: str) -> None:
        with self.transaction():
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            self._conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )
            self._conn.execute(
                """
                UPDATE sessions
                SET summary = '', message_count = 0, updated_at = ?
                WHERE session_id = ?
                """,
                (_utc_now(), session_id),
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_message(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=str(row["session_id"]),
        channel=str(row["channel"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=int(row["message_count"]),
        summary=str(row["summary"] or ""),
    )
