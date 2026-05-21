"""Session repository — implements SessionMemory over SQLite storage."""

from __future__ import annotations

import logging
from pathlib import Path

from domain.models import ChatTurn, MessageRole
from memory.context_window import ContextWindowBuilder
from memory.sqlite_store import SqliteStore
from runtime import settings

logger = logging.getLogger(__name__)


class SessionRepository:
    """
    Channel-agnostic conversation memory.
    Satisfies the SessionMemory port used by the conversation pipeline.
    """

    def __init__(
        self,
        store: SqliteStore | None = None,
        window: ContextWindowBuilder | None = None,
        *,
        migrate_legacy: bool = True,
    ) -> None:
        self._store = store or SqliteStore()
        self._window = window or ContextWindowBuilder(self._store)
        self._last_openings: dict[str, str] = {}
        if migrate_legacy:
            self._migrate_legacy_json()

    def get_last_assistant_opening(self, session_id: str) -> str:
        return self._last_openings.get(session_id, "")

    def record_assistant_opening(self, session_id: str, answer_text: str) -> None:
        from cognition.intent import extract_opening

        opening = extract_opening(answer_text)
        if opening:
            self._last_openings[session_id] = opening

    def is_empty(self, session_id: str) -> bool:
        return self._store.is_session_empty(session_id)

    def get_history(self, session_id: str) -> list[ChatTurn]:
        """Rolling window: summary of older turns + recent messages."""
        return self._window.build(session_id)

    def append(self, session_id: str, turn: ChatTurn, *, channel: str = "telegram") -> None:
        self._store.insert_message(
            session_id,
            turn.role.value,
            turn.content,
            channel=channel,
        )
        self._window.maybe_roll_summary(session_id)

    def get_summary(self, session_id: str) -> str:
        return self._store.get_session_summary(session_id)

    def search_history(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ChatTurn]:
        """Scalable retrieval of past messages by keyword (full history in DB)."""
        messages = self._store.search_messages(session_id, query, limit=limit)
        return [
            ChatTurn(role=MessageRole(msg.role), content=msg.content)
            for msg in messages
        ]

    def clear_session(self, session_id: str) -> None:
        self._store.clear_session(session_id)

    def _migrate_legacy_json(self) -> None:
        legacy_paths = [
            settings.SESSION_STORE_PATH,
            settings.LEGACY_DIALOGS_PATH,
        ]
        for path in legacy_paths:
            if path.suffix == ".json" and path.exists():
                count = self._store.migrate_from_json(path)
                if count:
                    logger.info("Legacy JSON migrated from %s", path)
