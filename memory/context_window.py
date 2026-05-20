"""Rolling context window: recent turns + conversation summary for the LLM."""

from __future__ import annotations

from domain.models import ChatTurn, MessageRole
from memory.sqlite_store import SqliteStore, StoredMessage
from runtime import settings

_SUMMARY_PREFIX = "Краткое содержание предыдущего диалога: "


class ContextWindowBuilder:
    """
    Builds the message list sent to the language model.
    Keeps the last N turns and injects a rolling summary of older messages.
    """

    def __init__(
        self,
        store: SqliteStore,
        *,
        max_turns: int | None = None,
        summarize_after_messages: int | None = None,
    ) -> None:
        self._store = store
        self._max_turns = max_turns or settings.MAX_SESSION_TURNS
        self._max_messages = self._max_turns * 2
        self._summarize_threshold = (
            summarize_after_messages
            if summarize_after_messages is not None
            else self._max_messages + 4
        )

    @property
    def max_messages(self) -> int:
        return self._max_messages

    def build(self, session_id: str) -> list[ChatTurn]:
        recent = self._store.get_recent_messages(
            session_id,
            limit=self._max_messages,
        )
        summary = self._store.get_session_summary(session_id)
        turns = [_to_chat_turn(msg) for msg in recent]

        if summary:
            return [
                ChatTurn(
                    role=MessageRole.SYSTEM,
                    content=_SUMMARY_PREFIX + summary,
                ),
                *turns,
            ]
        return turns

    def maybe_roll_summary(self, session_id: str) -> str | None:
        """
        When history exceeds the threshold, summarize older messages
        and store the result for future context windows.
        """
        total = self._store.count_messages(session_id)
        if total <= self._summarize_threshold:
            return None

        recent = self._store.get_recent_messages(
            session_id,
            limit=self._max_messages,
        )
        if not recent:
            return None

        oldest_recent_id = recent[0].id
        older = self._store.get_messages_before_id(session_id, oldest_recent_id)
        if not older:
            return None

        existing = self._store.get_session_summary(session_id)
        new_part = _summarize_messages(older)
        merged = _merge_summaries(existing, new_part)
        self._store.append_summary_record(
            session_id,
            merged,
            covers_until_message_id=oldest_recent_id - 1,
        )
        return merged


def _to_chat_turn(message: StoredMessage) -> ChatTurn:
    return ChatTurn(role=MessageRole(message.role), content=message.content)


def _summarize_messages(messages: list[StoredMessage], max_lines: int = 12) -> str:
    """Compact extractive summary of archived turns (no extra LLM call)."""
    lines: list[str] = []
    for msg in messages[-max_lines:]:
        label = "Клиент" if msg.role == "user" else "Ассистент"
        snippet = msg.content.replace("\n", " ").strip()[:160]
        if snippet:
            lines.append(f"{label}: {snippet}")
    return " ".join(lines)


def _merge_summaries(existing: str, new_part: str) -> str:
    if not existing:
        return new_part
    if not new_part:
        return existing
    combined = f"{existing} {new_part}"
    return combined[: settings.MEMORY_SUMMARY_MAX_CHARS]
