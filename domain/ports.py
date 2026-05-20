"""Abstract ports — infrastructure adapters implement these contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from domain.models import (
    AssistantResponse,
    ChatTurn,
    GroundedContext,
    KnowledgeFragment,
    UserInquiry,
)


class KnowledgeRetriever(Protocol):
    async def retrieve(self, query: str, limit: int = 4) -> list[KnowledgeFragment]: ...


class LanguageModel(Protocol):
    async def generate(
        self,
        system: str,
        history: list[ChatTurn],
        user_text: str,
    ) -> str: ...

    async def generate_stream(
        self,
        system: str,
        history: list[ChatTurn],
        user_text: str,
    ) -> AsyncIterator[str]: ...


class SessionMemory(Protocol):
    def is_empty(self, session_id: str) -> bool: ...

    def get_history(self, session_id: str) -> list[ChatTurn]: ...

    def append(self, session_id: str, turn: ChatTurn) -> None: ...

    def get_summary(self, session_id: str) -> str: ...

    def search_history(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ChatTurn]: ...


class InquiryGuard(Protocol):
    def validate(self, text: str) -> str | None: ...


class RequestThrottle(Protocol):
    def acquire(self, session_id: str) -> bool: ...

    def cooldown_seconds(self, session_id: str) -> int: ...


class ResponseComposer(Protocol):
    def build_system_prompt(self, context: GroundedContext) -> str: ...

    def finalize(
        self,
        raw_text: str,
        context: GroundedContext,
    ) -> AssistantResponse: ...


class SupportPipelinePort(Protocol):
    async def handle(self, inquiry: UserInquiry) -> PipelineResult: ...
