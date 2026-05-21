"""Core use-case pipeline: guard → retrieve → generate → remember."""

from __future__ import annotations

import logging

from cognition.engine import GroundedAnswerEngine
from cognition.intent import QueryIntent, classify_query_intent
from domain.models import (
    AssistantResponse,
    ChatTurn,
    KnowledgeFragment,
    MessageRole,
    PipelineResult,
    UserInquiry,
)
from knowledge.context import build_grounded_context, log_retrieval_context_stats
from knowledge.faq import match_faq
from domain.ports import KnowledgeRetriever, SessionMemory
from runtime import settings

logger = logging.getLogger(__name__)


class SupportConversationPipeline:
    """
    Application service coordinating retrieval-augmented support responses.
    Channel adapters delegate here — no business logic in Telegram handlers.
    """

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        engine: GroundedAnswerEngine,
        memory: SessionMemory,
        guard,
        throttle,
    ) -> None:
        self._retriever = retriever
        self._engine = engine
        self._memory = memory
        self._guard = guard
        self._throttle = throttle

    async def _resolve_fragments(
        self,
        query: str,
        intent: QueryIntent,
    ) -> list[KnowledgeFragment]:
        """FAQ first; BM25 retrieval when no confident FAQ match."""
        faq_hit = match_faq(query, intent)
        if faq_hit:
            _category, fragments = faq_hit
            return fragments
        return await self._retriever.retrieve(
            query,
            limit=settings.RETRIEVAL_TOP_K,
            min_score=settings.RETRIEVAL_MIN_SCORE,
            intent=intent,
        )

    async def handle(self, inquiry: UserInquiry) -> PipelineResult:
        session_id = inquiry.session_id
        is_first = self._memory.is_empty(session_id)

        rejection = self._guard.validate(inquiry.text)
        if rejection:
            return PipelineResult(
                reply=AssistantResponse(text=rejection, grounded=False),
                is_first_contact=is_first,
                blocked=True,
                block_reason="input_guard",
            )

        if not self._throttle.acquire(session_id):
            wait = self._throttle.cooldown_seconds(session_id)
            return PipelineResult(
                reply=AssistantResponse(
                    text=(
                        f"Слишком много запросов. Подождите {wait} сек. "
                        f"или позвоните: {settings.COMPANY_PHONE}."
                    ),
                    grounded=False,
                ),
                is_first_contact=is_first,
                blocked=True,
                block_reason="throttle",
            )

        intent = classify_query_intent(inquiry.text)
        last_opening = self._memory.get_last_assistant_opening(session_id)

        fragments = await self._resolve_fragments(inquiry.text, intent)
        context = build_grounded_context(
            fragments,
            intent=intent,
            last_assistant_opening=last_opening,
        )
        log_retrieval_context_stats(inquiry.text, fragments, context)
        logger.info(
            "Query routing | intent=%s retrieval_profile=%s response_style=%s",
            intent.name,
            intent.retrieval_profile,
            intent.response_style,
        )
        history = self._memory.get_history(session_id)

        response = await self._engine.answer(inquiry.text, context, history)

        self._memory.append(session_id, ChatTurn(MessageRole.USER, inquiry.text))
        self._memory.append(
            session_id,
            ChatTurn(MessageRole.ASSISTANT, response.text),
        )
        self._memory.record_assistant_opening(session_id, response.text)

        logger.info(
            "session=%s query_len=%d answer_len=%d citations=%d grounded=%s",
            session_id,
            len(inquiry.text),
            len(response.text),
            len(response.citations),
            response.grounded,
        )

        return PipelineResult(
            reply=response,
            is_first_contact=is_first,
            query_intent=intent.name,
        )

    async def handle_stream(self, inquiry: UserInquiry):
        """Async generator for streaming channel presenters."""
        session_id = inquiry.session_id
        is_first = self._memory.is_empty(session_id)

        rejection = self._guard.validate(inquiry.text)
        if rejection:
            yield (
                "done",
                PipelineResult(
                    reply=AssistantResponse(text=rejection, grounded=False),
                    is_first_contact=is_first,
                    blocked=True,
                    block_reason="input_guard",
                ),
            )
            return

        if not self._throttle.acquire(session_id):
            wait = self._throttle.cooldown_seconds(session_id)
            yield (
                "done",
                PipelineResult(
                    reply=AssistantResponse(
                        text=(
                            f"Слишком много запросов. Подождите {wait} сек. "
                            f"или позвоните: {settings.COMPANY_PHONE}."
                        ),
                        grounded=False,
                    ),
                    is_first_contact=is_first,
                    blocked=True,
                    block_reason="throttle",
                ),
            )
            return

        intent = classify_query_intent(inquiry.text)
        last_opening = self._memory.get_last_assistant_opening(session_id)

        fragments = await self._resolve_fragments(inquiry.text, intent)
        context = build_grounded_context(
            fragments,
            intent=intent,
            last_assistant_opening=last_opening,
        )
        log_retrieval_context_stats(inquiry.text, fragments, context)
        logger.info(
            "Query routing | intent=%s retrieval_profile=%s response_style=%s",
            intent.name,
            intent.retrieval_profile,
            intent.response_style,
        )
        history = self._memory.get_history(session_id)

        raw = ""
        async for token in self._engine.answer_stream(inquiry.text, context, history):
            raw += token
            yield ("token", token)

        response = self._engine.finalize_stream(raw, context)
        self._memory.append(session_id, ChatTurn(MessageRole.USER, inquiry.text))
        self._memory.append(
            session_id,
            ChatTurn(MessageRole.ASSISTANT, response.text),
        )
        self._memory.record_assistant_opening(session_id, response.text)

        yield (
            "done",
            PipelineResult(
                reply=response,
                is_first_contact=is_first,
                query_intent=intent.name,
            ),
        )
