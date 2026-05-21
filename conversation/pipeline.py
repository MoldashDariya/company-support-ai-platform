"""Core use-case pipeline: guard → FAQ or retrieve → generate → remember."""

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
from faq import UNSUPPORTED_FALLBACK, resolve_faq, telegram_intent_key
from knowledge.context import build_grounded_context, log_retrieval_context_stats
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

    def _faq_response(self, inquiry: UserInquiry, faq_match) -> PipelineResult:
        """Curated FAQ answer — no BM25 retrieval, no LLM."""
        session_id = inquiry.session_id
        is_first = self._memory.is_empty(session_id)
        intent_key = telegram_intent_key(faq_match.intent)
        response = AssistantResponse(text=faq_match.answer, grounded=True)

        self._memory.append(session_id, ChatTurn(MessageRole.USER, inquiry.text))
        self._memory.append(session_id, ChatTurn(MessageRole.ASSISTANT, response.text))
        self._memory.record_assistant_opening(session_id, response.text)

        logger.info(
            "FAQ answer | intent=%s retrieval=false answer_len=%d",
            faq_match.intent,
            len(response.text),
        )
        return PipelineResult(
            reply=response,
            is_first_contact=is_first,
            query_intent=intent_key,
        )

    async def _retrieve_fragments(
        self,
        query: str,
        intent: QueryIntent,
    ) -> list[KnowledgeFragment]:
        """BM25 retrieval — only for general / unknown intents."""
        logger.info("BM25 retrieval | intent=%s", intent.name)
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

        faq_match = resolve_faq(inquiry.text)
        if faq_match:
            return self._faq_response(inquiry, faq_match)

        intent = classify_query_intent(inquiry.text)
        last_opening = self._memory.get_last_assistant_opening(session_id)

        fragments = await self._retrieve_fragments(inquiry.text, intent)
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

        if not context.has_sufficient_evidence:
            logger.info("Unsupported question | fallback=true retrieval_used=true")
            fallback = AssistantResponse(text=UNSUPPORTED_FALLBACK, grounded=False)
            self._memory.append(session_id, ChatTurn(MessageRole.USER, inquiry.text))
            self._memory.append(session_id, ChatTurn(MessageRole.ASSISTANT, fallback.text))
            return PipelineResult(
                reply=fallback,
                is_first_contact=is_first,
                query_intent=intent.name,
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

        faq_match = resolve_faq(inquiry.text)
        if faq_match:
            yield ("token", faq_match.answer)
            yield ("done", self._faq_response(inquiry, faq_match))
            return

        intent = classify_query_intent(inquiry.text)
        last_opening = self._memory.get_last_assistant_opening(session_id)

        fragments = await self._retrieve_fragments(inquiry.text, intent)
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

        if not context.has_sufficient_evidence:
            logger.info("Unsupported question | fallback=true retrieval_used=true")
            yield ("token", UNSUPPORTED_FALLBACK)
            yield (
                "done",
                PipelineResult(
                    reply=AssistantResponse(text=UNSUPPORTED_FALLBACK, grounded=False),
                    is_first_contact=is_first,
                    query_intent=intent.name,
                ),
            )
            return

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
