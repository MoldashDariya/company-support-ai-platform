"""Grounded answer generation — retrieval context + constrained LLM inference."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from cognition.composer import GroundedResponseComposer
from cognition.provider import CerebrasLanguageModel
from domain.models import AssistantResponse, ChatTurn, GroundedContext

logger = logging.getLogger(__name__)


class GroundedAnswerEngine:
    """Orchestrates evidence-conditioned language model calls."""

    def __init__(
        self,
        llm: CerebrasLanguageModel | None = None,
        composer: GroundedResponseComposer | None = None,
    ) -> None:
        self._llm = llm or CerebrasLanguageModel()
        self._composer = composer or GroundedResponseComposer()

    async def answer(
        self,
        user_text: str,
        context: GroundedContext,
        history: list[ChatTurn],
    ) -> AssistantResponse:
        raw = ""
        async for chunk in self.answer_stream(user_text, context, history):
            raw += chunk
        return self._composer.finalize(raw, context)

    async def answer_stream(
        self,
        user_text: str,
        context: GroundedContext,
        history: list[ChatTurn],
    ) -> AsyncIterator[str]:
        if not context.has_sufficient_evidence:
            from cognition.prompts import EMPTY_RETRIEVAL_FALLBACK

            logger.info(
                "Answer generated | len=%d fallback_used=true retrieved_context_used=false",
                len(EMPTY_RETRIEVAL_FALLBACK),
            )
            yield EMPTY_RETRIEVAL_FALLBACK
            return

        system = self._composer.build_system_prompt(context)
        try:
            async for token in self._llm.generate_stream(system, history, user_text):
                yield token
        except Exception:
            logger.exception(
                "LLM stream failed | system_chars=%d history_turns=%d",
                len(system),
                len(history),
            )
            from cognition.postprocessor import synthesize_from_context

            recovered = synthesize_from_context(context)
            logger.warning(
                "LLM stream failed; using context synthesis | len=%d",
                len(recovered),
            )
            yield recovered

    def finalize_stream(
        self,
        raw_text: str,
        context: GroundedContext,
    ) -> AssistantResponse:
        return self._composer.finalize(raw_text, context)
