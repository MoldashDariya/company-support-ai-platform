"""Compose system prompts and delegate response finalization to the postprocessor."""

from __future__ import annotations

from cognition.postprocessor import GroundedResponsePostprocessor
from cognition.prompts import build_grounded_system_prompt
from domain.models import AssistantResponse, GroundedContext


class GroundedResponseComposer:
    def __init__(self, postprocessor: GroundedResponsePostprocessor | None = None) -> None:
        self._postprocessor = postprocessor or GroundedResponsePostprocessor()

    def build_system_prompt(self, context: GroundedContext) -> str:
        from cognition.intent import QueryIntent
        from runtime import settings

        evidence = context.formatted
        if len(evidence) > settings.MAX_GROUNDED_CONTEXT_CHARS:
            evidence = evidence[: settings.MAX_GROUNDED_CONTEXT_CHARS] + "\n[…]"
        intent = QueryIntent(
            context.query_intent,
            context.retrieval_profile,
            context.response_style,
        )
        return build_grounded_system_prompt(
            evidence,
            context.citations,
            intent=intent,
            last_assistant_opening=context.last_assistant_opening,
        )

    def finalize(self, raw_text: str, context: GroundedContext) -> AssistantResponse:
        return self._postprocessor.process(raw_text, context)
