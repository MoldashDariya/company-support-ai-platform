"""Compose system prompts and delegate response finalization to the postprocessor."""

from __future__ import annotations

from cognition.postprocessor import GroundedResponsePostprocessor
from cognition.prompts import build_grounded_system_prompt
from domain.models import AssistantResponse, GroundedContext


class GroundedResponseComposer:
    def __init__(self, postprocessor: GroundedResponsePostprocessor | None = None) -> None:
        self._postprocessor = postprocessor or GroundedResponsePostprocessor()

    def build_system_prompt(self, context: GroundedContext) -> str:
        return build_grounded_system_prompt(context.formatted, context.citations)

    def finalize(self, raw_text: str, context: GroundedContext) -> AssistantResponse:
        return self._postprocessor.process(raw_text, context)
