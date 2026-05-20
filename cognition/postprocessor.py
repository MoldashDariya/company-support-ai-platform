"""Post-process raw LLM output into grounded responses with mandatory citations."""

from __future__ import annotations

import re

from cognition.citations import citations_to_source_labels, format_citations_block
from domain.models import AssistantResponse, GroundedContext
from runtime import settings

# Model sometimes adds its own source footer — strip before appending structured block
_MODEL_SOURCE_FOOTER_RE = re.compile(
    r"\n{1,2}(?:📎|📚|🔗)?\s*Источник(?:и|а)?\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)


class GroundedResponsePostprocessor:
    """Ensures every grounded answer ends with human-readable source references."""

    def process(self, raw_text: str, context: GroundedContext) -> AssistantResponse:
        if not context.has_sufficient_evidence:
            return AssistantResponse(
                text=self._insufficient_evidence_text(),
                sources=[],
                citations=[],
                grounded=False,
            )

        answer = self._normalize_answer(raw_text)
        citations = context.citations or _citations_from_fragments(context.fragments)

        if not citations:
            return AssistantResponse(
                text=answer,
                sources=[],
                citations=[],
                grounded=True,
            )

        answer = self._strip_model_source_footer(answer)
        answer = self._append_citations(answer, citations)

        return AssistantResponse(
            text=answer,
            sources=citations_to_source_labels(citations),
            citations=citations,
            grounded=True,
        )

    def _normalize_answer(self, raw_text: str) -> str:
        answer = (raw_text or "").strip()
        if not answer:
            return (
                f"Не удалось сформировать ответ. Позвоните: {settings.COMPANY_PHONE}."
            )
        if len(answer) > settings.MAX_REPLY_CHARS:
            # Reserve space for citation block
            budget = settings.MAX_REPLY_CHARS - 400
            if budget > 200:
                answer = answer[: budget - 3] + "..."
            else:
                answer = answer[: settings.MAX_REPLY_CHARS - 3] + "..."
        return answer

    def _strip_model_source_footer(self, text: str) -> str:
        return _MODEL_SOURCE_FOOTER_RE.sub("", text).rstrip()

    def _append_citations(self, answer: str, citations: list) -> str:
        block = format_citations_block(citations)
        if block and block not in answer:
            return f"{answer}{block}"
        return answer

    def _insufficient_evidence_text(self) -> str:
        from cognition.prompts import INSUFFICIENT_EVIDENCE_FALLBACK

        return INSUFFICIENT_EVIDENCE_FALLBACK


def _citations_from_fragments(context_fragments):
    from knowledge.context import build_citations_from_fragments

    return build_citations_from_fragments(context_fragments)
