"""Post-process raw LLM output into grounded responses with mandatory citations."""

from __future__ import annotations

import logging
import re

from cognition.citations import citations_to_source_labels, format_citations_block
from cognition.intent import QueryIntent, extract_opening, openings_similar
from domain.models import AssistantResponse, GroundedContext
from runtime import settings

logger = logging.getLogger(__name__)

_MODEL_SOURCE_FOOTER_RE = re.compile(
    r"\n{1,2}(?:📎|📚|🔗)?\s*Источник(?:и|а)?\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)

_REFUSAL_PHRASE_RE = re.compile(
    r"(в моей базе знаний нет|я не нашел|я не нашёл|не нашел информации|"
    r"не нашёл информации|нет точной информации|нет этой информации|"
    r"не могу ответить|информации нет в базе)",
    re.IGNORECASE,
)

_EXCESSIVE_CITATION_RE = re.compile(r"\[\d+\]")

_CLICHE_PHRASES = (
    "широкий ассортимент",
    "опытные специалисты",
    "высокое качество",
    "надёжные поставщики",
    "надежные поставщики",
)

_INTENT_OPENERS: dict[str, str] = {
    "company_overview": "Кратко о компании:",
    "services": "По услугам магазина:",
    "products": "В ассортименте:",
    "delivery": "По доставке:",
    "tinting": "По колеровке и подбору цвета:",
    "technologies": "По технологиям покрытий:",
    "contacts": "Связаться с нами:",
    "loyalty_program": "По акциям и программам:",
    "interiors_design": "Для интерьера и декора:",
    "brands": "Представленные бренды:",
}


class GroundedResponsePostprocessor:
    """Ensures grounded answers synthesize from evidence; compact citations."""

    def process(self, raw_text: str, context: GroundedContext) -> AssistantResponse:
        if not context.has_sufficient_evidence:
            from cognition.prompts import EMPTY_RETRIEVAL_FALLBACK

            _log_answer_stats(
                answer_len=len(EMPTY_RETRIEVAL_FALLBACK),
                fallback_used=True,
                context_used=False,
                context=context,
            )
            return AssistantResponse(
                text=EMPTY_RETRIEVAL_FALLBACK,
                sources=[],
                citations=[],
                grounded=False,
            )

        answer = self._normalize_answer(raw_text)
        fallback_used = False

        if not answer or _is_refusal_answer(answer):
            answer = synthesize_from_context(context)
            fallback_used = True

        answer = self._strip_model_source_footer(answer)
        answer = _trim_inline_citations(answer, max_refs=2)
        answer = _dedupe_cliches(answer)
        answer = _apply_word_limits(answer)
        answer = _avoid_repeated_opening(answer, context)

        citations = (context.citations or _citations_from_fragments(context.fragments))[:2]
        if citations:
            answer = self._append_citations(answer, citations)

        _log_answer_stats(
            answer_len=len(answer),
            fallback_used=fallback_used,
            context_used=True,
            context=context,
        )

        return AssistantResponse(
            text=answer,
            sources=citations_to_source_labels(citations),
            citations=citations,
            grounded=True,
        )

    def _normalize_answer(self, raw_text: str) -> str:
        return (raw_text or "").strip()

    def _strip_model_source_footer(self, text: str) -> str:
        return _MODEL_SOURCE_FOOTER_RE.sub("", text).rstrip()

    def _append_citations(self, answer: str, citations: list) -> str:
        block = format_citations_block(citations, max_items=2)
        if block and block not in answer:
            return f"{answer}{block}"
        return answer


def synthesize_from_context(context: GroundedContext) -> str:
    """Intent-aware synthesis when the LLM refuses or returns empty."""
    fragments = context.fragments
    if not fragments:
        from cognition.prompts import EMPTY_RETRIEVAL_FALLBACK

        return EMPTY_RETRIEVAL_FALLBACK

    intent_name = context.query_intent or "general"
    opener = _INTENT_OPENERS.get(intent_name, "По вашему вопросу:")

    if intent_name == "delivery":
        return _synthesize_delivery(fragments, opener)
    if intent_name == "products":
        return _synthesize_products(fragments, opener)
    if intent_name == "tinting":
        return _synthesize_tinting(fragments, opener)
    if intent_name == "technologies":
        return _synthesize_technologies(fragments, opener)
    if intent_name == "services":
        return _synthesize_services(fragments, opener)
    if intent_name == "company_overview":
        return _synthesize_overview(fragments, opener)
    if intent_name == "contacts":
        return _synthesize_contacts(fragments, opener)
    if intent_name in ("loyalty_program", "brands", "interiors_design"):
        return _synthesize_from_snippets(fragments, opener)

    return _synthesize_from_snippets(fragments, opener)


def _synthesize_overview(fragments: list, opener: str) -> str:
    caps: list[str] = []
    for fragment in fragments:
        if _is_about_fragment(fragment.section.lower(), fragment.source_url.lower()):
            caps.extend(_capability_lines(fragment.body))
    if caps:
        unique = _unique_phrases(caps)[:6]
        body = (
            f"«{settings.COMPANY_NAME}» — магазин лакокрасочных материалов: "
            f"краски, покрытия, инструменты, колеровка и доставка. "
            f"Ключевое: {', '.join(unique)}."
        )
        return _apply_word_limits(f"{opener} {body}")
    return _synthesize_from_snippets(fragments, opener)


def _synthesize_delivery(fragments: list, opener: str) -> str:
    facts: list[str] = []
    for fragment in fragments:
        if "доставк" in (fragment.section + fragment.body).lower():
            snippet = fragment.body.replace("\n", " ")[:220]
            if snippet:
                facts.append(snippet)
    if facts:
        return _apply_word_limits(f"{opener} {' '.join(facts[:2])}")
    return _synthesize_from_snippets(fragments, opener)


def _synthesize_products(fragments: list, opener: str) -> str:
    items: list[str] = []
    for fragment in fragments:
        body = fragment.body.lower()
        if any(k in body for k in ("краск", "лак", "инструмент", "грунт", "эмаль")):
            items.extend(_capability_lines(fragment.body)[:5])
    if items:
        return _apply_word_limits(
            f"{opener} В каталоге: {', '.join(_unique_phrases(items)[:8])}."
        )
    return _synthesize_from_snippets(fragments, opener)


def _synthesize_tinting(fragments: list, opener: str) -> str:
    for fragment in fragments:
        hay = (fragment.section + fragment.body).lower()
        if any(k in hay for k in ("колер", "ral", "ncs", "цвет")):
            return _apply_word_limits(
                f"{opener} {fragment.body.replace(chr(10), ' ')[:350]}"
            )
    return _synthesize_from_snippets(fragments, opener)


def _synthesize_technologies(fragments: list, opener: str) -> str:
    for fragment in fragments:
        hay = (fragment.section + fragment.body).lower()
        if any(k in hay for k in ("технолог", "покрыт", "влаг", "стойк", "фасад")):
            return _apply_word_limits(
                f"{opener} {fragment.body.replace(chr(10), ' ')[:350]}"
            )
    return _synthesize_from_snippets(fragments, opener)


def _synthesize_services(fragments: list, opener: str) -> str:
    parts: list[str] = []
    for fragment in fragments:
        hay = (fragment.section + fragment.source_url).lower()
        sec = fragment.section.lower()
        if "доставк" in sec or "колер" in sec or "наши возможности" in sec:
            parts.extend(_capability_lines(fragment.body)[:4])
    if parts:
        return _apply_word_limits(
            f"{opener} Доступно: {', '.join(_unique_phrases(parts)[:6])}."
        )
    return _synthesize_from_snippets(fragments, opener)


def _synthesize_contacts(fragments: list, opener: str) -> str:
    for fragment in fragments:
        if "астана" in fragment.body.lower() or "телефон" in fragment.body.lower():
            return _apply_word_limits(f"{opener} {fragment.body[:300]}")
    return (
        f"{opener} Телефон: {settings.COMPANY_PHONE}, сайт: {settings.COMPANY_SITE}"
    )


def _synthesize_from_snippets(fragments: list, opener: str) -> str:
    snippets = [
        f.body.replace("\n", " ").strip()[:200]
        for f in fragments[:2]
        if f.body.strip()
    ]
    if snippets:
        return _apply_word_limits(f"{opener} {' '.join(snippets)}")
    return f"{opener} Подробности на {settings.COMPANY_SITE}"


def _is_about_fragment(section: str, url: str) -> bool:
    if any(x in url for x in ("privacy", "offer", "agreement", "howto")):
        return False
    if url.rstrip("/").endswith("/about") or "наши возможности" in section:
        return True
    return "о магазине" in section and "политика" not in section


def _capability_lines(body: str) -> list[str]:
    lines: list[str] = []
    for line in body.split("\n"):
        clean = line.strip().lstrip("-•#").strip()
        if 3 < len(clean) < 70 and not clean.startswith("http"):
            lines.append(clean[0].lower() + clean[1:] if clean else clean)
    return lines


def _unique_phrases(phrases: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def _dedupe_cliches(text: str) -> str:
    result = text
    for cliche in _CLICHE_PHRASES:
        pattern = re.compile(re.escape(cliche), re.IGNORECASE)
        count = 0

        def repl(match: re.Match) -> str:
            nonlocal count
            count += 1
            return match.group(0) if count == 1 else ""

        result = pattern.sub(repl, result)
    return re.sub(r"\s{2,}", " ", result).strip()


def _apply_word_limits(text: str) -> str:
    words = text.split()
    if len(words) > settings.ANSWER_MAX_WORDS:
        words = words[: settings.ANSWER_MAX_WORDS]
        text = " ".join(words).rstrip(",;:") + "."
    return _trim_sentences(text, max_sentences=5)


def _avoid_repeated_opening(answer: str, context: GroundedContext) -> str:
    last = context.last_assistant_opening
    if not last or not openings_similar(extract_opening(answer), last):
        return answer

    opener = _INTENT_OPENERS.get(context.query_intent, "")
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    if len(sentences) > 1:
        rest = " ".join(sentences[1:])
        return f"{opener} {rest}".strip() if opener else rest
    if opener:
        return f"{opener} {answer}"
    return answer


def _trim_sentences(text: str, *, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max_sentences]).strip()


def _is_refusal_answer(text: str) -> bool:
    return bool(_REFUSAL_PHRASE_RE.search(text))


def _trim_inline_citations(text: str, *, max_refs: int) -> str:
    refs = _EXCESSIVE_CITATION_RE.findall(text)
    if len(refs) <= max_refs:
        return text
    seen = 0

    def repl(match: re.Match) -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= max_refs else ""

    return re.sub(r"\[\d+\]", repl, text).replace("  ", " ").strip()


def _log_answer_stats(
    *,
    answer_len: int,
    fallback_used: bool,
    context_used: bool,
    context: GroundedContext,
) -> None:
    logger.info(
        "Answer generated | len=%d fallback_used=%s retrieved_context_used=%s "
        "intent=%s response_style=%s",
        answer_len,
        fallback_used,
        context_used,
        context.query_intent,
        context.response_style,
    )


def _citations_from_fragments(context_fragments):
    from knowledge.context import build_citations_from_fragments

    return build_citations_from_fragments(context_fragments)
