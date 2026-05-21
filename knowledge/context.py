"""Format retrieved fragments into LLM-ready grounded context with citations."""

from __future__ import annotations

import logging

from domain.models import GroundedContext, KnowledgeFragment, SourceCitation

logger = logging.getLogger(__name__)


def build_grounded_context(
    fragments: list[KnowledgeFragment],
    min_score: float | None = None,
    *,
    intent: "QueryIntent | None" = None,
    last_assistant_opening: str = "",
) -> GroundedContext:
    from cognition.intent import QueryIntent, filter_fragments_for_intent
    if not fragments:
        return GroundedContext(
            fragments=[],
            formatted="[Нет релевантных фрагментов в базе знаний компании.]",
            citations=[],
            has_sufficient_evidence=False,
        )

    relevant = _filter_by_retrieval_source(fragments, min_score)
    if not relevant:
        relevant = fragments[:]

    from runtime import settings

    qi = intent or QueryIntent.general()
    relevant = filter_fragments_for_intent(relevant, qi)
    capped = relevant[: settings.RETRIEVAL_TOP_K]
    citations = build_citations_from_fragments(capped)
    ref_by_key = {c.dedupe_key(): c.ref_id for c in citations}
    tagged_fragments = [_attach_citation_ref(frag, ref_by_key) for frag in capped]

    parts = [_format_evidence_fragment(frag) for frag in tagged_fragments]
    formatted, trimmed_fragments = _truncate_formatted_context(parts, tagged_fragments)

    return GroundedContext(
        fragments=trimmed_fragments,
        formatted=formatted,
        citations=citations[: min(2, len(trimmed_fragments))],
        has_sufficient_evidence=True,
        query_intent=qi.name,
        retrieval_profile=qi.retrieval_profile,
        response_style=qi.response_style,
        last_assistant_opening=last_assistant_opening,
    )


def log_retrieval_context_stats(
    query: str,
    fragments: list[KnowledgeFragment],
    context: GroundedContext,
    *,
    system_prompt_chars: int | None = None,
) -> None:
    """Safe diagnostics: counts and char sizes only (no secrets or full bodies)."""
    from cognition.prompts import build_grounded_system_prompt

    preview = (query or "").strip().replace("\n", " ")[:120]
    if len((query or "").strip()) > 120:
        preview += "…"

    prompt_chars = system_prompt_chars
    if prompt_chars is None:
        prompt_chars = len(build_grounded_system_prompt(context.formatted, context.citations))

    logger.info(
        "Retrieval stats | query=%r chunks=%d grounded_chars=%d prompt_chars=%d",
        preview,
        len(fragments),
        len(context.formatted),
        prompt_chars,
    )
    for idx, fragment in enumerate(context.fragments[:4], start=1):
        section = (fragment.section or "—")[:70]
        logger.info(
            "  #%d score=%s section=%s body_chars=%d",
            idx,
            f"{fragment.score:.3f}" if fragment.score else "n/a",
            section,
            len(fragment.body),
        )


def _truncate_formatted_context(
    parts: list[str],
    fragments: list[KnowledgeFragment],
) -> tuple[str, list[KnowledgeFragment]]:
    from runtime import settings

    max_chars = settings.MAX_GROUNDED_CONTEXT_CHARS
    joined = "\n\n".join(parts)
    if len(joined) <= max_chars:
        return joined, fragments

    logger.warning(
        "Grounded context trimmed | before=%d after_cap=%d fragments=%d→",
        len(joined),
        max_chars,
        len(fragments),
    )

    kept_parts: list[str] = []
    kept_fragments: list[KnowledgeFragment] = []
    total = 0
    separator = 2  # len("\n\n")

    for part, fragment in zip(parts, fragments):
        chunk_len = len(part) + (separator if kept_parts else 0)
        if total + chunk_len > max_chars:
            remaining = max_chars - total - (separator if kept_parts else 0)
            if remaining > 200:
                kept_parts.append(part[:remaining] + "\n[…]")
                kept_fragments.append(fragment)
            break
        kept_parts.append(part)
        kept_fragments.append(fragment)
        total += chunk_len

    if not kept_parts and parts:
        kept_parts = [parts[0][:max_chars] + "\n[…]"]
        kept_fragments = [fragments[0]]

    return "\n\n".join(kept_parts), kept_fragments


def _filter_by_retrieval_source(
    fragments: list[KnowledgeFragment],
    min_score: float | None,
) -> list[KnowledgeFragment]:
    from runtime import settings

    threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
    if threshold <= 0:
        return fragments

    kept: list[KnowledgeFragment] = []
    for fragment in fragments:
        source = (fragment.retrieval_source or "").lower()
        if source == "semantic" and fragment.score > 0:
            if fragment.score >= settings.SEMANTIC_MIN_SCORE:
                kept.append(fragment)
        elif fragment.score == 0.0 or fragment.score >= threshold:
            kept.append(fragment)
    return kept


def build_citations_from_fragments(
    fragments: list[KnowledgeFragment],
) -> list[SourceCitation]:
    """Deduplicate fragment metadata into numbered source citations."""
    seen: dict[tuple[str, str, str], SourceCitation] = {}
    ref_id = 1

    for fragment in fragments:
        citation = _citation_from_fragment(fragment, ref_id=0)
        key = citation.dedupe_key()
        if key in seen:
            continue
        citation = SourceCitation(
            ref_id=ref_id,
            title=citation.title,
            section=citation.section,
            source_url=citation.source_url,
            language=citation.language,
            retrieval_source=fragment.retrieval_source,
        )
        seen[key] = citation
        ref_id += 1

    return list(seen.values())


def attach_citations_to_fragments(
    fragments: list[KnowledgeFragment],
    citations: list[SourceCitation],
) -> list[KnowledgeFragment]:
    ref_by_key = {c.dedupe_key(): c.ref_id for c in citations}
    return [_attach_citation_ref(frag, ref_by_key) for frag in fragments]


def extract_source_labels(fragments: list[KnowledgeFragment]) -> list[str]:
    """Legacy string labels — prefer structured citations."""
    citations = build_citations_from_fragments(fragments)
    return [c.to_short_label() for c in citations]


def _citation_from_fragment(fragment: KnowledgeFragment, ref_id: int) -> SourceCitation:
    meta = fragment.metadata
    section = meta.section if meta else fragment.section
    title = meta.title if meta else fragment.title
    return SourceCitation(
        ref_id=ref_id,
        title=title,
        section=section,
        source_url=fragment.source_url,
        language=fragment.language,
        retrieval_source=fragment.retrieval_source,
    )


def _attach_citation_ref(
    fragment: KnowledgeFragment,
    ref_by_key: dict[tuple[str, str, str], int],
) -> KnowledgeFragment:
    citation = _citation_from_fragment(fragment, ref_id=0)
    ref_id = ref_by_key.get(citation.dedupe_key(), 0)
    return fragment.with_citation_ref(ref_id)


def _format_evidence_fragment(fragment: KnowledgeFragment) -> str:
    from runtime import settings

    body = fragment.body
    if len(body) > settings.MAX_FRAGMENT_BODY_CHARS:
        body = body[: settings.MAX_FRAGMENT_BODY_CHARS] + "…"

    ref = fragment.citation_ref
    header = f"### [{ref}] {fragment.title}" if ref else f"### {fragment.title}"
    lines = [header, f"- Раздел: {fragment.section}"]
    if fragment.source_url:
        lines.append(f"- URL: {fragment.source_url}")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)
