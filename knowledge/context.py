"""Format retrieved fragments into LLM-ready grounded context with citations."""

from __future__ import annotations

from domain.models import GroundedContext, KnowledgeFragment, SourceCitation


def build_grounded_context(
    fragments: list[KnowledgeFragment],
    min_score: float | None = None,
) -> GroundedContext:
    from runtime import settings

    threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
    relevant = [f for f in fragments if f.score == 0.0 or f.score >= threshold]
    has_evidence = len(relevant) > 0

    if not has_evidence:
        return GroundedContext(
            fragments=[],
            formatted="[Нет релевантных фрагментов в базе знаний компании.]",
            citations=[],
            has_sufficient_evidence=False,
        )

    citations = build_citations_from_fragments(relevant)
    ref_by_key = {c.dedupe_key(): c.ref_id for c in citations}
    tagged_fragments = [
        _attach_citation_ref(frag, ref_by_key) for frag in relevant
    ]

    parts = [_format_evidence_fragment(frag) for frag in tagged_fragments]

    return GroundedContext(
        fragments=tagged_fragments,
        formatted="\n\n".join(parts),
        citations=citations,
        has_sufficient_evidence=True,
    )


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
    ref = fragment.citation_ref
    header = f"### [{ref}] {fragment.title}" if ref else f"### {fragment.title}"
    lines = [header, f"- Раздел: {fragment.section}"]
    if fragment.source_url:
        lines.append(f"- URL: {fragment.source_url}")
    lines.append(f"- Язык: {fragment.language}")
    if fragment.retrieval_source:
        lines.append(f"- Метод поиска: {fragment.retrieval_source}")
    lines.append("")
    lines.append(fragment.body)
    return "\n".join(lines)
