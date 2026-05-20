"""Human-readable citation formatting for grounded assistant responses."""

from __future__ import annotations

from domain.models import SourceCitation
from runtime import settings


def format_single_citation(citation: SourceCitation, *, include_url: bool = True) -> str:
    """Format one citation line: title, section, optional URL."""
    ref = f"{citation.ref_id}."
    title_part = f"«{citation.title}»"
    section_part = f"раздел «{citation.section}»" if citation.section else ""

    if section_part and citation.section != citation.title:
        main = f"{title_part} — {section_part}"
    else:
        main = title_part

    lines = [f"{ref} {main}"]
    if include_url and citation.source_url:
        lines.append(f"   {citation.source_url}")
    return "\n".join(lines)


def format_citations_block(
    citations: list[SourceCitation],
    *,
    include_urls: bool | None = None,
) -> str:
    """Build the footer block appended to every grounded answer."""
    if not citations:
        return ""

    show_urls = include_urls if include_urls is not None else settings.CITATION_INCLUDE_URLS
    ordered = sorted(citations, key=lambda c: c.ref_id)
    lines = ["", "📚 Источники (подтверждённые данные компании):"]
    for citation in ordered:
        lines.append(format_single_citation(citation, include_url=show_urls))
    return "\n".join(lines)


def citations_to_source_labels(citations: list[SourceCitation]) -> list[str]:
    return [c.to_short_label() for c in sorted(citations, key=lambda c: c.ref_id)]
