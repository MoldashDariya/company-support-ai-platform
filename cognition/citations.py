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


def format_citations_index_compact(citations: list[SourceCitation]) -> str:
    """Short index for the system prompt (not shown to user)."""
    if not citations:
        return ""
    lines = ["\nКарта фрагментов:"]
    for c in sorted(citations, key=lambda x: x.ref_id)[:5]:
        lines.append(f"[{c.ref_id}] {c.section[:80]}")
    return "\n".join(lines)


def format_citations_block(
    citations: list[SourceCitation],
    *,
    include_urls: bool | None = None,
    max_items: int = 2,
) -> str:
    """Compact footer — at most max_items sources, minimal clutter."""
    if not citations:
        return ""

    show_urls = include_urls if include_urls is not None else settings.CITATION_INCLUDE_URLS
    ordered = sorted(citations, key=lambda c: c.ref_id)[:max_items]
    lines = ["", "Источники:"]
    for citation in ordered:
        label = citation.title
        if citation.section and citation.section != citation.title:
            short = citation.section.split("›")[-1].strip()[:50]
            label = f"{label} — {short}"
        if show_urls and citation.source_url:
            lines.append(f"• {label}")
        else:
            lines.append(f"• {label}")
    if len(citations) > max_items:
        lines.append(f"• … ещё {len(citations) - max_items} на сайте")
    return "\n".join(lines)


def citations_to_source_labels(citations: list[SourceCitation]) -> list[str]:
    return [c.to_short_label() for c in sorted(citations, key=lambda c: c.ref_id)]
