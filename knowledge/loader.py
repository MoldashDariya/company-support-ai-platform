"""Load and parse the company knowledge corpus from markdown."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from domain.models import ChunkMetadata, KnowledgeFragment

_URL_RE = re.compile(r"https?://[^\s\)>\"]+")
_LANG_HINT_RE = re.compile(r"language:\s*([a-z]{2})", re.IGNORECASE)


def export_fragments_to_markdown(
    fragments: list[KnowledgeFragment],
    path: Path,
) -> None:
    """Persist ingested fragments for BM25 and manual review."""
    from datetime import datetime, timezone

    languages = {f.language for f in fragments if f.language}
    language = languages.pop() if len(languages) == 1 else "ru"
    sources = sorted({f.source_url for f in fragments if f.source_url})

    header_lines = [
        "# Company knowledge base (ingested)",
        "",
        f"language: {language}",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if sources:
        header_lines.append(f"Sources: {', '.join(sources)}")
    header_lines.append("")

    body_lines: list[str] = []
    for fragment in fragments:
        section = fragment.section or fragment.title
        body_lines.append(f"## {section}")
        if fragment.source_url:
            body_lines.append(f"Source: {fragment.source_url}")
        body_lines.append("")
        body_lines.append(fragment.body)
        body_lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header_lines + body_lines), encoding="utf-8")


def corpus_fingerprint(fragments: list[KnowledgeFragment]) -> str:
    payload = "|".join(f"{f.section}:{f.body}" for f in fragments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_markdown_corpus(
    path: Path,
    *,
    default_source_url: str = "",
    default_language: str = "ru",
    default_title: str | None = None,
) -> list[KnowledgeFragment]:
    raw = path.read_text(encoding="utf-8")
    header, sections = _split_header_and_sections(raw)

    source_urls = _extract_source_urls(header) or ([default_source_url] if default_source_url else [])
    primary_source = source_urls[0] if source_urls else default_source_url
    language = _extract_language(header) or default_language
    document_title = default_title or _extract_document_title(header)

    fragments: list[KnowledgeFragment] = []
    for section_text in sections:
        section_text = section_text.strip()
        if not section_text.startswith("## "):
            continue
        lines = section_text.split("\n", 1)
        section_name = lines[0].removeprefix("## ").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if not body:
            continue

        section_source, body = _extract_source_line(body)
        section_source = section_source or _section_source_url(body) or primary_source
        metadata = ChunkMetadata(
            title=section_name,
            section=section_name,
            source_url=section_source,
            language=language,
        )
        fragments.append(
            KnowledgeFragment(
                section=section_name,
                body=body,
                metadata=metadata,
            )
        )

    from ingestion.quality import filter_fragments

    fragments, _skipped = filter_fragments(fragments, log_skips=True)

    if not fragments and document_title:
        metadata = ChunkMetadata(
            title=document_title,
            section="document",
            source_url=primary_source,
            language=language,
        )
        fragments.append(
            KnowledgeFragment(section="document", body=raw.strip(), metadata=metadata)
        )

    return fragments


def _split_header_and_sections(raw: str) -> tuple[str, list[str]]:
    stripped = raw.strip()
    sections = re.split(r"\n(?=## )", stripped)
    if sections and sections[0].strip().startswith("## "):
        return "", sections
    if not sections:
        return stripped, []
    return sections[0], sections[1:]


def _extract_source_urls(header: str) -> list[str]:
    urls = _URL_RE.findall(header)
    seen: dict[str, None] = {}
    for url in urls:
        seen.setdefault(url.rstrip(".,;"), None)
    return list(seen.keys())


def _extract_language(header: str) -> str | None:
    match = _LANG_HINT_RE.search(header)
    return match.group(1).lower() if match else None


def _extract_document_title(header: str) -> str | None:
    for line in header.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return None


def _extract_source_line(body: str) -> tuple[str | None, str]:
    lines = body.splitlines()
    if lines and lines[0].lower().startswith("source:"):
        url = lines[0].split(":", 1)[1].strip()
        remainder = "\n".join(lines[1:]).strip()
        return url, remainder
    return None, body


def _section_source_url(body: str) -> str | None:
    urls = _URL_RE.findall(body)
    return urls[0].rstrip(".,;") if urls else None
