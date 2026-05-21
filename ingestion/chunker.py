"""Semantic chunking and deduplication for cleaned page text."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import logging

from domain.models import ChunkMetadata, KnowledgeFragment
from ingestion.html_cleaner import CleanedPage, normalize_whitespace
from ingestion.quality import chunk_skip_reason
from runtime import settings

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


@dataclass
class ChunkingStats:
    input_pages: int = 0
    raw_chunks: int = 0
    deduplicated: int = 0
    quality_skipped: int = 0
    final_chunks: int = 0


class SemanticChunker:
    """Split cleaned pages into retrieval-sized fragments with metadata."""

    def __init__(
        self,
        *,
        max_chars: int | None = None,
        min_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        self._max_chars = max_chars or settings.CHUNK_MAX_CHARS
        self._min_chars = min_chars or settings.CHUNK_MIN_CHARS
        self._overlap = overlap_chars or settings.CHUNK_OVERLAP_CHARS

    def chunk_pages(self, pages: list[CleanedPage]) -> tuple[list[KnowledgeFragment], ChunkingStats]:
        stats = ChunkingStats(input_pages=len(pages))
        raw_fragments: list[KnowledgeFragment] = []

        for page in pages:
            raw_fragments.extend(self._chunk_page(page))

        stats.raw_chunks = len(raw_fragments)
        unique = deduplicate_fragments(raw_fragments)
        stats.deduplicated = stats.raw_chunks - len(unique)
        quality_kept: list[KnowledgeFragment] = []
        for fragment in unique:
            reason = chunk_skip_reason(
                fragment.body,
                section=fragment.section,
                source_url=fragment.source_url,
            )
            if reason:
                stats.quality_skipped += 1
                logger.info(
                    "Skipped chunk | reason=%s section=%s url=%s",
                    reason,
                    (fragment.section or "")[:60],
                    fragment.source_url or "",
                )
                continue
            quality_kept.append(fragment)
        stats.final_chunks = len(quality_kept)
        if stats.quality_skipped:
            logger.info(
                "Chunk quality filter | kept=%d skipped=%d",
                stats.final_chunks,
                stats.quality_skipped,
            )
        return quality_kept, stats

    def _chunk_page(self, page: CleanedPage) -> list[KnowledgeFragment]:
        sections = self._split_into_sections(page.text)
        fragments: list[KnowledgeFragment] = []

        for section_name, section_text in sections:
            for part in self._split_by_size(section_text):
                if chunk_skip_reason(
                    part,
                    section=section_name,
                    source_url=page.source_url,
                ):
                    continue
                title = page.title
                full_section = f"{title} › {section_name}" if section_name != "content" else title
                metadata = ChunkMetadata(
                    title=title,
                    section=full_section,
                    source_url=page.source_url,
                    language=page.language,
                )
                fragments.append(
                    KnowledgeFragment(
                        section=full_section,
                        body=part,
                        metadata=metadata,
                    )
                )
        return fragments

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_name = "content"
        current_lines: list[str] = []

        for line in lines:
            heading = _parse_heading(line)
            if heading:
                if current_lines:
                    sections.append((current_name, "\n".join(current_lines)))
                current_name = heading
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_name, "\n".join(current_lines)))

        return sections if sections else [("content", text)]

    def _split_by_size(self, text: str) -> list[str]:
        text = normalize_whitespace(text)
        if len(text) <= self._max_chars:
            return [text] if len(text) >= self._min_chars else []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: list[str] = []
        buffer = ""

        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= self._max_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer)
            if len(para) <= self._max_chars:
                buffer = para
            else:
                chunks.extend(self._hard_split(para))
                buffer = ""

        if buffer:
            chunks.append(buffer)

        return self._apply_overlap(chunks)

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        length = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)
            if length + word_len > self._max_chars and current:
                parts.append(" ".join(current))
                current = [word]
                length = len(word)
            else:
                current.append(word)
                length += word_len

        if current:
            parts.append(" ".join(current))
        return parts

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self._overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped: list[str] = []
        prev_tail = ""
        for chunk in chunks:
            if prev_tail:
                merged = f"{prev_tail}\n\n{chunk}"
                overlapped.append(merged[: self._max_chars + self._overlap])
            else:
                overlapped.append(chunk)
            prev_tail = chunk[-self._overlap :] if len(chunk) > self._overlap else chunk
        return overlapped


def _parse_heading(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("#"):
        match = _HEADING_RE.match(stripped)
        if match:
            return match.group(2).strip()
    if len(stripped) < 80 and stripped.isupper() and " " in stripped:
        return stripped.title()
    return None


def deduplicate_fragments(fragments: list[KnowledgeFragment]) -> list[KnowledgeFragment]:
    seen: set[str] = set()
    unique: list[KnowledgeFragment] = []

    for fragment in fragments:
        key = _content_fingerprint(fragment.body, fragment.source_url, fragment.section)
        if key in seen:
            continue
        seen.add(key)
        unique.append(fragment)

    return unique


def _content_fingerprint(text: str, source_url: str = "", section: str = "") -> str:
    normalized = normalize_whitespace(text).lower()
    payload = f"{source_url}|{section}|{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
