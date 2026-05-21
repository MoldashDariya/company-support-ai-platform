"""Lightweight chunk quality filters — no ML, low memory."""

from __future__ import annotations

import logging
import re

from domain.models import KnowledgeFragment
from ingestion.html_cleaner import CleanedPage
from ingestion.url_filters import is_url_allowed, url_block_reason
from runtime import settings

logger = logging.getLogger(__name__)

_LEGAL_MARKERS: tuple[str, ...] = (
    "политика конфиденциальности",
    "персональные данные",
    "пользователь вправе",
    "потребитель вправе",
    "договор купли-продажи",
    "публичная оферта",
    "обработк",
    "конфиденциальност",
    "cookie",
    "банковская карта не предназначена",
    "срок действия карты",
)

_LOW_VALUE_MARKERS: tuple[str, ...] = (
    "введены неверно",
    "истек срок действия",
    "товарный вид",
    "индивидуально-определенные свойства",
    "###### ",
)

_NUMBERING_HEAVY = re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b")
_SECTION_LEGAL_HEADING = re.compile(
    r"(политик|конфиденциальност|оферт|соглашен|пользовательск)",
    re.IGNORECASE,
)


def filter_pages_by_url(pages: list[CleanedPage]) -> list[CleanedPage]:
    """Drop pages whose source URL is on the ingestion blacklist."""
    kept: list[CleanedPage] = []
    for page in pages:
        reason = url_block_reason(page.source_url)
        if reason:
            logger.info("Skipped URL before chunking | url=%s reason=%s", page.source_url, reason)
            continue
        kept.append(page)
    return kept


def chunk_skip_reason(
    body: str,
    *,
    section: str = "",
    source_url: str = "",
) -> str | None:
    """Return skip reason for a chunk body, or None if it should be kept."""
    text = (body or "").strip()
    if not text:
        return "empty"

    min_chars = settings.CHUNK_MIN_CHARS
    if len(text) < min_chars:
        return "too_short"

    if source_url and not is_url_allowed(source_url):
        return url_block_reason(source_url) or "blocked_url"

    haystack = f"{section} {text}".lower()

    if _SECTION_LEGAL_HEADING.search(section or ""):
        return "legal_heading"

    legal_hits = sum(1 for marker in _LEGAL_MARKERS if marker in haystack)
    if legal_hits >= 2:
        return "legal_policy"

    if _NUMBERING_HEAVY.search(text):
        return "excessive_numbering"

    if any(marker in haystack for marker in _LOW_VALUE_MARKERS):
        return "low_value"

    words = re.findall(r"[\wа-яё]{3,}", haystack, flags=re.IGNORECASE)
    if len(words) < 12:
        return "too_few_words"

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.35 and len(words) > 40 and not _NUMBERING_HEAVY.search(text):
        return "repetitive"

    return None


def filter_fragments(
    fragments: list[KnowledgeFragment],
    *,
    log_skips: bool = True,
) -> tuple[list[KnowledgeFragment], int]:
    """Remove low-quality fragments; return (kept, skipped_count)."""
    kept: list[KnowledgeFragment] = []
    skipped = 0

    for fragment in fragments:
        reason = chunk_skip_reason(
            fragment.body,
            section=fragment.section,
            source_url=fragment.source_url,
        )
        if reason:
            skipped += 1
            if log_skips:
                logger.info(
                    "Skipped chunk | reason=%s section=%s url=%s",
                    reason,
                    (fragment.section or "")[:60],
                    fragment.source_url or "",
                )
            continue
        kept.append(fragment)

    if skipped:
        logger.info("Chunk quality filter | kept=%d skipped=%d", len(kept), skipped)
    return kept, skipped
