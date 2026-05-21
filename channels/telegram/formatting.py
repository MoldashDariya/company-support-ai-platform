"""Telegram HTML formatting for conversion-oriented replies."""

from __future__ import annotations

import html
import re

from channels.telegram.link_router import conversion_nudge, inline_link_suggestions

_STRIP_LEGACY_SOURCES = re.compile(
    r"\n{1,2}(?:📚|📎)?\s*Источники.*$",
    re.IGNORECASE | re.DOTALL,
)

_BULLET_LINE_RE = re.compile(r"^[\s]*[-•]\s+", re.MULTILINE)


def prepare_telegram_message(
    text: str,
    *,
    intent: str,
    query: str = "",
    add_links: bool = True,
    add_nudge: bool = True,
) -> str:
    """Build compact HTML message with optional inline links and CTA nudge."""
    body = _STRIP_LEGACY_SOURCES.sub("", (text or "").strip())
    body = re.sub(r"\[\d+\]", "", body)
    body = _normalize_paragraphs(body)

    parts: list[str] = [html.escape(body)]

    if add_links:
        link_block = _format_inline_links(intent, query)
        if link_block:
            parts.append(link_block)

    if add_nudge:
        nudge = conversion_nudge(intent, query)
        if nudge:
            parts.append(f"\n<b>👉 Следующий шаг:</b> {html.escape(nudge)}")

    return "\n\n".join(p for p in parts if p).strip()


def _format_inline_links(intent: str, query: str) -> str:
    links = inline_link_suggestions(intent, query)
    if not links:
        return ""
    lines = ["<b>Полезные ссылки:</b>"]
    for label, url in links:
        safe_url = html.escape(url, quote=True)
        lines.append(f'• <a href="{safe_url}">{html.escape(label)}</a>')
    return "\n".join(lines)


def _normalize_paragraphs(text: str) -> str:
    """Light structure: double newlines between blocks, preserve bullets."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if out and out[-1] != "":
                out.append("")
            continue
        if _BULLET_LINE_RE.match(stripped):
            out.append("• " + _BULLET_LINE_RE.sub("", stripped))
        else:
            out.append(stripped)
    return "\n".join(out)


def streaming_preview_text(text: str, max_len: int = 350) -> str:
    """Plain-text preview while streaming (no HTML)."""
    clean = _STRIP_LEGACY_SOURCES.sub("", text or "").strip()
    if len(clean) > max_len:
        return clean[: max_len - 1] + "…"
    return clean or "…"
