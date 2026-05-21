"""URL allow/block rules for crawl and ingestion."""

from __future__ import annotations

import re

# Substrings that disqualify a URL (path or full URL, case-insensitive)
URL_BLOCKLIST_SUBSTRINGS: tuple[str, ...] = (
    "privacy",
    "policy",
    "agreement",
    "cookie",
    "terms",
    "login",
    "cart",
    "wishlist",
    "compare",
    "offer_agreement",
    "privacy_policy",
    "user-agreement",
    "personal-data",
)

_EXTRA_SKIP_PATH = re.compile(
    r"(/logout|/checkout|/register|/signup|/auth|/account|"
    r"/favourites|/barcode-scanner|"
    r"\?action=|add-to-cart|wp-admin|/product/|/variant)",
    re.IGNORECASE,
)


def url_block_reason(url: str) -> str | None:
    """Return a short reason if the URL must be skipped, else None."""
    if not url:
        return "empty_url"
    lowered = url.lower()
    for token in URL_BLOCKLIST_SUBSTRINGS:
        if token in lowered:
            return f"blacklist:{token}"
    if _EXTRA_SKIP_PATH.search(lowered):
        return "blacklist:path_pattern"
    return None


def is_url_allowed(url: str) -> bool:
    return url_block_reason(url) is None
