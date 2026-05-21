"""Backward-compatible shim — canonical FAQ logic lives in project root ``faq.py``."""

from faq import (  # noqa: F401
    FAQ_INTENTS,
    UNSUPPORTED_FALLBACK,
    FaqMatch,
    classify_intent,
    get_curated_answer,
    is_faq_intent,
    resolve_faq,
    telegram_intent_key,
)

__all__ = [
    "FAQ_INTENTS",
    "UNSUPPORTED_FALLBACK",
    "FaqMatch",
    "classify_intent",
    "get_curated_answer",
    "is_faq_intent",
    "resolve_faq",
    "telegram_intent_key",
]
