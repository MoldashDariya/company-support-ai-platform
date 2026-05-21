"""Telegram UI widgets — intent-based CTA keyboards."""

from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup

from channels.telegram.link_router import build_intent_keyboard


def intent_actions_keyboard(intent: str, query: str = "") -> InlineKeyboardMarkup | None:
    """Dynamic conversion CTAs — no static always-on buttons."""
    return build_intent_keyboard(intent, query)
