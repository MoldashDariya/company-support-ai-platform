"""Inbound message validation and prompt-injection heuristics."""

from __future__ import annotations

import re

from runtime import settings

_OFF_TOPIC_HINT = (
    f"Я помогаю с вопросами о «{settings.COMPANY_NAME}»: краски, доставка, салоны, "
    f"бренды, программа для дизайнеров. Задайте вопрос о компании или "
    f"позвоните: {settings.COMPANY_PHONE}."
)

_INJECTION_REGEX = (
    r"ты\s+теперь\s+",
    r"pretend\s+you\s+are",
    r"act\s+as\s+",
    r"роль:\s*",
    r"выведи\s+промпт",
    r"ignore\s+all\s+instructions",
)


class InputGuard:
    def validate(self, text: str) -> str | None:
        cleaned = text.strip()
        if not cleaned:
            return f"Напишите ваш вопрос о компании «{settings.COMPANY_NAME}»."
        if len(cleaned) > settings.MAX_USER_INPUT_CHARS:
            return (
                f"Сообщение слишком длинное "
                f"(макс. {settings.MAX_USER_INPUT_CHARS} символов)."
            )
        lower = cleaned.lower()
        for pattern in settings.INJECTION_PATTERNS:
            if pattern in lower:
                return _OFF_TOPIC_HINT
        if any(re.search(p, lower) for p in _INJECTION_REGEX):
            return _OFF_TOPIC_HINT
        return None
