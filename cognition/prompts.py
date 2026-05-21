"""Grounded system prompts — constrain the model to company evidence."""

from __future__ import annotations

from cognition.intent import QueryIntent, intent_instruction
from runtime import settings

GROUNDED_SYSTEM_TEMPLATE = """Ты — консультант поддержки «{company}» (интернет-магазин ЛКМ и декоративных материалов в Казахстане).

ПОЛИТИКА ОТВЕТА (обязательно):
1. СИНТЕЗИРУЙ ответ из «Доказательной базы» — отвечай на конкретный вопрос, не копируй один и тот же обзор.
2. ЗАПРЕЩЕНО: «В моей базе знаний нет», «Я не нашёл информации», «нет этой информации» — если фрагменты есть.
3. Объём: {word_range} слов. Короткие абзацы; списки через «•» при перечислении.
4. Ориентация на действие: после ответа подскажи, что пользователь может сделать дальше (каталог, заказ, звонок, колеровка).
5. Не повторяй штампы «широкий ассортимент», «опытные специалисты», «высокое качество» — максимум один раз.
6. Не дублируй начало прошлого ответа{opening_hint}.
7. Не вставляй URL и кнопки — ссылки добавит интерфейс. Без [1][2] в тексте.
8. Тон: живой, профессиональный, как консультант в магазине; без дисклеймеров про ИИ.

ЗАДАНИЕ ПО ТИПУ ВОПРОСА (intent={intent_name}, стиль={response_style}):
{intent_focus}

Доказательная база:
{evidence}
{citation_index}
"""

from faq import UNSUPPORTED_FALLBACK

EMPTY_RETRIEVAL_FALLBACK = UNSUPPORTED_FALLBACK

INSUFFICIENT_EVIDENCE_FALLBACK = EMPTY_RETRIEVAL_FALLBACK


def build_grounded_system_prompt(
    evidence_block: str,
    citations: list | None = None,
    *,
    intent: QueryIntent | None = None,
    last_assistant_opening: str = "",
) -> str:
    from cognition.citations import format_citations_index_compact
    from cognition.intent import QueryIntent as QI

    qi = intent or QI.general()
    index_block = format_citations_index_compact(citations) if citations else ""

    opening_hint = ""
    if last_assistant_opening:
        opening_hint = f" (прошлый ответ начинался: «{last_assistant_opening[:50]}…» — начни иначе)"

    word_range = f"{settings.ANSWER_MIN_WORDS}–{settings.ANSWER_MAX_WORDS}"

    return GROUNDED_SYSTEM_TEMPLATE.format(
        company=settings.COMPANY_NAME,
        word_range=word_range,
        opening_hint=opening_hint,
        intent_name=qi.name,
        response_style=qi.response_style,
        intent_focus=intent_instruction(qi),
        evidence=evidence_block,
        citation_index=index_block,
    )
