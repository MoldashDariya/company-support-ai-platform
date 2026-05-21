"""Rule-based FAQ fallback — no embeddings, runs before BM25 retrieval."""

from __future__ import annotations

import logging
import re

from cognition.intent import QueryIntent
from domain.models import ChunkMetadata, KnowledgeFragment
from runtime import settings

logger = logging.getLogger(__name__)

_FAQ_SOURCE = settings.COMPANY_SITE.rstrip("/")

# category -> (match patterns, canonical answer body)
_FAQ_CONTENT: dict[str, tuple[tuple[str, ...], str]] = {
    "services": (
        (
            "услуг",
            "сервис",
            "чем занимается",
            "о компании",
            "о магазине",
            "что делаете",
            "возможност",
        ),
        (
            f"«{settings.COMPANY_NAME}» — специализированный магазин лакокрасочных материалов, "
            "декора и инструментов. Услуги: консультация по подбору покрытий и систем, "
            "профессиональная колеровка, доставка по Астане и области, бонусы и программа "
            "для дизайнеров. В салонах работают опытные специалисты."
        ),
    ),
    "delivery": (
        (
            "доставк",
            "курьер",
            "привез",
            "срок достав",
            "стоимость достав",
            "когда привезут",
        ),
        (
            "Доставка товара — курьером по тарифу службы доставки магазина. "
            "Менеджер связывается после оформления заказа. "
            "График: пн–пт 10:00–20:00, суббота до 14:00, воскресенье — без доставки. "
            "Заказы в выходные доставляются в понедельник. "
            "Служба доставки: +7 778 061 5000, info.online@abis.kz."
        ),
    ),
    "contacts": (
        (
            "контакт",
            "телефон",
            "позвонить",
            "связаться",
            "email",
            "почт",
            "whatsapp",
        ),
        (
            f"Связаться с «{settings.COMPANY_NAME}»: телефон {settings.COMPANY_PHONE}, "
            f"сайт {settings.COMPANY_SITE}, email info.online@abis.kz. "
            "Служба доставки: +7 778 061 5000."
        ),
    ),
    "office": (
        (
            "офис",
            "салон",
            "магазин где",
            "точк",
            "филиал",
            "шоурум",
        ),
        (
            f"У «{settings.COMPANY_NAME}» современные салоны в Астане с консультацией "
            "по ЛКМ, декору и колеровке. Актуальные адреса и график — в разделе "
            f"«О магазине» на {settings.COMPANY_SITE}."
        ),
    ),
    "address": (
        (
            "адрес",
            "где находит",
            "как добраться",
            "локация",
            "находитесь",
        ),
        (
            "Доставка — по адресу из заказа; при другом адресе сообщите менеджеру доставки. "
            f"Адреса салонов в Астане — на сайте {settings.COMPANY_SITE} в разделе «О магазине»."
        ),
    ),
    "coloring": (
        (
            "колер",
            "колеровк",
            "оттен",
            "ral",
            "ncs",
            "палитр",
            "подбор цвет",
        ),
        (
            "Колеровка и подбор оттенков (в т.ч. RAL/NCS) — одна из ключевых услуг магазина. "
            f"Подробности на странице колеровки: {_FAQ_SOURCE}/tinting"
        ),
    ),
    "catalog": (
        (
            "каталог",
            "ассортимент",
            "что прода",
            "какие товар",
            "бренд",
            "краск",
            "лак",
            "инструмент",
        ),
        (
            "Ассортимент: краски, лаки, эмали, грунты, декоративные материалы, инструменты "
            "и бренды (Dulux, Marshall, Dufa и др.). Каталог и актуальные позиции — "
            f"на {settings.COMPANY_SITE}"
        ),
    ),
}

_INTENT_TO_FAQ: dict[str, str] = {
    "services": "services",
    "company_overview": "services",
    "delivery": "delivery",
    "contacts": "contacts",
    "tinting": "coloring",
    "products": "catalog",
    "brands": "catalog",
}

_MIN_PATTERN_HITS = 1
_MIN_SCORE = 2


def _pattern_score(normalized: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if pattern in normalized)


def match_faq(query: str, intent: QueryIntent) -> tuple[str, list[KnowledgeFragment]] | None:
    """
    Return FAQ category and synthetic fragments if the query matches strongly enough.
    Otherwise None — caller should run BM25 retrieval.
    """
    normalized = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if len(normalized) < 3:
        return None

    best_category: str | None = None
    best_score = 0

    for category, (patterns, _body) in _FAQ_CONTENT.items():
        score = _pattern_score(normalized, patterns)
        mapped = _INTENT_TO_FAQ.get(intent.name)
        if mapped == category:
            score += 2
        if score > best_score:
            best_score = score
            best_category = category

    if not best_category or best_score < _MIN_SCORE:
        return None

    patterns, body = _FAQ_CONTENT[best_category]
    has_pattern = _pattern_score(normalized, patterns) >= _MIN_PATTERN_HITS
    has_intent = _INTENT_TO_FAQ.get(intent.name) == best_category
    if not has_pattern and not has_intent:
        return None

    section_titles = {
        "services": "Услуги и возможности",
        "delivery": "Доставка",
        "contacts": "Контакты",
        "office": "Салоны",
        "address": "Адрес и доставка",
        "coloring": "Колеровка",
        "catalog": "Ассортимент",
    }
    section = section_titles.get(best_category, "FAQ")
    source = f"{_FAQ_SOURCE}/faq#{best_category}"

    fragment = KnowledgeFragment(
        section=f"FAQ › {section}",
        body=body,
        score=1.0,
        metadata=ChunkMetadata(
            title=section,
            section=f"FAQ › {section}",
            source_url=source,
            language=settings.CORPUS_LANGUAGE,
        ),
        retrieval_source="faq",
    )
    logger.info(
        "FAQ match | category=%s intent=%s score=%d",
        best_category,
        intent.name,
        best_score,
    )
    return best_category, [fragment]
