"""Lightweight query intent classification for retrieval and answer policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.models import KnowledgeFragment

# (intent_name, patterns) — first match wins (order = specificity)
_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "delivery",
        (
            "доставк",
            "курьер",
            "привез",
            "срок достав",
            "стоимость достав",
            "shipping",
        ),
    ),
    (
        "contacts",
        (
            "контакт",
            "телефон",
            "адрес",
            "как связаться",
            "где находит",
            "салон",
            "магазин где",
        ),
    ),
    (
        "loyalty_program",
        (
            "бонус",
            "лояльност",
            "карт",
            "программа для дизайнер",
            "дизайнеров",
            "скидк",
            "акци",
            "промо",
        ),
    ),
    (
        "tinting",
        (
            "колер",
            "колеровк",
            "ral",
            "ncs",
            "оттен",
            "цвет",
            "палитр",
            "tinting",
            "tint",
        ),
    ),
    (
        "technologies",
        (
            "технолог",
            "влагостой",
            "прочност",
            "покрыти",
            "гидроизол",
            "фасад",
            "эксплуатац",
            "стойкост",
        ),
    ),
    (
        "interiors_design",
        (
            "интерьер",
            "дизайн",
            "декор",
            "обои",
            "лепнин",
            "interior",
            "design",
            "3d-панел",
        ),
    ),
    (
        "brands",
        (
            "бренд",
            "производител",
            "dulux",
            "marshall",
            "dufa",
            "oikos",
            "hammerite",
            "pinotex",
        ),
    ),
    (
        "products",
        (
            "краск",
            "лак",
            "эмаль",
            "грунт",
            "инструмент",
            "герметик",
            "клей",
            "шпатлев",
            "каталог",
            "товар",
            "ассортимент",
            "что прода",
            "какие есть",
            "для дерева",
            "для металла",
            "для стен",
        ),
    ),
    (
        "services",
        (
            "услуг",
            "сервис",
            "консультац",
            "подбор",
            "помощь",
            "обслуживан",
        ),
    ),
    (
        "company_overview",
        (
            "чем занимается",
            "о компании",
            "о магазине",
            "кто вы",
            "что такое центр красок",
            "расскажите о компании",
            "чем занимается компания",
        ),
    ),
)

_RETRIEVAL_PROFILES: dict[str, str] = {
    "company_overview": "about_capabilities",
    "services": "about_delivery_tinting",
    "products": "articles_brands_materials",
    "delivery": "delivery_pages",
    "tinting": "tinting_articles",
    "technologies": "articles_coatings",
    "contacts": "about_contacts",
    "loyalty_program": "promotions_loyalty",
    "interiors_design": "interiors_articles",
    "brands": "brands_articles",
    "general": "balanced",
}

_RESPONSE_STYLES: dict[str, str] = {
    "company_overview": "concise_business_summary",
    "services": "service_list",
    "products": "product_categories",
    "delivery": "delivery_facts",
    "tinting": "tinting_expert",
    "technologies": "technical_benefits",
    "contacts": "contact_direct",
    "loyalty_program": "promo_loyalty",
    "interiors_design": "design_inspiration",
    "brands": "brand_highlights",
    "general": "adaptive",
}

_INTENT_INSTRUCTIONS: dict[str, str] = {
    "company_overview": (
        "Фокус: чем занимается «{company}» — продажа ЛКМ, покрытий, инструментов; "
        "колеровка и доставка одним предложением. Без перечисления всего подряд."
    ),
    "services": (
        "Фокус: услуги — доставка, колеровка, консультация по подбору материалов, "
        "обслуживание клиентов. Не уходи в общий обзор компании."
    ),
    "products": (
        "Фокус: товары — краски, лаки, декоративные материалы, инструменты, бренды. "
        "Перечисли релевантные категории из фрагментов."
    ),
    "delivery": (
        "Фокус: доставка — сроки, зона, стоимость, правила из фрагментов. "
        "Не повторяй общий список возможностей магазина."
    ),
    "tinting": (
        "Фокус: колеровка, подбор цвета, RAL/NCS, палитры. Практические детали из базы."
    ),
    "technologies": (
        "Фокус: технологии покрытий — стойкость, влагостойкость, фасад/интерьер, "
        "современные системы LKM. Без маркетинговых штампов."
    ),
    "contacts": (
        "Фокус: как связаться — телефон, салоны, сайт. Кратко и по делу."
    ),
    "loyalty_program": (
        "Фокус: бонусы, акции, программа для дизайнеров, скидки — только из фрагментов."
    ),
    "interiors_design": (
        "Фокус: интерьер, декор, дизайн-решения, материалы для отделки."
    ),
    "brands": (
        "Фокус: бренды и производители, представленные в магазине."
    ),
    "general": (
        "Ответь строго по сути вопроса, используя наиболее релевантные фрагменты."
    ),
}

# Fragment relevance for retrieval filtering (haystack match)
_PROFILE_SIGNALS: dict[str, tuple[str, ...]] = {
    "about_capabilities": ("наши возможности", "/about", "о магазине"),
    "about_delivery_tinting": ("доставк", "колер", "/about", "услуг"),
    "articles_brands_materials": ("статьи", "бренд", "краск", "каталог", "инструмент"),
    "delivery_pages": ("доставк", "/about/delivery"),
    "tinting_articles": ("колер", "ral", "ncs", "цвет", "tint"),
    "articles_coatings": ("технолог", "покрыт", "влаг", "фасад", "статьи"),
    "about_contacts": ("астана", "контакт", "телефон", "/about"),
    "promotions_loyalty": ("акци", "скид", "бонус", "промо", "дизайнер"),
    "interiors_articles": ("интерьер", "дизайн", "декор", "статьи"),
    "brands_articles": ("бренд", "dulux", "marshall", "dufa", "статьи"),
    "balanced": (),
}


@dataclass(frozen=True)
class QueryIntent:
    name: str
    retrieval_profile: str
    response_style: str

    @classmethod
    def general(cls) -> QueryIntent:
        return cls("general", _RETRIEVAL_PROFILES["general"], _RESPONSE_STYLES["general"])


def classify_query_intent(query: str) -> QueryIntent:
    normalized = (query or "").lower().strip()
    for name, patterns in _INTENT_RULES:
        if any(p in normalized for p in patterns):
            return QueryIntent(
                name=name,
                retrieval_profile=_RETRIEVAL_PROFILES.get(name, "balanced"),
                response_style=_RESPONSE_STYLES.get(name, "adaptive"),
            )
    return QueryIntent.general()


def intent_instruction(intent: QueryIntent) -> str:
    from runtime import settings

    template = _INTENT_INSTRUCTIONS.get(intent.name, _INTENT_INSTRUCTIONS["general"])
    return template.format(company=settings.COMPANY_NAME)


def fragment_profile_score(fragment: KnowledgeFragment, intent: QueryIntent) -> float:
    """Higher = better match for intent retrieval profile."""
    signals = _PROFILE_SIGNALS.get(intent.retrieval_profile, ())
    if not signals:
        return 0.0

    haystack = " ".join(
        [
            (fragment.section or "").lower(),
            (fragment.title or "").lower(),
            (fragment.source_url or "").lower(),
            (fragment.body or "")[:300].lower(),
        ]
    )
    score = 0.0
    for signal in signals:
        if signal in haystack:
            score += 0.15
    return score


def filter_fragments_for_intent(
    fragments: list[KnowledgeFragment],
    intent: QueryIntent,
) -> list[KnowledgeFragment]:
    """Reorder and prefer fragments aligned with query intent."""
    if not fragments or intent.name == "general":
        return fragments

    scored = [
        (fragment, fragment.score + fragment_profile_score(fragment, intent) * 2.0)
        for fragment in fragments
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [f.with_score(s, f.retrieval_source) for f, s in scored]


def extract_opening(text: str) -> str:
    """First sentence or line for anti-repetition memory."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = re.match(r"^[^.!?]+[.!?]?", cleaned)
    if match:
        return match.group(0).strip()[:120]
    return cleaned[:120]


def openings_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_norm = re.sub(r"\W+", "", a.lower())[:60]
    b_norm = re.sub(r"\W+", "", b.lower())[:60]
    if len(a_norm) < 20 or len(b_norm) < 20:
        return a_norm == b_norm
    return a_norm in b_norm or b_norm in a_norm
