"""
Rule-based intent classification and curated FAQ answers.

No ML, no embeddings — used before BM25 retrieval on Render free tier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from runtime import settings

logger = logging.getLogger(__name__)

# Intents answered from curated text only (no BM25 retrieval)
FAQ_INTENTS: frozenset[str] = frozenset(
    {
        "vacancies",
        "contacts",
        "office",
        "address",
        "delivery",
        "services",
        "catalog",
        "coloring",
        "technologies",
        "products",
    }
)

_SITE = settings.COMPANY_SITE.rstrip("/")
_PHONE = settings.COMPANY_PHONE
_COMPANY = settings.COMPANY_NAME

# (intent, patterns) — first match wins; more specific intents first
_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "vacancies",
        (
            "ваканс",
            "нанимает",
            "нанимаете",
            "есть работа",
            "ищете сотруд",
            "трудоустрой",
            "резюме",
            "hr",
            "карьер",
            "работа в",
            "вы нанимаете",
        ),
    ),
    (
        "delivery",
        (
            "доставк",
            "курьер",
            "привез",
            "срок достав",
            "стоимость достав",
            "когда привезут",
            "shipping",
        ),
    ),
    (
        "contacts",
        (
            "контакт",
            "телефон",
            "позвонить",
            "связаться",
            "как связаться",
            "whatsapp",
            "email",
            "почт",
            "написать вам",
        ),
    ),
    (
        "office",
        (
            "где офис",
            "офис",
            "салон",
            "шоурум",
            "точк продаж",
            "филиал",
            "магазин где",
            "график работы",
        ),
    ),
    (
        "address",
        (
            "адрес",
            "где находит",
            "как добраться",
            "локация",
            "находитесь",
            "где вы",
        ),
    ),
    (
        "coloring",
        (
            "колер",
            "колеровк",
            "оттен",
            "ral",
            "ncs",
            "палитр",
            "подбор цвет",
            "tinting",
        ),
    ),
    (
        "technologies",
        (
            "технолог",
            "какие технологии",
            "влагостой",
            "систем покрыт",
            "гидроизол",
            "фасадн",
            "стойкост покрыт",
        ),
    ),
    (
        "catalog",
        (
            "каталог",
            "ассортимент",
            "что прода",
            "какие товар",
            "прайс",
            "посмотреть товар",
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
            "для дерева",
            "для металла",
            "для стен",
            "для ванн",
            "бренд",
            "dulux",
            "marshall",
        ),
    ),
    (
        "services",
        (
            "услуг",
            "сервис",
            "чем занимается",
            "о компании",
            "о магазине",
            "что делаете",
            "возможност",
            "консультац",
            "подбор материал",
        ),
    ),
)

_CURATED_ANSWERS: dict[str, str] = {
    "vacancies": (
        f"По вакансиям в «{_COMPANY}» актуальные предложения публикуются на сайте "
        f"{_SITE} и у HR-отдела. Напишите на info.online@abis.kz или позвоните "
        f"{_PHONE} — подскажут, есть ли открытые позиции и как отправить резюме."
    ),
    "contacts": (
        f"Связаться с «{_COMPANY}»: телефон {_PHONE}, сайт {_SITE}, "
        "email info.online@abis.kz. Служба доставки: +7 778 061 5000."
    ),
    "office": (
        f"У «{_COMPANY}» салоны в Астане с консультацией по ЛКМ, декору и колеровке. "
        f"Актуальные адреса и график — в разделе «О магазине» на {_SITE}."
    ),
    "address": (
        "Доставка — по адресу из заказа; другой адрес можно согласовать с менеджером доставки. "
        f"Адреса салонов в Астане — на {_SITE} в разделе «О магазине»."
    ),
    "delivery": (
        "Доставка курьером по тарифу службы доставки. Менеджер связывается после заказа. "
        "График: пн–пт 10:00–20:00, суббота до 14:00, воскресенье — без доставки. "
        "Заказы в выходные — в понедельник. Служба доставки: +7 778 061 5000."
    ),
    "services": (
        f"«{_COMPANY}» — магазин лакокрасочных материалов, декора и инструментов. "
        "Услуги: подбор покрытий, колеровка, доставка по Астане и области, "
        "бонусы и программа для дизайнеров, консультация в салонах."
    ),
    "catalog": (
        f"Каталог на {_SITE}: краски, лаки, грунты, декоративные материалы, инструменты "
        "и бренды (Dulux, Marshall, Dufa и др.). Выберите категорию и оформите заказ онлайн."
    ),
    "coloring": (
        "Колеровка и подбор оттенков (RAL/NCS) — ключевая услуга магазина. "
        f"Подробности: {_SITE}/tinting — можно подобрать цвет под ваш интерьер."
    ),
    "technologies": (
        "Мы работаем с современными системами ЛКМ: влагостойкие и фасадные покрытия, "
        "грунты, лаки, декоративные материалы. Специалисты подберут технологию под "
        f"поверхность и условия эксплуатации — консультация в салоне или по {_PHONE}."
    ),
    "products": (
        "В ассортименте: краски для стен, дерева, металла, ванных комнат; лаки, эмали, "
        "грунты, инструменты. Уточните задачу (помещение, поверхность) — подскажем "
        f"линейку и бренд. Каталог: {_SITE}"
    ),
}

UNSUPPORTED_FALLBACK = (
    f"По этому вопросу у меня нет готового ответа в базе знаний. "
    f"Напишите менеджеру: {_PHONE}, {_SITE} или info.online@abis.kz — "
    "помогут с подбором материалов, заказом и доставкой."
)


@dataclass(frozen=True)
class FaqMatch:
    intent: str
    answer: str


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").lower()).strip()


def classify_intent(query: str) -> str:
    """Return FAQ intent name or ``general`` when BM25 retrieval should run."""
    normalized = _normalize(query)
    if len(normalized) < 2:
        return "general"
    for name, patterns in _INTENT_RULES:
        if any(pattern in normalized for pattern in patterns):
            logger.info("Intent classified | intent=%s query=%r", name, query[:80])
            return name
    return "general"


def is_faq_intent(intent: str) -> bool:
    return intent in FAQ_INTENTS


def get_curated_answer(intent: str) -> str:
    return _CURATED_ANSWERS.get(intent, UNSUPPORTED_FALLBACK)


def resolve_faq(query: str) -> FaqMatch | None:
    """
    If the query maps to a known FAQ intent, return curated answer (no retrieval).
    Otherwise None — caller runs BM25.
    """
    intent = classify_intent(query)
    if not is_faq_intent(intent):
        return None
    answer = get_curated_answer(intent)
    logger.info("FAQ match | intent=%s retrieval=false", intent)
    return FaqMatch(intent=intent, answer=answer)


def telegram_intent_key(faq_intent: str) -> str:
    """Map FAQ intent to Telegram CTA / routing keys used elsewhere."""
    return {
        "vacancies": "contacts",
        "office": "contacts",
        "address": "contacts",
        "catalog": "products",
        "coloring": "tinting",
        "products": "products",
        "delivery": "delivery",
        "contacts": "contacts",
        "services": "services",
        "technologies": "technologies",
    }.get(faq_intent, faq_intent)
