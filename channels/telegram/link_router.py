"""Centralized URL map and intent-based CTA routing for Telegram."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from runtime import settings

_SITE = (os.getenv("COMPANY_SITE") or settings.COMPANY_SITE or "https://centr-krasok.kz/").rstrip("/")

# --- Configurable deep links (override via env if needed) ---
LINKS: dict[str, str] = {
    "site": _SITE + "/",
    "catalog": os.getenv("TG_URL_CATALOG", f"{_SITE}/catalog"),
    "interior_paints": os.getenv(
        "TG_URL_INTERIOR_PAINTS",
        f"{_SITE}/catalog/interiernye-kraski",
    ),
    "bathroom_paints": os.getenv(
        "TG_URL_BATHROOM_PAINTS",
        f"{_SITE}/catalog/dlya_kukhni_i_vannoy",
    ),
    "wood_paints": os.getenv(
        "TG_URL_WOOD_PAINTS",
        f"{_SITE}/catalog/kraski-po-derevu",
    ),
    "metal_paints": os.getenv(
        "TG_URL_METAL_PAINTS",
        f"{_SITE}/catalog/kraski-po-metallu",
    ),
    "delivery": os.getenv("TG_URL_DELIVERY", f"{_SITE}/about/delivery"),
    "about": os.getenv("TG_URL_ABOUT", f"{_SITE}/about"),
    "promotions": os.getenv("TG_URL_PROMOTIONS", f"{_SITE}/promotions"),
    "articles": os.getenv("TG_URL_ARTICLES", f"{_SITE}/articles"),
    "tinting": os.getenv("TG_URL_TINTING", f"{_SITE}/about/howto"),
    "interiors": os.getenv(
        "TG_URL_INTERIORS",
        f"{_SITE}/catalog/dekorativnaya-lepnina",
    ),
    "designer_program": os.getenv("TG_URL_DESIGNER", f"{_SITE}/promotions"),
}

_BATHROOM_QUERY_RE = re.compile(
    r"(ванн|санузел|кухн.*ванн|влагостойк.*интерьер)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CtaButton:
    label: str
    url_key: str

    @property
    def url(self) -> str:
        return resolve_url(self.url_key)


def resolve_url(key: str) -> str:
    return _normalize_https(LINKS.get(key, LINKS["site"])) or LINKS["site"]


def _normalize_https(url: str | None) -> str | None:
    if not url or not str(url).strip():
        return None
    u = str(url).strip()
    if u.startswith("http://"):
        u = "https://" + u[7:]
    elif not u.startswith("https://"):
        u = "https://" + u.lstrip("/")
    parsed = urlparse(u)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return u


def _whatsapp_url() -> str:
    digits = "".join(c for c in settings.COMPANY_PHONE if c.isdigit())
    return f"https://wa.me/{digits}" if digits else resolve_url("about")


def _phone_tel_url() -> str | None:
    """Telegram inline buttons require https — use WhatsApp or about for call intent."""
    return _whatsapp_url()


def is_bathroom_paint_query(query: str) -> bool:
    return bool(_BATHROOM_QUERY_RE.search(query or ""))


def resolve_cta_buttons(intent: str, query: str = "") -> list[CtaButton]:
    """Return up to 4 unique CTA buttons for intent + query nuance."""
    q = (query or "").lower().strip()
    intent = intent or "general"

    if is_bathroom_paint_query(q):
        return _dedupe_buttons(
            [
                CtaButton("🛁 Краски для ванной", "bathroom_paints"),
                CtaButton("🎨 Колеровка", "tinting"),
            ]
        )

    mapping: dict[str, list[CtaButton]] = {
        "products": [
            CtaButton("🛒 Каталог", "catalog"),
            CtaButton("🎨 Интерьерные краски", "interior_paints"),
        ],
        "brands": [
            CtaButton("🛒 Каталог", "catalog"),
            CtaButton("📰 Бренды и статьи", "articles"),
        ],
        "delivery": [
            CtaButton("🚚 Доставка", "delivery"),
            CtaButton("📞 Связаться", "about"),
        ],
        "contacts": [
            CtaButton("📞 Позвонить", "about"),
            CtaButton("💬 WhatsApp", "whatsapp"),
        ],
        "tinting": [
            CtaButton("🎨 Колеровка", "tinting"),
            CtaButton("🌈 Интерьерные краски", "interior_paints"),
        ],
        "interiors_design": [
            CtaButton("🏠 Интерьер и декор", "interiors"),
            CtaButton("🎨 Интерьерные краски", "interior_paints"),
        ],
        "technologies": [
            CtaButton("📖 Статьи о покрытиях", "articles"),
            CtaButton("🛒 Каталог", "catalog"),
        ],
        "services": [
            CtaButton("🚚 Доставка", "delivery"),
            CtaButton("🎨 Колеровка", "tinting"),
        ],
        "loyalty_program": [
            CtaButton("🎁 Акции", "promotions"),
            CtaButton("🛒 Каталог", "catalog"),
        ],
        "company_overview": [
            CtaButton("🛒 Каталог", "catalog"),
            CtaButton("📞 Связаться", "about"),
        ],
        "vacancies": [
            CtaButton("📞 Связаться", "about"),
            CtaButton("🌐 Сайт", "site"),
        ],
        "office": [
            CtaButton("📞 Контакты", "about"),
            CtaButton("🌐 Сайт", "site"),
        ],
        "address": [
            CtaButton("📞 Контакты", "about"),
            CtaButton("🚚 Доставка", "delivery"),
        ],
        "catalog": [
            CtaButton("🛒 Каталог", "catalog"),
            CtaButton("📞 Связаться", "about"),
        ],
    }

    if "дерев" in q:
        return _dedupe_buttons(
            [
                CtaButton("🪵 Краски по дереву", "wood_paints"),
                CtaButton("🛒 Каталог", "catalog"),
            ]
        )
    if "металл" in q:
        return _dedupe_buttons(
            [
                CtaButton("🔩 Краски по металлу", "metal_paints"),
                CtaButton("🛒 Каталог", "catalog"),
            ]
        )

    buttons = list(mapping.get(intent, mapping["company_overview"]))
    if intent == "contacts":
        buttons = [
            CtaButton("📞 Позвонить", "about"),
            CtaButton("💬 WhatsApp", "whatsapp"),
        ]
    return _dedupe_buttons(buttons)


def _dedupe_buttons(buttons: list[CtaButton]) -> list[CtaButton]:
    seen: set[str] = set()
    out: list[CtaButton] = []
    for btn in buttons:
        key = btn.url_key if btn.url_key != "whatsapp" else "wa"
        if key in seen:
            continue
        seen.add(key)
        out.append(btn)
    if len(out) > 4:
        out = out[:4]
    return out


def build_intent_keyboard(
    intent: str,
    query: str = "",
) -> InlineKeyboardMarkup | None:
    """Context-aware inline keyboard — max 2 rows, 2 buttons per row."""
    buttons = resolve_cta_buttons(intent, query)
    if not buttons:
        return None

    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for cta in buttons:
        url = _whatsapp_url() if cta.url_key == "whatsapp" else cta.url
        safe = _normalize_https(url)
        if not safe:
            continue
        row.append(InlineKeyboardButton(text=cta.label, url=safe))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    if not rows:
        return None
    return InlineKeyboardMarkup(inline_keyboard=rows[:2])


def inline_link_suggestions(intent: str, query: str = "") -> list[tuple[str, str]]:
    """Label + URL pairs for clickable lines inside the message body."""
    suggestions: list[tuple[str, str]] = []
    for cta in resolve_cta_buttons(intent, query)[:2]:
        if cta.url_key == "whatsapp":
            url = _whatsapp_url()
            label = "написать в WhatsApp"
        else:
            url = cta.url
            label = _link_label(cta)
        if url:
            suggestions.append((label, url))
    return suggestions


def conversion_nudge(intent: str, query: str = "") -> str:
    """Short next-step line (plain text; HTML applied in formatter)."""
    nudges = {
        "products": "Подберите подходящую категорию в каталоге или оформите заказ на сайте.",
        "delivery": "Уточнить доставку можно на странице доставки или у менеджера.",
        "contacts": "Мы на связи — выберите удобный способ ниже.",
        "tinting": "Для точного оттенка воспользуйтесь колеровкой или разделом интерьерных красок.",
        "interiors_design": "Посмотрите решения для интерьера и декора в каталоге.",
        "services": "Можем помочь с доставкой, колеровкой и подбором материалов.",
        "company_overview": "Готовы помочь с подбором — загляните в каталог или напишите нам.",
        "loyalty_program": "Актуальные акции — в разделе промо на сайте.",
        "vacancies": "Актуальные вакансии и контакты HR — на сайте или по телефону.",
        "office": "Адреса салонов и график — в разделе «О магазине» на сайте.",
        "address": "Адреса салонов и условия доставки — на сайте или у менеджера.",
        "catalog": "Откройте каталог на сайте и выберите нужную категорию.",
    }
    if is_bathroom_paint_query(query):
        return "Для ванной выберите влагостойкие краски — ссылки на подборку ниже."
    return nudges.get(intent, "Следующий шаг — в кнопках под сообщением.")


def _link_label(cta: CtaButton) -> str:
    labels = {
        "catalog": "каталог",
        "interior_paints": "интерьерные краски",
        "bathroom_paints": "краски для ванной и кухни",
        "wood_paints": "краски по дереву",
        "metal_paints": "краски по металлу",
        "delivery": "условия доставки",
        "tinting": "колеровка и подбор цвета",
        "interiors": "интерьер и декор",
        "about": "контакты магазина",
        "promotions": "акции",
        "articles": "полезные статьи",
    }
    return labels.get(cta.url_key, cta.label.replace("🛒 ", "").replace("🎨 ", ""))
