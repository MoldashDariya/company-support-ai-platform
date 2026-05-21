"""Metadata- and intent-aware reranking on top of semantic / hybrid retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from cognition.intent import QueryIntent, classify_query_intent, fragment_profile_score
from domain.models import KnowledgeFragment
from runtime import settings

logger = logging.getLogger(__name__)

_ABOUT_QUERY_PATTERNS = (
    "чем занимается",
    "чем занимается компания",
    "что продаете",
    "что продаёте",
    "какие услуги",
    "о компании",
    "о магазине",
    "что такое центр красок",
    "расскажите о компании",
)

_SECTION_BOOST_KEYWORDS: tuple[tuple[str, float], ...] = (
    ("о компании", 0.12),
    ("о магазине", 0.12),
    ("наши возможности", 0.18),
    ("услуги", 0.10),
    ("доставка", 0.10),
    ("колеровка", 0.10),
    ("колер", 0.08),
    ("interiors", 0.10),
    ("interior", 0.08),
    ("интерьер", 0.10),
    ("tinting", 0.10),
    ("tint", 0.06),
    ("промо", 0.06),
    ("акции", 0.05),
    ("статьи", 0.06),
)

_SECTION_PENALTY_KEYWORDS: tuple[tuple[str, float], ...] = (
    ("политика конфиденциальности", 0.22),
    ("privacy", 0.22),
    ("оферта", 0.18),
    ("offer agreement", 0.18),
    ("договор", 0.12),
    ("персональных данных", 0.15),
    ("#####", 0.08),
)

_URL_PENALTY_FRAGMENTS: tuple[tuple[str, float], ...] = (
    ("privacy", 0.25),
    ("policy", 0.20),
    ("offer", 0.20),
    ("agreement", 0.18),
    ("howto", 0.12),
)

_URL_BOOST_FRAGMENTS: tuple[tuple[str, float], ...] = (
    ("/about/delivery", 0.10),
    ("/about$", 0.0),  # handled via exact path check
    ("/promotions", 0.06),
    ("/articles", 0.06),
)

_ABOUT_URL_PATHS = ("/about", "/about/")
_ABOUT_URL_EXCLUDE = ("privacy", "policy", "offer", "agreement", "howto")


@dataclass(frozen=True)
class RerankDetail:
    fragment: KnowledgeFragment
    original_score: float
    reranked_score: float
    adjustments: tuple[str, ...]


def is_about_company_query(query: str) -> bool:
    normalized = (query or "").lower().strip()
    return any(pattern in normalized for pattern in _ABOUT_QUERY_PATTERNS)


def rerank_fragments(
    query: str,
    fragments: list[KnowledgeFragment],
    *,
    limit: int | None = None,
    intent: QueryIntent | None = None,
) -> list[KnowledgeFragment]:
    """Apply intent routing, metadata boosts/penalties, and weak-score filtering."""
    if not fragments:
        return []

    cap = min(limit or settings.RETRIEVAL_TOP_K, settings.RETRIEVAL_TOP_K)
    intent = intent or classify_query_intent(query)
    about_intent = intent.name == "company_overview"
    details: list[RerankDetail] = []

    for fragment in fragments:
        original = _base_score(fragment)
        delta, notes = _metadata_adjustments(fragment, intent=intent, about_intent=about_intent)
        reranked = max(0.0, original + delta)

        if reranked < settings.RETRIEVAL_WEAK_THRESHOLD and original < settings.SEMANTIC_MIN_SCORE:
            continue

        details.append(
            RerankDetail(
                fragment=fragment,
                original_score=original,
                reranked_score=reranked,
                adjustments=tuple(notes),
            )
        )

    if about_intent and not _has_about_hit(details):
        details.extend(_about_fallback_candidates(fragments))

    details.sort(key=lambda d: d.reranked_score, reverse=True)
    selected = details[:cap]

    _log_rerank(query, intent, selected)
    return [
        d.fragment.with_score(d.reranked_score, d.fragment.retrieval_source or "reranked")
        for d in selected
    ]


def merge_candidate_pool(
    *rankings: list[KnowledgeFragment],
) -> list[KnowledgeFragment]:
    """Union semantic/sparse/RRF lists, keeping the best base score per fragment."""
    best: dict[str, KnowledgeFragment] = {}

    for ranking in rankings:
        for fragment in ranking:
            key = _fragment_key(fragment)
            if key not in best or _base_score(fragment) > _base_score(best[key]):
                best[key] = fragment

    return list(best.values())


def _base_score(fragment: KnowledgeFragment) -> float:
    source = (fragment.retrieval_source or "").lower()
    score = fragment.score or 0.0
    if source == "sparse":
        return min(score / 8.0, 1.0)
    if source in ("semantic", "hybrid", "reranked"):
        return score
    return score


def _metadata_adjustments(
    fragment: KnowledgeFragment,
    *,
    intent: QueryIntent,
    about_intent: bool,
) -> tuple[float, list[str]]:
    section = (fragment.section or "").lower()
    title = (fragment.title or "").lower()
    url = (fragment.source_url or "").lower()
    body_sample = (fragment.body or "")[:400].lower()
    haystack = f"{section} {title} {url} {body_sample}"

    delta = 0.0
    notes: list[str] = []

    for keyword, boost in _SECTION_BOOST_KEYWORDS:
        if keyword in haystack:
            delta += boost
            notes.append(f"+section:{keyword}")

    for keyword, penalty in _SECTION_PENALTY_KEYWORDS:
        if keyword in haystack:
            delta -= penalty
            notes.append(f"-section:{keyword}")

    for fragment_url, penalty in _URL_PENALTY_FRAGMENTS:
        if fragment_url in url:
            delta -= penalty
            notes.append(f"-url:{fragment_url}")

    for fragment_url, boost in _URL_BOOST_FRAGMENTS:
        if fragment_url in url:
            delta += boost
            notes.append(f"+url:{fragment_url}")

    if _is_primary_about_url(url):
        delta += 0.22
        notes.append("+url:about_main")

    profile_boost = fragment_profile_score(fragment, intent) * 0.35
    if profile_boost > 0:
        delta += profile_boost
        notes.append(f"+profile:{intent.retrieval_profile}")

    if intent.name == "delivery":
        if "доставк" in haystack or "/about/delivery" in url:
            delta += 0.35
            notes.append("+intent:delivery")
        if "наши возможности" in section:
            delta -= 0.30
            notes.append("-intent:skip_capabilities")
    elif intent.name == "products" and any(
        k in haystack for k in ("краск", "инструмент", "статьи", "бренд")
    ):
        delta += 0.20
        notes.append("+intent:products")
    elif intent.name == "tinting" and any(k in haystack for k in ("колер", "ral", "ncs", "цвет")):
        delta += 0.25
        notes.append("+intent:tinting")
    elif intent.name == "technologies" and any(
        k in haystack for k in ("технолог", "покрыт", "влаг", "стойк")
    ):
        delta += 0.22
        notes.append("+intent:technologies")
    elif intent.name == "loyalty_program" and any(
        k in haystack for k in ("акци", "скид", "бонус", "промо")
    ):
        delta += 0.22
        notes.append("+intent:loyalty")
    elif intent.name in ("interiors_design", "brands") and "статьи" in url:
        delta += 0.18
        notes.append(f"+intent:{intent.name}")

    if about_intent:
        intent_delta, intent_notes = _about_intent_adjustments(fragment, url, section)
        delta += intent_delta
        notes.extend(intent_notes)

    if intent.name != "company_overview" and _is_primary_about_url(url) and "наши возможности" in section:
        delta -= 0.12
        notes.append("-intent:generic_about")

    return delta, notes


def _about_intent_adjustments(
    fragment: KnowledgeFragment,
    url: str,
    section: str,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    delta = 0.0

    if _is_primary_about_url(url):
        delta += 0.35
        notes.append("+intent:about_page")
    elif "/about/delivery" in url:
        delta += 0.15
        notes.append("+intent:delivery")

    for marker in ("наши возможности", "о магазине", "о компании"):
        if marker in section:
            delta += 0.30
            notes.append(f"+intent:{marker}")
            break

    if "помощь" in section and "privacy" not in url:
        delta -= 0.08
        notes.append("-intent:help_page")

    if any(excl in url for excl in _ABOUT_URL_EXCLUDE):
        delta -= 0.40
        notes.append("-intent:legal_url")

    return delta, notes


def _is_primary_about_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    if any(excl in lowered for excl in _ABOUT_URL_EXCLUDE):
        return False
    path = (urlparse(lowered).path or "/").rstrip("/")
    return path == "/about"


def _has_about_hit(details: list[RerankDetail]) -> bool:
    for detail in details[:8]:
        url = detail.fragment.source_url.lower()
        section = detail.fragment.section.lower()
        if _is_primary_about_url(url) or "наши возможности" in section:
            return True
    return False


def _about_fallback_candidates(
    all_fragments: list[KnowledgeFragment],
) -> list[RerankDetail]:
    """Inject strong about-page candidates when intent routing finds none in the pool."""
    extras: list[RerankDetail] = []
    for fragment in all_fragments:
        url = fragment.source_url.lower()
        section = fragment.section.lower()
        if not (_is_primary_about_url(url) or "наши возможности" in section):
            continue
        original = _base_score(fragment)
        extras.append(
            RerankDetail(
                fragment=fragment,
                original_score=original,
                reranked_score=max(original, 0.45) + 0.25,
                adjustments=("+intent:fallback_about",),
            )
        )
    return extras


def _log_rerank(query: str, intent: QueryIntent, selected: list[RerankDetail]) -> None:
    preview = (query or "").strip().replace("\n", " ")[:100]
    if len((query or "").strip()) > 100:
        preview += "…"

    logger.info(
        "Rerank | query=%r intent=%s retrieval_profile=%s selected=%d",
        preview,
        intent.name,
        intent.retrieval_profile,
        len(selected),
    )
    for idx, detail in enumerate(selected[:5], start=1):
        section = (detail.fragment.section or "—")[:70]
        adj = ",".join(detail.adjustments[:4]) if detail.adjustments else "—"
        if len(detail.adjustments) > 4:
            adj += ",…"
        logger.info(
            "  #%d original=%.4f reranked=%.4f adj=[%s] section=%s",
            idx,
            detail.original_score,
            detail.reranked_score,
            adj,
            section,
        )


def _fragment_key(fragment: KnowledgeFragment) -> str:
    return f"{fragment.source_url}|{fragment.section}|{hash(fragment.body)}"
