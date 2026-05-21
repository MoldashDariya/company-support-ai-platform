"""HTML extraction and noise removal for crawled pages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, Tag

from ingestion.crawler import CrawledPage
from runtime import settings

logger = logging.getLogger(__name__)

_NOISE_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "button",
        "menu",
    }
)

_JSON_LD_TYPES = frozenset({"application/ld+json", "application/json"})

_NOISE_CLASS_ID_RE = re.compile(
    r"(nav|menu|breadcrumb|sidebar|footer|header|cookie|banner|modal|popup|"
    r"social|share|widget|cart|search|login|signup|subscribe|newsletter|"
    r"pagination|toolbar|filter-bar|sort-bar|hydration|__next)",
    re.IGNORECASE,
)

_MAIN_CONTENT_SELECTORS = (
    "main",
    "article",
    '[role="main"]',
    ".content",
    ".main-content",
    ".page-content",
    "#content",
    "#main",
    ".faq",
)

_CONTENT_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "dt",
    "dd",
    "blockquote",
    "summary",
)

_FAQ_CONTAINERS = (".faq", ".faq-item", '[itemtype*="FAQPage"]', "details")


@dataclass(frozen=True)
class CleanedPage:
    source_url: str
    title: str
    text: str
    language: str


class HtmlCleaner:
    """Strip boilerplate, JSON-LD, and hydration payloads; emit compact text."""

    def clean(self, page: CrawledPage) -> CleanedPage | None:
        soup = BeautifulSoup(page.html, "lxml")

        self._strip_scripts_and_json_ld(soup)
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        self._remove_hydration_nodes(soup)
        self._remove_noise_by_class_id(soup)

        root = self._find_main_content(soup) or soup.body or soup
        title = self._extract_title(soup, root)
        language = self._detect_language(soup, root)
        text = self._extract_text(root)
        text = _dedupe_blocks(text)
        text = normalize_whitespace(text)
        text = text[: settings.CLEAN_MAX_PAGE_CHARS]

        if len(text) < 80:
            logger.debug("Skipping thin page %s (%d chars)", page.url, len(text))
            return None

        return CleanedPage(
            source_url=page.url,
            title=title,
            text=text,
            language=language,
        )

    def _strip_scripts_and_json_ld(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all("script")):
            script_type = (tag.get("type") or "").lower()
            if script_type in _JSON_LD_TYPES or tag.string:
                tag.decompose()
                continue
            id_attr = (tag.get("id") or "").lower()
            if id_attr in {"__next_data__", "__nuxt__", "app-state", "initial-state"}:
                tag.decompose()
                continue
            tag.decompose()

    def _remove_hydration_nodes(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(True)):
            if not isinstance(tag, Tag):
                continue
            tag_id = (tag.get("id") or "").lower()
            if tag_id in {"__next", "__nuxt", "root"} and tag.name == "div":
                # Keep root container but strip heavy child script payloads already removed
                continue
            data_attrs = " ".join(
                f"{k}={v}" for k, v in tag.attrs.items() if k.startswith("data-")
            ).lower()
            if "hydration" in data_attrs or "application/json" in data_attrs:
                tag.decompose()

    def _remove_noise_by_class_id(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(True)):
            if not isinstance(tag, Tag) or tag.attrs is None:
                continue
            classes = tag.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            attrs = " ".join(
                filter(
                    None,
                    [" ".join(classes), tag.get("id", "") or ""],
                )
            )
            if attrs and _NOISE_CLASS_ID_RE.search(attrs):
                tag.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> Tag | None:
        best: Tag | None = None
        best_len = 0
        for selector in _MAIN_CONTENT_SELECTORS:
            for node in soup.select(selector):
                length = len(node.get_text(strip=True))
                if length > best_len:
                    best = node
                    best_len = length
        if best_len > 120:
            return best
        return None

    def _extract_title(self, soup: BeautifulSoup, root: Tag) -> str:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            if title:
                return title
        h1 = root.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def _detect_language(self, soup: BeautifulSoup, root: Tag) -> str:
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return str(html_tag["lang"]).split("-")[0].lower()
        sample = root.get_text(" ", strip=True)[:2000]
        cyrillic = len(re.findall(r"[а-яё]", sample, re.IGNORECASE))
        latin = len(re.findall(r"[a-z]", sample, re.IGNORECASE))
        return "ru" if cyrillic >= latin else "en"

    def _extract_text(self, root: Tag) -> str:
        blocks: list[str] = []
        seen_faq: set[str] = set()

        for container in root.select(", ".join(_FAQ_CONTAINERS)):
            faq_text = container.get_text("\n", strip=True)
            if faq_text and faq_text not in seen_faq:
                seen_faq.add(faq_text)
                blocks.append(f"## FAQ\n{faq_text[:1200]}")

        for element in root.find_all(_CONTENT_TAGS):
            if self._is_inside_noise(element):
                continue
            line = element.get_text(" ", strip=True)
            if not line or len(line) < 3:
                continue
            if len(line) > 600:
                line = line[:600] + "…"
            if element.name.startswith("h") and len(element.name) == 2:
                level = min(int(element.name[1]), 6)
                blocks.append(f"{'#' * level} {line}")
            elif element.name == "dt":
                blocks.append(f"**{line}**")
            elif element.name == "dd":
                blocks.append(line)
            elif element.name == "summary":
                blocks.append(f"### {line}")
            else:
                blocks.append(line)

        if not blocks:
            fallback = root.get_text("\n", strip=True)
            blocks = [fallback[: settings.CLEAN_MAX_PAGE_CHARS]]

        return "\n".join(blocks)

    @staticmethod
    def _is_inside_noise(element: Tag) -> bool:
        parent = element.parent
        while parent and isinstance(parent, Tag):
            name = (parent.name or "").lower()
            if name in _NOISE_TAGS:
                return True
            parent = parent.parent
        return False


def _dedupe_blocks(text: str) -> str:
    lines = text.splitlines()
    deduped: list[str] = []
    prev = ""
    for line in lines:
        normalized = line.strip()
        if normalized and normalized == prev:
            continue
        deduped.append(line)
        prev = normalized
    return "\n".join(deduped)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
