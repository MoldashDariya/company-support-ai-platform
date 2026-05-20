"""HTML extraction and noise removal for crawled pages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, Tag

from ingestion.crawler import CrawledPage

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

_NOISE_CLASS_ID_RE = re.compile(
    r"(nav|menu|breadcrumb|sidebar|footer|header|cookie|banner|modal|popup|"
    r"social|share|widget|cart|search|login|signup|subscribe|newsletter)",
    re.IGNORECASE,
)

_MAIN_CONTENT_SELECTORS = (
    "main",
    "article",
    '[role="main"]',
    ".content",
    ".main-content",
    "#content",
    "#main",
)


@dataclass(frozen=True)
class CleanedPage:
    source_url: str
    title: str
    text: str
    language: str


class HtmlCleaner:
    """Strip boilerplate and produce normalized plain text per page."""

    def clean(self, page: CrawledPage) -> CleanedPage | None:
        soup = BeautifulSoup(page.html, "lxml")

        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        self._remove_noise_by_class_id(soup)

        root = self._find_main_content(soup) or soup.body or soup
        title = self._extract_title(soup, root)
        language = self._detect_language(soup, root)
        text = self._extract_text(root)

        text = normalize_whitespace(text)
        if len(text) < 80:
            logger.debug("Skipping thin page %s (%d chars)", page.url, len(text))
            return None

        return CleanedPage(
            source_url=page.url,
            title=title,
            text=text,
            language=language,
        )

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
        for selector in _MAIN_CONTENT_SELECTORS:
            node = soup.select_one(selector)
            if node and len(node.get_text(strip=True)) > 120:
                return node
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
        for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
            line = element.get_text(" ", strip=True)
            if not line:
                continue
            if element.name in {"h1", "h2", "h3", "h4"}:
                level = int(element.name[1])
                blocks.append(f"{'#' * level} {line}")
            else:
                blocks.append(line)
        if not blocks:
            blocks = [root.get_text("\n", strip=True)]
        return "\n".join(blocks)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
