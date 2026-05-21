"""Multi-page website crawler — editorial pages only (no catalog/product listings)."""

from __future__ import annotations

import heapq
import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import httpx

from runtime import settings

logger = logging.getLogger(__name__)

_SKIP_PATH_PATTERNS = re.compile(
    r"(/catalog|/cart|/login|/logout|/checkout|/register|/signup|/auth|/account|"
    r"/wishlist|/compare|/favourites|/barcode-scanner|"
    r"\?action=|add-to-cart|wp-admin|/product/|/variant)",
    re.IGNORECASE,
)

_HYDRATION_MARKERS = re.compile(
    r"(__NEXT_DATA__|window\.__INITIAL_STATE__|application/ld\+json|"
    r'"@type"\s*:\s*"Product"|hydration)',
    re.IGNORECASE,
)

_SKIP_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".css",
    ".js",
    ".xml",
    ".json",
    ".ico",
    ".woff",
    ".woff2",
)

# Lower score = crawled sooner
_PRIORITY_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^/about/delivery|/about/howto", re.I), 0),
    (re.compile(r"^/about", re.I), 1),
    (re.compile(r"^/articles", re.I), 2),
    (re.compile(r"^/useful|полез|polezn", re.I), 3),
    (re.compile(r"koler|kolер|tint|колер", re.I), 4),
    (re.compile(r"interior|interern|интерьер", re.I), 5),
    (re.compile(r"^/promotions", re.I), 6),
    (re.compile(r"^/news", re.I), 7),
    (re.compile(r"^/$", re.I), 8),
]


@dataclass(frozen=True)
class CrawledPage:
    url: str
    html: str
    status_code: int


@dataclass
class CrawlReport:
    pages: list[CrawledPage] = field(default_factory=list)
    skipped_urls: list[str] = field(default_factory=list)
    failed_urls: list[tuple[str, str]] = field(default_factory=list)


class WebsiteCrawler:
    """Priority-aware crawler limited to editorial / service pages."""

    def __init__(
        self,
        seed_urls: list[str] | None = None,
        *,
        max_pages: int | None = None,
        request_delay_sec: float | None = None,
        timeout_sec: float | None = None,
        user_agent: str | None = None,
    ) -> None:
        seeds = seed_urls or settings.ingestion_seed_urls()
        if not seeds:
            seeds = [settings.COMPANY_SITE]

        self._seeds = [_normalize_url(url) for url in seeds if _is_crawlable(_normalize_url(url))]
        if not self._seeds:
            self._seeds = [_normalize_url(settings.COMPANY_SITE.rstrip("/") + "/")]
        self._max_pages = max_pages or settings.CRAWL_MAX_PAGES
        self._delay = request_delay_sec if request_delay_sec is not None else settings.CRAWL_REQUEST_DELAY_SEC
        self._timeout = timeout_sec or settings.CRAWL_TIMEOUT_SEC
        self._user_agent = user_agent or settings.CRAWL_USER_AGENT
        self._allowed_netloc = urlparse(self._seeds[0]).netloc
        self._enqueue_counter = 0

    def crawl(self) -> CrawlReport:
        report = CrawlReport()
        visited: set[str] = set()
        heap: list[tuple[int, int, str]] = []
        for seed in self._seeds:
            self._push_url(heap, seed)

        headers = {"User-Agent": self._user_agent, "Accept": "text/html,application/xhtml+xml"}

        with httpx.Client(
            headers=headers,
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            while heap and len(report.pages) < self._max_pages:
                _prio, _seq, url = heapq.heappop(heap)
                if url in visited:
                    continue
                visited.add(url)

                if not _is_crawlable(url):
                    report.skipped_urls.append(url)
                    continue

                try:
                    response = client.get(url)
                    if response.status_code >= 400:
                        report.failed_urls.append((url, f"HTTP {response.status_code}"))
                        continue

                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type and "application/xhtml" not in content_type:
                        report.skipped_urls.append(url)
                        continue

                    html = response.text
                    skip_reason = _html_skip_reason(html)
                    if skip_reason:
                        report.skipped_urls.append(url)
                        logger.debug("Skipped %s: %s", url, skip_reason)
                        continue

                    report.pages.append(
                        CrawledPage(url=url, html=html, status_code=response.status_code)
                    )
                    logger.info("Crawled %s (%d bytes)", url, len(html))

                    for link in self._extract_links(url, html):
                        if link not in visited:
                            self._push_url(heap, link)

                    if self._delay > 0:
                        time.sleep(self._delay)
                except httpx.HTTPError as exc:
                    report.failed_urls.append((url, str(exc)))
                    logger.warning("Failed to crawl %s: %s", url, exc)

        logger.info(
            "Crawl finished: %d pages, %d skipped, %d failed",
            len(report.pages),
            len(report.skipped_urls),
            len(report.failed_urls),
        )
        return report

    def _push_url(self, heap: list[tuple[int, int, str]], url: str) -> None:
        normalized = _normalize_url(url)
        if not _is_crawlable(normalized):
            return
        priority = _url_priority(normalized)
        self._enqueue_counter += 1
        heapq.heappush(heap, (priority, self._enqueue_counter, normalized))

    def _extract_links(self, base_url: str, html: str) -> list[str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        links: list[str] = []
        seen: set[str] = set()
        for tag in soup.find_all("a", href=True):
            href = tag.get("href", "").strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            absolute = _normalize_url(urljoin(base_url, href))
            if urlparse(absolute).netloc != self._allowed_netloc:
                continue
            if absolute not in seen and _is_crawlable(absolute):
                seen.add(absolute)
                links.append(absolute)
        return links


def _is_crawlable(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    full = url.lower()
    if _SKIP_PATH_PATTERNS.search(full):
        return False
    if parsed.query and _should_skip_query(parsed):
        return False
    path = (parsed.path or "/").lower()
    if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return False
    return _is_allowed_editorial_path(path)


def _is_allowed_editorial_path(path: str) -> bool:
    """Allow homepage, about, articles, useful, tinting/interior, promotions, news."""
    path = path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    if path.startswith("/catalog"):
        return False

    allowed_prefixes = (
        "/",
        "/about",
        "/articles",
        "/useful",
        "/promotions",
        "/news",
    )
    for prefix in allowed_prefixes:
        if path == prefix or (prefix != "/" and path.startswith(prefix)):
            return True

    editorial_keywords = (
        "koler",
        "kolер",
        "tint",
        "колер",
        "interior",
        "interern",
        "интерьер",
        "polezn",
        "полез",
    )
    lowered = path.lower()
    return any(keyword in lowered for keyword in editorial_keywords)


def _html_skip_reason(html: str) -> str | None:
    if len(html) > settings.CRAWL_MAX_HTML_BYTES:
        return f"html>{settings.CRAWL_MAX_HTML_BYTES}b"
    if _HYDRATION_MARKERS.search(html) and len(html) > 80_000:
        return "large_hydration_payload"
    return None


def _should_skip_query(parsed) -> bool:
    query = parse_qs(parsed.query)
    blocked_keys = {
        "pagen_1",
        "showall_1",
        "set_filter",
        "filter",
        "filters",
        "sort",
        "page",
        "view",
        "q",
        "variant",
    }
    lowered = {k.lower() for k in query}
    if lowered.intersection(blocked_keys):
        return True
    return any(
        key.lower().startswith("pagen") or key.lower().startswith("showall")
        for key in query
    )


def _url_priority(url: str) -> int:
    path = urlparse(url).path or "/"
    for pattern, score in _PRIORITY_RULES:
        if pattern.search(path):
            return score
    return 50


def _normalize_url(url: str) -> str:
    cleaned, _frag = urldefrag(url.strip())
    parsed = urlparse(cleaned)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(path=path, fragment="").geturl()
