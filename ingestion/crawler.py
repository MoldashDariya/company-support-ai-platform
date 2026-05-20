"""Multi-page website crawler with same-domain link discovery."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from runtime import settings

logger = logging.getLogger(__name__)

_SKIP_PATH_PATTERNS = re.compile(
    r"(/catalog/compare|/favourites|/barcode-scanner|/cart|/login|/checkout|"
    r"\?action=|add-to-cart|wp-admin)",
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
    """Breadth-first crawler limited to a single site origin."""

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

        self._seeds = [_normalize_url(url) for url in seeds]
        self._max_pages = max_pages or settings.CRAWL_MAX_PAGES
        self._delay = request_delay_sec if request_delay_sec is not None else settings.CRAWL_REQUEST_DELAY_SEC
        self._timeout = timeout_sec or settings.CRAWL_TIMEOUT_SEC
        self._user_agent = user_agent or settings.CRAWL_USER_AGENT
        self._allowed_netloc = urlparse(self._seeds[0]).netloc

    def crawl(self) -> CrawlReport:
        report = CrawlReport()
        visited: set[str] = set()
        queue: list[str] = list(self._seeds)

        headers = {"User-Agent": self._user_agent, "Accept": "text/html,application/xhtml+xml"}

        with httpx.Client(
            headers=headers,
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            while queue and len(report.pages) < self._max_pages:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                if not self._is_crawlable(url):
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
                    report.pages.append(
                        CrawledPage(url=url, html=html, status_code=response.status_code)
                    )
                    logger.info("Crawled %s (%d bytes)", url, len(html))

                    for link in self._extract_links(url, html):
                        if link not in visited and link not in queue:
                            queue.append(link)

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

    def _is_crawlable(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != self._allowed_netloc:
            return False
        full = url.lower()
        if _SKIP_PATH_PATTERNS.search(full):
            return False
        path = (parsed.path or "").lower()
        if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
            return False
        # Skip catalog listing pages; editorial URLs are preferred
        if path.startswith("/catalog"):
            return False
        return True

    def _extract_links(self, base_url: str, html: str) -> list[str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            href = tag.get("href", "").strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            absolute = _normalize_url(urljoin(base_url, href))
            if urlparse(absolute).netloc == self._allowed_netloc:
                links.append(absolute)
        return links


def _normalize_url(url: str) -> str:
    cleaned, _frag = urldefrag(url.strip())
    parsed = urlparse(cleaned)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(path=path, fragment="").geturl()
