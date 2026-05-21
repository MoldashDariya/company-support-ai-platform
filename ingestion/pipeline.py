"""End-to-end ingestion: crawl → clean → chunk → index."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domain.models import KnowledgeFragment
from ingestion.chunker import ChunkingStats, SemanticChunker
from ingestion.crawler import CrawlReport, WebsiteCrawler
from ingestion.html_cleaner import CleanedPage, HtmlCleaner
from ingestion.quality import filter_pages_by_url
from knowledge.loader import corpus_fingerprint, export_fragments_to_markdown
from knowledge.retriever import SparseRetriever
from runtime import settings

if TYPE_CHECKING:
    from knowledge.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    pages_crawled: int = 0
    pages_cleaned: int = 0
    chunks_indexed: int = 0
    chunks_deduplicated: int = 0
    chunks_quality_skipped: int = 0
    crawl_failures: int = 0
    corpus_path: str = ""
    chroma_collection: str = ""
    corpus_hash: str = ""
    top_sections: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class KnowledgeIngestionPipeline:
    """Production-style pipeline that rebuilds the knowledge base from the website."""

    def __init__(
        self,
        *,
        seed_urls: list[str] | None = None,
        corpus_path: Path | None = None,
    ) -> None:
        self._seeds = seed_urls
        self._corpus_path = corpus_path or settings.CORPUS_PATH
        self._crawler = WebsiteCrawler(seed_urls=self._seeds)
        self._cleaner = HtmlCleaner()
        self._chunker = SemanticChunker()

    def run(self) -> IngestionResult:
        result = IngestionResult(
            corpus_path=str(self._corpus_path),
            chroma_collection=settings.CHROMA_COLLECTION,
        )

        logger.info("Starting knowledge ingestion from %s", self._seeds or settings.ingestion_seed_urls())

        crawl_report = self._crawler.crawl()
        result.pages_crawled = len(crawl_report.pages)
        result.crawl_failures = len(crawl_report.failed_urls)
        result.errors.extend(f"{url}: {err}" for url, err in crawl_report.failed_urls)

        cleaned = filter_pages_by_url(self._clean_pages(crawl_report))
        result.pages_cleaned = len(cleaned)

        if not cleaned:
            result.errors.append("No usable content extracted from crawled pages.")
            logger.error("Ingestion aborted: no cleaned pages")
            return result

        fragments, chunk_stats = self._chunker.chunk_pages(cleaned)
        result.chunks_indexed = chunk_stats.final_chunks
        result.chunks_deduplicated = chunk_stats.deduplicated
        result.chunks_quality_skipped = chunk_stats.quality_skipped
        result.top_sections = _top_sections(fragments)

        if not fragments:
            result.errors.append("Chunking produced zero fragments.")
            logger.error("Ingestion aborted: no chunks")
            return result

        export_fragments_to_markdown(fragments, self._corpus_path)
        rebuild_semantic_index(fragments)
        rebuild_sparse_index(fragments)

        result.corpus_hash = corpus_fingerprint(fragments)
        logger.info(
            "Ingestion complete | pages=%d chunks=%d skipped_urls=%d hash=%s",
            result.pages_cleaned,
            result.chunks_indexed,
            len(crawl_report.skipped_urls),
            result.corpus_hash[:12],
        )
        return result

    def _clean_pages(self, report: CrawlReport) -> list[CleanedPage]:
        cleaned: list[CleanedPage] = []
        for page in report.pages:
            doc = self._cleaner.clean(page)
            if doc:
                cleaned.append(doc)
        return cleaned


def _top_sections(fragments: list[KnowledgeFragment], limit: int = 15) -> list[str]:
    from collections import Counter

    counts = Counter((f.section or "—")[:100] for f in fragments)
    return [f"{name} ({count})" for name, count in counts.most_common(limit)]


def rebuild_semantic_index(fragments: list[KnowledgeFragment]) -> Any | None:
    """Embed fragments and rebuild the ChromaDB collection (optional)."""
    if not settings.semantic_retrieval_enabled():
        logger.info("Semantic index skipped (ENABLE_SEMANTIC_SEARCH=false)")
        return None

    from knowledge.embeddings import SentenceEmbeddingService
    from knowledge.semantic_retriever import SemanticRetriever
    from knowledge.vector_store import ChromaVectorStore

    embeddings = SentenceEmbeddingService(settings.EMBEDDING_MODEL)
    store = ChromaVectorStore(
        persist_dir=str(settings.CHROMA_PERSIST_DIR),
        collection_name=settings.CHROMA_COLLECTION,
    )
    semantic = SemanticRetriever(embeddings, store)
    semantic.index(fragments)
    return semantic


def rebuild_sparse_index(fragments: list[KnowledgeFragment]) -> SparseRetriever | None:
    """Rebuild BM25 index in memory (used on next HybridRetriever bootstrap)."""
    if not settings.sparse_retrieval_enabled():
        return None
    sparse = SparseRetriever()
    sparse.index(fragments)
    logger.info("BM25 sparse index rebuilt (%d chunks)", sparse.fragment_count)
    return sparse


def rebuild_all_indexes(fragments: list[KnowledgeFragment]) -> None:
    """Rebuild semantic (Chroma) and sparse (BM25) indexes from fragments."""
    rebuild_semantic_index(fragments)
    rebuild_sparse_index(fragments)
