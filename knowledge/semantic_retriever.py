"""Dense semantic retrieval via sentence embeddings and ChromaDB."""

from __future__ import annotations

import logging
from pathlib import Path

from domain.models import KnowledgeFragment
from knowledge.embeddings import SentenceEmbeddingService
from knowledge.loader import corpus_fingerprint, load_markdown_corpus
from knowledge.vector_store import ChromaVectorStore, embedding_text
from runtime import settings

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Vector-similarity retriever with metadata-filtered search."""

    def __init__(
        self,
        embedding_service: SentenceEmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._embeddings = embedding_service
        self._store = vector_store
        self._fragments: list[KnowledgeFragment] = []

    @property
    def fragment_count(self) -> int:
        return len(self._fragments) or self._store.document_count

    def index(self, fragments: list[KnowledgeFragment]) -> None:
        self._fragments = fragments
        if not fragments:
            return

        texts = [embedding_text(frag) for frag in fragments]
        vectors = self._embeddings.embed_documents(texts)
        self._store.upsert_fragments(fragments, vectors)
        self._store.set_corpus_hash(corpus_fingerprint(fragments))
        logger.info("Semantic index built: %d chunks", len(fragments))

    def load_from_path(self, path: Path) -> None:
        fragments = load_markdown_corpus(
            path,
            default_source_url=settings.COMPANY_SITE,
            default_language=settings.CORPUS_LANGUAGE,
        )
        fingerprint = corpus_fingerprint(fragments)
        if (
            self._store.document_count == len(fragments)
            and self._store.stored_corpus_hash() == fingerprint
        ):
            self._fragments = fragments
            logger.info("Reusing existing Chroma index (%d chunks)", len(fragments))
            return
        self.index(fragments)

    async def retrieve(
        self,
        query: str,
        limit: int = 4,
        min_score: float = 0.0,
        *,
        language: str | None = None,
        source_url: str | None = None,
        section: str | None = None,
    ) -> list[KnowledgeFragment]:
        if self._store.document_count == 0:
            return []

        query_vector = self._embeddings.embed_query(query)
        where = _build_metadata_filter(language=language, source_url=source_url, section=section)

        hits = self._store.query(
            query_vector,
            limit=limit * 2,
            where=where or None,
        )

        results: list[KnowledgeFragment] = []
        for fragment, similarity in hits:
            if similarity < min_score:
                continue
            results.append(
                fragment.with_score(similarity, retrieval_source="semantic")
            )
            if len(results) >= limit:
                break

        return results if results else [f.with_score(f.score, "semantic") for f, _ in hits[:limit]]


def _build_metadata_filter(
    *,
    language: str | None,
    source_url: str | None,
    section: str | None,
) -> dict | None:
    clauses: list[dict] = []
    if language:
        clauses.append({"language": language})
    if source_url:
        clauses.append({"source_url": source_url})
    if section:
        clauses.append({"section": section})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
