"""Hybrid retrieval: reciprocal rank fusion of sparse (BM25) and semantic search."""

from __future__ import annotations

import logging
from pathlib import Path

from domain.models import KnowledgeFragment
from knowledge.loader import load_markdown_corpus
from knowledge.retriever import SparseRetriever
from knowledge.semantic_retriever import SemanticRetriever
from runtime import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines optional BM25 sparse retrieval with dense semantic search.
    Implements the KnowledgeRetriever port used by the conversation pipeline.
    """

    def __init__(
        self,
        semantic: SemanticRetriever | None,
        sparse: SparseRetriever | None = None,
        *,
        enable_sparse: bool = True,
        rrf_k: int = 60,
    ) -> None:
        self._semantic = semantic
        self._sparse = sparse
        self._enable_sparse = enable_sparse and sparse is not None
        self._rrf_k = rrf_k
        self._fragments: list[KnowledgeFragment] = []

    @property
    def fragment_count(self) -> int:
        if self._fragments:
            return len(self._fragments)
        if self._semantic:
            return self._semantic.fragment_count
        return self._sparse.fragment_count if self._sparse else 0

    @classmethod
    def from_corpus(cls, corpus_path: Path | None = None) -> HybridRetriever:
        path = corpus_path or settings.CORPUS_PATH
        fragments = load_markdown_corpus(
            path,
            default_source_url=settings.COMPANY_SITE,
            default_language=settings.CORPUS_LANGUAGE,
        )
        return cls.from_fragments(fragments, corpus_path=path)

    @classmethod
    def from_fragments(
        cls,
        fragments: list[KnowledgeFragment],
        *,
        corpus_path: Path | None = None,
    ) -> HybridRetriever:
        """Wire semantic + sparse indexes from an in-memory fragment list."""
        path = corpus_path or settings.CORPUS_PATH

        semantic: SemanticRetriever | None = None
        if settings.RETRIEVAL_MODE in ("semantic", "hybrid"):
            from knowledge.embeddings import SentenceEmbeddingService
            from knowledge.vector_store import ChromaVectorStore

            embeddings = SentenceEmbeddingService(settings.EMBEDDING_MODEL)
            store = ChromaVectorStore(
                persist_dir=str(settings.CHROMA_PERSIST_DIR),
                collection_name=settings.CHROMA_COLLECTION,
            )
            semantic = SemanticRetriever(embeddings, store)
            if path.exists() and path.stat().st_size > 0:
                semantic.load_from_path(path)
            elif fragments:
                semantic.index(fragments)
            else:
                logger.warning("No corpus file and no fragments; semantic index empty")

        sparse: SparseRetriever | None = None
        if settings.ENABLE_BM25 and settings.RETRIEVAL_MODE in ("sparse", "hybrid"):
            sparse = SparseRetriever()
            if fragments:
                sparse.index(fragments)

        retriever = cls(
            semantic=semantic,
            sparse=sparse,
            enable_sparse=settings.ENABLE_BM25,
            rrf_k=settings.HYBRID_RRF_K,
        )
        retriever._fragments = fragments
        logger.info(
            "Hybrid retriever ready | mode=%s bm25=%s chunks=%d",
            settings.RETRIEVAL_MODE,
            settings.ENABLE_BM25,
            retriever.fragment_count,
        )
        return retriever

    async def retrieve(
        self,
        query: str,
        limit: int = 4,
        min_score: float = 0.0,
        *,
        language: str | None = None,
    ) -> list[KnowledgeFragment]:
        lang = language or settings.CORPUS_LANGUAGE
        mode = settings.RETRIEVAL_MODE

        if mode == "sparse" and self._sparse:
            hits = await self._sparse.retrieve(query, limit=limit, min_score=min_score)
            return self._propagate_citation_refs(hits)

        if mode == "semantic" and self._semantic:
            hits = await self._semantic.retrieve(
                query,
                limit=limit,
                min_score=settings.SEMANTIC_MIN_SCORE,
                language=lang,
            )
            return self._propagate_citation_refs(hits)

        if self._semantic:
            return await self._hybrid_retrieve(query, limit, min_score, language=lang)

        if self._sparse:
            hits = await self._sparse.retrieve(query, limit=limit, min_score=min_score)
            return self._propagate_citation_refs(hits)
        return []

    @staticmethod
    def _propagate_citation_refs(
        fragments: list[KnowledgeFragment],
    ) -> list[KnowledgeFragment]:
        from knowledge.context import attach_citations_to_fragments, build_citations_from_fragments

        citations = build_citations_from_fragments(fragments)
        return attach_citations_to_fragments(fragments, citations)

    async def _hybrid_retrieve(
        self,
        query: str,
        limit: int,
        min_score: float,
        *,
        language: str | None,
    ) -> list[KnowledgeFragment]:
        rankings: list[list[KnowledgeFragment]] = []

        semantic_hits = await self._semantic.retrieve(
            query,
            limit=limit * 2,
            min_score=settings.SEMANTIC_MIN_SCORE,
            language=language,
        )
        if semantic_hits:
            rankings.append(semantic_hits)

        if self._enable_sparse and self._sparse:
            sparse_hits = await self._sparse.retrieve(
                query,
                limit=limit * 2,
                min_score=min_score,
            )
            if sparse_hits:
                rankings.append(
                    [h.with_score(h.score, "sparse") for h in sparse_hits]
                )

        if not rankings:
            return []

        if len(rankings) == 1:
            fused = rankings[0][:limit]
        else:
            fused = _reciprocal_rank_fusion(rankings, k=self._rrf_k, limit=limit)

        fused = [f.with_score(f.score, "hybrid") for f in fused]
        return self._propagate_citation_refs(fused)


def _fragment_key(fragment: KnowledgeFragment) -> str:
    return f"{fragment.section}::{hash(fragment.body)}"


def _reciprocal_rank_fusion(
    rankings: list[list[KnowledgeFragment]],
    *,
    k: int,
    limit: int,
) -> list[KnowledgeFragment]:
    scores: dict[str, float] = {}
    best: dict[str, KnowledgeFragment] = {}

    for ranking in rankings:
        for rank, fragment in enumerate(ranking):
            key = _fragment_key(fragment)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best or fragment.score > best[key].score:
                best[key] = fragment

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[KnowledgeFragment] = []
    for key, rrf_score in ordered[:limit]:
        fragment = best[key]
        fused.append(fragment.with_score(rrf_score, "hybrid"))
    return fused
