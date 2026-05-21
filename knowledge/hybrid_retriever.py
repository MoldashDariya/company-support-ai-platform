"""Hybrid retrieval: reciprocal rank fusion of sparse (BM25) and optional semantic search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognition.intent import QueryIntent, classify_query_intent
from domain.models import KnowledgeFragment
from knowledge.loader import load_markdown_corpus
from knowledge.reranker import merge_candidate_pool, rerank_fragments
from knowledge.retriever import SparseRetriever
from runtime import settings

if TYPE_CHECKING:
    from knowledge.semantic_retriever import SemanticRetriever

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
        """Wire sparse and optional semantic indexes from an in-memory fragment list."""
        path = corpus_path or settings.CORPUS_PATH

        semantic = _build_semantic_retriever(path, fragments)

        sparse: SparseRetriever | None = None
        if settings.sparse_retrieval_enabled():
            sparse = SparseRetriever()
            if fragments:
                sparse.index(fragments)

        if not semantic and not sparse:
            logger.warning(
                "No retrieval backend enabled; check ENABLE_SEMANTIC_SEARCH and ENABLE_BM25"
            )

        retriever = cls(
            semantic=semantic,
            sparse=sparse,
            enable_sparse=settings.sparse_retrieval_enabled(),
            rrf_k=settings.HYBRID_RRF_K,
        )
        retriever._fragments = fragments
        logger.info(
            "Retriever ready | mode=%s semantic=%s bm25=%s chunks=%d",
            settings.effective_retrieval_mode(),
            semantic is not None,
            sparse is not None,
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
        intent: QueryIntent | None = None,
    ) -> list[KnowledgeFragment]:
        lang = language or settings.CORPUS_LANGUAGE
        mode = settings.effective_retrieval_mode()
        effective_limit = min(limit, settings.RETRIEVAL_TOP_K)
        query_intent = intent or classify_query_intent(query)

        if mode == "sparse" and self._sparse:
            hits = await self._sparse.retrieve(
                query,
                limit=settings.RETRIEVAL_CANDIDATE_POOL,
                min_score=min_score,
            )
            ranked = rerank_fragments(query, hits, limit=effective_limit, intent=query_intent)
            return self._propagate_citation_refs(ranked)

        if mode == "semantic" and self._semantic:
            hits = await self._semantic.retrieve(
                query,
                limit=settings.RETRIEVAL_CANDIDATE_POOL,
                min_score=settings.SEMANTIC_MIN_SCORE,
                language=lang,
            )
            ranked = rerank_fragments(query, hits, limit=effective_limit, intent=query_intent)
            return self._propagate_citation_refs(ranked)

        if self._semantic and self._enable_sparse and self._sparse:
            return await self._hybrid_retrieve(
                query, effective_limit, min_score, language=lang, intent=query_intent
            )

        if self._sparse:
            hits = await self._sparse.retrieve(
                query,
                limit=settings.RETRIEVAL_CANDIDATE_POOL,
                min_score=min_score,
            )
            ranked = rerank_fragments(query, hits, limit=effective_limit, intent=query_intent)
            return self._propagate_citation_refs(ranked)
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
        intent: QueryIntent,
    ) -> list[KnowledgeFragment]:
        pool_size = settings.RETRIEVAL_CANDIDATE_POOL
        rankings: list[list[KnowledgeFragment]] = []

        if self._semantic:
            semantic_hits = await self._semantic.retrieve(
                query,
                limit=pool_size,
                min_score=settings.SEMANTIC_MIN_SCORE,
                language=language,
            )
            if semantic_hits:
                rankings.append(semantic_hits)

        if self._enable_sparse and self._sparse:
            sparse_hits = await self._sparse.retrieve(
                query,
                limit=pool_size,
                min_score=min_score,
            )
            if sparse_hits:
                rankings.append(
                    [h.with_score(h.score, "sparse") for h in sparse_hits]
                )

        if not rankings:
            return []

        if len(rankings) == 1:
            candidates = rankings[0]
        else:
            rrf_ranked = _reciprocal_rank_fusion(
                rankings,
                k=self._rrf_k,
                limit=pool_size,
            )
            candidates = merge_candidate_pool(*rankings, rrf_ranked)

        all_known = self._fragments or candidates
        ranked = rerank_fragments(query, candidates, limit=limit, intent=intent)
        if not ranked and all_known:
            ranked = rerank_fragments(query, all_known, limit=limit, intent=intent)

        return self._propagate_citation_refs(ranked)


def _build_semantic_retriever(
    corpus_path: Path,
    fragments: list[KnowledgeFragment],
) -> Any | None:
    if not settings.semantic_retrieval_enabled():
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
    if corpus_path.exists() and corpus_path.stat().st_size > 0:
        semantic.load_from_path(corpus_path)
    elif fragments:
        semantic.index(fragments)
    else:
        logger.warning("No corpus file and no fragments; semantic index empty")
    return semantic


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
