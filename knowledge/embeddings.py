"""Sentence embedding service — lazy-loaded when ENABLE_SEMANTIC_SEARCH=true."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SentenceEmbeddingService:
    """Wraps sentence-transformers for document and query encoding."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector[0].tolist()
