"""ChromaDB-backed vector store with metadata-aware similarity search."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from domain.models import ChunkMetadata, KnowledgeFragment

logger = logging.getLogger(__name__)

_CHROMA_METADATA_KEYS = ("title", "section", "source_url", "language")


class ChromaVectorStore:
    """Persistent vector index with structured chunk metadata."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._hash_path = Path(persist_dir) / ".corpus_hash"
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def stored_corpus_hash(self) -> str | None:
        if not self._hash_path.exists():
            return None
        return self._hash_path.read_text(encoding="utf-8").strip()

    def set_corpus_hash(self, corpus_hash: str) -> None:
        self._hash_path.parent.mkdir(parents=True, exist_ok=True)
        self._hash_path.write_text(corpus_hash, encoding="utf-8")

    def upsert_fragments(
        self,
        fragments: list[KnowledgeFragment],
        embeddings: list[list[float]],
    ) -> None:
        if len(fragments) != len(embeddings):
            raise ValueError("fragments and embeddings length mismatch")

        if not fragments:
            return

        self._reset_collection()

        ids = [_chunk_id(frag) for frag in fragments]
        documents = [_document_text(frag) for frag in fragments]
        metadatas = [_to_chroma_metadata(frag) for frag in fragments]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks in Chroma collection '%s'", len(fragments), self._collection_name)

    def query(
        self,
        query_embedding: list[float],
        *,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[KnowledgeFragment, float]]:
        if self._collection.count() == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(limit, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[tuple[KnowledgeFragment, float]] = []
        for doc_id, document, meta, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - float(distance)
            fragment = _fragment_from_chroma(doc_id, document, meta, similarity)
            hits.append((fragment, similarity))

        return hits

    def _reset_collection(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _chunk_id(fragment: KnowledgeFragment) -> str:
    url = fragment.source_url or ""
    key = f"{url}|{fragment.section}|{fragment.body[:120]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def embedding_text(fragment: KnowledgeFragment) -> str:
    """Rich text for embedding (title + section + body)."""
    return f"{fragment.title} | {fragment.section}\n{fragment.body}"


def _document_text(fragment: KnowledgeFragment) -> str:
    """Stored document is the chunk body (used when rehydrating hits)."""
    return fragment.body


def _to_chroma_metadata(fragment: KnowledgeFragment) -> dict[str, str]:
    meta = fragment.metadata
    if meta is None:
        return {
            "title": fragment.section,
            "section": fragment.section,
            "source_url": "",
            "language": "ru",
        }
    return {
        "title": meta.title,
        "section": meta.section,
        "source_url": meta.source_url,
        "language": meta.language,
    }


def _fragment_from_chroma(
    doc_id: str,
    document: str | None,
    meta: dict[str, Any] | None,
    similarity: float,
) -> KnowledgeFragment:
    del doc_id  # stable id; body comes from stored document
    meta = meta or {}
    body = document or ""

    chunk_meta = ChunkMetadata(
        title=str(meta.get("title", "")),
        section=str(meta.get("section", "")),
        source_url=str(meta.get("source_url", "")),
        language=str(meta.get("language", "ru")),
    )
    return KnowledgeFragment(
        section=chunk_meta.section,
        body=body,
        score=similarity,
        metadata=chunk_meta,
        retrieval_source="semantic",
    )
