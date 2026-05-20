"""Sparse lexical retrieval (BM25) over the company knowledge corpus."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from domain.models import KnowledgeFragment
from knowledge.loader import load_markdown_corpus

_STOPWORDS = frozenset(
    "и в на с по для что как где это не о от до из к у же ли бы а то все".split()
)


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[\wа-яё]+", text.lower(), flags=re.IGNORECASE)
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


class SparseRetriever:
    """BM25-based sparse retriever — optional leg of hybrid RAG."""

    def __init__(self, fragments: list[KnowledgeFragment] | None = None) -> None:
        self._fragments: list[KnowledgeFragment] = fragments or []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    @property
    def fragment_count(self) -> int:
        return len(self._fragments)

    def index(self, fragments: list[KnowledgeFragment]) -> None:
        self._fragments = fragments
        self._corpus_tokens = [
            _tokenize(f"{f.title} {f.section} {f.body}") for f in self._fragments
        ]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def load_from_path(self, path, **loader_kwargs) -> None:
        self.index(load_markdown_corpus(path, **loader_kwargs))

    async def retrieve(
        self,
        query: str,
        limit: int = 4,
        min_score: float = 0.0,
    ) -> list[KnowledgeFragment]:
        if not self._bm25 or not self._fragments:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return self._top_fragments(limit)

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(scores, self._fragments),
            key=lambda pair: pair[0],
            reverse=True,
        )

        results: list[KnowledgeFragment] = []
        for score, fragment in ranked:
            if score <= min_score:
                continue
            results.append(
                fragment.with_score(float(score), retrieval_source="sparse")
            )
            if len(results) >= limit:
                break

        return results if results else self._top_fragments(limit)

    def _top_fragments(self, limit: int) -> list[KnowledgeFragment]:
        return [
            f.with_score(0.0, retrieval_source="sparse")
            for f in self._fragments[:limit]
        ]
