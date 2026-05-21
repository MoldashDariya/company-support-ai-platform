"""Core domain entities for the support assistant platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class ChunkMetadata:
    """Structured provenance for a knowledge chunk."""

    title: str
    section: str
    source_url: str
    language: str


@dataclass(frozen=True)
class SourceCitation:
    """Human-readable reference to a grounded evidence source."""

    ref_id: int
    title: str
    section: str
    source_url: str = ""
    language: str = "ru"
    retrieval_source: str = ""

    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.title.strip(), self.section.strip(), self.source_url.strip())

    def to_short_label(self) -> str:
        """Compact label for logs and legacy consumers."""
        if self.source_url:
            return f"{self.title} — {self.section} ({self.source_url})"
        return f"{self.title} — {self.section}"


@dataclass(frozen=True)
class KnowledgeFragment:
    """A single indexed passage from the company knowledge corpus."""

    section: str
    body: str
    score: float = 0.0
    metadata: ChunkMetadata | None = None
    retrieval_source: str = ""
    citation_ref: int = 0

    @property
    def label(self) -> str:
        return self.metadata.title if self.metadata else self.section

    @property
    def title(self) -> str:
        return self.metadata.title if self.metadata else self.section

    @property
    def source_url(self) -> str:
        if self.metadata:
            return self.metadata.source_url
        return ""

    @property
    def language(self) -> str:
        if self.metadata:
            return self.metadata.language
        return "ru"

    def with_score(self, score: float, retrieval_source: str = "") -> KnowledgeFragment:
        return KnowledgeFragment(
            section=self.section,
            body=self.body,
            score=score,
            metadata=self.metadata,
            retrieval_source=retrieval_source or self.retrieval_source,
            citation_ref=self.citation_ref,
        )

    def with_citation_ref(self, ref_id: int) -> KnowledgeFragment:
        return KnowledgeFragment(
            section=self.section,
            body=self.body,
            score=self.score,
            metadata=self.metadata,
            retrieval_source=self.retrieval_source,
            citation_ref=ref_id,
        )


@dataclass(frozen=True)
class ChatTurn:
    role: MessageRole
    content: str

    def to_llm_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class UserInquiry:
    """Inbound user message normalized for processing."""

    session_id: str
    text: str
    channel: str = "telegram"


@dataclass
class GroundedContext:
    """Retrieved evidence bundle passed to the language model."""

    fragments: list[KnowledgeFragment] = field(default_factory=list)
    formatted: str = ""
    citations: list[SourceCitation] = field(default_factory=list)
    has_sufficient_evidence: bool = True
    query_intent: str = "general"
    retrieval_profile: str = "balanced"
    response_style: str = "adaptive"
    last_assistant_opening: str = ""


@dataclass
class AssistantResponse:
    """Final assistant output with grounding metadata."""

    text: str
    sources: list[str] = field(default_factory=list)
    citations: list[SourceCitation] = field(default_factory=list)
    grounded: bool = True


@dataclass
class PipelineResult:
    """Outcome of the conversation pipeline for a single inquiry."""

    reply: AssistantResponse
    is_first_contact: bool = False
    blocked: bool = False
    block_reason: str | None = None
    query_intent: str = "general"
