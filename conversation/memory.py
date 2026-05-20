"""Backward-compatible export — prefer memory.session_repository."""

from memory.session_repository import SessionRepository

# Legacy alias
JsonSessionStore = SessionRepository

__all__ = ["SessionRepository", "JsonSessionStore"]
