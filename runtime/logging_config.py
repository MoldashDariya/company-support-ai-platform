"""Structured logging setup with secret redaction."""

from __future__ import annotations

import logging
import re
import sys


class _SecretRedactor(logging.Filter):
    _token_re = re.compile(r"\d{8,}:[A-Za-z0-9_-]{20,}")
    _openai_key_re = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")
    _bearer_re = re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE)

    def _redact(self, text: str) -> str:
        text = self._token_re.sub("[REDACTED]", text)
        text = self._openai_key_re.sub("[REDACTED]", text)
        text = self._bearer_re.sub("Bearer [REDACTED]", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            record.args = tuple(
                self._redact(a) if isinstance(a, str) else a for a in record.args
            )
        return True


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger().addFilter(_SecretRedactor())
