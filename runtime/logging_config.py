"""Structured logging setup with secret redaction."""

from __future__ import annotations

import logging
import re
import sys


class _SecretRedactor(logging.Filter):
    _token_re = re.compile(r"\d{8,}:[A-Za-z0-9_-]{20,}")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._token_re.sub("[REDACTED]", record.msg)
        return True


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger().addFilter(_SecretRedactor())
