"""Per-session request throttling to protect LLM API quota."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from runtime import settings


class SlidingWindowThrottle:
    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def acquire(self, session_id: str) -> bool:
        now = time.time()
        window_start = now - settings.THROTTLE_WINDOW_SEC
        hits = self._windows[session_id]
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= settings.THROTTLE_MAX_REQUESTS:
            return False
        hits.append(now)
        return True

    def cooldown_seconds(self, session_id: str) -> int:
        hits = self._windows[session_id]
        if not hits:
            return settings.THROTTLE_WINDOW_SEC
        wait = settings.THROTTLE_WINDOW_SEC - (time.time() - hits[0])
        return max(1, int(wait) + 1)
