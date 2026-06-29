"""15-minute self-cleaning cache."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


def make_cache_key(*parts: Any) -> str:
    """Hash any (str/dict/list/tuple/None) into a stable key."""
    s = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


class TTLCache:
    """A very lightweight thread-safe TTL+LRU cache."""

    def __init__(self, ttl_seconds: int = 900, max_entries: int = 256):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            ts, value = hit
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_entries:
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest_key, None)
            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
