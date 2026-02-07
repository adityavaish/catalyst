"""
Response Cache — avoids redundant LLM calls for identical/similar requests.

Three caching strategies:
  1. Exact match   — hash(endpoint_path + method + input) → cached response
  2. TTL-based     — configurable per-endpoint expiry
  3. Semantic      — optional fuzzy matching via embedding similarity (future)

For deterministic endpoints (temperature=0), caching is especially effective.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached response."""
    key: str
    value: Any
    created_at: float
    ttl: float  # seconds, 0 = infinite
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl


class ResponseCache:
    """
    In-memory LRU response cache with TTL support.

    For production, swap this with Redis-backed cache by subclassing.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    @staticmethod
    def make_key(endpoint_path: str, method: str, input_data: Any) -> str:
        """Create a deterministic cache key from request data."""
        raw = json.dumps(
            {"path": endpoint_path, "method": method, "input": input_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached response, or None if miss/expired."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._stats["misses"] += 1
                return None
            entry.hits += 1
            self._stats["hits"] += 1
            logger.debug("Cache HIT: %s (hits=%d)", key[:12], entry.hits)
            return entry.value

    async def set(self, key: str, value: Any, ttl: float | None = None):
        """Store a response in cache."""
        async with self._lock:
            # Evict if at capacity (simple: remove oldest)
            if len(self._store) >= self._max_size and key not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
                self._stats["evictions"] += 1

            self._store[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl if ttl is not None else self._default_ttl,
            )

    async def invalidate(self, key: str):
        """Remove a specific entry."""
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self):
        """Flush the entire cache."""
        async with self._lock:
            self._store.clear()

    @property
    def stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "size": len(self._store),
            "max_size": self._max_size,
            "hit_rate": round(self._stats["hits"] / total, 3) if total > 0 else 0,
        }
