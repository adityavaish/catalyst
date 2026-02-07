"""
Tests for cache and circuit breaker.
"""

import asyncio

import pytest

from src.cache import ResponseCache
from src.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from src.models import CatalystResponse


# ---------------------------------------------------------------------------
# ResponseCache tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_miss():
    cache = ResponseCache(max_size=10, default_ttl=60)
    result = await cache.get("nonexistent")
    assert result is None
    assert cache.stats["misses"] == 1


@pytest.mark.asyncio
async def test_cache_hit():
    cache = ResponseCache(max_size=10, default_ttl=60)
    resp = CatalystResponse(status_code=200, data={"hello": "world"})
    await cache.set("key1", resp)
    result = await cache.get("key1")
    assert result is not None
    assert result.data == {"hello": "world"}
    assert cache.stats["hits"] == 1


@pytest.mark.asyncio
async def test_cache_ttl_expiry():
    cache = ResponseCache(max_size=10, default_ttl=0.1)  # 100ms TTL
    resp = CatalystResponse(status_code=200, data="cached")
    await cache.set("key1", resp, ttl=0.1)
    # Should hit immediately
    assert await cache.get("key1") is not None
    # Wait for expiry
    await asyncio.sleep(0.15)
    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_cache_eviction():
    cache = ResponseCache(max_size=2, default_ttl=60)
    await cache.set("a", CatalystResponse(data="1"))
    await cache.set("b", CatalystResponse(data="2"))
    await cache.set("c", CatalystResponse(data="3"))  # should evict "a"
    assert await cache.get("a") is None
    assert await cache.get("b") is not None
    assert await cache.get("c") is not None
    assert cache.stats["evictions"] == 1


@pytest.mark.asyncio
async def test_cache_make_key():
    k1 = ResponseCache.make_key("/api/test", "POST", {"x": 1})
    k2 = ResponseCache.make_key("/api/test", "POST", {"x": 1})
    k3 = ResponseCache.make_key("/api/test", "POST", {"x": 2})
    assert k1 == k2  # same input → same key
    assert k1 != k3  # different input → different key


@pytest.mark.asyncio
async def test_cache_invalidate():
    cache = ResponseCache(max_size=10, default_ttl=60)
    await cache.set("key1", CatalystResponse(data="x"))
    await cache.invalidate("key1")
    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_cache_hit_rate():
    cache = ResponseCache(max_size=10, default_ttl=60)
    await cache.set("a", CatalystResponse(data="1"))
    await cache.get("a")  # hit
    await cache.get("a")  # hit
    await cache.get("b")  # miss
    stats = cache.stats
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.667, abs=0.01)


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_starts_closed():
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_passes_through_on_success():
    cb = CircuitBreaker(failure_threshold=3)
    result = await cb.call(asyncio.coroutine(lambda: "ok"))
    assert result == "ok"


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)

    async def fail():
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)

    assert cb.state == CircuitState.OPEN

    # Next call should be short-circuited
    with pytest.raises(CircuitOpenError):
        await cb.call(fail)

    assert cb.stats["total_short_circuits"] == 1


@pytest.mark.asyncio
async def test_circuit_recovers():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, success_threshold=1)

    async def fail():
        raise ValueError("boom")

    async def succeed():
        return "ok"

    # Trip the circuit
    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout
    await asyncio.sleep(0.15)

    # Should go to HALF_OPEN and succeed
    result = await cb.call(succeed)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_timeout():
    cb = CircuitBreaker(failure_threshold=3, timeout=0.1)

    async def slow():
        await asyncio.sleep(1)
        return "never"

    with pytest.raises(asyncio.TimeoutError):
        await cb.call(slow)
