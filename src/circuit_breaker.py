"""
Circuit Breaker — protects against slow/failing LLM providers.

States:
  CLOSED   → Normal operation, requests pass through
  OPEN     → LLM is failing, return fallback immediately (fail fast)
  HALF_OPEN → Testing if LLM has recovered (allow one probe request)

Prevents cascading failures and provides fast error responses when
the LLM provider is degraded.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class CircuitState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Async circuit breaker for LLM calls.

    Args:
        failure_threshold: Consecutive failures before opening
        recovery_timeout:  Seconds to wait before probing (OPEN → HALF_OPEN)
        success_threshold: Consecutive successes in HALF_OPEN to close
        timeout:           Per-call timeout in seconds (0 = no timeout)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        timeout: float = 60.0,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._timeout = timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_short_circuits = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func: Callable[..., Coroutine], *args: Any, **kwargs: Any) -> Any:
        """
        Execute ``func`` through the circuit breaker.
        Raises ``CircuitOpenError`` if the circuit is open.
        """
        async with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("Circuit breaker → HALF_OPEN (probing)")
                else:
                    self._total_short_circuits += 1
                    raise CircuitOpenError(
                        f"Circuit is OPEN — LLM calls blocked for "
                        f"{self._recovery_timeout - (time.time() - self._last_failure_time):.0f}s more"
                    )

        try:
            if self._timeout > 0:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self._timeout)
            else:
                result = await func(*args, **kwargs)
        except Exception as e:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def _on_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker → CLOSED (recovered)")
            else:
                self._failure_count = 0

    async def _on_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker → OPEN (probe failed)")
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker → OPEN after %d consecutive failures",
                    self._failure_count,
                )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_short_circuits": self._total_short_circuits,
        }


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and blocking calls."""
    pass
