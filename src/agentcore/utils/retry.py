"""Generic retry with jittered backoff — no agentcore type dependencies."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay_ms: float = 1000
    max_delay_ms: float = 30000
    backoff_factor: float = 2.0
    jitter: bool = True
    # Which error categories are retryable (strings, no enum dependency)
    retryable_categories: set[str] = field(default_factory=lambda: {
        "transient", "rate_limit", "unknown",
    })


def classify_error(error: Exception) -> str:
    """Classify an exception into a category string.

    Pure function, no agentcore dependencies. Returns one of:
    "transient", "rate_limit", "auth", "context_too_long",
    "invalid_request", "permanent", "unknown"
    """
    error_str = str(error).lower()

    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return "rate_limit"
    if "context" in error_str and ("too long" in error_str or "exceed" in error_str or "limit" in error_str):
        return "context_too_long"
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return "auth"
    if "500" in error_str or "502" in error_str or "503" in error_str or "internal server" in error_str:
        return "transient"
    if "timeout" in error_str or "timed out" in error_str or "connection" in error_str:
        return "transient"
    if "400" in error_str or "invalid" in error_str:
        return "invalid_request"
    if "404" in error_str or "not found" in error_str:
        return "permanent"

    return "unknown"


def compute_delay(attempt: int, config: RetryConfig, retry_after_ms: float = 0) -> float:
    """Compute delay in milliseconds before next retry."""
    if retry_after_ms > 0:
        return retry_after_ms

    delay = config.base_delay_ms * (config.backoff_factor ** attempt)
    delay = min(delay, config.max_delay_ms)

    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    classify: Callable[[Exception], str] | None = None,
    on_retry: Callable[[Exception, str, int], Awaitable[None] | None] | None = None,
    **kwargs: Any,
) -> T:
    """Call an async function with retry logic.

    Args:
        fn: Async function to call.
        *args: Positional arguments to fn.
        config: Retry configuration. Defaults to RetryConfig().
        classify: Error classifier. Defaults to classify_error.
        on_retry: Optional callback(error, category, attempt) before each retry.
        **kwargs: Keyword arguments to fn.

    Returns:
        The result of fn.

    Raises:
        The last exception if all retries are exhausted, or a non-retryable error.
    """
    cfg = config or RetryConfig()
    classify_fn = classify or classify_error

    last_error: Exception | None = None

    for attempt in range(cfg.max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as error:
            last_error = error
            category = classify_fn(error)

            if category not in cfg.retryable_categories:
                raise

            if attempt >= cfg.max_attempts - 1:
                raise

            delay_ms = compute_delay(attempt, cfg)

            if on_retry:
                result = on_retry(error, category, attempt)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result

            logger.info(
                "Retrying after %.0fms (attempt %d/%d, category=%s)",
                delay_ms, attempt + 1, cfg.max_attempts, category,
            )
            await asyncio.sleep(delay_ms / 1000)

    raise last_error  # type: ignore[misc]
