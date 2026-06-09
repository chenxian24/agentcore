"""Tool execution retry policies.

Provides pluggable retry strategies for transient tool failures.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class RetryPolicy(ABC):
    """Abstract retry policy for tool execution failures."""

    @abstractmethod
    def should_retry(self, attempt: int, error: Exception | str) -> bool:
        """Return True if the tool call should be retried."""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Return the delay in seconds before the next retry attempt."""

    def on_retry(self, attempt: int, error: Exception | str, delay: float) -> None:
        """Called before a retry. Override for logging/metrics."""
        logger.info("Retrying tool (attempt %d) after %.1fs: %s", attempt + 1, delay, error)


class NoRetryPolicy(RetryPolicy):
    """Never retry (default)."""

    def should_retry(self, attempt: int, error: Exception | str) -> bool:
        return False

    def get_delay(self, attempt: int) -> float:
        return 0.0


class ExponentialBackoffPolicy(RetryPolicy):
    """Retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubled each attempt).
        max_delay: Maximum delay in seconds.
        retryable_errors: Optional set of error substrings that trigger retry.
            If empty, all errors are retried.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable_errors: set[str] | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._retryable_errors = retryable_errors or set()

    def should_retry(self, attempt: int, error: Exception | str) -> bool:
        if attempt >= self._max_retries:
            return False
        if not self._retryable_errors:
            return True
        error_str = str(error).lower()
        return any(pattern.lower() in error_str for pattern in self._retryable_errors)

    def get_delay(self, attempt: int) -> float:
        delay = self._base_delay * (2 ** attempt)
        return min(delay, self._max_delay)


class FixedDelayPolicy(RetryPolicy):
    """Retry with a fixed delay.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Fixed delay in seconds between retries.
        retryable_errors: Optional set of error substrings that trigger retry.
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 2.0,
        retryable_errors: set[str] | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._delay = delay
        self._retryable_errors = retryable_errors or set()

    def should_retry(self, attempt: int, error: Exception | str) -> bool:
        if attempt >= self._max_retries:
            return False
        if not self._retryable_errors:
            return True
        error_str = str(error).lower()
        return any(pattern.lower() in error_str for pattern in self._retryable_errors)

    def get_delay(self, attempt: int) -> float:
        return self._delay
