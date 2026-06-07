"""Retry policy with error classification and jittered backoff."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Classification of API errors for retry decisions."""

    TRANSIENT = "transient"  # 429, 500, 502, 503 — retry with backoff
    CONTEXT_TOO_LONG = "context_too_long"  # Token limit — compress and retry
    AUTH = "auth"  # 401, 403 — try fallback provider
    RATE_LIMIT = "rate_limit"  # 429 with retry-after — wait and retry
    INVALID_REQUEST = "invalid_request"  # 400 — fix messages and retry
    PERMANENT = "permanent"  # 404, unsupported — no retry
    UNKNOWN = "unknown"


class ErrorClassifier:
    """Classifies exceptions into error categories for retry decisions."""

    def classify(self, error: Exception) -> ErrorCategory:
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for common API error patterns
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return ErrorCategory.RATE_LIMIT
        if "context" in error_str and ("too long" in error_str or "exceed" in error_str or "limit" in error_str):
            return ErrorCategory.CONTEXT_TOO_LONG
        if "401" in error_str or "403" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
            return ErrorCategory.AUTH
        if "500" in error_str or "502" in error_str or "503" in error_str or "internal server" in error_str:
            return ErrorCategory.TRANSIENT
        if "timeout" in error_str or "timed out" in error_str or "connection" in error_str:
            return ErrorCategory.TRANSIENT
        if "400" in error_str or "invalid" in error_str:
            return ErrorCategory.INVALID_REQUEST
        if "404" in error_str or "not found" in error_str:
            return ErrorCategory.PERMANENT

        return ErrorCategory.UNKNOWN


@dataclass
class RetryState:
    """Tracks retry state for a single operation."""

    attempt: int = 0
    max_attempts: int = 3
    last_error: Exception | None = None
    last_category: ErrorCategory = ErrorCategory.UNKNOWN
    total_wait_ms: float = 0

    @property
    def should_retry(self) -> bool:
        return self.attempt < self.max_attempts

    @property
    def is_context_too_long(self) -> bool:
        return self.last_category == ErrorCategory.CONTEXT_TOO_LONG

    @property
    def is_auth_error(self) -> bool:
        return self.last_category == ErrorCategory.AUTH


@dataclass
class RetryPolicy:
    """Configurable retry policy with jittered backoff.

    Extensions can subclass or configure this to customize retry behavior.
    """

    max_attempts: int = 3
    base_delay_ms: float = 1000
    max_delay_ms: float = 30000
    backoff_factor: float = 2.0
    jitter: bool = True

    # Which error categories to retry
    retryable_categories: set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.CONTEXT_TOO_LONG,
            ErrorCategory.INVALID_REQUEST,
            ErrorCategory.UNKNOWN,
        }
    )

    def should_retry(self, category: ErrorCategory, attempt: int) -> bool:
        """Check if we should retry given the error category and attempt number."""
        if attempt >= self.max_attempts:
            return False
        return category in self.retryable_categories

    def compute_delay(self, attempt: int, retry_after_ms: float = 0) -> float:
        """Compute delay in milliseconds before next retry."""
        if retry_after_ms > 0:
            return retry_after_ms

        delay = self.base_delay_ms * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay_ms)

        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def create_state(self) -> RetryState:
        return RetryState(max_attempts=self.max_attempts)
