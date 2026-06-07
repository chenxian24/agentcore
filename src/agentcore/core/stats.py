"""Request statistics collector for model API interactions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RequestStats:
    """Snapshot of a single model API request."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    finish_reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregateStats:
    """Aggregated statistics over multiple requests."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    by_model: dict[str, AggregateStats] = field(default_factory=dict)


class StatsCollector:
    """Collects and aggregates model API request statistics.

    Thread-safe. Stores all data in memory — callers decide on persistence.
    """

    def __init__(self, max_recent: int = 200) -> None:
        self._max_recent = max_recent
        self._recent: list[RequestStats] = []
        self._aggregate = AggregateStats()
        self._model_aggregates: dict[str, AggregateStats] = {}
        self._total_latency: float = 0.0
        self._model_total_latency: dict[str, float] = {}
        self._lock = Lock()

    def record(self, stats: RequestStats) -> None:
        """Record a single request's statistics."""
        with self._lock:
            self._recent.append(stats)
            if len(self._recent) > self._max_recent:
                self._recent = self._recent[-self._max_recent:]

            # Update global aggregate
            self._aggregate.total_requests += 1
            if stats.success:
                self._aggregate.successful_requests += 1
            else:
                self._aggregate.failed_requests += 1
            self._aggregate.total_prompt_tokens += stats.prompt_tokens
            self._aggregate.total_completion_tokens += stats.completion_tokens
            self._aggregate.total_reasoning_tokens += stats.reasoning_tokens
            self._aggregate.total_tokens += stats.total_tokens
            self._total_latency += stats.latency_ms
            if self._aggregate.total_requests > 0:
                self._aggregate.avg_latency_ms = self._total_latency / self._aggregate.total_requests

            # Update per-model aggregate
            model = stats.model or "unknown"
            if model not in self._model_aggregates:
                self._model_aggregates[model] = AggregateStats()
                self._model_total_latency[model] = 0.0

            ma = self._model_aggregates[model]
            ma.total_requests += 1
            if stats.success:
                ma.successful_requests += 1
            else:
                ma.failed_requests += 1
            ma.total_prompt_tokens += stats.prompt_tokens
            ma.total_completion_tokens += stats.completion_tokens
            ma.total_reasoning_tokens += stats.reasoning_tokens
            ma.total_tokens += stats.total_tokens
            self._model_total_latency[model] += stats.latency_ms
            if ma.total_requests > 0:
                ma.avg_latency_ms = self._model_total_latency[model] / ma.total_requests

            self._aggregate.by_model = dict(self._model_aggregates)

    def get_aggregate(self) -> AggregateStats:
        """Return aggregated statistics (includes per-model breakdown)."""
        with self._lock:
            return self._aggregate

    def get_recent(self, limit: int = 50) -> list[RequestStats]:
        """Return the most recent request stats."""
        with self._lock:
            return list(self._recent[-limit:])

    def reset(self) -> None:
        """Clear all collected statistics."""
        with self._lock:
            self._recent.clear()
            self._aggregate = AggregateStats()
            self._model_aggregates.clear()
            self._total_latency = 0.0
            self._model_total_latency.clear()
