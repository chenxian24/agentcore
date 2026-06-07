"""Context sources: pluggable providers of context information.

A ContextSource provides additional context (file contents, git state, memory, etc.)
that gets aggregated into the message context before compression.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """A piece of context from a source."""

    source_name: str
    content: str
    priority: int = 100  # lower = higher priority, kept first
    token_estimate: int = 0  # 0 = auto-estimate
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4


class ContextSource(ABC):
    """Abstract base for context providers.

    Extensions implement this to inject domain-specific context:
    - FileContextSource: relevant file contents
    - GitContextSource: git status, recent commits
    - MemoryContextSource: recalled memories
    - SearchContextSource: search results
    - LSPContextSource: language server diagnostics
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def priority(self) -> int:
        """Default priority for chunks from this source. Lower = kept first."""
        return 100

    @abstractmethod
    async def gather(self, query: str, max_tokens: int, **kwargs: Any) -> list[ContextChunk]:
        """Gather context chunks relevant to the query.

        Args:
            query: The user's input or current focus
            max_tokens: Maximum tokens to allocate to this source
            **kwargs: Additional context (session, messages, etc.)

        Returns:
            List of ContextChunk objects, ordered by relevance
        """
        ...


class ContextAggregator:
    """Aggregates context from multiple sources.

    Gathers context from all registered sources, merges by priority,
    and fits within a token budget. The result is prepended to the
    message context before compression.
    """

    def __init__(self, max_tokens: int = 10000) -> None:
        self._sources: dict[str, ContextSource] = {}
        self._max_tokens = max_tokens

    def register(self, source: ContextSource) -> None:
        self._sources[source.name] = source

    def unregister(self, name: str) -> None:
        self._sources.pop(name, None)

    def get(self, name: str) -> ContextSource | None:
        return self._sources.get(name)

    def list_sources(self) -> list[str]:
        return list(self._sources.keys())

    async def gather(
        self,
        query: str,
        max_tokens: int | None = None,
        source_names: list[str] | None = None,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        """Gather context from all (or specified) sources.

        Returns chunks sorted by priority, trimmed to fit within max_tokens.
        """
        budget = max_tokens or self._max_tokens
        sources = (
            [self._sources[n] for n in source_names if n in self._sources]
            if source_names
            else list(self._sources.values())
        )

        # Gather from all sources concurrently
        import asyncio
        tasks = []
        per_source_budget = budget // len(sources) if sources else 0
        for source in sources:
            tasks.append(self._safe_gather(source, query, per_source_budget, **kwargs))

        results = await asyncio.gather(*tasks)

        # Merge and sort by priority
        all_chunks: list[ContextChunk] = []
        for chunks in results:
            all_chunks.extend(chunks)
        all_chunks.sort(key=lambda c: c.priority)

        # Trim to fit budget
        result: list[ContextChunk] = []
        used = 0
        for chunk in all_chunks:
            if used + chunk.token_estimate > budget:
                break
            result.append(chunk)
            used += chunk.token_estimate

        return result

    async def _safe_gather(
        self,
        source: ContextSource,
        query: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        try:
            return await source.gather(query, max_tokens, **kwargs)
        except Exception:
            logger.error("Context source '%s' failed", source.name, exc_info=True)
            return []
