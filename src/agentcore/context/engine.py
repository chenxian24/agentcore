"""Context engine with pluggable compression strategies and context pipeline.

Provides two levels of API:
1. Simple: ContextEngine.compress() with pluggable strategies (backward compatible)
2. Advanced: ContextPipeline with gather → rank → dedupe → budget → compress → render
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentcore.core.message import Message
    from agentcore.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Strategy API (backward compatible)
# ---------------------------------------------------------------------------

class ContextStrategy(ABC):
    """Protocol for context compression strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        system_prompt_tokens: int = 0,
    ) -> list[Message]:
        """Return a subset/compression of messages that fits within max_tokens."""
        ...

    def on_session_start(self, session: Any) -> None:
        """Called when a new session starts. Override to initialize state."""

    def on_turn_end(self, messages: list[Message], response: Any) -> None:
        """Called after each LLM turn. Override to track state."""

    def should_compress(self, messages: list[Message], max_tokens: int) -> bool:
        """Return True if compression is needed. Default: always compress."""
        return True

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text. Default: len(text) // 4."""
        return len(text) // 4


class SlidingWindowStrategy(ContextStrategy):
    """Keep the most recent messages that fit in the budget."""

    @property
    def name(self) -> str:
        return "sliding_window"

    def __init__(self, reserve_tokens: int = 4096) -> None:
        self._reserve = reserve_tokens

    def _estimate_tokens(self, text: str) -> int:
        return self.estimate_tokens(text)

    def _estimate_message_tokens(self, msg: Message) -> int:
        """Estimate tokens for a message including tool_calls."""
        cost = self._estimate_tokens(msg.content)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                cost += self._estimate_tokens(tc.function.name)
                cost += self._estimate_tokens(tc.function.arguments)
                cost += 10  # overhead per tool call
        if msg.tool_call_id:
            cost += 10
        return cost

    def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        system_prompt_tokens: int = 0,
    ) -> list[Message]:
        available = max_tokens - system_prompt_tokens - self._reserve
        result: list[Message] = []
        used = 0
        for msg in reversed(messages):
            cost = self._estimate_message_tokens(msg)
            if used + cost > available:
                break
            result.insert(0, msg)
            used += cost
        return result


class SummarizationStrategy(ContextStrategy):
    """Keep recent N messages verbatim, mark older ones for summarization.

    Returns a summary placeholder message with metadata indicating
    that actual summarization is needed. A hook can intercept
    POST_BUILD_MESSAGES to perform the LLM call.
    """

    @property
    def name(self) -> str:
        return "summarization"

    def __init__(self, keep_recent: int = 10) -> None:
        self._keep_recent = keep_recent

    def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        system_prompt_tokens: int = 0,
    ) -> list[Message]:
        if len(messages) <= self._keep_recent:
            return messages

        old = messages[: -self._keep_recent]
        recent = messages[-self._keep_recent :]

        from agentcore.core.message import Message

        summary_text = f"[{len(old)} earlier messages summarized]"
        summary_msg = Message.system(summary_text)
        summary_msg.metadata["needs_summarization"] = True
        summary_msg.metadata["source_messages"] = old

        return [summary_msg] + recent


# ---------------------------------------------------------------------------
# Context Pipeline API (advanced)
# ---------------------------------------------------------------------------

class ContextSource(str, Enum):
    """Types of context sources."""

    SESSION_HISTORY = "session_history"
    SYSTEM_PROMPT = "system_prompt"
    USER_MEMORY = "user_memory"
    PROJECT_INSTRUCTION = "project_instruction"
    FILE_SNIPPET = "file_snippet"
    TOOL_RESULT_SUMMARY = "tool_result_summary"
    EXTERNAL_SEARCH = "external_search"


@dataclass
class ContextChunk:
    """A single piece of context with metadata for pipeline processing."""

    content: str
    source: ContextSource
    priority: int = 100  # lower = higher priority
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4


class ContextSourceProvider(ABC):
    """Abstract provider that gathers context chunks from a source."""

    @property
    @abstractmethod
    def source_type(self) -> ContextSource: ...

    @abstractmethod
    async def gather(self, **kwargs: Any) -> list[ContextChunk]:
        """Gather context chunks from this source."""
        ...


class ContextPipeline:
    """Full context pipeline: gather → rank → dedupe → budget → compress → render.

    Each stage is pluggable. Default implementations are provided for
    ranking (by priority), deduping (by content hash), and budgeting
    (by token estimate).
    """

    def __init__(self, max_tokens: int = 128000, reserve_tokens: int | None = None) -> None:
        self._sources: list[ContextSourceProvider] = []
        self._max_tokens = max_tokens
        self._reserve_tokens = reserve_tokens if reserve_tokens is not None else min(4096, max_tokens // 4)

    def add_source(self, source: ContextSourceProvider) -> None:
        """Register a context source provider."""
        self._sources.append(source)

    async def gather(self, **kwargs: Any) -> list[ContextChunk]:
        """Gather chunks from all sources."""
        chunks: list[ContextChunk] = []
        for source in self._sources:
            try:
                gathered = await source.gather(**kwargs)
                chunks.extend(gathered)
            except Exception:
                pass  # Source failure isolation
        return chunks

    @staticmethod
    def rank(chunks: list[ContextChunk]) -> list[ContextChunk]:
        """Sort chunks by priority (lower number = higher priority)."""
        return sorted(chunks, key=lambda c: c.priority)

    @staticmethod
    def dedupe(chunks: list[ContextChunk]) -> list[ContextChunk]:
        """Remove duplicate chunks based on content hash."""
        seen: set[str] = set()
        result: list[ContextChunk] = []
        for chunk in chunks:
            h = hash(chunk.content)
            if h not in seen:
                seen.add(h)
                result.append(chunk)
        return result

    def budget(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        """Select chunks that fit within the token budget."""
        available = self._max_tokens - self._reserve_tokens
        result: list[ContextChunk] = []
        used = 0
        for chunk in chunks:
            if used + chunk.token_estimate > available:
                break
            result.append(chunk)
            used += chunk.token_estimate
        return result

    async def process(self, **kwargs: Any) -> list[ContextChunk]:
        """Run the full pipeline: gather → rank → dedupe → budget."""
        chunks = await self.gather(**kwargs)
        chunks = self.rank(chunks)
        chunks = self.dedupe(chunks)
        chunks = self.budget(chunks)
        return chunks


class ContextEngine:
    """Manages context strategies and applies them.

    Extensions can register custom strategies (e.g. 'hermes_compression')
    that the config's context.strategy field can reference.

    Args:
        default_strategy: Name of the default compression strategy.
        tokenizer: Optional custom tokenizer for accurate token estimation.
            If not provided, strategies use their built-in len(text)//4 approximation.
    """

    def __init__(self, default_strategy: str = "sliding_window", tokenizer: Any = None) -> None:
        self._strategies: dict[str, ContextStrategy] = {
            "sliding_window": SlidingWindowStrategy(),
            "summarization": SummarizationStrategy(),
        }
        self._default = default_strategy
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        """The custom tokenizer, if one was provided."""
        return self._tokenizer

    def register_strategy(self, strategy: ContextStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get_strategy(self, name: str | None = None) -> ContextStrategy:
        name = name or self._default
        strategy = self._strategies.get(name)
        if not strategy:
            raise ValueError(f"Unknown context strategy: {name}")
        return strategy

    def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        strategy_name: str | None = None,
        system_prompt_tokens: int = 0,
    ) -> list[Message]:
        strategy = self.get_strategy(strategy_name)
        return strategy.compress(messages, max_tokens, system_prompt_tokens)

    def list_strategies(self) -> list[str]:
        return list(self._strategies.keys())

    def on_session_start(self, session: Any, strategy_name: str | None = None) -> None:
        """Forward session start event to the active strategy."""
        self.get_strategy(strategy_name).on_session_start(session)

    def on_turn_end(self, messages: list[Message], response: Any, strategy_name: str | None = None) -> None:
        """Forward turn end event to the active strategy."""
        self.get_strategy(strategy_name).on_turn_end(messages, response)

    def should_compress(self, messages: list[Message], max_tokens: int, strategy_name: str | None = None) -> bool:
        """Check if compression is needed via the active strategy."""
        return self.get_strategy(strategy_name).should_compress(messages, max_tokens)
