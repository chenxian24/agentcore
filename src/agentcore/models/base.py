"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content blocks for multimodal messages
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    """Text content block."""

    type: str = "text"
    text: str


class ImageURL(BaseModel):
    """Image URL descriptor."""

    url: str  # data:image/... base64 or https://
    detail: str = "auto"  # auto | low | high


class ImageBlock(BaseModel):
    """Image content block."""

    type: str = "image_url"
    image_url: ImageURL


class ToolCallFunction(BaseModel):
    """Function call inside a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool/function call from the model."""

    id: str = ""
    type: str = "function"
    function: ToolCallFunction = Field(default_factory=lambda: ToolCallFunction(name="", arguments=""))


class ToolResultBlock(BaseModel):
    """Tool result content block (for tool role messages)."""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""


# Union of all content block types
ContentBlock = Union[TextBlock, ImageBlock, ToolResultBlock]


# ---------------------------------------------------------------------------
# Thinking / reasoning configuration
# ---------------------------------------------------------------------------

class ThinkingConfig(BaseModel):
    """Thinking/reasoning mode configuration.

    Maps to:
    - Anthropic: thinking = {"type": "enabled", "budget_tokens": N}
    - OpenAI o-series: reasoning_effort
    - DeepSeek: enable_deep_think
    """

    enabled: bool = False
    budget_tokens: int = 10000
    type: str = "enabled"  # "enabled" | "disabled"


# ---------------------------------------------------------------------------
# Unified chat request parameters
# ---------------------------------------------------------------------------

class ChatParams(BaseModel):
    """Unified chat request parameters aligned with mainstream API standards.

    All fields are optional with sensible defaults. Providers map these to
    their native API format, silently ignoring unsupported fields.
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    response_format: dict[str, str] | None = None  # {"type": "json_object"}
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Union[str, dict[str, Any]] = ""  # "auto" | "none" | "required" | {"type":"function",...}
    thinking: ThinkingConfig | None = None

    @classmethod
    def from_legacy(
        cls,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatParams:
        """Create ChatParams from legacy positional arguments."""
        return cls(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Message and response models
# ---------------------------------------------------------------------------

class LLMMessage(BaseModel):
    """A message in LLM format.

    Supports both simple text and multimodal content blocks.
    For tool role messages, set tool_call_id and put result in content.
    """

    role: str  # system | user | assistant | tool
    content: Union[str, list[ContentBlock]] = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str = ""
    thinking: str = ""  # thinking/reasoning content (Claude extended thinking, etc.)
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens, reasoning_tokens
    finish_reason: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool execution protocol and streaming events
# ---------------------------------------------------------------------------


class ToolExecutor(Protocol):
    """Protocol for tool execution.

    Any object with an async `execute(ToolCall) -> dict` method satisfies
    this protocol. The engine uses it to run tool calls requested by the
    model without caring about the underlying implementation.

    Return value convention: {"success": bool, "output": ..., "error": str}
    """

    async def execute(self, tool_call: ToolCall) -> dict[str, Any]: ...


@dataclass
class ToolEvent:
    """Structured event emitted during tool-augmented streaming."""

    type: str  # "tool_start" | "tool_end"
    tool_name: str = ""
    arguments: str = ""
    result: dict[str, Any] | None = None
    tool_call_id: str = ""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Supports both the legacy positional-argument interface and the new
    ChatParams-based interface. Subclasses should implement the ChatParams
    variant; the legacy signatures have default implementations that
    delegate to the new ones.
    """

    # --- capability flags (override in subclasses) ---

    @property
    def supports_thinking(self) -> bool:
        return False

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def supports_vision(self) -> bool:
        return False

    # --- primary interface (ChatParams) ---

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response.

        Yields text chunks. Thinking content and tool calls are available
        via the response metadata after the stream completes.
        """
        ...

    @abstractmethod
    async def embed(self, texts: list[str], model: str = "") -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...
