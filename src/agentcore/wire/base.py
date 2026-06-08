"""WireProtocol — abstract serialization layer between agent and LLM API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from agentcore.wire.types import WireEvent, WireResponse


class WireProtocol(ABC):
    """Pluggable wire protocol for LLM API communication.

    Each protocol handles serialization/deserialization for a specific API format
    (Chat Completions, Responses, Gemini, etc.). The agent loop works with
    normalized WireResponse/WireEvent types and never touches raw wire formats.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol identifier: 'chat_completions', 'responses', etc."""
        ...

    @abstractmethod
    def get_endpoint(self) -> str:
        """API path: '/v1/chat/completions', '/v1/responses', etc."""
        ...

    @abstractmethod
    def build_request_body(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the HTTP request body for this protocol.

        Args:
            messages: Internal message list (role/content/tool_calls format).
            model: Model identifier.
            tools: Tool definitions in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            stream: Whether to request streaming.
            **kwargs: Protocol-specific parameters (thinking, reasoning_effort, etc.).

        Returns:
            JSON-serializable request body dict.
        """
        ...

    @abstractmethod
    def parse_response(self, raw: dict[str, Any]) -> WireResponse:
        """Parse a non-streaming API response into normalized form.

        Args:
            raw: Raw JSON response from the API.

        Returns:
            Normalized WireResponse.
        """
        ...

    @abstractmethod
    def parse_stream_event(self, raw: dict[str, Any]) -> WireEvent | None:
        """Parse a single streaming event into normalized form.

        Args:
            raw: Raw JSON event from the SSE/WebSocket stream.

        Returns:
            Normalized WireEvent, or None if the event should be skipped.
        """
        ...

    @abstractmethod
    def is_stream_done(self, raw: dict[str, Any]) -> bool:
        """Check if this streaming event signals the end of the stream.

        Args:
            raw: Raw JSON event from the stream.

        Returns:
            True if the stream is complete.
        """
        ...

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool definitions to this protocol's format.

        Default: pass through (OpenAI format). Override for protocols
        that use a different tool schema (e.g., Anthropic's input_schema).

        Args:
            tools: Tool definitions in OpenAI format.

        Returns:
            Tool definitions in this protocol's format.
        """
        return tools
