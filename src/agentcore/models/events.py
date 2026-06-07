"""Unified stream event model for agent streaming.

Provides a provider-agnostic event interface for streaming responses.
Providers emit StreamEvents; consumers (runners, UIs) subscribe to them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.models.base import ToolCall


class StreamEventType(str, Enum):
    """Types of stream events."""

    MESSAGE_START = "message_start"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE_DONE = "message_done"
    ERROR = "error"


class StreamEvent(BaseModel):
    """A single event in a streaming response.

    Providers emit these during stream_chat(). Consumers filter by type
    to handle text display, tool execution, reasoning visualization, etc.
    """

    type: StreamEventType
    text: str = ""
    tool_call: ToolCall | None = None
    tool_name: str = ""
    tool_call_id: str = ""
    tool_result: dict[str, Any] | None = None
    thinking: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = ""
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text(cls, delta: str) -> StreamEvent:
        return cls(type=StreamEventType.TEXT_DELTA, text=delta)

    @classmethod
    def reasoning(cls, delta: str) -> StreamEvent:
        return cls(type=StreamEventType.REASONING_DELTA, thinking=delta)

    @classmethod
    def tool_call_start(cls, tool_call: ToolCall) -> StreamEvent:
        return cls(
            type=StreamEventType.TOOL_CALL_DELTA,
            tool_call=tool_call,
            tool_name=tool_call.function.name,
        )

    @classmethod
    def tool_call_done(cls, tool_call: ToolCall) -> StreamEvent:
        return cls(
            type=StreamEventType.TOOL_CALL,
            tool_call=tool_call,
            tool_name=tool_call.function.name,
        )

    @classmethod
    def tool_result_done(cls, tool_call_id: str, result: dict[str, Any]) -> StreamEvent:
        return cls(
            type=StreamEventType.TOOL_RESULT,
            tool_call_id=tool_call_id,
            tool_result=result,
        )

    @classmethod
    def message_start(cls) -> StreamEvent:
        return cls(type=StreamEventType.MESSAGE_START)

    @classmethod
    def message_done(
        cls,
        content: str = "",
        finish_reason: str = "",
        usage: dict[str, int] | None = None,
    ) -> StreamEvent:
        return cls(
            type=StreamEventType.MESSAGE_DONE,
            text=content,
            finish_reason=finish_reason,
            usage=usage or {},
        )

    @classmethod
    def error(cls, error: str) -> StreamEvent:
        return cls(type=StreamEventType.ERROR, error=error)
