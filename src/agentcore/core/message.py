"""Message models for agent conversations."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.models.base import ToolCall


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A conversation message."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    role: MessageRole
    content: str
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_llm_message(self) -> dict[str, Any]:
        """Convert to LLM provider format."""
        msg: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return msg

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def assistant_with_tools(cls, content: str, tool_calls: list[ToolCall]) -> Message:
        """Create an assistant message that includes tool calls."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str = "", name: str = "") -> Message:
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
