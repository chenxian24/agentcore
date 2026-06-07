"""MessageAdapter — unified conversion between Message and LLMMessage.

Single source of truth for message format conversion.
Runners should not define their own _message_to_llm helpers.
"""

from __future__ import annotations

from typing import Any

from agentcore.core.message import Message, MessageRole
from agentcore.models.base import LLMMessage, LLMResponse, ToolCall, ToolCallFunction


class MessageAdapter:
    """Converts between domain Message and provider LLMMessage formats."""

    @staticmethod
    def to_llm(message: Message) -> LLMMessage:
        """Convert a domain Message to an LLMMessage for provider consumption."""
        role = message.role.value if isinstance(message.role, MessageRole) else str(message.role)

        # Build content
        content = message.content or ""

        # Build the LLM message
        llm = LLMMessage(role=role, content=content)

        # Preserve tool call metadata
        if message.tool_call_id:
            llm.tool_call_id = message.tool_call_id
        if message.tool_calls:
            llm.tool_calls = message.tool_calls
        if message.name:
            llm.name = message.name

        return llm

    @staticmethod
    def to_llm_list(messages: list[Message]) -> list[LLMMessage]:
        """Convert a list of domain Messages to LLMMessages."""
        return [MessageAdapter.to_llm(m) for m in messages]

    @staticmethod
    def from_response(response: LLMResponse, role: str = "assistant") -> Message:
        """Create a domain Message from an LLMResponse."""
        msg = Message(
            role=MessageRole.ASSISTANT if role == "assistant" else MessageRole(role),
            content=response.content or "",
        )
        if response.tool_calls:
            msg.tool_calls = response.tool_calls
        return msg

    @staticmethod
    def from_tool_result(tool_call_id: str, output: str, name: str = "") -> Message:
        """Create a tool-role Message from a tool execution result."""
        return Message(
            role=MessageRole.TOOL,
            content=output,
            tool_call_id=tool_call_id,
            name=name,
        )
