"""Session management for agent conversations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.config.schema import AgentConfig
from agentcore.core.message import Message, MessageRole


class Session(BaseModel):
    """A conversation session — a message list with metadata.

    Context compression and prompt building are external concerns,
    handled by mechanism-layer modules before calling the provider.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    config: AgentConfig = Field(default_factory=AgentConfig)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_context_messages(self, max_messages: int = 100) -> list[Message]:
        """Return recent messages. For context compression, use agentcore.context externally."""
        return self.messages[-max_messages:]

    def get_history(self) -> list[dict[str, Any]]:
        return [m.to_llm_message() for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now(timezone.utc)

    def get_message_count(self) -> int:
        return len(self.messages)

    def get_last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None
