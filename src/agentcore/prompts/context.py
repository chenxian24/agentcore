"""Context window management for conversations."""

from __future__ import annotations

from typing import Any

from agentcore.core.message import Message, MessageRole


class ContextManager:
    """Manages conversation context within token limits."""

    def __init__(
        self,
        max_tokens: int = 128000,
        strategy: str = "sliding_window",
        reserve_tokens: int = 4096,
    ) -> None:
        self._max_tokens = max_tokens
        self._strategy = strategy
        self._reserve_tokens = reserve_tokens
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def available_tokens(self) -> int:
        return self._max_tokens - self._reserve_tokens

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _total_tokens(self) -> int:
        return sum(self._estimate_tokens(m.content) for m in self._messages)

    def get_context_messages(self, system_prompt: str = "") -> list[Message]:
        messages = list(self._messages)
        if self._strategy == "sliding_window":
            messages = self._apply_sliding_window(messages, system_prompt)
        return messages

    def _apply_sliding_window(
        self, messages: list[Message], system_prompt: str
    ) -> list[Message]:
        system_tokens = self._estimate_tokens(system_prompt) if system_prompt else 0
        available = self.available_tokens - system_tokens

        result: list[Message] = []
        used_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self._estimate_tokens(msg.content)
            if used_tokens + msg_tokens > available:
                break
            result.insert(0, msg)
            used_tokens += msg_tokens

        return result

    def get_history_text(self) -> str:
        lines = []
        for msg in self._messages:
            lines.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(lines)
