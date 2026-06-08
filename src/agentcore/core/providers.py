"""Provider interfaces for skills and memory — extension points for agent capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SkillProvider(ABC):
    """Provides skills (knowledge packs) to the agent.

    Skills are markdown-based instructions that augment the system prompt
    with domain-specific knowledge. Implementations load skills from
    files, databases, or remote sources.
    """

    @abstractmethod
    def list_skills(self) -> list[dict[str, Any]]:
        """List all available skills.

        Returns:
            List of dicts with at least 'name' and 'description' keys.
        """

    @abstractmethod
    def get_skill(self, name: str) -> str | None:
        """Get skill content by name.

        Args:
            name: Skill identifier.

        Returns:
            Skill content (markdown), or None if not found.
        """

    @abstractmethod
    def format_for_system_prompt(self) -> str:
        """Format all active skills for injection into the system prompt.

        Returns:
            Formatted string listing available skills and their content.
        """

    def match_skill(self, user_input: str) -> str | None:
        """Match user input to a skill (e.g., '/skill-name').

        Args:
            user_input: Raw user input.

        Returns:
            Skill name if matched, None otherwise.
        """
        if user_input.startswith("/"):
            name = user_input.split()[0][1:]  # Remove leading '/'
            if self.get_skill(name) is not None:
                return name
        return None


class MemoryProvider(ABC):
    """Provides persistent memory to the agent.

    Memory providers store and recall information across sessions.
    They inject relevant context into the system prompt and can
    be triggered to store new information after each turn.
    """

    @abstractmethod
    async def get_context(self) -> str:
        """Get memory context for injection into the system prompt.

        Returns:
            Formatted memory context string.
        """

    @abstractmethod
    async def store(self, key: str, value: str) -> None:
        """Store a memory entry.

        Args:
            key: Memory key/identifier.
            value: Memory content.
        """

    @abstractmethod
    async def recall(self, query: str, limit: int = 5) -> list[str]:
        """Recall memories relevant to a query.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of relevant memory entries.
        """

    async def on_turn_end(self, messages: list[dict[str, Any]]) -> None:
        """Hook called after each agent turn. Override to auto-extract memories."""
