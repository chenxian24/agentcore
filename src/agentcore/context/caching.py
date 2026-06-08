"""Prompt caching — inject cache control markers for supported providers.

Anthropic supports prompt caching via cache_control breakpoints on messages.
This reduces latency and cost for long system prompts and conversation histories.
"""

from __future__ import annotations

from typing import Any


class PromptCacheManager:
    """Manages prompt caching markers for supported providers."""

    def apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        provider_name: str,
    ) -> list[dict[str, Any]]:
        """Apply cache control breakpoints to messages.

        Anthropic: adds cache_control to system message and last 3 user/assistant messages.
        Other providers: returns messages unchanged.

        Args:
            messages: LLM messages in OpenAI format.
            provider_name: Provider identifier ('anthropic', 'openai', etc.).

        Returns:
            Messages with cache_control markers (if applicable).
        """
        if provider_name.lower() not in ("anthropic", "claude"):
            return messages

        result = [self._copy_msg(m) for m in messages]

        # Cache the system message (first message if role=system)
        if result and result[0].get("role") == "system":
            result[0]["cache_control"] = {"type": "ephemeral"}

        # Cache the last 3 non-system messages
        non_system_indices = [i for i, m in enumerate(result) if m.get("role") != "system"]
        for idx in non_system_indices[-3:]:
            result[idx]["cache_control"] = {"type": "ephemeral"}

        return result

    def _copy_msg(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Shallow copy a message dict."""
        return dict(msg)
