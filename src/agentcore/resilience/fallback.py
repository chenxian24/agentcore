"""Fallback provider chain: try primary, then fallback providers on failure."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentcore.models.base import BaseLLMProvider, ChatParams, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class FallbackEntry:
    """A fallback provider with its configuration."""

    name: str
    provider: BaseLLMProvider
    params_override: dict[str, Any] = field(default_factory=dict)
    active: bool = True


class FallbackProviderChain:
    """Manages a chain of fallback providers.

    When the primary provider fails, the chain tries each fallback in order.
    Extensions can register fallback providers (e.g. different API keys,
    different models, different providers entirely).

    Usage:
        chain = FallbackProviderChain(primary=primary_provider)
        chain.add("backup-openai", backup_provider)
        chain.add("local-ollama", local_provider)
        response = await chain.chat(messages, params)
    """

    def __init__(self, primary: BaseLLMProvider | None = None) -> None:
        self._primary = primary
        self._fallbacks: list[FallbackEntry] = []
        self._current_index: int = -1  # -1 = primary

    @property
    def primary(self) -> BaseLLMProvider | None:
        return self._primary

    def set_primary(self, provider: BaseLLMProvider) -> None:
        self._primary = provider
        self._current_index = -1

    def add(
        self,
        name: str,
        provider: BaseLLMProvider,
        params_override: dict[str, Any] | None = None,
    ) -> None:
        self._fallbacks.append(FallbackEntry(
            name=name,
            provider=provider,
            params_override=params_override or {},
        ))

    def remove(self, name: str) -> None:
        self._fallbacks = [f for f in self._fallbacks if f.name != name]

    def get_current_provider(self) -> BaseLLMProvider | None:
        """Get the currently active provider (primary or fallback)."""
        if self._current_index == -1:
            return self._primary
        if 0 <= self._current_index < len(self._fallbacks):
            return self._fallbacks[self._current_index].provider
        return self._primary

    def get_current_name(self) -> str:
        if self._current_index == -1:
            return "primary"
        if 0 <= self._current_index < len(self._fallbacks):
            return self._fallbacks[self._current_index].name
        return "primary"

    def activate_next_fallback(self) -> BaseLLMProvider | None:
        """Switch to the next fallback provider. Returns None if exhausted."""
        next_index = self._current_index + 1
        if next_index < len(self._fallbacks):
            entry = self._fallbacks[next_index]
            if entry.active:
                self._current_index = next_index
                logger.info("Switched to fallback provider '%s'", entry.name)
                return entry.provider
        return None

    def reset_to_primary(self) -> None:
        """Reset back to the primary provider."""
        self._current_index = -1

    def apply_params_override(self, params: ChatParams) -> ChatParams:
        """Apply fallback-specific parameter overrides."""
        if self._current_index >= 0 and self._current_index < len(self._fallbacks):
            override = self._fallbacks[self._current_index].params_override
            if override:
                # Create a new params dict with overrides
                data = params.model_dump()
                data.update(override)
                return ChatParams(**data)
        return params

    async def chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams,
        error_classifier: Any = None,
    ) -> LLMResponse:
        """Try primary, then each fallback, until one succeeds.

        This is a convenience method; the engine's _run_tool_loop
        uses FallbackProviderChain.activate_next_fallback() directly.
        """
        provider = self.get_current_provider()
        if not provider:
            raise RuntimeError("No provider available")

        effective_params = self.apply_params_override(params)
        return await provider.chat(messages=messages, params=effective_params)

    def list_providers(self) -> list[dict[str, Any]]:
        """List all providers in the chain."""
        result = []
        if self._primary:
            result.append({"name": "primary", "active": True, "current": self._current_index == -1})
        for i, entry in enumerate(self._fallbacks):
            result.append({
                "name": entry.name,
                "active": entry.active,
                "current": self._current_index == i,
            })
        return result
