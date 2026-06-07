"""Model provider registry."""

from __future__ import annotations

from typing import Any

from agentcore.config.schema import ModelConfig, ModelProvider
from agentcore.models.anthropic import AnthropicProvider
from agentcore.models.base import BaseLLMProvider
from agentcore.models.local import LocalProvider
from agentcore.models.openai import OpenAIProvider


class ModelRegistry:
    """Registry for LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in provider factories."""
        self._factories: dict[str, type[BaseLLMProvider]] = {
            ModelProvider.OPENAI: OpenAIProvider,
            ModelProvider.ANTHROPIC: AnthropicProvider,
            ModelProvider.LOCAL: LocalProvider,
        }

    def register(self, name: str, provider: BaseLLMProvider) -> None:
        """Register a provider instance."""
        self._providers[name] = provider

    def register_factory(self, name: str, factory: type[BaseLLMProvider]) -> None:
        """Register a provider factory."""
        self._factories[name] = factory

    def get(self, name: str) -> BaseLLMProvider | None:
        """Get a registered provider by name."""
        return self._providers.get(name)

    def create_from_config(self, config: ModelConfig) -> BaseLLMProvider:
        """Create and register a provider from config.

        Provider name can be a ModelProvider enum value or any string
        matching a previously registered factory.
        """
        provider_name = config.provider.value if isinstance(config.provider, ModelProvider) else config.provider

        factory = self._factories.get(provider_name)
        if not factory:
            raise ValueError(f"Unknown provider: {provider_name}")

        kwargs: dict[str, Any] = {"default_model": config.model}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.timeout:
            kwargs["timeout"] = config.timeout

        provider = factory(**kwargs)  # type: ignore[call-arg]
        self._providers[provider_name] = provider
        return provider

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
