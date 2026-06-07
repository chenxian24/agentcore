"""Configuration schemas for AgentCore."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Built-in provider identifiers.

    ModelConfig.provider accepts any string — these are convenience constants.
    Plugins can register additional providers via PluginContext.register_provider().
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ThinkingConfig(BaseModel):
    """Thinking/reasoning mode configuration."""

    enabled: bool = False
    budget_tokens: int = 10000
    type: str = "enabled"


class ModelConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str = ModelProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 60.0
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    response_format: dict[str, str] | None = None
    thinking: ThinkingConfig | None = None


class SystemPromptConfig(BaseModel):
    """System prompt configuration."""

    template: str = "You are a helpful assistant."
    variables: dict[str, Any] = Field(default_factory=dict)


class ContextConfig(BaseModel):
    """Context management configuration."""

    max_tokens: int = 128000
    strategy: str = "sliding_window"  # sliding_window | summarization
    summary_prompt: str = "Summarize the following conversation concisely:"


class RuntimeConfig(BaseModel):
    """Agent runtime behavior configuration."""

    max_tool_rounds: int = 20
    max_subtask_agents: int = 5


class PluginConfig(BaseModel):
    """Configuration for a plugin."""

    name: str
    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Main agent configuration."""

    name: str = "default"
    description: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    system_prompt: SystemPromptConfig = Field(default_factory=SystemPromptConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    plugins: list[PluginConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
