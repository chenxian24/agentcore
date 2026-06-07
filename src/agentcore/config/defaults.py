"""Default configuration values for AgentCore."""

from agentcore.config.schema import (
    AgentConfig,
    ContextConfig,
    ModelConfig,
    SystemPromptConfig,
)

DEFAULT_MODEL_CONFIG = ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096,
)

DEFAULT_SYSTEM_PROMPT = SystemPromptConfig(
    template="You are a helpful assistant.",
    variables={},
)

DEFAULT_CONTEXT_CONFIG = ContextConfig(
    max_tokens=128000,
    strategy="sliding_window",
)

DEFAULT_CONFIG = AgentConfig(
    name="default",
    description="Default agent configuration",
    model=DEFAULT_MODEL_CONFIG,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    context=DEFAULT_CONTEXT_CONFIG,
    metadata={},
)
