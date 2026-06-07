"""Provider capability declarations.

Providers expose ProviderCapabilities so that AgentRuntime, prompt builders,
and fallback logic can make fine-grained decisions about what to enable.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProviderCapabilities(BaseModel):
    """Fine-grained capability flags for an LLM provider.

    Used by:
    - AgentRuntime: decide whether to enable tools, vision, streaming tools
    - PromptBuilder: inject different tool descriptions based on capabilities
    - Fallback: check if a provider can handle the current request
    """

    chat: bool = True
    streaming: bool = True
    tools: bool = False
    streaming_tools: bool = False
    vision: bool = False
    embeddings: bool = False
    json_object: bool = False
    json_schema: bool = False
    reasoning: list[str] = []  # e.g. ["extended_thinking", "reasoning_effort"]
    image_input_formats: list[str] = []  # e.g. ["png", "jpeg", "webp"]

    def supports(self, capability: str) -> bool:
        """Check if a specific capability is supported."""
        val = getattr(self, capability, None)
        if isinstance(val, bool):
            return val
        if isinstance(val, list):
            return len(val) > 0
        return False
