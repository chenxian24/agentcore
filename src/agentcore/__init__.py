"""AgentCore — Atomic agent engine.

Layered architecture:
  Atomic layer:  agentcore.*         — Message, Provider, Config, Engine, Stats, Utils
  Mechanism layer: agentcore.<mod>.* — hooks, tools, plugins, context, events, mcp, resilience, ...
  Application layer: extensions/     — Hermes, OpenCode, OpenClaw built from core + mechanisms
"""

from agentcore.config.schema import AgentConfig, ModelConfig, SystemPromptConfig
from agentcore.core.adapter import MessageAdapter
from agentcore.core.engine import AgentEngine
from agentcore.core.message import Message, MessageRole
from agentcore.core.session import Session
from agentcore.core.stats import AggregateStats, RequestStats, StatsCollector
from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ThinkingConfig,
    ToolCall,
    ToolExecutor,
)
from agentcore.models.capabilities import ProviderCapabilities
from agentcore.models.events import StreamEvent, StreamEventType
from agentcore.models.registry import ModelRegistry
from agentcore.runtime import AgentRuntime
from agentcore.tools.pipeline import ToolPipeline
from agentcore.tools.result import ToolResult
from agentcore.utils import run_loop, with_retry

__all__ = [
    # Types
    "Message",
    "MessageRole",
    "LLMMessage",
    "LLMResponse",
    "ChatParams",
    "ToolCall",
    "ToolExecutor",
    "ThinkingConfig",
    # Provider
    "BaseLLMProvider",
    "ModelRegistry",
    "ProviderCapabilities",
    # Streaming
    "StreamEvent",
    "StreamEventType",
    # Config
    "AgentConfig",
    "ModelConfig",
    "SystemPromptConfig",
    # Engine & Runtime
    "AgentEngine",
    "AgentRuntime",
    "MessageAdapter",
    "Session",
    # Tools
    "ToolPipeline",
    "ToolResult",
    # Stats
    "StatsCollector",
    "RequestStats",
    "AggregateStats",
    # Utils
    "run_loop",
    "with_retry",
]
__version__ = "0.6.0"
