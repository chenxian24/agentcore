"""AgentCore — Atomic agent engine.

Layered architecture:
  Atomic layer:  agentcore.*         — Message, Provider, Config, Engine, Stats, Utils
  Mechanism layer: agentcore.<mod>.* — hooks, tools, plugins, context, events, mcp, resilience, ...
  Application layer: extensions/     — Hermes, OpenCode, OpenClaw built from core + mechanisms
"""

from agentcore.agents.manager import SubAgentManager, SubAgentTask, SubAgentResult
from agentcore.config.schema import AgentConfig, ModelConfig, SystemPromptConfig
from agentcore.context.caching import PromptCacheManager
from agentcore.context.engine import ContextEngine
from agentcore.core.adapter import MessageAdapter
from agentcore.core.engine import AgentEngine
from agentcore.core.message import Message, MessageRole
from agentcore.core.providers import MemoryProvider, SkillProvider
from agentcore.core.session import Session
from agentcore.core.session_store import JsonlSessionStore, MemorySessionStore, SessionStore
from agentcore.core.stats import AggregateStats, RequestStats, StatsCollector
from agentcore.delivery.channel import Channel, ChannelMessage, DeliveryManager
from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ThinkingConfig,
    ThinkingLevel,
    ToolCall,
    ToolExecutor,
)
from agentcore.models.capabilities import ProviderCapabilities
from agentcore.models.events import StreamEvent, StreamEventType
from agentcore.models.registry import ModelRegistry
from agentcore.resilience.fallback import FallbackProviderChain
from agentcore.middleware import LoggingMiddleware, MessageTransformMiddleware, Middleware, MiddlewareContext, MiddlewareStack
from agentcore.runtime import AgentRuntime
from agentcore.tokenizer import SimpleTokenizer, TiktokenTokenizer, Tokenizer
from agentcore.tools.pipeline import ToolPipeline
from agentcore.tools.retry import ExponentialBackoffPolicy, FixedDelayPolicy, NoRetryPolicy, RetryPolicy
from agentcore.tools.registry import ToolRegistry
from agentcore.tools.repair import ToolCallRepairer
from agentcore.tools.result import ToolResult
from agentcore.utils import run_loop, with_retry
from agentcore.wire import (
    ChatCompletionsProtocol,
    ResponsesProtocol,
    WireEvent,
    WireProtocol,
    WireResponse,
    WireToolCall,
)

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
    "ThinkingLevel",
    # Provider
    "BaseLLMProvider",
    "ModelRegistry",
    "ProviderCapabilities",
    # Wire Protocol
    "WireProtocol",
    "ChatCompletionsProtocol",
    "ResponsesProtocol",
    "WireResponse",
    "WireToolCall",
    "WireEvent",
    # Streaming
    "StreamEvent",
    "StreamEventType",
    # Config
    "AgentConfig",
    "ModelConfig",
    "SystemPromptConfig",
    # Context
    "ContextEngine",
    "PromptCacheManager",
    # Engine & Runtime
    "AgentEngine",
    "AgentRuntime",
    "MessageAdapter",
    "Session",
    # Session persistence
    "SessionStore",
    "MemorySessionStore",
    "JsonlSessionStore",
    # Skills & Memory
    "SkillProvider",
    "MemoryProvider",
    # Tools
    "ToolPipeline",
    "ToolRegistry",
    "ToolResult",
    "ToolCallRepairer",
    # Sub-agents
    "SubAgentManager",
    "SubAgentTask",
    "SubAgentResult",
    # Resilience
    "FallbackProviderChain",
    # Delivery
    "Channel",
    "ChannelMessage",
    "DeliveryManager",
    # Stats
    "StatsCollector",
    "RequestStats",
    "AggregateStats",
    # Utils
    "run_loop",
    "with_retry",
]
__version__ = "0.7.0"
