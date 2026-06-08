"""Wire protocol abstraction — pluggable serialization for LLM APIs."""

from agentcore.wire.base import WireProtocol
from agentcore.wire.chat_completions import ChatCompletionsProtocol
from agentcore.wire.responses import ResponsesProtocol
from agentcore.wire.types import WireEvent, WireResponse, WireToolCall

__all__ = [
    "WireProtocol",
    "ChatCompletionsProtocol",
    "ResponsesProtocol",
    "WireResponse",
    "WireToolCall",
    "WireEvent",
]
