"""Hook system types: hook names, context, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentcore.models.base import ChatParams, LLMMessage, LLMResponse, ToolCall


class HookName(str, Enum):
    """All hook points in the agent lifecycle.

    HookManager also accepts arbitrary string hook names for plugin-defined hooks.
    These are the built-in constants for common lifecycle points.
    """

    # Engine lifecycle
    ENGINE_INIT = "engine_init"
    ENGINE_SHUTDOWN = "engine_shutdown"

    # LLM call lifecycle
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"

    # Tool lifecycle
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"

    # Context / message building
    PRE_BUILD_MESSAGES = "pre_build_messages"
    POST_BUILD_MESSAGES = "post_build_messages"

    # Session lifecycle
    SESSION_CREATED = "session_created"
    SESSION_MESSAGE_ADDED = "session_message_added"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_RESET = "session_reset"

    # Transform hooks — allow plugins to modify outputs
    TRANSFORM_TOOL_RESULT = "transform_tool_result"
    TRANSFORM_LLM_OUTPUT = "transform_llm_output"
    TRANSFORM_TERMINAL_OUTPUT = "transform_terminal_output"

    # Gateway / dispatch
    PRE_GATEWAY_DISPATCH = "pre_gateway_dispatch"

    # Sub-agent lifecycle
    SUBAGENT_STOP = "subagent_stop"

    # Approval workflow
    PRE_APPROVAL_REQUEST = "pre_approval_request"
    POST_APPROVAL_RESPONSE = "post_approval_response"


@dataclass
class HookContext:
    """Mutable context passed through hook chain. Hooks modify this in-place."""

    name: HookName | str
    engine: Any = None
    session: Any = None
    messages: list[LLMMessage] = field(default_factory=list)
    params: ChatParams | None = None
    tool_call: ToolCall | None = None
    tool_result: dict[str, Any] = field(default_factory=dict)
    response: LLMResponse | None = None
    user_input: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel: bool = False
    skip: bool = False
    abort_loop: bool = False  # Set True to abort entire tool loop
    transform_result: str | None = None  # Used by TRANSFORM_* hooks to return modified output

    # Mechanism instances — set by application layer when dispatching
    hooks: Any = None
    tools: Any = None
    context: Any = None
    events: Any = None

    @property
    def hook_name(self) -> str:
        """Get the hook name as a string, regardless of whether it's HookName or str."""
        if isinstance(self.name, HookName):
            return self.name.value
        return self.name


@dataclass
class HookResult:
    """Result returned by a hook handler."""

    success: bool = True
    data: Any = None
    error: str = ""
