"""General-purpose tool registry. Accepts callable handlers and ToolExecutor objects.

The registry itself implements the ToolExecutor protocol, so it can be passed
directly to AgentEngine.chat(tool_executor=registry).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agentcore.models.base import ToolCall, ToolExecutor

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
AnyToolHandler = ToolHandler | ToolExecutor


@dataclass
class ToolEntry:
    """A registered tool."""

    name: str
    handler: AnyToolHandler
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    check_fn: Callable[[ToolCall], Awaitable[bool]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for tools that the agent can call.

    Supports two handler types:
    - Callable handlers: async def(args: dict) -> dict  (most common)
    - ToolExecutor objects: anything with async execute(ToolCall) -> dict

    The registry also implements ToolExecutor itself, so it can be passed
    directly to AgentEngine.chat() as the tool_executor argument.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        handler: AnyToolHandler,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        check_fn: Callable[[ToolCall], Awaitable[bool]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool handler."""
        self._tools[name] = ToolEntry(
            name=name,
            handler=handler,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            check_fn=check_fn,
            metadata=metadata or {},
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[ToolEntry]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for all registered tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters,
                },
            }
            for entry in self._tools.values()
        ]

    async def execute(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute a tool call by looking up the handler in the registry.

        Implements the ToolExecutor protocol so the registry can be passed
        directly to AgentEngine.chat(tool_executor=registry).
        """
        name = tool_call.function.name
        entry = self._tools.get(name)
        if not entry:
            return {"success": False, "output": None, "error": f"Tool '{name}' not found"}

        if entry.check_fn:
            try:
                allowed = await entry.check_fn(tool_call)
                if not allowed:
                    return {"success": False, "output": None, "error": f"Tool '{name}' check failed"}
            except Exception as e:
                return {"success": False, "output": None, "error": f"Tool '{name}' check error: {e}"}

        try:
            args = json.loads(tool_call.function.arguments) if isinstance(
                tool_call.function.arguments, str
            ) else tool_call.function.arguments
        except (json.JSONDecodeError, TypeError):
            args = {}

        try:
            if hasattr(entry.handler, 'execute') and not callable(entry.handler):
                return await entry.handler.execute(tool_call)
            elif callable(entry.handler):
                if isinstance(args, dict):
                    result = await entry.handler(**args)
                else:
                    result = await entry.handler(args)
                if isinstance(result, dict):
                    if "success" in result:
                        return result
                    if "output" in result or "error" in result:
                        return {"success": "error" not in result or not result.get("error"), "output": result.get("output"), "error": result.get("error", "")}
                return {"success": True, "output": result, "error": ""}
            else:
                return {"success": False, "output": None, "error": f"Invalid handler type for '{name}'"}
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", name, e)
            return {"success": False, "output": None, "error": str(e)}
