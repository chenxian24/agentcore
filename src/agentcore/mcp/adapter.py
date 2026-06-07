"""MCP Tool Adapter — bridges MCP tools into AgentCore's ToolRegistry.

Converts MCP tool definitions to AgentCore format and wraps MCP tool
calls as ToolRegistry-compatible handlers.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from agentcore.mcp.protocol import MCPProtocolClient
from agentcore.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Bridges MCP server tools into an AgentCore ToolRegistry.

    Usage:
        adapter = MCPToolAdapter(protocol_client, "my-server")
        await adapter.register_tools(tool_registry)
        # Now "my-server__tool_name" is available in the registry
    """

    def __init__(
        self,
        protocol: MCPProtocolClient,
        server_name: str,
        prefix: str = "",
    ) -> None:
        self._protocol = protocol
        self._server_name = server_name
        self._prefix = prefix or server_name
        self._registered_names: list[str] = []

    @property
    def registered_names(self) -> list[str]:
        return list(self._registered_names)

    async def register_tools(self, registry: ToolRegistry) -> int:
        """Discover MCP tools and register them in the AgentCore registry.

        Returns the number of tools registered.
        Tools are prefixed with "{server_name}__" to avoid name collisions.
        """
        mcp_tools = await self._protocol.list_tools()
        count = 0

        for tool_def in mcp_tools:
            mcp_name = tool_def.get("name", "")
            if not mcp_name:
                continue

            # Prefix to avoid collisions
            agentcore_name = f"{self._prefix}__{mcp_name}"
            description = tool_def.get("description", f"MCP tool: {mcp_name}")
            parameters = tool_def.get("inputSchema", {"type": "object", "properties": {}})

            # Create a closure that captures the MCP tool name
            def _make_handler(tool_name: str):
                async def handler(**kwargs: Any) -> dict[str, Any]:
                    return await self._protocol.call_tool(tool_name, kwargs)
                return handler

            registry.register(
                name=agentcore_name,
                handler=_make_handler(mcp_name),
                description=description,
                parameters=parameters,
                metadata={"mcp_server": self._server_name, "mcp_tool": mcp_name},
            )
            self._registered_names.append(agentcore_name)
            count += 1

        logger.info("MCP adapter '%s': registered %d tools", self._server_name, count)
        return count

    def unregister_tools(self, registry: ToolRegistry) -> None:
        """Remove all tools registered by this adapter."""
        for name in self._registered_names:
            registry.unregister(name)
        self._registered_names.clear()
