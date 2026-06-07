"""MCP manager: manages multiple MCP server connections and tool routing."""

from __future__ import annotations

import logging
from typing import Any

from agentcore.mcp.client import MCPClient
from agentcore.models.base import ToolCall

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages connections to multiple MCP servers.

    Provides unified tool discovery across all servers and routes
    tool calls to the correct server.

    Usage:
        manager = MCPManager()
        await manager.add_server("filesystem", command="npx", args=["-y", "@mcp/fs", "/path"])
        await manager.add_server("github", host="127.0.0.1", port=3001)
        tools = manager.get_all_tools()  # merged from all servers
        result = await manager.call_tool(ToolCall(...))
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name

    async def add_server_stdio(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Add and connect to an MCP server via stdio."""
        client = MCPClient(name)
        await client.connect_stdio(command, args, cwd)
        self._clients[name] = client
        self._register_tools(name, client.tools)

    async def add_server_tcp(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        """Add and connect to an MCP server via TCP."""
        client = MCPClient(name)
        await client.connect_tcp(host, port)
        self._clients[name] = client
        self._register_tools(name, client.tools)

    async def remove_server(self, name: str) -> None:
        """Disconnect and remove a server."""
        client = self._clients.pop(name, None)
        if client:
            await client.disconnect()
            # Remove tool mappings
            self._tool_to_server = {
                t: s for t, s in self._tool_to_server.items() if s != name
            }

    def _register_tools(self, server_name: str, tools: list[dict[str, Any]]) -> None:
        """Register tools from a server into the routing table."""
        for tool_def in tools:
            tool_name = tool_def.get("name", "")
            if tool_name:
                prefixed = f"mcp_{server_name}_{tool_name}"
                self._tool_to_server[prefixed] = server_name
                self._tool_to_server[tool_name] = server_name  # also register unprefixed

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions from all connected servers (OpenAI format)."""
        all_tools = []
        for name, client in self._clients.items():
            for tool_def in client.tools:
                prefixed_name = f"mcp_{name}_{tool_def['name']}"
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": prefixed_name,
                        "description": tool_def.get("description", ""),
                        "parameters": tool_def.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                })
        return all_tools

    def get_server_tools(self, server_name: str) -> list[dict[str, Any]]:
        """Get tool definitions from a specific server."""
        client = self._clients.get(server_name)
        if not client:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": f"mcp_{server_name}_{t['name']}",
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in client.tools
        ]

    async def call_tool(self, tool_call: ToolCall) -> dict[str, Any]:
        """Route a tool call to the correct MCP server."""
        tool_name = tool_call.function.name

        # Try prefixed name first: mcp_servername_toolname
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return {"error": f"MCP tool '{tool_name}' not found in any server"}

        client = self._clients.get(server_name)
        if not client or not client.connected:
            return {"error": f"MCP server '{server_name}' not connected"}

        # Extract actual tool name (remove mcp_ prefix and server name)
        actual_tool = tool_name
        prefix = f"mcp_{server_name}_"
        if tool_name.startswith(prefix):
            actual_tool = tool_name[len(prefix):]

        # Parse arguments
        import json
        args = tool_call.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        return await client.call_tool(actual_tool, args)

    def get_client(self, name: str) -> MCPClient | None:
        return self._clients.get(name)

    def list_servers(self) -> list[str]:
        return list(self._clients.keys())

    @property
    def connected_count(self) -> int:
        return sum(1 for c in self._clients.values() if c.connected)

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception:
                logger.error("Error disconnecting MCP '%s'", client.name, exc_info=True)
        self._clients.clear()
        self._tool_to_server.clear()
