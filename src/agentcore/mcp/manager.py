"""MCP manager: manages multiple MCP server connections and tool routing."""

from __future__ import annotations

import logging
from typing import Any

from agentcore.mcp.protocol import MCPProtocolClient
from agentcore.mcp.transport import MCPTransport, StdioTransport, TcpTransport
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
        self._clients: dict[str, MCPProtocolClient] = {}
        self._transports: dict[str, MCPTransport] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}  # server_name -> tool defs
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name

    async def add_server_stdio(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Add and connect to an MCP server via stdio."""
        transport = StdioTransport(command, args, cwd)
        await transport.connect()
        client = MCPProtocolClient(transport)
        await client.initialize()
        tools = await client.list_tools()

        self._transports[name] = transport
        self._clients[name] = client
        self._tools[name] = tools
        self._register_tools(name, tools)
        logger.info("MCP '%s' connected (stdio): %d tools", name, len(tools))

    async def add_server_tcp(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        """Add and connect to an MCP server via TCP."""
        transport = TcpTransport(host, port)
        await transport.connect()
        client = MCPProtocolClient(transport)
        await client.initialize()
        tools = await client.list_tools()

        self._transports[name] = transport
        self._clients[name] = client
        self._tools[name] = tools
        self._register_tools(name, tools)
        logger.info("MCP '%s' connected (tcp): %d tools", name, len(tools))

    async def add_server(
        self,
        name: str,
        transport: MCPTransport,
    ) -> None:
        """Add and connect to an MCP server with a pre-built transport."""
        await transport.connect()
        client = MCPProtocolClient(transport)
        await client.initialize()
        tools = await client.list_tools()

        self._transports[name] = transport
        self._clients[name] = client
        self._tools[name] = tools
        self._register_tools(name, tools)
        logger.info("MCP '%s' connected: %d tools", name, len(tools))

    async def load_from_config(self, mcp_servers: dict[str, Any]) -> None:
        """Load MCP servers from configuration dict.

        Config format:
            mcp_servers:
              filesystem:
                command: npx
                args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
              github:
                command: npx
                args: ["-y", "@modelcontextprotocol/server-github"]
                env:
                  GITHUB_TOKEN: "${GITHUB_TOKEN}"
              custom:
                host: localhost
                port: 3000
        """
        for name, server_cfg in mcp_servers.items():
            if not server_cfg:
                continue
            try:
                if hasattr(server_cfg, "model_dump"):
                    cfg = server_cfg.model_dump()
                else:
                    cfg = dict(server_cfg)

                command = cfg.get("command", "")
                host = cfg.get("host", "")
                port = cfg.get("port", 0)

                if command:
                    await self.add_server_stdio(
                        name=name,
                        command=command,
                        args=cfg.get("args"),
                        cwd=cfg.get("cwd") or None,
                    )
                elif host and port:
                    await self.add_server_tcp(
                        name=name,
                        host=host,
                        port=port,
                    )
                else:
                    logger.warning("MCP server '%s': missing command or host+port", name)
            except Exception:
                logger.error("Failed to connect MCP server '%s'", name, exc_info=True)

    async def remove_server(self, name: str) -> None:
        """Disconnect and remove a server."""
        transport = self._transports.pop(name, None)
        self._clients.pop(name, None)
        self._tools.pop(name, None)
        if transport:
            await transport.close()
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
                self._tool_to_server[tool_name] = server_name

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions from all connected servers (OpenAI format)."""
        all_tools = []
        for name, tools in self._tools.items():
            for tool_def in tools:
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
        tools = self._tools.get(server_name, [])
        return [
            {
                "type": "function",
                "function": {
                    "name": f"mcp_{server_name}_{t['name']}",
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    async def call_tool(self, tool_call: ToolCall) -> dict[str, Any]:
        """Route a tool call to the correct MCP server."""
        tool_name = tool_call.function.name

        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return {"error": f"MCP tool '{tool_name}' not found in any server"}

        client = self._clients.get(server_name)
        if not client:
            return {"error": f"MCP server '{server_name}' not connected"}

        actual_tool = tool_name
        prefix = f"mcp_{server_name}_"
        if tool_name.startswith(prefix):
            actual_tool = tool_name[len(prefix):]

        import json
        args = tool_call.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        return await client.call_tool(actual_tool, args)

    def get_client(self, name: str) -> MCPProtocolClient | None:
        return self._clients.get(name)

    def list_servers(self) -> list[str]:
        return list(self._clients.keys())

    @property
    def connected_count(self) -> int:
        return len(self._clients)

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name, transport in self._transports.items():
            try:
                await transport.close()
            except Exception:
                logger.error("Error disconnecting MCP '%s'", name, exc_info=True)
        self._clients.clear()
        self._transports.clear()
        self._tools.clear()
        self._tool_to_server.clear()
