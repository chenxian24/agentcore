"""MCP Protocol layer — JSON-RPC semantics over a transport.

Handles: initialize handshake, tools/list, tools/call.
Does not handle: transport framing, tool adapter logic.
"""

from __future__ import annotations

import logging
from typing import Any

from agentcore.mcp.transport import MCPTransport

logger = logging.getLogger(__name__)


class MCPProtocolClient:
    """MCP protocol client that speaks JSON-RPC over a transport.

    Responsibilities:
    - MCP initialize handshake
    - tools/list discovery
    - tools/call execution
    - JSON-RPC request/response correlation
    """

    def __init__(self, transport: MCPTransport, client_name: str = "agentcore", client_version: str = "0.6.0") -> None:
        self._transport = transport
        self._client_name = client_name
        self._client_version = client_version
        self._req_id = 0
        self._server_info: dict[str, Any] = {}

    @property
    def server_info(self) -> dict[str, Any]:
        return self._server_info

    async def initialize(self) -> dict[str, Any]:
        """Perform MCP protocol initialization."""
        resp = await self._request({
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
            },
        })
        result = resp.get("result", {})
        self._server_info = result.get("serverInfo", {})

        await self._notify({"method": "notifications/initialized"})

        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover tools from the server."""
        resp = await self._request({"method": "tools/list", "params": {}})
        return resp.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool and return the result."""
        resp = await self._request({
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        result = resp.get("result", {})
        parts = result.get("content", [])
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if result.get("isError"):
            return {"error": text}
        return {"output": text}

    async def _request(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        self._req_id += 1
        msg.update({"jsonrpc": "2.0", "id": self._req_id})
        await self._transport.send_json(msg)
        return await self._transport.receive_json(timeout=30)

    async def _notify(self, msg: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg["jsonrpc"] = "2.0"
        await self._transport.send_json(msg)
