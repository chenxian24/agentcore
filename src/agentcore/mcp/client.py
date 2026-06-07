"""MCP client: connects to an MCP server and provides tool discovery + execution."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for a single MCP server connection.

    Supports TCP connections (primary) with stdio fallback.
    Handles the MCP protocol handshake, tool discovery, and tool execution.
    """

    def __init__(self, server_name: str) -> None:
        self._name = server_name
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._req_id = 0
        self._tools: list[dict[str, Any]] = []
        self._connected = False
        self._server_info: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    @property
    def server_info(self) -> dict[str, Any]:
        return self._server_info

    async def connect_tcp(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Connect to an MCP server via TCP."""
        self._reader, self._writer = await asyncio.open_connection(host, port)
        await self._initialize()

    async def connect_stdio(self, command: str, args: list[str] | None = None, cwd: str | None = None) -> None:
        """Connect to an MCP server via stdio (subprocess)."""
        import sys
        cmd = [command] + (args or [])
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        self._reader = None  # We'll use proc.stdout directly
        self._writer = None  # We'll use proc.stdin directly
        await self._initialize()

    async def _initialize(self) -> None:
        """Perform MCP protocol initialization."""
        resp = await self._send({
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agentcore", "version": "0.3.0"},
            },
        })
        self._server_info = resp.get("result", {}).get("serverInfo", {})

        await self._notify({"method": "notifications/initialized"})

        # Discover tools
        resp = await self._send({"method": "tools/list", "params": {}})
        self._tools = resp.get("result", {}).get("tools", [])
        self._connected = True
        logger.info("MCP '%s' connected: %d tools", self._name, len(self._tools))

    async def _send(self, msg: dict) -> dict:
        """Send a JSON-RPC request and wait for response."""
        self._req_id += 1
        msg.update({"jsonrpc": "2.0", "id": self._req_id})
        payload = json.dumps(msg) + "\n"

        if self._writer:
            self._writer.write(payload.encode("utf-8"))
            await self._writer.drain()
            line = await asyncio.wait_for(self._reader.readline(), timeout=30)
        elif self._proc:
            self._proc.stdin.write(payload.encode())
            await self._proc.stdin.drain()
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=30)
        else:
            raise RuntimeError("MCP client not connected")

        if not line:
            raise RuntimeError(f"MCP server '{self._name}' disconnected")
        return json.loads(line)

    async def _notify(self, msg: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg["jsonrpc"] = "2.0"
        payload = json.dumps(msg) + "\n"

        if self._writer:
            self._writer.write(payload.encode("utf-8"))
            await self._writer.drain()
        elif self._proc:
            self._proc.stdin.write(payload.encode())
            await self._proc.stdin.drain()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server."""
        resp = await self._send({
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        result = resp.get("result", {})
        parts = result.get("content", [])
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if result.get("isError"):
            return {"error": text}
        return {"output": text}

    async def list_tools(self) -> list[dict[str, Any]]:
        """Refresh and return the tool list from the server."""
        resp = await self._send({"method": "tools/list", "params": {}})
        self._tools = resp.get("result", {}).get("tools", [])
        return self._tools

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._writer:
            self._writer.close()
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()
        self._connected = False
