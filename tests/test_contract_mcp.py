"""Contract tests for MCP Transport/Protocol/Adapter layers."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentcore.mcp.protocol import MCPProtocolClient
from agentcore.mcp.transport import MCPTransport
from agentcore.tools.registry import ToolRegistry


class MockTransport(MCPTransport):
    """In-memory transport for testing MCP protocol without real subprocesses."""

    def __init__(self) -> None:
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._recv_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def send(self, data: bytes) -> None:
        await self._send_queue.put(data)

    async def receive(self, timeout: float = 30.0) -> bytes:
        return await asyncio.wait_for(self._recv_queue.get(), timeout=timeout)

    async def close(self) -> None:
        self._connected = False

    # Test helpers

    async def respond(self, response: dict) -> None:
        """Simulate server sending a response."""
        await self._recv_queue.put((json.dumps(response) + "\n").encode("utf-8"))

    async def receive_request(self, timeout: float = 1.0) -> dict:
        """Receive the next request sent by the client."""
        data = await asyncio.wait_for(self._send_queue.get(), timeout=timeout)
        return json.loads(data.decode("utf-8"))


@pytest.mark.asyncio
async def test_mcp_protocol_initialize():
    transport = MockTransport()
    await transport.connect()
    client = MCPProtocolClient(transport, "test-client", "0.1.0")

    # Start initialize in background
    async def handle():
        req = await transport.receive_request()
        assert req["method"] == "initialize"
        await transport.respond({
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test-server", "version": "1.0"},
                "capabilities": {"tools": {}},
            },
        })
        # Handle notifications/initialized
        notif = await transport.receive_request()
        assert notif["method"] == "notifications/initialized"

    task = asyncio.create_task(handle())
    result = await client.initialize()
    await task
    assert client.server_info["name"] == "test-server"


@pytest.mark.asyncio
async def test_mcp_protocol_list_tools():
    transport = MockTransport()
    await transport.connect()
    client = MCPProtocolClient(transport)

    async def handle():
        # Skip initialize
        req = await transport.receive_request()
        await transport.respond({"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "2024-11-05", "serverInfo": {}, "capabilities": {}}})
        notif = await transport.receive_request()

        # Handle tools/list
        req = await transport.receive_request()
        assert req["method"] == "tools/list"
        await transport.respond({
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {"tools": [{"name": "echo", "description": "Echo tool"}]},
        })

    task = asyncio.create_task(handle())
    await client.initialize()
    tools = await client.list_tools()
    await task
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"


@pytest.mark.asyncio
async def test_mcp_protocol_call_tool():
    transport = MockTransport()
    await transport.connect()
    client = MCPProtocolClient(transport)

    async def handle():
        # Skip initialize
        req = await transport.receive_request()
        await transport.respond({"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "2024-11-05", "serverInfo": {}, "capabilities": {}}})
        notif = await transport.receive_request()

        # Handle tools/call
        req = await transport.receive_request()
        assert req["method"] == "tools/call"
        assert req["params"]["name"] == "echo"
        await transport.respond({
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {"content": [{"type": "text", "text": "hello"}]},
        })

    task = asyncio.create_task(handle())
    await client.initialize()
    result = await client.call_tool("echo", {"input": "hi"})
    await task
    assert result == {"output": "hello"}


@pytest.mark.asyncio
async def test_mcp_protocol_call_tool_error():
    transport = MockTransport()
    await transport.connect()
    client = MCPProtocolClient(transport)

    async def handle():
        req = await transport.receive_request()
        await transport.respond({"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "2024-11-05", "serverInfo": {}, "capabilities": {}}})
        notif = await transport.receive_request()

        req = await transport.receive_request()
        await transport.respond({
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {"content": [{"type": "text", "text": "not found"}], "isError": True},
        })

    task = asyncio.create_task(handle())
    await client.initialize()
    result = await client.call_tool("bad", {})
    await task
    assert "error" in result
