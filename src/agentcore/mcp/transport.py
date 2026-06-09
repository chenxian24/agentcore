"""MCP Transport layer — raw byte stream abstraction.

Provides transport implementations for different connection types:
- StdioTransport: subprocess stdin/stdout
- TcpTransport: TCP socket connection

Transports handle framing (newline-delimited JSON) but not protocol semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_BUFFER_LIMIT = 10 * 1024 * 1024  # 10 MB — must exceed largest MCP response


class MCPTransport(ABC):
    """Abstract transport for MCP JSON-RPC communication."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection."""
        ...

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """Send raw bytes."""
        ...

    @abstractmethod
    async def receive(self, timeout: float = 30.0) -> bytes:
        """Receive raw bytes with timeout."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        ...

    async def send_json(self, obj: dict[str, Any]) -> None:
        """Send a JSON object as newline-delimited JSON."""
        payload = json.dumps(obj) + "\n"
        await self.send(payload.encode("utf-8"))

    async def receive_json(self, timeout: float = 30.0) -> dict[str, Any]:
        """Receive a JSON object from newline-delimited JSON."""
        data = await self.receive(timeout)
        return json.loads(data)


class StdioTransport(MCPTransport):
    """Transport over subprocess stdin/stdout."""

    def __init__(self, command: str, args: list[str] | None = None, cwd: str | None = None) -> None:
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None

    async def connect(self) -> None:
        cmd = [self._command] + self._args
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            limit=_DEFAULT_BUFFER_LIMIT,
        )

    async def send(self, data: bytes) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Not connected")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def receive(self, timeout: float = 30.0) -> bytes:
        if not self._proc or not self._proc.stdout:
            raise RuntimeError("Not connected")
        line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        if not line:
            raise RuntimeError("MCP server disconnected (stdio)")
        return line

    async def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()
            self._proc = None


class TcpTransport(MCPTransport):
    """Transport over TCP socket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port, limit=_DEFAULT_BUFFER_LIMIT,
        )

    async def send(self, data: bytes) -> None:
        if not self._writer:
            raise RuntimeError("Not connected")
        self._writer.write(data)
        await self._writer.drain()

    async def receive(self, timeout: float = 30.0) -> bytes:
        if not self._reader:
            raise RuntimeError("Not connected")
        line = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
        if not line:
            raise RuntimeError("MCP server disconnected (tcp)")
        return line

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None
            self._reader = None
