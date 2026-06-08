"""MCP (Model Context Protocol) client/manager built into agentcore."""

from agentcore.mcp.adapter import MCPToolAdapter
from agentcore.mcp.manager import MCPManager
from agentcore.mcp.protocol import MCPProtocolClient
from agentcore.mcp.transport import MCPTransport, StdioTransport, TcpTransport

__all__ = [
    "MCPManager",
    "MCPProtocolClient",
    "MCPToolAdapter",
    "MCPTransport",
    "StdioTransport",
    "TcpTransport",
]
