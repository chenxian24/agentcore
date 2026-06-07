"""Standardized tool result model.

All tool executions (native, MCP, skill) return ToolResult.
This ensures consistent handling across the tool pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized result from tool execution.

    Convention:
    - success=True: tool ran successfully, output contains the result
    - success=False: tool failed, error contains the reason
    - metadata carries extra info (timing, source, etc.)
    """

    success: bool = True
    output: Any = None
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any = None, **metadata: Any) -> ToolResult:
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        return cls(success=False, error=error, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the legacy dict format used by tool executors."""
        if self.success:
            return {"output": self.output}
        return {"output": self.output, "error": self.error}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        """Create from legacy dict format."""
        if "error" in data and data["error"]:
            return cls.fail(data["error"], output=data.get("output"))
        return cls.ok(data.get("output"))
