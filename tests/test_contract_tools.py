"""Tool contract tests — ToolRegistry, ToolEntry."""

from __future__ import annotations

import json

import pytest

from agentcore.models.base import ToolCall, ToolCallFunction
from agentcore.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


async def _echo_handler(args: dict) -> dict:
    return {"output": args.get("input", "")}


async def _fail_handler(args: dict) -> dict:
    raise RuntimeError("handler error")


class TestToolRegistry:
    def test_register_and_get(self, registry: ToolRegistry):
        registry.register("echo", _echo_handler, description="Echo input")
        entry = registry.get("echo")
        assert entry is not None
        assert entry.name == "echo"
        assert entry.description == "Echo input"

    def test_has(self, registry: ToolRegistry):
        assert not registry.has("echo")
        registry.register("echo", _echo_handler)
        assert registry.has("echo")

    def test_unregister(self, registry: ToolRegistry):
        registry.register("echo", _echo_handler)
        registry.unregister("echo")
        assert not registry.has("echo")

    def test_list_names(self, registry: ToolRegistry):
        registry.register("a", _echo_handler)
        registry.register("b", _echo_handler)
        names = registry.list_names()
        assert "a" in names
        assert "b" in names

    def test_get_tool_definitions(self, registry: ToolRegistry):
        registry.register(
            "echo",
            _echo_handler,
            description="Echo",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}},
        )
        defs = registry.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_execute_success(self, registry: ToolRegistry):
        registry.register("echo", _echo_handler)
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments='{"input": "hello"}'))
        result = await registry.execute(tc)
        assert result == {"output": "hello"}

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry: ToolRegistry):
        tc = ToolCall(function=ToolCallFunction(name="nonexistent", arguments="{}"))
        result = await registry.execute(tc)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_invalid_json(self, registry: ToolRegistry):
        registry.register("echo", _echo_handler)
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments="not json"))
        result = await registry.execute(tc)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_handler_exception(self, registry: ToolRegistry):
        registry.register("fail", _fail_handler)
        tc = ToolCall(function=ToolCallFunction(name="fail", arguments="{}"))
        result = await registry.execute(tc)
        assert "error" in result
