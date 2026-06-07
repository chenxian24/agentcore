"""Contract tests for ToolPipeline, PluginManager state machine, ContextPipeline."""

from __future__ import annotations

import pytest

from agentcore.context.engine import ContextChunk, ContextPipeline, ContextSource
from agentcore.models.base import ToolCall, ToolCallFunction
from agentcore.plugins.base import Plugin, PluginContext
from agentcore.plugins.manager import PluginManager, PluginState
from agentcore.tools.pipeline import ToolPipeline
from agentcore.tools.policy import PatternPolicy, PolicyDecision, PolicyPipeline
from agentcore.tools.registry import ToolRegistry


# --- ToolPipeline tests ---

async def _echo_handler(args: dict) -> dict:
    return {"output": args.get("input", "")}


@pytest.fixture
def tool_pipeline() -> ToolPipeline:
    reg = ToolRegistry()
    reg.register("echo", _echo_handler, description="Echo", parameters={"type": "object", "properties": {"input": {"type": "string"}}})
    return ToolPipeline(reg)


class TestToolPipeline:
    @pytest.mark.asyncio
    async def test_execute_success(self, tool_pipeline: ToolPipeline):
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments='{"input": "hi"}'))
        result = await tool_pipeline.execute(tc)
        assert result.success is True
        assert result.output == "hi"

    @pytest.mark.asyncio
    async def test_execute_invalid_json(self, tool_pipeline: ToolPipeline):
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments="bad"))
        result = await tool_pipeline.execute(tc)
        assert result.success is False
        assert "Invalid JSON" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_pipeline: ToolPipeline):
        tc = ToolCall(function=ToolCallFunction(name="nonexistent", arguments="{}"))
        result = await tool_pipeline.execute(tc)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_policy_deny(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_handler)
        policy = PolicyPipeline()
        policy.add(PatternPolicy("deny-echo", deny_patterns=["echo"]))
        pipeline = ToolPipeline(reg, policy)
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments="{}"))
        result = await pipeline.execute(tc)
        assert result.success is False
        assert "denied" in result.error

    @pytest.mark.asyncio
    async def test_transformer(self, tool_pipeline: ToolPipeline):
        async def upper_transform(tc: ToolCall, result):
            if result.success and isinstance(result.output, str):
                result.output = result.output.upper()
            return result
        tool_pipeline.add_transformer(upper_transform)
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments='{"input": "hello"}'))
        result = await tool_pipeline.execute(tc)
        assert result.output == "HELLO"


# --- PluginManager state machine tests ---

class DummyPlugin(Plugin):
    def __init__(self, name: str, deps: list[str] | None = None, fail_setup: bool = False):
        self._name = name
        self._deps = deps or []
        self._fail_setup = fail_setup
        self.setup_called = False
        self.teardown_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return f"Dummy {self._name}"

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def setup(self, ctx: PluginContext) -> None:
        if self._fail_setup:
            raise RuntimeError(f"{self._name} setup failed")
        self.setup_called = True

    async def teardown(self, ctx: PluginContext) -> None:
        self.teardown_called = True


class TestPluginManager:
    @pytest.mark.asyncio
    async def test_register_and_state(self):
        pm = PluginManager()
        p = DummyPlugin("test")
        pm.register(p)
        assert pm.get_state("test") == PluginState.REGISTERED

    @pytest.mark.asyncio
    async def test_dependency_order(self):
        pm = PluginManager()
        a = DummyPlugin("a")
        b = DummyPlugin("b", deps=["a"])
        pm.register(b)
        pm.register(a)
        await pm.initialize_all()
        order = pm._load_order
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_cyclic_dependency_detection(self):
        pm = PluginManager()
        a = DummyPlugin("a", deps=["b"])
        b = DummyPlugin("b", deps=["a"])
        pm.register(a)
        pm.register(b)
        with pytest.raises(ValueError, match="Cyclic"):
            await pm.initialize_all()

    @pytest.mark.asyncio
    async def test_missing_dependency_warning(self):
        pm = PluginManager(strict_dependencies=False)
        a = DummyPlugin("a", deps=["nonexistent"])
        pm.register(a)
        # Should not raise (soft warning)
        await pm.initialize_all()
        assert pm.get_state("a") == PluginState.READY

    @pytest.mark.asyncio
    async def test_missing_dependency_strict(self):
        pm = PluginManager(strict_dependencies=True)
        a = DummyPlugin("a", deps=["nonexistent"])
        pm.register(a)
        with pytest.raises(ValueError, match="not registered"):
            await pm.initialize_all()

    @pytest.mark.asyncio
    async def test_failed_plugin_state(self):
        pm = PluginManager(strict_dependencies=False)
        good = DummyPlugin("good")
        bad = DummyPlugin("bad", fail_setup=True)
        pm.register(good)
        pm.register(bad)
        await pm.initialize_all()
        assert pm.get_state("good") == PluginState.READY
        assert pm.get_state("bad") == PluginState.FAILED

    @pytest.mark.asyncio
    async def test_shutdown_calls_teardown(self):
        pm = PluginManager()
        p = DummyPlugin("test")
        pm.register(p)
        await pm.initialize_all()
        await pm.shutdown_all()
        assert p.teardown_called is True
        assert pm.get_state("test") == PluginState.STOPPED

    @pytest.mark.asyncio
    async def test_list_plugins_includes_state(self):
        pm = PluginManager()
        p = DummyPlugin("test")
        pm.register(p)
        await pm.initialize_all()
        info = pm.list_plugins()
        assert len(info) == 1
        assert info[0]["state"] == "ready"


# --- ContextPipeline tests ---

class TestContextPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        p = ContextPipeline(max_tokens=10000)
        result = await p.process()
        assert result == []

    def test_rank_by_priority(self):
        chunks = [
            ContextChunk(content="low", source=ContextSource.SESSION_HISTORY, priority=200),
            ContextChunk(content="high", source=ContextSource.SYSTEM_PROMPT, priority=10),
        ]
        ranked = ContextPipeline.rank(chunks)
        assert ranked[0].content == "high"

    def test_dedupe(self):
        chunks = [
            ContextChunk(content="same", source=ContextSource.SESSION_HISTORY),
            ContextChunk(content="same", source=ContextSource.SESSION_HISTORY),
            ContextChunk(content="different", source=ContextSource.SESSION_HISTORY),
        ]
        deduped = ContextPipeline.dedupe(chunks)
        assert len(deduped) == 2

    def test_budget(self):
        p = ContextPipeline(max_tokens=100)  # ~25 tokens available after reserve
        chunks = [
            ContextChunk(content="a" * 80, source=ContextSource.SESSION_HISTORY, token_estimate=20),
            ContextChunk(content="b" * 80, source=ContextSource.SESSION_HISTORY, token_estimate=20),
            ContextChunk(content="c" * 80, source=ContextSource.SESSION_HISTORY, token_estimate=20),
        ]
        budgeted = p.budget(chunks)
        total = sum(c.token_estimate for c in budgeted)
        assert total <= 100 - p._reserve_tokens
