"""Context contract tests — ContextEngine, strategies."""

from __future__ import annotations

import pytest

from agentcore.context.engine import ContextEngine, SlidingWindowStrategy, SummarizationStrategy
from agentcore.core.message import Message, MessageRole


class TestSlidingWindowStrategy:
    def test_name(self):
        s = SlidingWindowStrategy()
        assert s.name == "sliding_window"

    def test_compress_under_budget(self):
        s = SlidingWindowStrategy(reserve_tokens=100)
        msgs = [Message.user("hi"), Message.assistant("hello")]
        result = s.compress(msgs, max_tokens=10000)
        assert len(result) == 2

    def test_compress_over_budget(self):
        s = SlidingWindowStrategy(reserve_tokens=100)
        msgs = [Message.user(f"message {i}") for i in range(100)]
        result = s.compress(msgs, max_tokens=200)
        assert len(result) < 100

    def test_compress_empty(self):
        s = SlidingWindowStrategy()
        result = s.compress([], max_tokens=10000)
        assert result == []


class TestSummarizationStrategy:
    def test_name(self):
        s = SummarizationStrategy()
        assert s.name == "summarization"

    def test_compress_keeps_recent(self):
        s = SummarizationStrategy(keep_recent=3)
        msgs = [Message.user(f"msg {i}") for i in range(20)]
        result = s.compress(msgs, max_tokens=100)
        assert len(result) <= 4  # 1 summary + 3 recent


class TestContextEngine:
    def test_default_strategies(self):
        e = ContextEngine()
        names = e.list_strategies()
        assert "sliding_window" in names
        assert "summarization" in names

    def test_register_custom_strategy(self):
        e = ContextEngine()

        class CustomStrategy(SlidingWindowStrategy):
            @property
            def name(self) -> str:
                return "custom"

        e.register_strategy(CustomStrategy())
        assert "custom" in e.list_strategies()

    def test_get_default_strategy(self):
        e = ContextEngine()
        s = e.get_strategy()
        assert s.name == "sliding_window"

    def test_compress_delegates(self):
        e = ContextEngine()
        msgs = [Message.user("hi")]
        result = e.compress(msgs, max_tokens=10000)
        assert len(result) >= 1

    def test_unknown_strategy_raises(self):
        e = ContextEngine()
        with pytest.raises(ValueError):
            e.get_strategy("nonexistent")
