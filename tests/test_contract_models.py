"""Contract tests for new model types: StreamEvent, ToolResult, ProviderCapabilities, MessageAdapter."""

from __future__ import annotations

import pytest

from agentcore.core.adapter import MessageAdapter
from agentcore.core.message import Message, MessageRole
from agentcore.models.base import LLMResponse, ToolCall, ToolCallFunction
from agentcore.models.capabilities import ProviderCapabilities
from agentcore.models.events import StreamEvent, StreamEventType
from agentcore.tools.result import ToolResult


class TestStreamEvent:
    def test_text_factory(self):
        e = StreamEvent.text_delta("hello")
        assert e.type == StreamEventType.TEXT_DELTA
        assert e.text == "hello"

    def test_reasoning_factory(self):
        e = StreamEvent.reasoning("thinking...")
        assert e.type == StreamEventType.REASONING_DELTA
        assert e.thinking == "thinking..."

    def test_tool_call_factory(self):
        tc = ToolCall(function=ToolCallFunction(name="echo", arguments="{}"))
        e = StreamEvent.tool_call_done(tc)
        assert e.type == StreamEventType.TOOL_CALL
        assert e.tool_name == "echo"

    def test_message_done_factory(self):
        e = StreamEvent.message_done(content="response", finish_reason="stop")
        assert e.type == StreamEventType.MESSAGE_DONE
        assert e.text == "response"
        assert e.finish_reason == "stop"

    def test_error_factory(self):
        e = StreamEvent.error_event("something broke")
        assert e.type == StreamEventType.ERROR
        assert e.error == "something broke"


class TestToolResult:
    def test_ok(self):
        r = ToolResult.ok("output data")
        assert r.success is True
        assert r.output == "output data"
        assert r.error == ""

    def test_fail(self):
        r = ToolResult.fail("not found")
        assert r.success is False
        assert r.error == "not found"

    def test_to_dict_success(self):
        r = ToolResult.ok("data")
        d = r.to_dict()
        assert d == {"output": "data"}

    def test_to_dict_failure(self):
        r = ToolResult.fail("err", output="partial")
        d = r.to_dict()
        assert d["error"] == "err"

    def test_from_dict_success(self):
        r = ToolResult.from_dict({"output": "data"})
        assert r.success is True
        assert r.output == "data"

    def test_from_dict_failure(self):
        r = ToolResult.from_dict({"output": None, "error": "bad"})
        assert r.success is False
        assert r.error == "bad"


class TestProviderCapabilities:
    def test_defaults(self):
        c = ProviderCapabilities()
        assert c.chat is True
        assert c.streaming is True
        assert c.tools is False
        assert c.vision is False

    def test_supports_bool(self):
        c = ProviderCapabilities(tools=True)
        assert c.supports("tools") is True
        assert c.supports("vision") is False

    def test_supports_list(self):
        c = ProviderCapabilities(reasoning=["extended_thinking"])
        assert c.supports("reasoning") is True

    def test_supports_empty_list(self):
        c = ProviderCapabilities(reasoning=[])
        assert c.supports("reasoning") is False


class TestMessageAdapter:
    def test_to_llm_user(self):
        m = Message.user("hello")
        llm = MessageAdapter.to_llm(m)
        assert llm.role == "user"
        assert llm.content == "hello"

    def test_to_llm_assistant(self):
        m = Message.assistant("response")
        llm = MessageAdapter.to_llm(m)
        assert llm.role == "assistant"
        assert llm.content == "response"

    def test_to_llm_system(self):
        m = Message.system("system prompt")
        llm = MessageAdapter.to_llm(m)
        assert llm.role == "system"

    def test_to_llm_tool(self):
        m = Message.tool("result", tool_call_id="tc_123")
        llm = MessageAdapter.to_llm(m)
        assert llm.role == "tool"
        assert llm.tool_call_id == "tc_123"

    def test_to_llm_list(self):
        msgs = [Message.user("a"), Message.assistant("b")]
        llm_list = MessageAdapter.to_llm_list(msgs)
        assert len(llm_list) == 2
        assert llm_list[0].role == "user"
        assert llm_list[1].role == "assistant"

    def test_from_response(self):
        resp = LLMResponse(content="answer", model="test")
        m = MessageAdapter.from_response(resp)
        assert m.role == MessageRole.ASSISTANT
        assert m.content == "answer"

    def test_from_tool_result(self):
        m = MessageAdapter.from_tool_result("tc_1", "output", name="echo")
        assert m.role == MessageRole.TOOL
        assert m.content == "output"
        assert m.tool_call_id == "tc_1"
        assert m.name == "echo"
