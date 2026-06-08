"""ChatCompletionsProtocol — OpenAI Chat Completions API wire format."""

from __future__ import annotations

from typing import Any

from agentcore.wire.base import WireProtocol
from agentcore.wire.types import WireEvent, WireResponse, WireToolCall


class ChatCompletionsProtocol(WireProtocol):
    """Wire protocol for OpenAI-compatible Chat Completions API.

    Handles serialization for POST /v1/chat/completions and parsing
    of both streaming (SSE) and non-streaming responses.
    """

    @property
    def name(self) -> str:
        return "chat_completions"

    def get_endpoint(self) -> str:
        return "/v1/chat/completions"

    def build_request_body(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        if "stop" in kwargs and kwargs["stop"]:
            body["stop"] = kwargs["stop"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            body["seed"] = kwargs["seed"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            body["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs and kwargs["frequency_penalty"]:
            body["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs and kwargs["presence_penalty"]:
            body["presence_penalty"] = kwargs["presence_penalty"]
        if "response_format" in kwargs and kwargs["response_format"]:
            body["response_format"] = kwargs["response_format"]
        if "reasoning_effort" in kwargs and kwargs["reasoning_effort"]:
            body["reasoning_effort"] = kwargs["reasoning_effort"]
        return body

    def parse_response(self, raw: dict[str, Any]) -> WireResponse:
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message", {})

        tool_calls: list[WireToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            tool_calls.append(WireToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", "{}"),
            ))

        usage_raw = raw.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }

        return WireResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
            raw=raw,
        )

    def parse_stream_event(self, raw: dict[str, Any]) -> WireEvent | None:
        choices = raw.get("choices")
        if not choices:
            # Usage-only chunk at end of stream
            usage_raw = raw.get("usage")
            if usage_raw:
                return WireEvent(
                    type="done",
                    usage={
                        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
                        "completion_tokens": usage_raw.get("completion_tokens", 0),
                        "total_tokens": usage_raw.get("total_tokens", 0),
                    },
                )
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason", "")

        # Text delta
        content = delta.get("content")
        if content:
            return WireEvent(type="text_delta", delta=content)

        # Tool call deltas
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            fn_name = fn.get("name")
            fn_args_delta = fn.get("arguments", "")
            tc_id = tc.get("id", "")

            if fn_name:
                # Start of a new tool call
                return WireEvent(
                    type="tool_call_start",
                    tool_call=WireToolCall(id=tc_id, name=fn_name, arguments=""),
                )
            elif fn_args_delta:
                # Continuation of tool call arguments
                return WireEvent(
                    type="tool_call_delta",
                    delta=fn_args_delta,
                    tool_call=WireToolCall(id=tc_id, name="", arguments=fn_args_delta),
                )

        # Finish
        if finish_reason:
            return WireEvent(type="done", finish_reason=finish_reason)

        return None

    def is_stream_done(self, raw: dict[str, Any]) -> bool:
        choices = raw.get("choices")
        if choices and choices[0].get("finish_reason"):
            return True
        return False
