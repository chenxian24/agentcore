"""ResponsesProtocol — OpenAI Responses API wire format (skeleton).

The Responses API uses a fundamentally different structure from Chat Completions:
- Input: 'input' array instead of 'messages'
- Output: 'output' array with typed items (message, function_call, function_call_output)
- Streaming: event types like 'response.output_text.delta'

This is a skeleton implementation. Full implementation requires a running
Responses API endpoint for integration testing.
"""

from __future__ import annotations

from typing import Any

from agentcore.wire.base import WireProtocol
from agentcore.wire.types import WireEvent, WireResponse, WireToolCall


class ResponsesProtocol(WireProtocol):
    """Wire protocol for OpenAI Responses API (POST /v1/responses)."""

    @property
    def name(self) -> str:
        return "responses"

    def get_endpoint(self) -> str:
        return "/v1/responses"

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
        # Convert messages to Responses API input format
        input_items = self._messages_to_input(messages)

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": stream,
        }
        if tools:
            body["tools"] = self.convert_tools(tools)
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens
        return body

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI Chat Completions tools to Responses API format.

        Chat Completions: {"type": "function", "function": {"name", "description", "parameters"}}
        Responses:        {"type": "function", "name", "description", "parameters"}
        """
        result = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                result.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                result.append(tool)
        return result

    def parse_response(self, raw: dict[str, Any]) -> WireResponse:
        text_parts: list[str] = []
        tool_calls: list[WireToolCall] = []

        for item in raw.get("output", []):
            item_type = item.get("type", "")
            if item_type == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
            elif item_type == "function_call":
                tool_calls.append(WireToolCall(
                    id=item.get("call_id", item.get("id", "")),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", "{}"),
                ))

        usage_raw = raw.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("input_tokens", 0),
            "completion_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens": usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
        }

        return WireResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=raw.get("status", "completed"),
            raw=raw,
        )

    def parse_stream_event(self, raw: dict[str, Any]) -> WireEvent | None:
        event_type = raw.get("type", "")

        if event_type == "response.output_text.delta":
            return WireEvent(
                type="text_delta",
                delta=raw.get("delta", ""),
            )
        elif event_type == "response.function_call_arguments.delta":
            return WireEvent(
                type="tool_call_delta",
                delta=raw.get("delta", ""),
                tool_call=WireToolCall(
                    id=raw.get("item_id", ""),
                    name="",
                    arguments=raw.get("delta", ""),
                ),
            )
        elif event_type == "response.output_item.added":
            item = raw.get("item", {})
            if item.get("type") == "function_call":
                return WireEvent(
                    type="tool_call_start",
                    tool_call=WireToolCall(
                        id=item.get("call_id", item.get("id", "")),
                        name=item.get("name", ""),
                        arguments="",
                    ),
                )
        elif event_type == "response.completed":
            response = raw.get("response", {})
            usage_raw = response.get("usage", {})
            return WireEvent(
                type="done",
                finish_reason=response.get("status", "completed"),
                usage={
                    "prompt_tokens": usage_raw.get("input_tokens", 0),
                    "completion_tokens": usage_raw.get("output_tokens", 0),
                    "total_tokens": usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
                },
            )

        return None

    def is_stream_done(self, raw: dict[str, Any]) -> bool:
        return raw.get("type") == "response.completed"

    def _messages_to_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chat Completions messages to Responses API input format."""
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "system":
                input_items.append({
                    "role": "system",
                    "content": content,
                })
            elif role == "user":
                input_items.append({
                    "role": "user",
                    "content": content,
                })
            elif role == "assistant":
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        })
                if content:
                    input_items.append({
                        "role": "assistant",
                        "content": content,
                    })
            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id or "",
                    "output": content,
                })

        return input_items
