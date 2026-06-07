"""Anthropic Claude LLM provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    ImageBlock,
    LLMMessage,
    LLMResponse,
    TextBlock,
    ToolCall,
    ToolCallFunction,
    ToolResultBlock,
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        default_model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
        self._default_model = default_model
        self._last_stream_tool_calls: list[ToolCall] = []

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_thinking(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_content(self, content: str | list) -> str | list[dict[str, Any]]:
        """Convert LLMMessage content to Anthropic format."""
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": block.image_url.url,
                    },
                })
            elif isinstance(block, ToolResultBlock):
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                })
            else:
                blocks.append({"type": "text", "text": str(block)})
        return blocks

    def _split_system_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract system message and convert rest to Anthropic format."""
        system = ""
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls -> content blocks
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    converted = self._convert_content(msg.content)
                    if isinstance(converted, str):
                        content_blocks.append({"type": "text", "text": converted})
                    else:
                        content_blocks.extend(converted)
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": _safe_json_loads(tc.function.arguments),
                    })
                result.append({"role": "assistant", "content": content_blocks})
            elif msg.role == "tool":
                # Tool result message -> user message with tool_result block
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                    }],
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": self._convert_content(msg.content),
                })
        return system, result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response to LLMResponse."""
        content = ""
        thinking = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                thinking += block.thinking
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    type="function",
                    function=ToolCallFunction(
                        name=block.name,
                        arguments=_safe_json_dumps(block.input),
                    ),
                ))

        return LLMResponse(
            content=content,
            thinking=thinking,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "",
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Parameter building
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool definitions to Anthropic format.

        OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
        Anthropic: {"name", "description", "input_schema"}
        """
        result = []
        for tool in tools:
            if "function" in tool:
                fn = tool["function"]
                result.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Already in Anthropic format or unknown — pass through
                result.append(tool)
        return result

    def _build_params(self, params: ChatParams, system: str) -> dict[str, Any]:
        """Build Anthropic API params from ChatParams."""
        p: dict[str, Any] = {
            "model": params.model or self._default_model,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if system:
            p["system"] = system
        if params.top_p != 1.0:
            p["top_p"] = params.top_p
        if params.stop:
            p["stop_sequences"] = params.stop
        if params.tools:
            p["tools"] = self._convert_tools_to_anthropic(params.tools)
        if params.tool_choice:
            if isinstance(params.tool_choice, str):
                if params.tool_choice == "auto":
                    p["tool_choice"] = {"type": "auto"}
                elif params.tool_choice == "none":
                    pass  # Anthropic doesn't have "none", just omit tools
                elif params.tool_choice == "required":
                    p["tool_choice"] = {"type": "any"}
                else:
                    p["tool_choice"] = {"type": "auto"}
            else:
                p["tool_choice"] = params.tool_choice
        # Thinking / extended thinking
        if params.thinking and params.thinking.enabled:
            p["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.thinking.budget_tokens,
            }
            # Extended thinking requires temperature=1 and no top_p
            p["temperature"] = 1.0
            p.pop("top_p", None)
        return p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if params is None:
            params = ChatParams()

        system, msgs = self._split_system_messages(messages)
        api_params = self._build_params(params, system)
        api_params["messages"] = msgs

        response = await self._client.messages.create(**api_params)
        return self._parse_response(response)

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if params is None:
            params = ChatParams()

        system, msgs = self._split_system_messages(messages)
        api_params = self._build_params(params, system)
        api_params["messages"] = msgs

        async with self._client.messages.stream(**api_params) as stream:
            # Track tool use blocks across streaming events
            tool_blocks: dict[int, dict[str, str]] = {}
            current_block_type: str = ""
            current_block_idx: int = -1

            async for event in stream:
                if not hasattr(event, "type"):
                    continue

                if event.type == "content_block_start":
                    block = event.content_block
                    current_block_idx = event.index
                    if hasattr(block, "type"):
                        current_block_type = block.type
                        if block.type == "tool_use":
                            tool_blocks[current_block_idx] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }
                        else:
                            current_block_type = ""

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text"):
                        yield delta.text
                    elif hasattr(delta, "partial_json") and current_block_idx in tool_blocks:
                        tool_blocks[current_block_idx]["arguments"] += delta.partial_json

                elif event.type == "content_block_stop":
                    current_block_type = ""

            # Store accumulated tool calls for engine to retrieve
            if tool_blocks:
                self._last_stream_tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type="function",
                        function=ToolCallFunction(name=tc["name"], arguments=tc["arguments"]),
                    )
                    for tc in (tool_blocks[i] for i in sorted(tool_blocks))
                ]
            else:
                self._last_stream_tool_calls = []

    async def embed(self, texts: list[str], model: str = "") -> list[list[float]]:
        raise NotImplementedError("Anthropic does not provide an embedding API.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(s: str) -> Any:
    """JSON loads with fallback to raw string."""
    import json
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def _safe_json_dumps(obj: Any) -> str:
    """JSON dumps with fallback to str()."""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
