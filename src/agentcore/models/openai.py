"""OpenAI LLM provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    ImageBlock,
    ImageURL,
    LLMMessage,
    LLMResponse,
    TextBlock,
    ToolCall,
    ToolCallFunction,
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        default_model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
        self._default_model = default_model
        self._last_stream_tool_calls: list[ToolCall] = []

    @property
    def name(self) -> str:
        return "openai"

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
        """Convert LLMMessage content to OpenAI format."""
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": block.image_url.url,
                        "detail": block.image_url.detail,
                    },
                })
            else:
                # ToolResultBlock or other — convert to text
                blocks.append({"type": "text", "text": str(block)})
        return blocks

    def _convert_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Convert LLMMessage tool_calls to OpenAI format."""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLMMessage list to OpenAI format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": self._convert_content(msg.content)}
            if msg.name:
                m["name"] = msg.name
            if msg.role == "tool" and msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.role == "assistant" and msg.tool_calls:
                m["tool_calls"] = self._convert_tool_calls(msg.tool_calls)
                # OpenAI requires content to be null or omitted when tool_calls present
                if not msg.content:
                    m["content"] = None
            result.append(m)
        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_tool_calls(self, tool_calls: list[ChatCompletionMessageToolCall] | None) -> list[ToolCall]:
        """Parse OpenAI tool_calls to LLMResponse format."""
        if not tool_calls:
            return []
        return [
            ToolCall(
                id=tc.id,
                type=tc.type,
                function=ToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in tool_calls
        ]

    def _extract_reasoning_tokens(self, usage: Any) -> int:
        """Extract reasoning tokens from OpenAI usage object."""
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            details = usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                return details.reasoning_tokens
        return 0

    def _build_params(self, params: ChatParams) -> dict[str, Any]:
        """Build OpenAI API params from ChatParams."""
        p: dict[str, Any] = {
            "model": params.model or self._default_model,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.top_p != 1.0:
            p["top_p"] = params.top_p
        if params.stop:
            p["stop"] = params.stop
        if params.seed is not None:
            p["seed"] = params.seed
        if params.frequency_penalty:
            p["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty:
            p["presence_penalty"] = params.presence_penalty
        if params.response_format:
            p["response_format"] = params.response_format
        if params.tools:
            p["tools"] = params.tools
        if params.tool_choice:
            p["tool_choice"] = params.tool_choice
        # Thinking / reasoning — map to reasoning_effort for o-series models
        if params.thinking and params.thinking.enabled:
            if params.thinking.budget_tokens <= 4096:
                p["reasoning_effort"] = "low"
            elif params.thinking.budget_tokens <= 16384:
                p["reasoning_effort"] = "medium"
            else:
                p["reasoning_effort"] = "high"
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

        api_params = self._build_params(params)
        api_params["messages"] = self._to_openai_messages(messages)

        response = await self._client.chat.completions.create(**api_params)
        choice = response.choices[0]
        msg = choice.message

        usage = response.usage
        reasoning_tokens = self._extract_reasoning_tokens(usage)

        return LLMResponse(
            content=msg.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "reasoning_tokens": reasoning_tokens,
            },
            finish_reason=choice.finish_reason or "",
            tool_calls=self._parse_tool_calls(msg.tool_calls),
        )

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if params is None:
            params = ChatParams()

        api_params = self._build_params(params)
        api_params["messages"] = self._to_openai_messages(messages)
        api_params["stream"] = True

        stream = await self._client.chat.completions.create(**api_params)

        # Accumulate tool calls across chunks
        tool_call_acc: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Yield text content
            if delta.content:
                yield delta.content

            # Accumulate tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_call_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_acc[idx]["arguments"] += tc.function.arguments

        # Store accumulated tool calls so the engine can retrieve them
        if tool_call_acc:
            self._last_stream_tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type="function",
                    function=ToolCallFunction(name=tc["name"], arguments=tc["arguments"]),
                )
                for tc in (tool_call_acc[i] for i in sorted(tool_call_acc))
            ]
        else:
            self._last_stream_tool_calls = []

    async def embed(self, texts: list[str], model: str = "") -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=model or "text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]
