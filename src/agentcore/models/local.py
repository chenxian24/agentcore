"""Local model provider (Ollama-compatible)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ToolCall,
    ToolCallFunction,
)


class LocalProvider(BaseLLMProvider):
    """Local model provider via Ollama-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._last_stream_tool_calls: list = []

    @property
    def name(self) -> str:
        return "local"

    @property
    def supports_tools(self) -> bool:
        return True  # Ollama supports tools since v0.2.x

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _to_ollama_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLMMessage list to Ollama format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content if isinstance(msg.content, str) else str(msg.content)}
            if msg.role == "assistant" and msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": _safe_json_loads(tc.function.arguments) if tc.function.arguments else {},
                        }
                    }
                    for tc in msg.tool_calls
                ]
            if msg.role == "tool":
                m["content"] = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append(m)
        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_tool_calls(self, tool_calls: Any) -> list[ToolCall]:
        """Parse Ollama tool_calls to LLMResponse format."""
        if not tool_calls:
            return []
        result: list[ToolCall] = []
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            result.append(ToolCall(
                id=f"call_{i}",
                type="function",
                function=ToolCallFunction(
                    name=func.get("name", ""),
                    arguments=_safe_json_dumps(func.get("arguments", {})),
                ),
            ))
        return result

    # ------------------------------------------------------------------
    # Parameter building
    # ------------------------------------------------------------------

    def _build_options(self, params: ChatParams) -> dict[str, Any]:
        """Build Ollama options from ChatParams."""
        options: dict[str, Any] = {
            "temperature": params.temperature,
            "num_predict": params.max_tokens,
        }
        if params.top_p != 1.0:
            options["top_p"] = params.top_p
        if params.stop:
            options["stop"] = params.stop
        if params.seed is not None:
            options["seed"] = params.seed
        if params.frequency_penalty:
            options["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty:
            options["presence_penalty"] = params.presence_penalty
        return options

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

        body: dict[str, Any] = {
            "model": params.model or self._default_model,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": self._build_options(params),
        }
        if params.tools:
            body["tools"] = params.tools

        response = await self._client.post("/api/chat", json=body)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", params.model or self._default_model),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            finish_reason="stop" if data.get("done") else "",
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
        )

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if params is None:
            params = ChatParams()

        body: dict[str, Any] = {
            "model": params.model or self._default_model,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": self._build_options(params),
        }
        if params.tools:
            body["tools"] = params.tools

        async with self._client.stream("POST", "/api/chat", json=body) as response:
            response.raise_for_status()
            import json

            tool_calls_acc: list[dict[str, Any]] = []

            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        msg = chunk["message"]
                        if "content" in msg and msg["content"]:
                            yield msg["content"]
                        # Accumulate tool calls from Ollama streaming
                        if "tool_calls" in msg and msg["tool_calls"]:
                            for tc in msg["tool_calls"]:
                                func = tc.get("function", {})
                                tool_calls_acc.append({
                                    "id": tc.get("id", f"call_{len(tool_calls_acc)}"),
                                    "name": func.get("name", ""),
                                    "arguments": json.dumps(func.get("arguments", {})),
                                })

            # Store accumulated tool calls for engine to retrieve
            if tool_calls_acc:
                from agentcore.models.base import ToolCall, ToolCallFunction
                self._last_stream_tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type="function",
                        function=ToolCallFunction(name=tc["name"], arguments=tc["arguments"]),
                    )
                    for tc in tool_calls_acc
                ]
            else:
                self._last_stream_tool_calls = []

    async def embed(self, texts: list[str], model: str = "") -> list[list[float]]:
        result = []
        for text in texts:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": model or self._default_model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            result.append(data["embedding"])
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(obj: Any) -> Any:
    """JSON loads with fallback."""
    if isinstance(obj, (dict, list)):
        return obj
    import json
    try:
        return json.loads(obj)
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe_json_dumps(obj: Any) -> str:
    """JSON dumps with fallback."""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
