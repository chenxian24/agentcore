"""Atomic agent engine — provider + config + session + stats.

No hooks, no events, no plugins, no tools, no context, no resilience.
Those are mechanism-layer concerns, composed externally via utils.run_loop.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from agentcore.config.schema import AgentConfig
from agentcore.core.message import Message, MessageRole
from agentcore.core.session import Session
from agentcore.core.stats import RequestStats, StatsCollector
from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ToolCall,
    ToolExecutor,
)
from agentcore.models.registry import ModelRegistry
from agentcore.utils.run_loop import run_loop


class AgentEngine:
    """Atomic engine: provider + config + session + stats.

    Responsibilities:
    - Hold the LLM provider and config
    - Single LLM calls (chat, stream)
    - Tool-call loop via utils.run_loop (no built-in mechanism dispatch)
    - Session management
    - Stats recording

    Not responsible for (handled externally):
    - Hook dispatch (use agentcore.hooks)
    - Tool registration/definition injection (use agentcore.tools)
    - Context compression (use agentcore.context)
    - Event emission (use agentcore.events)
    - Plugin lifecycle (use agentcore.plugins)
    - Retry/fallback (use agentcore.resilience / agentcore.utils.with_retry)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._model_registry = ModelRegistry()
        self._provider = provider
        self._sessions: dict[str, Session] = {}
        self._stats = StatsCollector()

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def provider(self) -> BaseLLMProvider:
        return self._ensure_provider()

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def stats(self) -> StatsCollector:
        return self._stats

    def configure(self, config: AgentConfig) -> None:
        """Update config and recreate provider."""
        self._config = config
        self._provider = self._model_registry.create_from_config(config.model)

    def _ensure_provider(self) -> BaseLLMProvider:
        if self._provider is None:
            self._provider = self._model_registry.create_from_config(self._config.model)
        return self._provider

    def _build_chat_params(self, **overrides: Any) -> ChatParams:
        """Build ChatParams from config, with optional overrides."""
        mc = self._config.model
        return ChatParams(
            model=overrides.get("model", mc.model),
            temperature=overrides.get("temperature", mc.temperature),
            max_tokens=overrides.get("max_tokens", mc.max_tokens),
            top_p=overrides.get("top_p", mc.top_p),
            stop=overrides.get("stop", mc.stop),
            seed=overrides.get("seed", mc.seed),
            frequency_penalty=overrides.get("frequency_penalty", mc.frequency_penalty),
            presence_penalty=overrides.get("presence_penalty", mc.presence_penalty),
            response_format=overrides.get("response_format", mc.response_format),
            tools=overrides.get("tools", []),
            tool_choice=overrides.get("tool_choice", ""),
            thinking=overrides.get("thinking", mc.thinking),
        )

    # --- Session management ---

    def create_session(self, session_id: str = "") -> Session:
        session = Session(
            id=session_id or Session().id,
            config=self._config,
        )
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    # --- Stats ---

    def _record_stats(
        self,
        response: LLMResponse | None,
        latency_ms: float,
        error: str = "",
    ) -> None:
        usage = response.usage if response else {}
        self._stats.record(RequestStats(
            model=response.model if response else self._config.model.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            success=error == "",
            error=error,
            finish_reason=response.finish_reason if response else "",
        ))

    # --- Core chat methods ---

    async def chat(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
    ) -> LLMResponse:
        """Single LLM call. No tool loop, no hooks, no mechanism dispatch."""
        provider = self._ensure_provider()
        if params is None:
            params = self._build_chat_params()

        t0 = time.monotonic()
        try:
            response = await provider.chat(messages=messages, params=params)
            self._record_stats(response, (time.monotonic() - t0) * 1000)
        except Exception as e:
            self._record_stats(None, (time.monotonic() - t0) * 1000, error=str(e))
            raise
        return response

    async def stream(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
    ) -> AsyncIterator[str]:
        """Single streaming LLM call. Yields text chunks."""
        provider = self._ensure_provider()
        if params is None:
            params = self._build_chat_params()

        t0 = time.monotonic()
        full = ""
        try:
            async for chunk in provider.stream_chat(messages=messages, params=params):
                full += chunk
                yield chunk
            self._record_stats(
                LLMResponse(content=full, model=self._config.model.model),
                (time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            self._record_stats(None, (time.monotonic() - t0) * 1000, error=str(e))
            raise

    async def stream_with_tools(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        tool_executor: ToolExecutor | None = None,
        *,
        max_rounds: int = 10,
        on_tool_call=None,
        on_tool_result=None,
        on_chunk=None,
    ) -> LLMResponse:
        """Streaming LLM call with tool-call loop.

        Yields text chunks while streaming, then executes tool calls and loops.
        Falls back to non-streaming if the provider doesn't support streaming.
        """
        provider = self._ensure_provider()
        if params is None:
            params = self._build_chat_params()

        # Adapt tool_executor
        if tool_executor is not None:
            if hasattr(tool_executor, 'execute'):
                async def _executor(tc: ToolCall) -> dict:
                    return await tool_executor.execute(tc)
            else:
                async def _executor(tc: ToolCall) -> dict:
                    return await tool_executor(tc)
        else:
            _executor = None

        t0 = time.monotonic()
        full_content = ""

        try:
            for _round in range(max_rounds):
                # Stream this round
                async for chunk in provider.stream_chat(messages=messages, params=params):
                    full_content += chunk
                    if on_chunk:
                        on_chunk(chunk)

                # Check if provider accumulated tool calls
                tool_calls = getattr(provider, '_last_stream_tool_calls', [])
                if not tool_calls or _executor is None:
                    break

                # Build assistant message with tool calls
                messages.append(LLMMessage(
                    role="assistant",
                    content=full_content,
                    tool_calls=tool_calls,
                ))

                # Execute each tool call
                for tc in tool_calls:
                    if on_tool_call:
                        try:
                            result = on_tool_call(tc, messages)
                            if hasattr(result, '__await__'):
                                await result
                        except Exception:
                            pass

                    tool_result = await _executor(tc)

                    if on_tool_result:
                        transformed = on_tool_result(tc, tool_result, messages)
                        if hasattr(transformed, '__await__'):
                            transformed = await transformed
                        if transformed is not None:
                            tool_result = transformed

                    messages.append(LLMMessage(
                        role="tool",
                        content=str(tool_result.get("output", tool_result.get("error", ""))),
                        tool_call_id=tc.id,
                    ))

                # Reset for next round
                full_content = ""
                provider._last_stream_tool_calls = []

            response = LLMResponse(
                content=full_content,
                model=self._config.model.model,
                tool_calls=getattr(provider, '_last_stream_tool_calls', []),
            )
            self._record_stats(response, (time.monotonic() - t0) * 1000)
        except Exception as e:
            self._record_stats(None, (time.monotonic() - t0) * 1000, error=str(e))
            raise
        return response

    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        params: ChatParams | None = None,
        tool_executor: ToolExecutor | None = None,
        *,
        max_rounds: int = 10,
        on_tool_call=None,
        on_tool_result=None,
        on_response=None,
    ) -> LLMResponse:
        """LLM call with tool-call loop via utils.run_loop.

        When tool_executor is provided, runs the tool-call loop.
        When tool_executor is None, returns tool_calls in the response
        for the caller to handle (backward compatible).

        Hooks, policies, events, etc. are passed via the callback parameters.
        """
        provider = self._ensure_provider()
        if params is None:
            params = self._build_chat_params()

        t0 = time.monotonic()
        try:
            if tool_executor is not None:
                # Adapt ToolExecutor protocol or plain callable
                if hasattr(tool_executor, 'execute'):
                    async def _executor(tc: ToolCall) -> dict:
                        return await tool_executor.execute(tc)
                else:
                    async def _executor(tc: ToolCall) -> dict:
                        return await tool_executor(tc)

                response = await run_loop(
                    provider=provider,
                    messages=messages,
                    params=params,
                    tool_executor=_executor,
                    max_rounds=max_rounds,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_response=on_response,
                )
            else:
                response = await provider.chat(messages=messages, params=params)
            self._record_stats(response, (time.monotonic() - t0) * 1000)
        except Exception as e:
            self._record_stats(None, (time.monotonic() - t0) * 1000, error=str(e))
            raise
        return response
