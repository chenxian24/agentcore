"""Tool call loop — the pure core primitive of an agent.

No hooks, no events, no policies, no plugins. Just:
  provider.chat(messages) → tool_calls → tool_executor(tc) → append → repeat
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ToolCall,
)

logger = logging.getLogger(__name__)

# Type alias for the tool executor function
ToolExecutorFn = Callable[[ToolCall], Awaitable[dict[str, Any]]]

# Optional callbacks for external mechanism integration
OnToolCallFn = Callable[[ToolCall, list[LLMMessage]], Awaitable[None] | None]
OnToolResultFn = Callable[[ToolCall, dict[str, Any], list[LLMMessage]], Awaitable[dict[str, Any]] | dict[str, Any] | None]
OnResponseFn = Callable[[LLMResponse, list[LLMMessage]], Awaitable[None] | None]


async def run_loop(
    provider: BaseLLMProvider,
    messages: list[LLMMessage],
    params: ChatParams,
    tool_executor: ToolExecutorFn,
    *,
    max_rounds: int = 10,
    max_workers: int = 1,
    on_tool_call: OnToolCallFn | None = None,
    on_tool_result: OnToolResultFn | None = None,
    on_response: OnResponseFn | None = None,
) -> LLMResponse:
    """Execute the tool-call loop: call model → execute tools → repeat.

    This is the irreducible agent primitive. Everything else (hooks, policies,
    events, approval) is composed externally via the callback parameters.

    Args:
        provider: LLM provider to call.
        messages: Message list (mutated in-place — tool results are appended).
        params: Chat parameters (model, temperature, tools, etc.).
        tool_executor: Async function that executes a tool call and returns a result dict.
        max_rounds: Maximum tool-call rounds before stopping.
        max_workers: Max concurrent tool executions per round. 1 = sequential (default).
        on_tool_call: Optional callback before each tool execution.
            If it raises, the tool is skipped. If it sets metadata, it flows to on_tool_result.
        on_tool_result: Optional callback after each tool execution.
            Can transform the result by returning a new dict.
        on_response: Optional callback after each LLM response.

    Returns:
        The last LLMResponse (with accumulated tool_calls from all rounds).
    """
    last_response: LLMResponse | None = None
    all_tool_calls: list[ToolCall] = []

    for _round in range(max_rounds):
        response = await provider.chat(messages=messages, params=params)
        last_response = response

        if on_response:
            result = on_response(response, messages)
            if _is_awaitable(result):
                await result

        if not response.tool_calls:
            break

        messages.append(LLMMessage(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        for tc in response.tool_calls:
            all_tool_calls.append(tc)

        if max_workers > 1 and len(response.tool_calls) > 1:
            await _execute_tools_parallel(
                response.tool_calls, messages, tool_executor,
                on_tool_call, on_tool_result,
            )
        else:
            for tc in response.tool_calls:
                await _execute_single_tool(
                    tc, messages, tool_executor,
                    on_tool_call, on_tool_result,
                )

    return last_response or LLMResponse()


async def _execute_single_tool(
    tc: ToolCall,
    messages: list[LLMMessage],
    tool_executor: ToolExecutorFn,
    on_tool_call: OnToolCallFn | None,
    on_tool_result: OnToolResultFn | None,
) -> None:
    """Execute a single tool call and append result to messages."""
    if on_tool_call:
        try:
            result = on_tool_call(tc, messages)
            if _is_awaitable(result):
                await result
        except Exception as e:
            logger.debug("on_tool_call raised for %s: %s", tc.function.name, e)
            messages.append(LLMMessage(
                role="tool",
                content=str(e),
                tool_call_id=tc.id,
            ))
            return

    tool_result = await tool_executor(tc)

    if on_tool_result:
        transformed = on_tool_result(tc, tool_result, messages)
        if _is_awaitable(transformed):
            transformed = await transformed  # type: ignore[misc]
        if transformed is not None:
            tool_result = transformed

    messages.append(LLMMessage(
        role="tool",
        content=str(tool_result.get("output", tool_result.get("error", ""))),
        tool_call_id=tc.id,
    ))


async def _execute_tools_parallel(
    tool_calls: list[ToolCall],
    messages: list[LLMMessage],
    tool_executor: ToolExecutorFn,
    on_tool_call: OnToolCallFn | None,
    on_tool_result: OnToolResultFn | None,
) -> None:
    """Execute multiple tool calls concurrently, preserving result order."""
    import asyncio

    async def _run_one(tc: ToolCall) -> tuple[ToolCall, dict[str, Any]]:
        if on_tool_call:
            try:
                result = on_tool_call(tc, messages)
                if _is_awaitable(result):
                    await result
            except Exception as e:
                logger.debug("on_tool_call raised for %s: %s", tc.function.name, e)
                return tc, {"output": str(e), "error": str(e)}

        tool_result = await tool_executor(tc)

        if on_tool_result:
            transformed = on_tool_result(tc, tool_result, messages)
            if _is_awaitable(transformed):
                transformed = await transformed  # type: ignore[misc]
            if transformed is not None:
                tool_result = transformed

        return tc, tool_result

    results = await asyncio.gather(*(_run_one(tc) for tc in tool_calls))

    for tc, tool_result in results:
        messages.append(LLMMessage(
            role="tool",
            content=str(tool_result.get("output", tool_result.get("error", ""))),
            tool_call_id=tc.id,
        ))


def _is_awaitable(obj: Any) -> bool:
    import asyncio
    return asyncio.iscoroutine(obj) or asyncio.isfuture(obj)
