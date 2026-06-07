"""Tool Execution Pipeline — standardized tool call processing.

Pipeline stages:
  ToolCall → PolicyPipeline → ApprovalGateway → ToolExecutor → ResultTransformer → EventEmitter

This replaces scattered tool execution logic in runners.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from agentcore.models.base import ToolCall
from agentcore.tools.policy import PolicyContext, PolicyDecision, PolicyPipeline
from agentcore.tools.registry import ToolRegistry
from agentcore.tools.result import ToolResult

logger = logging.getLogger(__name__)

# Type aliases
ApprovalHandler = Callable[[ToolCall, dict[str, Any]], Awaitable[bool]]
ResultTransformer = Callable[[ToolCall, ToolResult], Awaitable[ToolResult]]
ToolEventEmitter = Callable[[str, dict[str, Any]], Awaitable[None]]


class ToolPipeline:
    """Standardized tool execution pipeline.

    Replaces the ad-hoc _tool_executor closures in runners.
    Stages: policy check → approval → execute → transform → emit.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyPipeline | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or PolicyPipeline()
        self._approval_handler: ApprovalHandler | None = None
        self._transformers: list[ResultTransformer] = []
        self._event_emitter: ToolEventEmitter | None = None

    def set_approval_handler(self, handler: ApprovalHandler) -> None:
        """Set the approval handler for ASK policy decisions."""
        self._approval_handler = handler

    def add_transformer(self, transformer: ResultTransformer) -> None:
        """Add a result transformer to the pipeline."""
        self._transformers.append(transformer)

    def set_event_emitter(self, emitter: ToolEventEmitter) -> None:
        """Set the event emitter for tool lifecycle events."""
        self._event_emitter = emitter

    async def execute(
        self,
        tool_call: ToolCall,
        policy_context: PolicyContext | dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute a tool call through the full pipeline.

        Args:
            tool_call: The tool call to execute.
            policy_context: Context for policy evaluation.

        Returns:
            ToolResult with the execution outcome.
        """
        tool_name = tool_call.function.name

        # Parse arguments
        try:
            arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError:
            return ToolResult.fail(f"Invalid JSON arguments for {tool_name}")

        # Stage 1: Policy evaluation
        decision = self._policy.evaluate(tool_name, arguments, policy_context)
        if decision == PolicyDecision.DENY:
            return ToolResult.fail(f"Tool {tool_name} denied by policy")
        if decision == PolicyDecision.ASK:
            if self._approval_handler:
                approved = await self._approval_handler(tool_call, arguments)
                if not approved:
                    return ToolResult.fail(f"Tool {tool_name} not approved")

        # Emit tool_start event
        if self._event_emitter:
            await self._event_emitter("tool.start", {
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call.id,
            })

        # Stage 2: Execute
        try:
            raw_result = await self._registry.execute(tool_call)
            result = ToolResult.from_dict(raw_result)
        except Exception as e:
            logger.error("Tool execution error: %s — %s", tool_name, e, exc_info=True)
            result = ToolResult.fail(str(e))

        # Stage 3: Transform results
        for transformer in self._transformers:
            try:
                result = await transformer(tool_call, result)
            except Exception as e:
                logger.warning("Result transformer error: %s", e)

        # Emit tool_end event
        if self._event_emitter:
            await self._event_emitter("tool.end", {
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            })

        return result
