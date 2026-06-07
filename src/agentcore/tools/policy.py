"""Tool policy pipeline: layered allow/deny/ask evaluation before tool execution."""

from __future__ import annotations

import fnmatch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PolicyDecision(str, Enum):
    """Decision a policy can make about a tool call."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # Ask user for confirmation


@dataclass
class PolicyContext:
    """Standard context for policy evaluation.

    Provides rich information about the current execution environment
    so policies can make informed decisions beyond just tool name matching.
    """

    sender_id: str = ""
    session_id: str = ""
    is_subagent: bool = False
    sandbox_mode: str = ""
    provider_name: str = ""
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolPolicy(ABC):
    """Base class for tool policies.

    A policy evaluates a tool call and returns a decision, or None
    to express no opinion (defer to the next policy in the pipeline).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision | None:
        """Evaluate the tool call. Return None to pass to next policy."""
        ...


@dataclass
class PatternPolicy(ToolPolicy):
    """Pattern-based policy: matches tool names against glob patterns.

    Examples:
        PatternPolicy("safety", deny_patterns=["execute_command", "rm_*"])
        PatternPolicy("approval", ask_patterns=["write_file", "delete_*"])
    """

    _name: str = "pattern"
    allow_patterns: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)
    ask_patterns: list[str] = field(default_factory=list)
    priority: int = 100

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision | None:
        for pattern in self.deny_patterns:
            if fnmatch.fnmatch(tool_name, pattern):
                return PolicyDecision.DENY
        for pattern in self.ask_patterns:
            if fnmatch.fnmatch(tool_name, pattern):
                return PolicyDecision.ASK
        for pattern in self.allow_patterns:
            if fnmatch.fnmatch(tool_name, pattern):
                return PolicyDecision.ALLOW
        return None


class ContextAwarePolicy(ToolPolicy):
    """Base class for policies that need rich execution context.

    Subclasses receive a PolicyContext with sender, session, sandbox,
    and provider information for more nuanced decisions.
    """

    @abstractmethod
    def evaluate_with_context(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: PolicyContext,
    ) -> PolicyDecision | None:
        """Evaluate with full PolicyContext."""
        ...

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision | None:
        """Adapter: converts dict context to PolicyContext and delegates."""
        if context is None:
            pc = PolicyContext()
        elif isinstance(context, PolicyContext):
            pc = context
        else:
            pc = PolicyContext(
                sender_id=context.get("sender_id", ""),
                session_id=context.get("session_id", ""),
                is_subagent=context.get("is_subagent", False),
                sandbox_mode=context.get("sandbox_mode", ""),
                provider_name=context.get("provider_name", ""),
                model_name=context.get("model_name", ""),
                metadata={k: v for k, v in context.items() if k not in {
                    "sender_id", "session_id", "is_subagent",
                    "sandbox_mode", "provider_name", "model_name",
                }},
            )
        return self.evaluate_with_context(tool_name, arguments, pc)


class PolicyPipeline:
    """Chains multiple policies. First non-None decision wins.

    Usage:
        pipeline = PolicyPipeline()
        pipeline.add(PatternPolicy("safety", deny_patterns=["rm_*"]))
        pipeline.add(PatternPolicy("review", ask_patterns=["write_*"]))
        decision = pipeline.evaluate("write_file", {"path": "/tmp/x"})
    """

    def __init__(self, default: PolicyDecision = PolicyDecision.ALLOW) -> None:
        self._policies: list[ToolPolicy] = []
        self._default = default

    def add(self, policy: ToolPolicy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: getattr(p, "priority", 100))

    def remove(self, name: str) -> None:
        self._policies = [p for p in self._policies if p.name != name]

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        context: PolicyContext | dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate all policies in order. First non-None decision wins.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool call arguments.
            context: PolicyContext or dict with execution context.
        """
        # Normalize context to dict for policies that expect it
        ctx_dict: dict[str, Any] | None = None
        if isinstance(context, PolicyContext):
            ctx_dict = {
                "sender_id": context.sender_id,
                "session_id": context.session_id,
                "is_subagent": context.is_subagent,
                "sandbox_mode": context.sandbox_mode,
                "provider_name": context.provider_name,
                "model_name": context.model_name,
                **context.metadata,
            }
        elif isinstance(context, dict):
            ctx_dict = context

        for policy in self._policies:
            decision = policy.evaluate(tool_name, arguments or {}, ctx_dict)
            if decision is not None:
                logger.debug(
                    "Policy '%s' decided %s for tool '%s'",
                    policy.name, decision.value, tool_name,
                )
                return decision
        return self._default

    def list_policies(self) -> list[dict[str, Any]]:
        return [{"name": p.name, "priority": getattr(p, "priority", 100)} for p in self._policies]
