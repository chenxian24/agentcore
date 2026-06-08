"""Normalized wire types — protocol-agnostic data models for LLM communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WireToolCall:
    """Normalized tool call from any protocol."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class WireResponse:
    """Normalized non-streaming response from any protocol."""

    text: str = ""
    tool_calls: list[WireToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WireEvent:
    """Normalized streaming event from any protocol."""

    type: str  # "text_delta" | "tool_call_start" | "tool_call_delta" | "done"
    delta: str = ""
    tool_call: WireToolCall | None = None
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
