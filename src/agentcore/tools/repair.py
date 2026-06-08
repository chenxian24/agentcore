"""Tool-call repair — detect and fix plain-text tool calls from LLMs.

Some LLMs (especially smaller ones) emit tool calls as markdown text blocks
instead of native tool_use/function_call format. This module detects and
repairs these into proper WireToolCall objects.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentcore.wire.types import WireToolCall

# Pattern: ```tool_name\n{...}\n``` or ```tool_name\narg: value\n```
_TOOL_BLOCK_RE = re.compile(
    r"```(\w+)\s*\n(.*?)```",
    re.DOTALL,
)

# Pattern: function_name({...}) or function_name({\n...\n})
_FUNCTION_CALL_RE = re.compile(
    r"(\w+)\s*\((\{.*?\})\)",
    re.DOTALL,
)


class ToolCallRepairer:
    """Detect and repair plain-text tool calls into WireToolCall objects.

    Handles common patterns:
    1. Markdown code blocks: ```tool_name\n{"arg": "val"}\n```
    2. Function call style: tool_name({"arg": "val"})
    3. JSON with tool name: {"name": "tool_name", "arguments": {...}}
    """

    def __init__(self, known_tools: set[str] | None = None) -> None:
        """
        Args:
            known_tools: Set of known tool names. If provided, only matches
                against these names. If None, matches any identifier.
        """
        self._known_tools = known_tools

    def detect(self, text: str) -> list[WireToolCall] | None:
        """Detect tool calls in plain text. Returns None if none found."""
        results: list[WireToolCall] = []

        # Try markdown code block pattern
        for match in _TOOL_BLOCK_RE.finditer(text):
            tool_name = match.group(1).strip()
            body = match.group(2).strip()
            if self._is_tool_name(tool_name):
                tc = self._parse_args(tool_name, body)
                if tc:
                    results.append(tc)

        # Try function call pattern
        if not results:
            for match in _FUNCTION_CALL_RE.finditer(text):
                tool_name = match.group(1).strip()
                args_str = match.group(2).strip()
                if self._is_tool_name(tool_name):
                    tc = self._parse_args(tool_name, args_str)
                    if tc:
                        results.append(tc)

        return results if results else None

    def _is_tool_name(self, name: str) -> bool:
        if self._known_tools is not None:
            return name in self._known_tools
        # Heuristic: tool names are lowercase with underscores/hyphens
        return bool(re.match(r"^[a-z][a-z0-9_-]*$", name))

    def _parse_args(self, tool_name: str, body: str) -> WireToolCall | None:
        """Try to parse the body as JSON arguments."""
        try:
            args = json.loads(body)
            if isinstance(args, dict):
                return WireToolCall(
                    id=f"repaired_{tool_name}",
                    name=tool_name,
                    arguments=json.dumps(args, ensure_ascii=False),
                )
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to parse as key-value pairs
        try:
            pairs = {}
            for line in body.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    pairs[key.strip()] = value.strip().strip('"').strip("'")
            if pairs:
                return WireToolCall(
                    id=f"repaired_{tool_name}",
                    name=tool_name,
                    arguments=json.dumps(pairs, ensure_ascii=False),
                )
        except Exception:
            pass

        return None
