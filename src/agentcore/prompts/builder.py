"""Dynamic system prompt builder that assembles context from tools, plugins, and config."""

from __future__ import annotations

from typing import Any

from agentcore.prompts.template import PromptTemplate, SystemPromptBuilder


class DynamicPromptBuilder:
    """Builds a system prompt dynamically from engine state.

    Gathers tool descriptions from ToolRegistry, formats them as text,
    and combines with base prompt and additional sections.

    Usage:
        builder = DynamicPromptBuilder(engine)
        prompt = builder.build()
    """

    def __init__(
        self,
        tool_registry: Any = None,
        base_prompt: str = "You are a helpful assistant.",
        identity: str = "",
        extra_sections: dict[str, str] | None = None,
    ) -> None:
        self._tools = tool_registry
        self._base = base_prompt
        self._identity = identity
        self._extra = extra_sections or {}

    def build(self, variables: dict[str, Any] | None = None) -> str:
        builder = SystemPromptBuilder()

        # Base prompt with variable substitution
        base = self._base
        if variables:
            template = PromptTemplate(base)
            base = template.render(variables)
        builder.set_base(base)

        # Identity section
        if self._identity:
            builder.add_section("Identity", self._identity)

        # Tools section
        if self._tools:
            tool_defs = self._tools.get_tool_definitions()
            if tool_defs:
                tools_text = self._format_tools(tool_defs)
                builder.add_section("Available Tools", tools_text)

        # Extra sections
        for title, content in self._extra.items():
            builder.add_section(title, content)

        return builder.build()

    def _format_tools(self, tool_defs: list[dict[str, Any]]) -> str:
        """Format tool definitions as human-readable text for the system prompt."""
        lines = []
        for td in tool_defs:
            func = td.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            properties = params.get("properties", {})
            required = params.get("required", [])

            line = f"- **{name}**: {desc}"
            if properties:
                param_parts = []
                for pname, pinfo in properties.items():
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    req_mark = " (required)" if pname in required else ""
                    param_parts.append(f"  - `{pname}` ({ptype}){req_mark}: {pdesc}")
                line += "\n" + "\n".join(param_parts)
            lines.append(line)
        return "\n".join(lines)
