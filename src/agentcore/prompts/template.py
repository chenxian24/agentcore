"""Prompt template engine using Jinja2."""

from __future__ import annotations

from typing import Any

from jinja2 import BaseLoader, Environment, TemplateSyntaxError


class PromptTemplate:
    """A prompt template that supports Jinja2 syntax."""

    def __init__(self, template: str) -> None:
        self._raw = template
        self._env = Environment(loader=BaseLoader(), autoescape=False)
        self._template = self._env.from_string(template)

    @property
    def raw(self) -> str:
        return self._raw

    def render(self, variables: dict[str, Any] | None = None) -> str:
        """Render the template with given variables."""
        try:
            return self._template.render(**(variables or {}))
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error: {e}") from e

    def get_variables(self) -> list[str]:
        """Extract variable names from the template."""
        import re
        return list(set(re.findall(r"\{\{\s*(\w+)", self._raw)))

    def __repr__(self) -> str:
        return f"PromptTemplate({self._raw[:50]}...)"


class SystemPromptBuilder:
    """Builds system prompts from components."""

    def __init__(self, base: str = "") -> None:
        self._base = base
        self._sections: list[str] = []

    def set_base(self, base: str) -> SystemPromptBuilder:
        self._base = base
        return self

    def add_section(self, title: str, content: str) -> SystemPromptBuilder:
        self._sections.append(f"## {title}\n{content}")
        return self

    def build(self, variables: dict[str, Any] | None = None) -> str:
        parts = []
        if self._base:
            template = PromptTemplate(self._base)
            parts.append(template.render(variables))
        parts.extend(self._sections)
        return "\n\n".join(parts)
