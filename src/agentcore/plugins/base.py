"""Plugin base class and context facade."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentcore.hooks.types import HookName


class PluginContext:
    """Facade provided to plugins during setup.

    Exposes mechanism instances directly — no engine reference.
    Plugins register hooks, tools, and commands through this interface.
    """

    def __init__(
        self,
        config: Any,
        hooks: Any = None,
        tools: Any = None,
        context: Any = None,
        events: Any = None,
        model_registry: Any = None,
        **extra: Any,
    ) -> None:
        self._config = config
        self._hooks = hooks
        self._tools = tools
        self._context = context
        self._events = events
        self._model_registry = model_registry
        self._extra = extra
        self._plugin_name: str = ""
        self._session: Any = None

    def register_hook(
        self,
        hook_name: HookName | str,
        handler: Any,
        priority: int = 100,
    ) -> None:
        if self._hooks:
            self._hooks.register(hook_name, handler, plugin_name=self._plugin_name, priority=priority)

    def register_tool(
        self,
        name: str,
        handler: Any,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        check_fn: Any = None,
    ) -> None:
        if self._tools:
            self._tools.register(name, handler, description, parameters, check_fn)

    def register_command(self, name: str, handler: Any, description: str = "") -> None:
        commands = self._config.metadata.setdefault("commands", {})
        commands[name] = {"handler": handler, "description": description}

    @property
    def config(self) -> Any:
        return self._config

    @property
    def hooks(self) -> Any:
        return self._hooks

    @property
    def tools(self) -> Any:
        return self._tools

    @property
    def context(self) -> Any:
        return self._context

    @property
    def events(self) -> Any:
        return self._events

    @property
    def session(self) -> Any:
        """The current active session. Set by the runtime after session creation."""
        return self._session

    def set_session(self, session: Any) -> None:
        """Set the active session reference (called by runtime)."""
        self._session = session

    def register_provider(self, name: str, factory: Any) -> None:
        """Register an LLM provider factory in the model registry."""
        if self._model_registry:
            self._model_registry.register_factory(name, factory)

    def get(self, key: str) -> Any:
        """Access extra context values (e.g. mcp_manager, provider)."""
        return self._extra.get(key)

    def register_tokenizer(self, tokenizer: Any) -> None:
        """Register a custom tokenizer for accurate token estimation.

        The tokenizer is stored in config.metadata["_tokenizer"] and can be
        used by the context engine and other components.
        """
        self._config.metadata["_tokenizer"] = tokenizer


class Plugin(ABC):
    """Base class for agentcore plugins.

    Lifecycle:
        1. __init__() -- lightweight, no I/O
        2. setup(PluginContext) -- register hooks, tools, commands
        3. on_engine_ready(PluginContext) -- called after all plugins set up
        4. teardown(PluginContext) -- cleanup on shutdown
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return ""

    @property
    def dependencies(self) -> list[str]:
        return []

    @abstractmethod
    async def setup(self, ctx: PluginContext) -> None: ...

    async def on_engine_ready(self, ctx: PluginContext) -> None:
        pass

    async def teardown(self, ctx: PluginContext) -> None:
        pass
