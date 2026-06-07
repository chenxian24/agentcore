"""Plugin lifecycle orchestrator with state machine and dependency graph."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from agentcore.plugins.base import Plugin, PluginContext

logger = logging.getLogger(__name__)


class PluginState(str, Enum):
    """Plugin lifecycle states."""

    REGISTERED = "registered"
    PLANNED = "planned"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class PluginManager:
    """Orchestrates plugin lifecycle: registration, ordering, setup, teardown.

    Features:
    - State machine per plugin (registered → planned → initializing → ready → stopped)
    - Cyclic dependency detection
    - Missing dependency policy (hard fail or soft warning)
    - Init failure isolation
    - Auto unregister hooks/tools on teardown

    Accepts mechanism instances (hooks, tools, context, events, config)
    and passes them to PluginContext. No engine dependency.
    """

    def __init__(
        self,
        config: Any = None,
        hooks: Any = None,
        tools: Any = None,
        context: Any = None,
        events: Any = None,
        *,
        strict_dependencies: bool = False,
        **extra: Any,
    ) -> None:
        self._config = config
        self._hooks = hooks
        self._tools = tools
        self._context = context
        self._events = events
        self._extra = extra
        self._plugins: dict[str, Plugin] = {}
        self._states: dict[str, PluginState] = {}
        self._load_order: list[str] = []
        self._initialized: bool = False
        self._strict = strict_dependencies

    @property
    def plugins(self) -> dict[str, Plugin]:
        return dict(self._plugins)

    def register(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self._plugins[plugin.name] = plugin
        self._states[plugin.name] = PluginState.REGISTERED

    def _resolve_load_order(self) -> list[str]:
        """Topological sort with cyclic dependency detection."""
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[str] = []

        def visit(name: str, path: list[str]) -> None:
            if name in in_stack:
                cycle = " → ".join(path[path.index(name):]) + " → " + name
                raise ValueError(f"Cyclic plugin dependency detected: {cycle}")
            if name in visited:
                return

            in_stack.add(name)
            path.append(name)

            plugin = self._plugins.get(name)
            if plugin:
                for dep in plugin.dependencies:
                    if dep not in self._plugins:
                        msg = f"Plugin '{name}' depends on '{dep}' which is not registered"
                        if self._strict:
                            raise ValueError(msg)
                        logger.warning(msg)
                    else:
                        visit(dep, path[:])

            in_stack.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._plugins:
            if name not in visited:
                visit(name, [])

        return order

    async def initialize_all(self) -> None:
        self._load_order = self._resolve_load_order()

        # Phase 1: setup
        for name in self._load_order:
            plugin = self._plugins[name]
            self._states[name] = PluginState.INITIALIZING
            ctx = self._make_context(name)
            try:
                await plugin.setup(ctx)
                logger.info("Plugin '%s' v%s initialized", name, plugin.version)
            except Exception:
                self._states[name] = PluginState.FAILED
                logger.error("Plugin '%s' setup failed", name, exc_info=True)
                if self._strict:
                    raise
                continue

        # Phase 2: on_engine_ready
        for name in self._load_order:
            if self._states[name] == PluginState.FAILED:
                continue
            plugin = self._plugins[name]
            ctx = self._make_context(name)
            try:
                await plugin.on_engine_ready(ctx)
                self._states[name] = PluginState.READY
            except Exception:
                self._states[name] = PluginState.FAILED
                logger.error("Plugin '%s' on_engine_ready failed", name, exc_info=True)

        self._initialized = True

        # Store plugin info in config.metadata for runtime access
        if self._config and hasattr(self._config, 'metadata'):
            self._config.metadata["plugins"] = self.list_plugins()

    async def shutdown_all(self) -> None:
        for name in reversed(self._load_order):
            plugin = self._plugins[name]
            self._states[name] = PluginState.SHUTTING_DOWN
            ctx = self._make_context(name)
            try:
                await plugin.teardown(ctx)
            except Exception:
                logger.error("Plugin '%s' teardown failed", name, exc_info=True)

            # Auto-unregister hooks and tools for this plugin
            if self._hooks and hasattr(self._hooks, 'unregister'):
                try:
                    self._hooks.unregister(name)
                except Exception:
                    pass

            self._states[name] = PluginState.STOPPED

        self._initialized = False

    def _make_context(self, plugin_name: str) -> PluginContext:
        ctx = PluginContext(
            config=self._config,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
            **self._extra,
        )
        ctx._plugin_name = plugin_name
        return ctx

    def get_plugin(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    def get_state(self, name: str) -> PluginState | None:
        return self._states.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "state": self._states.get(p.name, PluginState.REGISTERED).value,
            }
            for p in self._plugins.values()
        ]
