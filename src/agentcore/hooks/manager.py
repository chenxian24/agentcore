"""Hook registration and dispatch manager."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from agentcore.hooks.types import HookContext, HookName, HookResult

logger = logging.getLogger(__name__)

HookHandler = Callable[[HookContext], Awaitable[HookResult | None]]


def _hook_key(hook_name: HookName | str) -> str:
    """Normalize hook name to string key."""
    if isinstance(hook_name, HookName):
        return hook_name.value
    return hook_name


class HookManager:
    """Manages hook registration and dispatch.

    Hooks are ordered by priority (lower = earlier). Multiple hooks on the
    same point form a chain; each receives the (possibly mutated) context.
    If any hook sets ctx.cancel=True, the chain short-circuits.

    Supports both HookName enum values and arbitrary string hook names,
    allowing plugins to define custom hook points.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[tuple[int, str, HookHandler]]] = defaultdict(list)

    def register(
        self,
        hook_name: HookName | str,
        handler: HookHandler,
        plugin_name: str = "",
        priority: int = 100,
    ) -> None:
        """Register a handler for a hook point.

        Args:
            hook_name: HookName enum value or arbitrary string hook name.
            handler: Async callable receiving HookContext.
            plugin_name: Name of the registering plugin (for unregistration).
            priority: Lower values run first.
        """
        key = _hook_key(hook_name)
        self._handlers[key].append((priority, plugin_name, handler))
        self._handlers[key].sort(key=lambda x: x[0])

    def unregister(self, plugin_name: str) -> None:
        """Remove all handlers registered by a plugin."""
        for key in self._handlers:
            self._handlers[key] = [
                (p, n, h) for p, n, h in self._handlers[key] if n != plugin_name
            ]

    async def dispatch(self, ctx: HookContext) -> HookContext:
        """Run all handlers for ctx.name in priority order.

        Returns the (possibly mutated) context. If any hook sets
        ctx.cancel=True, the chain stops immediately.
        """
        key = _hook_key(ctx.name)
        for _priority, plugin_name, handler in self._handlers.get(key, []):
            if ctx.cancel:
                break
            try:
                result = await handler(ctx)
                if result and not result.success:
                    logger.warning(
                        "Hook %s from '%s' returned failure: %s",
                        key, plugin_name, result.error,
                    )
            except Exception:
                logger.error(
                    "Hook %s from '%s' raised an exception",
                    key, plugin_name, exc_info=True,
                )
        return ctx

    def has_hooks(self, hook_name: HookName | str) -> bool:
        """Check if any handlers are registered for the given hook point."""
        key = _hook_key(hook_name)
        return bool(self._handlers.get(key))

    def list_hooks(self) -> dict[str, list[str]]:
        """Return a summary of registered hooks: {hook_name: [plugin_names]}."""
        return {
            key: [n for _, n, _ in handlers]
            for key, handlers in self._handlers.items()
            if handlers
        }
