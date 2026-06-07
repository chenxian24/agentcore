"""Async event bus with topic-based pub/sub and wildcard support."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from fnmatch import fnmatch

logger = logging.getLogger(__name__)

EventHandler = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    """An event published on the bus."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    _handled: bool = field(default=False, repr=False)

    def stop_propagation(self) -> None:
        """Stop further handlers from processing this event."""
        self._handled = True


class EventBus:
    """Async event bus with topic-based subscriptions and wildcard patterns.

    Topics use dot notation: "tool.executed", "session.message.added"
    Wildcards: "tool.*" matches "tool.executed", "tool.failed", etc.

    Supports:
        - Exact topic matching
        - Wildcard subscriptions (fnmatch patterns)
        - Priority-ordered handlers
        - Async handlers
        - Event propagation control (stop_propagation)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[tuple[int, str, EventHandler]]] = {}
        self._history: list[Event] = []
        self._max_history = 1000

    def on(
        self,
        topic: str,
        handler: EventHandler,
        priority: int = 100,
        subscriber: str = "",
    ) -> None:
        """Subscribe to a topic (exact or wildcard pattern)."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append((priority, subscriber, handler))
        self._handlers[topic].sort(key=lambda x: x[0])

    def off(self, topic: str, handler: EventHandler | None = None, subscriber: str = "") -> None:
        """Unsubscribe from a topic. If handler is None, remove all for subscriber."""
        if topic not in self._handlers:
            return
        if handler is not None:
            self._handlers[topic] = [
                (p, s, h) for p, s, h in self._handlers[topic] if h != handler
            ]
        elif subscriber:
            self._handlers[topic] = [
                (p, s, h) for p, s, h in self._handlers[topic] if s != subscriber
            ]
        else:
            del self._handlers[topic]

    def off_all(self, subscriber: str) -> None:
        """Remove all handlers registered by a subscriber."""
        for topic in list(self._handlers.keys()):
            self._handlers[topic] = [
                (p, s, h) for p, s, h in self._handlers[topic] if s != subscriber
            ]
            if not self._handlers[topic]:
                del self._handlers[topic]

    async def emit(self, event: Event) -> Event:
        """Emit an event. Handlers run in priority order; stop_propagation halts the chain."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Collect matching handlers (exact match + wildcard match)
        matched: list[tuple[int, str, EventHandler]] = []
        for pattern, handlers in self._handlers.items():
            if pattern == event.topic or fnmatch(event.topic, pattern):
                matched.extend(handlers)

        # Sort by priority
        matched.sort(key=lambda x: x[0])

        for _priority, subscriber, handler in matched:
            if event._handled:
                break
            try:
                await handler(event)
            except Exception:
                logger.error(
                    "Event handler error: topic=%s subscriber=%s",
                    event.topic, subscriber, exc_info=True,
                )

        return event

    def emit_sync(self, event: Event) -> asyncio.Task[Event]:
        """Emit an event without awaiting (fire-and-forget)."""
        return asyncio.create_task(self.emit(event))

    def has_handlers(self, topic: str) -> bool:
        """Check if any handlers match a topic (exact or wildcard)."""
        for pattern in self._handlers:
            if pattern == topic or fnmatch(topic, pattern):
                if self._handlers[pattern]:
                    return True
        return False

    def list_topics(self) -> dict[str, list[str]]:
        """List all registered topics and their subscribers."""
        return {
            topic: [s for _, s, _ in handlers]
            for topic, handlers in self._handlers.items()
        }

    def get_history(self, topic: str | None = None, limit: int = 50) -> list[Event]:
        """Get recent events, optionally filtered by topic."""
        events = self._history
        if topic:
            events = [e for e in events if fnmatch(e.topic, topic)]
        return events[-limit:]
