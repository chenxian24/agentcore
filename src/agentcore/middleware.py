"""Composable middleware stack for the LLM call path.

Middleware wraps the full request/response cycle, allowing plugins to:
- Transform messages before sending to the LLM
- Transform responses after receiving from the LLM
- Add logging, metrics, retry logic, etc.

Usage:
    class MyMiddleware(Middleware):
        async def process(self, context: MiddlewareContext, next_fn: NextFn) -> LLMResponse:
            # Before LLM call
            context.messages.append(LLMMessage.system("extra instruction"))
            # Call next middleware / actual LLM
            response = await next_fn(context)
            # After LLM call
            return response

    stack = MiddlewareStack()
    stack.add(MyMiddleware())
    response = await stack.execute(context, provider.chat)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agentcore.models.base import ChatParams, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class MiddlewareContext:
    """Mutable context passed through the middleware chain."""
    messages: list[LLMMessage]
    params: ChatParams
    metadata: dict[str, Any] = field(default_factory=dict)


# Type for the "next" function in the chain
NextFn = Callable[[MiddlewareContext], Awaitable[LLMResponse]]


class Middleware(ABC):
    """Base class for middleware.

    Override `process` to wrap the LLM call. Call `await next_fn(context)`
    to proceed to the next middleware (or the actual LLM call).
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def process(self, context: MiddlewareContext, next_fn: NextFn) -> LLMResponse:
        """Process the request. Call next_fn to continue the chain."""


class MiddlewareStack:
    """Ordered stack of middleware that wraps LLM calls.

    Middleware is executed in order: first added = outermost wrapper.
    """

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    def add(self, middleware: Middleware) -> None:
        """Add middleware to the stack (appended, so it wraps inner middleware)."""
        self._middleware.append(middleware)

    def remove(self, name: str) -> bool:
        """Remove middleware by name. Returns True if found."""
        for i, m in enumerate(self._middleware):
            if m.name == name:
                self._middleware.pop(i)
                return True
        return False

    def list_middleware(self) -> list[str]:
        """List middleware names in execution order."""
        return [m.name for m in self._middleware]

    async def execute(
        self,
        context: MiddlewareContext,
        handler: Callable[[list[LLMMessage], ChatParams], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        """Execute the middleware chain, ending with the actual handler.

        Args:
            context: The middleware context with messages and params.
            handler: The actual LLM call function (e.g. provider.chat).

        Returns:
            The LLMResponse from the chain.
        """
        if not self._middleware:
            return await handler(context.messages, context.params)

        # Build the chain from inside out
        async def _handler(ctx: MiddlewareContext) -> LLMResponse:
            return await handler(ctx.messages, ctx.params)

        chain = _handler
        for mw in reversed(self._middleware):
            next_fn = chain
            current_mw = mw

            def _make_wrapper(middleware: Middleware, next_fn: NextFn) -> NextFn:
                async def wrapper(ctx: MiddlewareContext) -> LLMResponse:
                    return await middleware.process(ctx, next_fn)
                return wrapper

            chain = _make_wrapper(current_mw, next_fn)

        return await chain(context)


# --- Built-in middleware ---

class LoggingMiddleware(Middleware):
    """Logs LLM calls with timing and token counts."""

    @property
    def name(self) -> str:
        return "logging"

    async def process(self, context: MiddlewareContext, next_fn: NextFn) -> LLMResponse:
        import time
        msg_count = len(context.messages)
        logger.info("LLM call: %d messages, model=%s", msg_count, context.params.model)
        start = time.monotonic()

        response = await next_fn(context)

        elapsed = time.monotonic() - start
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info("LLM response: %.2fs, %d tokens, tool_calls=%d",
                     elapsed, tokens, len(response.tool_calls or []))
        return response


class MessageTransformMiddleware(Middleware):
    """Applies a transform function to messages before the LLM call."""

    def __init__(self, transform: Callable[[list[LLMMessage]], list[LLMMessage]]) -> None:
        self._transform = transform

    @property
    def name(self) -> str:
        return "message_transform"

    async def process(self, context: MiddlewareContext, next_fn: NextFn) -> LLMResponse:
        context.messages = self._transform(context.messages)
        return await next_fn(context)
