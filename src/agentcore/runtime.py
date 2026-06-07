"""AgentRuntime — standard agent composition layer.

Sits between AgentEngine (atomic) and runners (application).
Owns: hooks, tools, context, events, plugins, prompt, tool pipeline.
Runners delegate orchestration to AgentRuntime and only handle I/O.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Callable, Awaitable

from agentcore.config.schema import AgentConfig, SystemPromptConfig
from agentcore.context.engine import ContextEngine
from agentcore.core.adapter import MessageAdapter
from agentcore.core.engine import AgentEngine
from agentcore.core.message import Message, MessageRole
from agentcore.core.session import Session
from agentcore.events.bus import Event, EventBus
from agentcore.hooks.manager import HookManager
from agentcore.hooks.types import HookContext, HookName
from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ToolCall,
)
from agentcore.models.capabilities import ProviderCapabilities
from agentcore.models.events import StreamEvent, StreamEventType
from agentcore.plugins.base import Plugin, PluginContext
from agentcore.plugins.manager import PluginManager
from agentcore.tools.pipeline import ToolPipeline
from agentcore.tools.policy import PolicyContext, PolicyPipeline
from agentcore.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Standard agent composition layer.

    Responsibilities:
    - Bootstrap mechanism layer (hooks, tools, context, events, plugins)
    - Manage plugin lifecycle
    - Build system prompt
    - Execute chat pipeline (PRE_BUILD → compress → LLM → TRANSFORM → store)
    - Execute tool pipeline (PRE_TOOL → execute → POST_TOOL → transform)
    - Provide chat() and stream_events() high-level methods

    Not responsible for:
    - LLM provider implementation (that's BaseLLMProvider)
    - UI/CLI rendering (that's the runner)
    - Platform-specific I/O (that's the runner)
    """

    def __init__(
        self,
        engine: AgentEngine,
        hooks: HookManager | None = None,
        tools: ToolRegistry | None = None,
        context: ContextEngine | None = None,
        events: EventBus | None = None,
        plugins: PluginManager | None = None,
        policy: PolicyPipeline | None = None,
    ) -> None:
        self._engine = engine
        self._hooks = hooks or HookManager()
        self._tools = tools or ToolRegistry()
        self._context = context or ContextEngine()
        self._events = events or EventBus()
        self._policy = policy or PolicyPipeline()
        self._plugins = plugins or PluginManager(
            config=engine.config,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
        )
        self._tool_pipeline = ToolPipeline(self._tools, self._policy)
        self._session: Session | None = None
        self._initialized = False

    # --- Properties ---

    @property
    def engine(self) -> AgentEngine:
        return self._engine

    @property
    def hooks(self) -> HookManager:
        return self._hooks

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def context(self) -> ContextEngine:
        return self._context

    @property
    def events(self) -> EventBus:
        return self._events

    @property
    def plugins(self) -> PluginManager:
        return self._plugins

    @property
    def policy(self) -> PolicyPipeline:
        return self._policy

    @property
    def tool_pipeline(self) -> ToolPipeline:
        return self._tool_pipeline

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def config(self) -> AgentConfig:
        return self._engine.config

    # --- Lifecycle ---

    async def initialize(self, plugins: list[Plugin] | None = None) -> None:
        """Initialize the runtime: register plugins, setup hooks/tools/events.

        Args:
            plugins: Optional list of plugins to register before initialization.
        """
        if plugins:
            for plugin in plugins:
                self._plugins.register(plugin)

        # Wire up tool pipeline event emitter
        self._tool_pipeline.set_event_emitter(self._emit_tool_event)

        # Create plugin context
        ctx = PluginContext(
            config=self._engine.config,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
            model_registry=self._engine.model_registry,
        )

        # Let plugins register tools/hooks
        for plugin in self._plugins.plugins.values():
            try:
                await plugin.setup(ctx)
            except Exception as e:
                logger.error("Plugin setup failed: %s — %s", plugin.name, e)

        # Engine init hook
        await self._hooks.dispatch(HookContext(name=HookName.ENGINE_INIT, engine=self._engine))

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the runtime: teardown plugins, dispatch shutdown hooks."""
        await self._hooks.dispatch(HookContext(name=HookName.ENGINE_SHUTDOWN, engine=self._engine))
        await self._plugins.shutdown_all()
        self._initialized = False

    # --- Session management ---

    def create_session(self, session_id: str = "") -> Session:
        """Create a new session and dispatch SESSION_START."""
        self._session = self._engine.create_session(session_id)
        return self._session

    async def start_session(self, session_id: str = "") -> Session:
        """Create session and dispatch SESSION_START hook."""
        session = self.create_session(session_id)
        await self._hooks.dispatch(HookContext(
            name=HookName.SESSION_START,
            engine=self._engine,
            session=session,
        ))
        return session

    async def end_session(self) -> None:
        """Dispatch SESSION_END hook."""
        if self._session:
            await self._hooks.dispatch(HookContext(
                name=HookName.SESSION_END,
                engine=self._engine,
                session=self._session,
            ))

    # --- Chat pipeline ---

    async def chat(
        self,
        user_input: str,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[ToolCall, dict], Awaitable[None]] | None = None,
        on_response: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Full chat pipeline: PRE_BUILD → add user msg → compress → LLM → TRANSFORM → store.

        Returns the assistant's response text.
        """
        session = self._session
        if not session:
            raise RuntimeError("No active session. Call create_session() first.")

        config = self._engine.config

        # Stage 1: PRE_BUILD_MESSAGES hook
        ctx = HookContext(
            name=HookName.PRE_BUILD_MESSAGES,
            engine=self._engine,
            session=session,
            user_input=user_input,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
        )
        await self._hooks.dispatch(ctx)
        if ctx.cancel:
            return ctx.metadata.get("cancel_reason", "Cancelled by hook")

        # Stage 2: Add user message
        session.add_message(Message.user(user_input))

        # Stage 3: Context compression
        compressed = self._context.compress(
            session.messages,
            max_tokens=config.context.max_tokens,
        )

        # Stage 4: Convert to LLM messages
        llm_messages = MessageAdapter.to_llm_list(compressed)

        # Stage 5: Build ChatParams
        tool_defs = self._tools.get_tool_definitions()
        params = ChatParams(
            model=config.model.model,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
            tools=tool_defs,
        )

        # Stage 6: LLM call with tool loop
        tool_executor = self._build_tool_executor(on_tool_call, on_tool_result)
        response = await self._engine.chat_with_tools(
            messages=llm_messages,
            params=params,
            tool_executor=tool_executor,
            max_rounds=10,
        )

        # Stage 7: TRANSFORM_LLM_OUTPUT hook
        response_content = response.content or ""
        transform_ctx = HookContext(
            name=HookName.TRANSFORM_LLM_OUTPUT,
            engine=self._engine,
            session=session,
            response=response,
            transform_result=response_content,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
        )
        await self._hooks.dispatch(transform_ctx)
        if transform_ctx.transform_result is not None:
            response_content = transform_ctx.transform_result

        # Stage 8: Store assistant message
        session.add_message(Message.assistant(response_content))

        # Notify callback
        if on_response:
            await on_response(response_content)

        return response_content

    async def stream_events(
        self,
        user_input: str,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[ToolCall, dict], Awaitable[None]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat pipeline that yields StreamEvents.

        Events: message_start, text_delta, reasoning_delta, tool_call,
                tool_result, message_done, error.
        """
        session = self._session
        if not session:
            raise RuntimeError("No active session. Call create_session() first.")

        config = self._engine.config

        # PRE_BUILD_MESSAGES
        ctx = HookContext(
            name=HookName.PRE_BUILD_MESSAGES,
            engine=self._engine,
            session=session,
            user_input=user_input,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
        )
        await self._hooks.dispatch(ctx)
        if ctx.cancel:
            yield StreamEvent.error(ctx.metadata.get("cancel_reason", "Cancelled"))
            return

        session.add_message(Message.user(user_input))

        compressed = self._context.compress(session.messages, max_tokens=config.context.max_tokens)
        llm_messages = MessageAdapter.to_llm_list(compressed)

        tool_defs = self._tools.get_tool_definitions()
        params = ChatParams(
            model=config.model.model,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
            tools=tool_defs,
        )

        yield StreamEvent.message_start()

        # Stream text chunks
        full_content = ""
        try:
            async for chunk in self._engine.stream(llm_messages, params):
                full_content += chunk
                yield StreamEvent.text(chunk)
        except Exception as e:
            yield StreamEvent.error(str(e))
            return

        # TRANSFORM_LLM_OUTPUT
        transform_ctx = HookContext(
            name=HookName.TRANSFORM_LLM_OUTPUT,
            engine=self._engine,
            session=session,
            transform_result=full_content,
            hooks=self._hooks,
            tools=self._tools,
            context=self._context,
            events=self._events,
        )
        await self._hooks.dispatch(transform_ctx)
        if transform_ctx.transform_result is not None:
            full_content = transform_ctx.transform_result

        session.add_message(Message.assistant(full_content))

        yield StreamEvent.message_done(content=full_content)

    # --- Internal helpers ---

    def _build_tool_executor(
        self,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[ToolCall, dict], Awaitable[None]] | None = None,
    ) -> Any:
        """Build a tool executor function for chat_with_tools."""
        runtime = self

        async def _executor(tc: ToolCall) -> dict:
            # Notify callback
            if on_tool_call:
                await on_tool_call(tc)

            # Build policy context
            policy_ctx = PolicyContext(
                session_id=runtime.session.id if runtime.session else "",
                provider_name=runtime.config.model.provider,
                model_name=runtime.config.model.model,
            )

            # Execute through pipeline
            result = await runtime._tool_pipeline.execute(tc, policy_ctx)

            # Notify callback
            if on_tool_result:
                await on_tool_result(tc, result.to_dict())

            return result.to_dict()

        return _executor

    async def _emit_tool_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a tool lifecycle event on the event bus."""
        await self._events.emit(Event(topic=event_type, data=data, source="tool_pipeline"))
