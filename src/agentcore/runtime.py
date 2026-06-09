"""AgentRuntime — standard agent composition layer.

Sits between AgentEngine (atomic) and runners (application).
Owns: hooks, tools, context, events, plugins, prompt, tool pipeline.
Runners delegate orchestration to AgentRuntime and only handle I/O.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from agentcore.agents.manager import SubAgentManager, SubAgentTask
from agentcore.config.schema import AgentConfig, SystemPromptConfig
from agentcore.context.caching import PromptCacheManager
from agentcore.context.engine import ContextEngine
from agentcore.core.adapter import MessageAdapter
from agentcore.core.engine import AgentEngine
from agentcore.core.message import Message, MessageRole
from agentcore.core.providers import MemoryProvider, SkillProvider
from agentcore.core.session import Session
from agentcore.core.session_store import MemorySessionStore, SessionStore
from agentcore.delivery.channel import ChannelMessage, DeliveryManager
from agentcore.events.bus import Event, EventBus
from agentcore.hooks.manager import HookManager
from agentcore.hooks.types import HookContext, HookName
from agentcore.models.base import (
    BaseLLMProvider,
    ChatParams,
    LLMMessage,
    LLMResponse,
    ThinkingLevel,
    ToolCall,
    ToolCallFunction,
)
from agentcore.models.capabilities import ProviderCapabilities
from agentcore.models.events import StreamEvent, StreamEventType
from agentcore.plugins.base import Plugin, PluginContext
from agentcore.plugins.manager import PluginManager
from agentcore.resilience.fallback import FallbackProviderChain
from agentcore.tools.pipeline import ToolPipeline
from agentcore.tools.policy import PolicyContext, PolicyPipeline
from agentcore.tools.registry import ToolRegistry
from agentcore.tools.repair import ToolCallRepairer

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Standard agent composition layer.

    Responsibilities:
    - Bootstrap mechanism layer (hooks, tools, context, events, plugins)
    - Manage plugin lifecycle
    - Build system prompt
    - Execute the unified agent loop (streaming + tool execution + hooks)
    - Provide legacy chat() and stream_events() for backward compatibility

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
        session_store: SessionStore | None = None,
        skill_provider: SkillProvider | None = None,
        memory_provider: MemoryProvider | None = None,
        tool_repairer: ToolCallRepairer | None = None,
        cache_manager: PromptCacheManager | None = None,
        thinking_level: ThinkingLevel | None = None,
        sub_agent_manager: SubAgentManager | None = None,
        fallback_chain: FallbackProviderChain | None = None,
        delivery: DeliveryManager | None = None,
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

        # New integrations
        self._session_store = session_store or MemorySessionStore()
        self._skill_provider = skill_provider
        self._memory_provider = memory_provider
        self._tool_repairer = tool_repairer or ToolCallRepairer()
        self._cache_manager = cache_manager or PromptCacheManager()
        self._thinking_level = thinking_level
        self._sub_agents = sub_agent_manager
        self._fallback = fallback_chain
        self._delivery = delivery

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

    @property
    def session_store(self) -> SessionStore:
        return self._session_store

    @property
    def sub_agent_manager(self) -> SubAgentManager | None:
        return self._sub_agents

    @property
    def fallback_chain(self) -> FallbackProviderChain | None:
        return self._fallback

    @property
    def delivery(self) -> DeliveryManager | None:
        return self._delivery

    # --- Lifecycle ---

    async def initialize(self, plugins: list[Plugin] | None = None) -> None:
        """Initialize the runtime: register plugins, setup hooks/tools/events."""
        if plugins:
            for plugin in plugins:
                self._plugins.register(plugin)

        self._tool_pipeline.set_event_emitter(self._emit_tool_event)

        # Sync PluginManager's mechanism references with runtime's
        self._plugins._hooks = self._hooks
        self._plugins._tools = self._tools
        self._plugins._context = self._context
        self._plugins._events = self._events
        self._plugins._config = self._engine.config
        self._plugins._extra["model_registry"] = self._engine.model_registry

        # Initialize all plugins (PluginManager calls setup + on_engine_ready)
        await self._plugins.initialize_all()

        await self._hooks.dispatch(HookContext(name=HookName.ENGINE_INIT, engine=self._engine))
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the runtime: teardown plugins, dispatch shutdown hooks."""
        if self._session:
            await self.end_session()
        await self._hooks.dispatch(HookContext(name=HookName.ENGINE_SHUTDOWN, engine=self._engine))
        await self._plugins.shutdown_all()
        self._initialized = False

    # --- Session management ---

    async def create_session(self, session_id: str = "") -> Session:
        """Create a new session and dispatch SESSION_CREATED."""
        self._session = self._engine.create_session(session_id)
        # Make session accessible to plugins via config.metadata
        if self._engine.config and hasattr(self._engine.config, 'metadata'):
            self._engine.config.metadata["_current_session"] = self._session
        # SESSION_CREATED hook
        await self._hooks.dispatch(HookContext(
            name=HookName.SESSION_CREATED,
            engine=self._engine,
            session=self._session,
        ))
        return self._session

    async def start_session(self, session_id: str = "") -> Session:
        """Create session and dispatch SESSION_START hook."""
        session = await self.create_session(session_id)
        await self._hooks.dispatch(HookContext(
            name=HookName.SESSION_START,
            engine=self._engine,
            session=session,
        ))
        self._context.on_session_start(session)
        return session

    async def end_session(self) -> None:
        """Dispatch SESSION_END hook and persist session."""
        if self._session:
            await self._hooks.dispatch(HookContext(
                name=HookName.SESSION_END,
                engine=self._engine,
                session=self._session,
            ))
            await self._persist_session()

    async def _persist_session(self) -> None:
        """Persist the current session to the store."""
        if self._session and self._session_store:
            # Filter out non-serializable values from metadata (functions, objects, etc.)
            clean_metadata = {}
            for k, v in self._session.metadata.items():
                if k.startswith("_"):
                    continue
                try:
                    json.dumps(v)
                    clean_metadata[k] = v
                except (TypeError, ValueError):
                    pass

            config_data = {}
            if hasattr(self._session.config, "model_dump"):
                try:
                    config_data = self._session.config.model_dump(exclude_none=True)
                except Exception:
                    pass
            # Strip non-serializable metadata from config
            if "metadata" in config_data:
                raw_meta = config_data["metadata"]
                clean_meta = {}
                for mk, mv in raw_meta.items():
                    if mk.startswith("_"):
                        continue
                    try:
                        json.dumps(mv)
                        clean_meta[mk] = mv
                    except (TypeError, ValueError):
                        pass
                config_data["metadata"] = clean_meta

            data = {
                "config": config_data,
                "messages": [m.model_dump() for m in self._session.messages],
                "metadata": clean_metadata,
                "created_at": self._session.created_at.isoformat(),
                "updated_at": self._session.updated_at.isoformat(),
            }
            try:
                await self._session_store.save(self._session.id, data)
            except (TypeError, ValueError) as e:
                logger.warning("Session persistence failed (non-serializable data): %s", e)

    # --- Unified agent loop (streaming + tools) ---

    async def run(
        self,
        user_input: str,
        *,
        max_rounds: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """Unified agent loop: stream LLM output while executing tools.

        This is the primary API. It streams text to the caller while
        simultaneously executing tool calls and looping until the model
        produces a final text response.

        Yields:
            StreamEvent objects: message_start, text_delta, tool_call,
            tool_result, message_done, error.
        """
        session = self._session
        if not session:
            yield StreamEvent.error_event("No active session. Call create_session() first.")
            return

        config = self._engine.config
        provider = self._engine.provider

        # SESSION_START on first call (if not already started)
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
            reason = ctx.metadata.get("cancel_reason", "")
            if reason:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=reason + "\n")
            return

        # Add user message
        session.add_message(Message.user(user_input))

        yield StreamEvent.message_start()

        # Agent loop: stream → tools → repeat
        for round_num in range(max_rounds):
            # Context compression
            if self._context.should_compress(session.messages, config.context.max_tokens):
                compressed = self._context.compress(
                    session.messages,
                    max_tokens=config.context.max_tokens,
                )
            else:
                compressed = session.messages

            # Convert to LLM messages
            llm_messages = MessageAdapter.to_llm_list(compressed)

            # Inject skills into system prompt
            if self._skill_provider and round_num == 0:
                skills_text = self._skill_provider.format_for_system_prompt()
                if skills_text:
                    llm_messages = self._inject_system_content(llm_messages, skills_text)

            # Inject memory context
            if self._memory_provider and round_num == 0:
                try:
                    memory_text = await self._memory_provider.get_context()
                    if memory_text:
                        llm_messages = self._inject_system_content(llm_messages, memory_text)
                except Exception as e:
                    logger.warning("Memory context injection failed: %s", e)

            # Apply prompt caching
            llm_messages = self._cache_manager.apply_cache_control(
                [m.model_dump() for m in llm_messages],
                config.model.provider,
            )
            llm_messages = [LLMMessage(**m) for m in llm_messages]

            # POST_BUILD_MESSAGES hook
            build_ctx = HookContext(
                name=HookName.POST_BUILD_MESSAGES,
                engine=self._engine,
                session=session,
                messages=llm_messages,
                hooks=self._hooks,
                tools=self._tools,
                context=self._context,
                events=self._events,
            )
            await self._hooks.dispatch(build_ctx)

            # Build ChatParams
            tool_defs = self._tools.get_tool_definitions()
            params = ChatParams(
                model=config.model.model,
                temperature=config.model.temperature,
                max_tokens=config.model.max_tokens,
                tools=tool_defs,
            )

            # Apply thinking level
            if self._thinking_level:
                thinking_kwargs = provider.map_thinking_level(self._thinking_level, config.model.model)
                if "thinking" in thinking_kwargs:
                    params.thinking = thinking_kwargs["thinking"]

            # PRE_LLM_CALL hook
            pre_llm_ctx = HookContext(
                name=HookName.PRE_LLM_CALL,
                engine=self._engine,
                session=session,
                messages=llm_messages,
                params=params,
                hooks=self._hooks,
                tools=self._tools,
                context=self._context,
                events=self._events,
            )
            await self._hooks.dispatch(pre_llm_ctx)
            if pre_llm_ctx.cancel:
                yield StreamEvent.error_event("LLM call cancelled by hook")
                return

            # Stream LLM response (with fallback support)
            full_content = ""
            tool_calls: list[ToolCall] = []
            try:
                stream_iter = self._stream_with_fallback(llm_messages, params)
                async for chunk in stream_iter:
                    full_content += chunk
                    yield StreamEvent.text_delta(chunk)

                # Get tool calls accumulated during streaming
                active_provider = self._get_active_provider()
                if hasattr(active_provider, "_last_stream_tool_calls"):
                    tool_calls = active_provider._last_stream_tool_calls
                    active_provider._last_stream_tool_calls = []

            except Exception as e:
                yield StreamEvent.error_event(str(e))
                return

            # POST_LLM_CALL hook
            post_llm_ctx = HookContext(
                name=HookName.POST_LLM_CALL,
                engine=self._engine,
                session=session,
                response=LLMResponse(
                    content=full_content,
                    model=config.model.model,
                    tool_calls=tool_calls,
                ),
                hooks=self._hooks,
                tools=self._tools,
                context=self._context,
                events=self._events,
            )
            await self._hooks.dispatch(post_llm_ctx)

            # Tool-call repair: check if text contains plain-text tool calls
            if not tool_calls and full_content:
                repaired = self._tool_repairer.detect(full_content)
                if repaired:
                    for tc in repaired:
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            function=ToolCallFunction(name=tc.name, arguments=tc.arguments),
                        ))
                    full_content = ""  # Clear text since it was a tool call

            # If no tool calls, this is the final response
            if not tool_calls:
                # TRANSFORM_LLM_OUTPUT hook
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

                # Store assistant message
                session.add_message(Message.assistant(full_content))

                # Context lifecycle
                self._context.on_turn_end(session.messages, full_content)

                # Memory provider turn end
                if self._memory_provider:
                    try:
                        await self._memory_provider.on_turn_end(
                            [m.model_dump() for m in session.messages]
                        )
                    except Exception as e:
                        logger.warning("Memory turn-end hook failed: %s", e)

                # Persist session
                await self._persist_session()

                # Deliver result through delivery manager
                await self._deliver_result(full_content, session)

                yield StreamEvent.message_done(content=full_content)
                return

            # Execute tool calls
            assistant_msg = Message.assistant_with_tools(full_content or "", tool_calls)
            session.add_message(assistant_msg)

            for tc in tool_calls:
                # PRE_TOOL_CALL hook
                pre_tool_ctx = HookContext(
                    name=HookName.PRE_TOOL_CALL,
                    engine=self._engine,
                    session=session,
                    tool_call=tc,
                    hooks=self._hooks,
                    tools=self._tools,
                    context=self._context,
                    events=self._events,
                )
                await self._hooks.dispatch(pre_tool_ctx)
                if pre_tool_ctx.cancel:
                    result_data = {"success": False, "output": None, "error": "Tool call cancelled by hook"}
                else:
                    # Route to sub-agent if applicable, otherwise normal pipeline
                    result_data = await self._execute_tool_with_subagent(tc, session, config)

                # TRANSFORM_TOOL_RESULT hook
                transform_tool_ctx = HookContext(
                    name=HookName.TRANSFORM_TOOL_RESULT,
                    engine=self._engine,
                    session=session,
                    tool_call=tc,
                    tool_result=result_data,
                    hooks=self._hooks,
                    tools=self._tools,
                    context=self._context,
                    events=self._events,
                )
                await self._hooks.dispatch(transform_tool_ctx)
                if transform_tool_ctx.tool_result is not None:
                    result_data = transform_tool_ctx.tool_result

                # POST_TOOL_CALL hook
                post_tool_ctx = HookContext(
                    name=HookName.POST_TOOL_CALL,
                    engine=self._engine,
                    session=session,
                    tool_call=tc,
                    tool_result=result_data,
                    hooks=self._hooks,
                    tools=self._tools,
                    context=self._context,
                    events=self._events,
                )
                await self._hooks.dispatch(post_tool_ctx)

                # Store tool result
                tool_output = str(result_data.get("output", result_data.get("error", "")))
                session.add_message(Message.tool(tool_output, tool_call_id=tc.id))

                yield StreamEvent.tool_result_done(tc.id, result_data)

            # Loop continues for next round

        # Exhausted max rounds
        yield StreamEvent.message_done(content=full_content or "[Max tool rounds reached]")

    # --- Legacy APIs (backward compatible) ---

    async def chat(
        self,
        user_input: str,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[ToolCall, dict], Awaitable[None]] | None = None,
        on_response: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Legacy chat pipeline. Use run() for new code."""
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
            return ctx.metadata.get("cancel_reason", "Cancelled by hook")

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

        tool_executor = self._build_tool_executor(on_tool_call, on_tool_result)
        response = await self._engine.chat_with_tools(
            messages=llm_messages,
            params=params,
            tool_executor=tool_executor,
            max_rounds=10,
        )

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

        session.add_message(Message.assistant(response_content))
        if on_response:
            await on_response(response_content)

        return response_content

    async def stream_events(
        self,
        user_input: str,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[ToolCall, dict], Awaitable[None]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Legacy streaming pipeline (no tool execution). Use run() for tools + streaming."""
        session = self._session
        if not session:
            yield StreamEvent.error_event("No active session. Call create_session() first.")
            return

        config = self._engine.config

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
            reason = ctx.metadata.get("cancel_reason", "")
            if reason:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=reason + "\n")
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

        full_content = ""
        try:
            async for chunk in self._engine.stream(llm_messages, params):
                full_content += chunk
                yield StreamEvent.text_delta(chunk)
        except Exception as e:
            yield StreamEvent.error_event(str(e))
            return

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
            if on_tool_call:
                await on_tool_call(tc)

            policy_ctx = PolicyContext(
                session_id=runtime.session.id if runtime.session else "",
                provider_name=runtime.config.model.provider,
                model_name=runtime.config.model.model,
            )

            result = await runtime._tool_pipeline.execute(tc, policy_ctx)

            if on_tool_result:
                await on_tool_result(tc, result.to_dict())

            return result.to_dict()

        return _executor

    def _inject_system_content(
        self,
        messages: list[LLMMessage],
        content: str,
    ) -> list[LLMMessage]:
        """Inject content into the system message."""
        if not messages:
            return messages

        result = list(messages)
        if result[0].role == "system":
            existing = result[0].content if isinstance(result[0].content, str) else ""
            result[0] = LLMMessage(role="system", content=f"{existing}\n\n{content}")
        else:
            result.insert(0, LLMMessage(role="system", content=content))
        return result

    async def _emit_tool_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a tool lifecycle event on the event bus."""
        await self._events.emit(Event(topic=event_type, data=data, source="tool_pipeline"))

    def _get_active_provider(self) -> BaseLLMProvider:
        """Get the currently active LLM provider (primary or fallback)."""
        if self._fallback:
            current = self._fallback.get_current_provider()
            if current:
                return current
        return self._engine.provider

    async def _stream_with_fallback(
        self,
        messages: list[LLMMessage],
        params: ChatParams,
    ) -> AsyncIterator[str]:
        """Stream LLM response with fallback provider support.

        If the primary provider fails with a retryable error, tries each
        fallback provider in order. Non-retryable errors propagate immediately.
        """
        if not self._fallback:
            # No fallback configured — use engine directly
            async for chunk in self._engine.stream(messages, params):
                yield chunk
            return

        # Try primary first, then fallbacks
        while True:
            provider = self._get_active_provider()
            try:
                # Apply fallback-specific params override
                effective_params = self._fallback.apply_params_override(params) if self._fallback else params
                async for chunk in provider.stream_chat(messages, effective_params):
                    yield chunk
                return  # Success
            except Exception as e:
                # Check if we should try fallback
                from agentcore.utils.retry import classify_error
                category = classify_error(e)
                if category in ("permanent", "invalid_request", "context_too_long"):
                    raise  # Not retryable

                next_provider = self._fallback.activate_next_fallback()
                if not next_provider:
                    raise  # Exhausted all fallbacks
                logger.info("Fallback: switched to '%s' after error: %s", self._fallback.get_current_name(), e)

    async def _execute_tool_with_subagent(
        self,
        tc: ToolCall,
        session: Session,
        config: AgentConfig,
    ) -> dict[str, Any]:
        """Execute a tool call, routing to SubAgentManager if applicable.

        Tool calls named 'delegate_task' or 'delegate_subagent' are routed
        to the SubAgentManager. All others go through the normal pipeline.
        """
        # Check for sub-agent delegation
        if self._sub_agents and tc.function.name in ("delegate_task", "delegate_subagent"):
            import json
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except (json.JSONDecodeError, TypeError):
                args = {}

            task = SubAgentTask(
                description=args.get("description", ""),
                prompt=args.get("prompt", ""),
                metadata=args.get("metadata", {}),
            )

            # Dispatch SUBAGENT_STOP hook
            result = await self._sub_agents.delegate(task)
            await self._hooks.dispatch(HookContext(
                name=HookName.SUBAGENT_STOP,
                engine=self._engine,
                session=session,
                tool_call=tc,
                tool_result={"task_id": task.task_id, "success": result.success, "output": result.output, "error": result.error},
                hooks=self._hooks,
                tools=self._tools,
                context=self._context,
                events=self._events,
            ))

            if result.success:
                return {"success": True, "output": result.output, "error": ""}
            return {"success": False, "output": None, "error": result.error}

        # Normal tool pipeline
        policy_ctx = PolicyContext(
            session_id=session.id,
            provider_name=config.model.provider,
            model_name=config.model.model,
        )
        try:
            tool_result = await self._tool_pipeline.execute(tc, policy_ctx)
            return tool_result.to_dict()
        except Exception as e:
            return {"success": False, "output": None, "error": str(e)}

    async def _deliver_result(self, content: str, session: Session) -> None:
        """Deliver the final result through the delivery manager."""
        if not self._delivery:
            return
        try:
            msg = ChannelMessage(
                channel_id=session.id,
                content=content,
            )
            await self._delivery.send(msg)
        except Exception as e:
            logger.warning("Delivery failed: %s", e)
