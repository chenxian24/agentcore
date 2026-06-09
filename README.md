# AgentCore

Atomic agent engine — 通过分层架构构建任意智能体系统的基础框架。

```
Application layer:  extensions/         — Hermes, OpenCode, OpenClaw, Codex（由 core + mechanisms 组合而成）
Mechanism layer:    agentcore.<mod>.*   — hooks, tools, plugins, context, events, mcp, resilience, middleware
Atomic layer:       agentcore.*         — Message, Provider, Config, Engine, Runtime, Wire, Stats, Utils
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  extensions/ — runners (hermes, opencode, openclaw, codex)│
├─────────────────────────────────────────────────────────┤
│                    Mechanism Layer                        │
│  ┌────────┐ ┌──────┐ ┌────────┐ ┌───────┐ ┌──────────┐ │
│  │ Hooks  │ │Tools │ │Plugins │ │Context│ │  Events  │ │
│  └────────┘ └──────┘ └────────┘ └───────┘ └──────────┘ │
│  ┌────────┐ ┌──────┐ ┌────────┐ ┌───────┐ ┌──────────┐ │
│  │  MCP   │ │Resil.│ │Delivery│ │Middlew│ │Tokenizer │ │
│  └────────┘ └──────┘ └────────┘ └───────┘ └──────────┘ │
├─────────────────────────────────────────────────────────┤
│                     Atomic Layer                         │
│  AgentEngine · AgentRuntime · Session · Config · Wire   │
│  LLM Providers · ToolRegistry · Stats · Utils           │
└─────────────────────────────────────────────────────────┘
```

## Features

- **LLM Providers** — OpenAI / Anthropic / 本地模型，统一的 `BaseLLMProvider` 接口
- **Wire Protocol** — `ChatCompletionsProtocol` / `ResponsesProtocol`，可插拔序列化
- **Agent Engine** — chat、streaming、tool-call loop，`AgentEngine` 是最底层原子
- **Agent Runtime** — 组合层：hooks、tools、context、events、plugins、prompt、pipeline
- **Plugin System** — `Plugin` ABC + `PluginManager`（依赖解析、生命周期管理）
- **Hook System** — 生命周期事件钩子（ENGINE_INIT、PRE_CHAT、POST_TOOL 等）
- **Tool Pipeline** — 注册、策略检查、审批门控、重试、结果转换、事件发射
- **Tool Retry** — `ExponentialBackoffPolicy` / `FixedDelayPolicy` / `NoRetryPolicy`
- **Middleware** — 可组合中间件栈，环绕 LLM 调用路径（日志、转换、指标）
- **Context** — 上下文压缩、滑动窗口、摘要策略、Prompt Cache
- **Tokenizer** — 可插拔分词器（`SimpleTokenizer` / `TiktokenTokenizer`）
- **Session** — 会话管理 + 持久化（`MemorySessionStore` / `JsonlSessionStore`）
- **Sub-agents** — `SubAgentManager` 子 agent 创建、任务委派、结果收集
- **MCP** — Model Context Protocol 客户端（stdio / TCP 传输）
- **Resilience** — Fallback provider chain、错误分类
- **Delivery** — Channel 抽象，消息投递管理
- **Stats** — 请求级 + 聚合级统计收集

## Installation

```bash
pip install -e .

# With tiktoken tokenizer support
pip install -e ".[tiktoken]"

# With API server
pip install -e ".[api]"
```

## Quick Start

```python
from agentcore import AgentEngine, AgentConfig, AgentRuntime

config = AgentConfig(...)
engine = AgentEngine(config)
runtime = AgentRuntime(engine=engine, plugins=plugin_manager)
await runtime.initialize()

session = await runtime.create_session("my-session")
async for event in runtime.run("Hello"):
    print(event)
```

## Public API

47 个导出符号，覆盖完整 agent 构建需求：

| Category | Exports |
|----------|---------|
| Types | `Message`, `MessageRole`, `LLMMessage`, `LLMResponse`, `ChatParams`, `ToolCall`, `ToolExecutor`, `ThinkingConfig`, `ThinkingLevel` |
| Provider | `BaseLLMProvider`, `ModelRegistry`, `ProviderCapabilities` |
| Wire | `WireProtocol`, `ChatCompletionsProtocol`, `ResponsesProtocol`, `WireResponse`, `WireToolCall`, `WireEvent` |
| Streaming | `StreamEvent`, `StreamEventType` |
| Config | `AgentConfig`, `ModelConfig`, `SystemPromptConfig` |
| Context | `ContextEngine`, `PromptCacheManager` |
| Engine | `AgentEngine`, `AgentRuntime`, `MessageAdapter`, `Session` |
| Session | `SessionStore`, `MemorySessionStore`, `JsonlSessionStore` |
| Skills | `SkillProvider`, `MemoryProvider` |
| Tools | `ToolPipeline`, `ToolRegistry`, `ToolResult`, `ToolCallRepairer` |
| Agents | `SubAgentManager`, `SubAgentTask`, `SubAgentResult` |
| Resilience | `FallbackProviderChain` |
| Delivery | `Channel`, `ChannelMessage`, `DeliveryManager` |
| Stats | `StatsCollector`, `RequestStats`, `AggregateStats` |
| Utils | `run_loop`, `with_retry` |

## Module Structure

```
agentcore/
├── __init__.py          # Public API (47 exports)
├── __main__.py          # CLI: version, serve
├── runtime.py           # AgentRuntime — 组合层
├── middleware.py         # MiddlewareStack — 可组合中间件
├── tokenizer.py         # Tokenizer ABC + implementations
├── agents/              # SubAgentManager, SubAgentTask
├── config/              # AgentConfig, ModelConfig, load/save
├── context/             # ContextEngine, strategies, cache
├── core/                # AgentEngine, Session, Message, Stats, SessionStore
├── delivery/            # Channel, DeliveryManager
├── events/              # EventBus
├── hooks/               # HookManager, HookName
├── mcp/                 # MCPManager, transport, protocol
├── models/              # BaseLLMProvider, ToolCall, StreamEvent, providers
├── plugins/             # Plugin, PluginContext, PluginManager
├── prompts/             # PromptTemplate, DynamicPromptBuilder
├── resilience/          # FallbackProviderChain, ErrorClassifier
├── tools/               # ToolRegistry, ToolPipeline, policies, retry
├── utils/               # run_loop, with_retry
└── wire/                # WireProtocol, ChatCompletions, Responses
```

## License

MIT
