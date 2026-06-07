# Agent Core

Core agent framework providing LLM provider abstraction, agent engine, configuration, session management, and plugin system.

## Features

- **LLM Providers** — OpenAI, Anthropic, local model support
- **Agent Engine** — Chat, streaming, tool-call loop
- **Configuration** — YAML-based config with schema validation
- **Plugins** — Plugin lifecycle, manifest, manager
- **Hooks** — Pre/post request hooks
- **Tools** — Tool registry, pipeline, policy
- **Context** — Context compression and source management
- **Events** — Event bus for decoupled communication
- **MCP** — Model Context Protocol client/adapter

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from agentcore.core.engine import AgentEngine
from agentcore.models.openai import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
engine = AgentEngine(provider=provider)

response = await engine.chat([{"role": "user", "content": "Hello"}])
```

## License

MIT
