"""Configuration loader/saver for AgentCore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agentcore.config.schema import AgentConfig


def load_config(path: str | Path) -> AgentConfig:
    """Load configuration from a YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any]
    if path.suffix in (".yaml", ".yml"):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return AgentConfig.model_validate(raw)


def save_config(config: AgentConfig, path: str | Path) -> None:
    """Save configuration to a YAML or JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump()
    if path.suffix in (".yaml", ".yml"):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    elif path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config_from_dict(data: dict[str, Any]) -> AgentConfig:
    """Load configuration from a dictionary."""
    return AgentConfig.model_validate(data)
