"""Plugin system for agentcore extensions."""

from agentcore.plugins.base import Plugin, PluginContext
from agentcore.plugins.manager import PluginManager

__all__ = ["Plugin", "PluginContext", "PluginManager"]
