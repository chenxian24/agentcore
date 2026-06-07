"""Plugin manifest: declarative metadata for plugins.

A manifest describes what a plugin provides (tools, hooks, commands, capabilities)
and when it should be activated (conditions based on config, environment, capabilities).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActivationTrigger(str, Enum):
    """When a plugin should be activated."""

    STARTUP = "startup"  # Always activate on engine startup
    ON_COMMAND = "on_command"  # Activate when a specific command is used
    ON_PROVIDER = "on_provider"  # Activate when a specific model provider is used
    ON_CAPABILITY = "on_capability"  # Activate when a capability is requested
    ON_DEMAND = "on_demand"  # Activate only when explicitly loaded


@dataclass
class ToolContract:
    """A tool that a plugin provides."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookContract:
    """A hook that a plugin registers."""

    hook_name: str
    priority: int = 100


@dataclass
class PluginManifest:
    """Declarative metadata for a plugin.

    Extensions declare manifests so the activation planner can decide
    whether to load them without importing their code.
    """

    # Identity
    name: str
    version: str = "0.1.0"
    description: str = ""

    # Dependencies
    dependencies: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    # What the plugin provides
    tools: list[ToolContract] = field(default_factory=list)
    hooks: list[HookContract] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)

    # When to activate
    triggers: list[ActivationTrigger] = field(default_factory=lambda: [ActivationTrigger.ON_DEMAND])

    # Activation conditions (all must be true)
    requires_providers: list[str] = field(default_factory=list)  # e.g. ["openai", "anthropic"]
    requires_config: list[str] = field(default_factory=list)  # config keys that must exist
    requires_env: list[str] = field(default_factory=list)  # env vars that must exist
    requires_capabilities: list[str] = field(default_factory=list)  # capabilities that must be available

    # Configuration schema
    config_schema: dict[str, Any] = field(default_factory=dict)

    # Metadata
    author: str = ""
    homepage: str = ""
    tags: list[str] = field(default_factory=list)


class ActivationPlanner:
    """Decides which plugins to activate based on their manifests and current context."""

    def __init__(self) -> None:
        self._manifests: dict[str, PluginManifest] = {}

    def register(self, manifest: PluginManifest) -> None:
        self._manifests[manifest.name] = manifest

    def unregister(self, name: str) -> None:
        self._manifests.pop(name, None)

    def get_manifest(self, name: str) -> PluginManifest | None:
        return self._manifests.get(name)

    def list_manifests(self) -> list[PluginManifest]:
        return list(self._manifests.values())

    def plan(
        self,
        provider: str = "",
        config: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        capabilities: set[str] | None = None,
        explicit_names: set[str] | None = None,
    ) -> list[PluginManifest]:
        """Return the list of plugins that should be activated.

        Args:
            provider: Current model provider name (e.g. "openai")
            config: Current configuration dict
            env: Environment variables
            capabilities: Currently available capabilities
            explicit_names: Plugin names explicitly requested by user
        """
        import os

        config = config or {}
        env = env or os.environ
        capabilities = capabilities or set()
        explicit_names = explicit_names or set()

        activated: list[PluginManifest] = []
        activated_names: set[str] = set()

        for manifest in self._manifests.values():
            if manifest.name in activated_names:
                continue

            # Check conflicts
            if any(c in activated_names for c in manifest.conflicts):
                continue

            # Check if explicitly requested
            if manifest.name in explicit_names:
                activated.append(manifest)
                activated_names.add(manifest.name)
                continue

            # Check activation triggers
            should_activate = False

            for trigger in manifest.triggers:
                if trigger == ActivationTrigger.STARTUP:
                    should_activate = True
                    break
                elif trigger == ActivationTrigger.ON_DEMAND:
                    # Only activate if explicitly requested (handled above)
                    pass
                elif trigger == ActivationTrigger.ON_PROVIDER:
                    if provider and provider in manifest.requires_providers:
                        should_activate = True
                        break
                elif trigger == ActivationTrigger.ON_CAPABILITY:
                    if capabilities & set(manifest.requires_capabilities):
                        should_activate = True
                        break

            if not should_activate:
                continue

            # Check conditions
            if manifest.requires_providers and provider not in manifest.requires_providers:
                continue
            if manifest.requires_config:
                if not all(k in config for k in manifest.requires_config):
                    continue
            if manifest.requires_env:
                if not all(k in env for k in manifest.requires_env):
                    continue
            if manifest.requires_capabilities:
                if not capabilities.issuperset(manifest.requires_capabilities):
                    continue

            activated.append(manifest)
            activated_names.add(manifest.name)

        # Resolve dependencies
        changed = True
        while changed:
            changed = False
            for manifest in list(activated):
                for dep_name in manifest.dependencies:
                    if dep_name not in activated_names:
                        dep_manifest = self._manifests.get(dep_name)
                        if dep_manifest:
                            activated.append(dep_manifest)
                            activated_names.add(dep_name)
                            changed = True

        return activated
