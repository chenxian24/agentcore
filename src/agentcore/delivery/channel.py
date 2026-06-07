"""Channel abstraction for message delivery.

A Channel represents a communication endpoint (CLI, Discord, Slack, WebSocket, etc.)
that can send messages to and receive messages from users.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    REACTION = "reaction"
    SYSTEM = "system"


@dataclass
class ChannelMessage:
    """A message sent through a channel."""

    channel_id: str
    content: str
    message_type: MessageType = MessageType.TEXT
    sender_id: str = ""
    sender_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_to: str = ""  # ID of message being replied to
    attachments: list[dict[str, Any]] = field(default_factory=list)


class Channel(ABC):
    """Abstract base for a communication channel.

    Extensions implement this for specific platforms (Discord, Slack, CLI, etc.)
    The DeliveryManager routes messages through registered channels.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique channel name (e.g. 'discord', 'cli', 'websocket')."""
        ...

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the channel."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the channel connection."""
        ...

    @abstractmethod
    async def send(self, message: ChannelMessage) -> bool:
        """Send a message through the channel. Returns True if successful."""
        ...

    async def receive(self) -> AsyncIterator[ChannelMessage]:
        """Receive messages from the channel. Override for push-based channels."""
        return
        yield  # make it an async generator

    @abstractmethod
    async def list_channels(self) -> list[dict[str, Any]]:
        """List available sub-channels (e.g. Discord guilds, Slack workspaces)."""
        ...

    async def on_message(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        """Register a handler for incoming messages. Override for push-based channels."""
        pass

    @property
    def connected(self) -> bool:
        """Whether the channel is currently connected."""
        return False


class DeliveryManager:
    """Routes messages to the appropriate channel.

    Extensions register channels. The manager selects the right channel
    based on channel_id prefix or explicit routing.
    """

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._default: str = ""
        self._message_handlers: list[Callable[[ChannelMessage], Awaitable[None]]] = []

    def register(self, channel: Channel, default: bool = False) -> None:
        self._channels[channel.name] = channel
        if default or not self._default:
            self._default = channel.name

    def unregister(self, name: str) -> None:
        self._channels.pop(name, None)
        if self._default == name:
            self._default = next(iter(self._channels), "")

    def get(self, name: str) -> Channel | None:
        return self._channels.get(name)

    def get_default(self) -> Channel | None:
        return self._channels.get(self._default)

    def list_channels(self) -> list[Channel]:
        return list(self._channels.values())

    async def send(self, message: ChannelMessage, channel_name: str = "") -> bool:
        """Send a message through the specified or default channel."""
        name = channel_name or self._infer_channel(message.channel_id)
        channel = self._channels.get(name)
        if not channel:
            logger.error("Channel '%s' not found", name)
            return False
        return await channel.send(message)

    async def broadcast(self, message: ChannelMessage) -> dict[str, bool]:
        """Send a message to all connected channels."""
        results = {}
        for name, channel in self._channels.items():
            if channel.connected:
                results[name] = await channel.send(message)
        return results

    def on_message(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        """Register a global handler for incoming messages from any channel."""
        self._message_handlers.append(handler)

    async def connect_all(self) -> None:
        for channel in self._channels.values():
            try:
                await channel.connect()
            except Exception:
                logger.error("Failed to connect channel '%s'", channel.name, exc_info=True)

    async def disconnect_all(self) -> None:
        for channel in self._channels.values():
            try:
                await channel.disconnect()
            except Exception:
                logger.error("Failed to disconnect channel '%s'", channel.name, exc_info=True)

    def _infer_channel(self, channel_id: str) -> str:
        """Infer channel name from channel_id prefix (e.g. 'discord:123' -> 'discord')."""
        if ":" in channel_id:
            return channel_id.split(":", 1)[0]
        return self._default
