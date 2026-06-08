"""Session persistence — pluggable storage backends for conversation sessions."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Abstract session storage backend."""

    @abstractmethod
    async def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist session data."""

    @abstractmethod
    async def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session data by ID. Returns None if not found."""

    @abstractmethod
    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all stored sessions (metadata only)."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete a session by ID."""


class MemorySessionStore(SessionStore):
    """In-memory session storage (default, non-persistent)."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def save(self, session_id: str, data: dict[str, Any]) -> None:
        self._store[session_id] = data

    async def load(self, session_id: str) -> dict[str, Any] | None:
        return self._store.get(session_id)

    async def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {"id": sid, "updated_at": d.get("updated_at", "")}
            for sid, d in self._store.items()
        ]

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class JsonlSessionStore(SessionStore):
    """JSONL file-based session storage.

    Each session is stored as a directory under `base_path/<session_id>/`.
    Messages are appended to `messages.jsonl`; metadata is in `meta.json`.
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self._base / session_id

    async def save(self, session_id: str, data: dict[str, Any]) -> None:
        sdir = self._session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "id": session_id,
            "config": data.get("config", {}),
            "metadata": data.get("metadata", {}),
            "created_at": data.get("created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (sdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save messages as JSONL
        messages = data.get("messages", [])
        with (sdir / "messages.jsonl").open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    async def load(self, session_id: str) -> dict[str, Any] | None:
        sdir = self._session_dir(session_id)
        meta_path = sdir / "meta.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        messages = []
        msg_path = sdir / "messages.jsonl"
        if msg_path.exists():
            with msg_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))

        return {
            "id": session_id,
            "config": meta.get("config", {}),
            "metadata": meta.get("metadata", {}),
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "messages": messages,
        }

    async def list_sessions(self) -> list[dict[str, Any]]:
        result = []
        for sdir in self._base.iterdir():
            if sdir.is_dir():
                meta_path = sdir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    result.append({
                        "id": sdir.name,
                        "updated_at": meta.get("updated_at", ""),
                    })
        result.sort(key=lambda x: x["updated_at"], reverse=True)
        return result

    async def delete(self, session_id: str) -> None:
        import shutil
        sdir = self._session_dir(session_id)
        if sdir.exists():
            shutil.rmtree(sdir)
