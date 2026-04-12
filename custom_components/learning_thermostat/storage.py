"""Persistent storage for learning thermostat state."""

from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

STORAGE_VERSION = 1
STORAGE_MINOR_VERSION = 1
STORAGE_KEY = f"{DOMAIN}_state"
STORAGE_DATA_KEY = "entries"


def build_entry_payload(
    learning_state: dict[str, Any] | None,
    learning_last_tick: str | None,
    current_temperature: float | None,
    outside_temperature: float | None,
) -> dict[str, Any]:
    """Build one stored entry payload."""
    return {
        "learning_state": learning_state,
        "learning_last_tick": learning_last_tick,
        "current_temperature": current_temperature,
        "outside_temperature": outside_temperature,
    }


class LearningStateStore:
    """Wrap Home Assistant storage for persisted learning state."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            STORAGE_KEY,
            minor_version=STORAGE_MINOR_VERSION,
        )
        self._lock = asyncio.Lock()

    async def async_get_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Return one stored entry."""
        async with self._lock:
            data = await self._store.async_load() or {STORAGE_DATA_KEY: {}}
            entries = data.get(STORAGE_DATA_KEY, {})
            if not isinstance(entries, dict):
                return None
            entry = entries.get(entry_id)
            return dict(entry) if isinstance(entry, dict) else None

    async def async_save_entry(self, entry_id: str, payload: dict[str, Any]) -> None:
        """Persist one entry payload."""
        async with self._lock:
            data = await self._store.async_load() or {STORAGE_DATA_KEY: {}}
            entries = data.get(STORAGE_DATA_KEY)
            if not isinstance(entries, dict):
                entries = {}
            entries = dict(entries)
            entries[entry_id] = dict(payload)
            await self._store.async_save({STORAGE_DATA_KEY: entries})

    async def async_delete_entry(self, entry_id: str) -> None:
        """Remove one stored entry."""
        async with self._lock:
            data = await self._store.async_load() or {STORAGE_DATA_KEY: {}}
            entries = data.get(STORAGE_DATA_KEY)
            if not isinstance(entries, dict):
                return
            if entry_id not in entries:
                return
            updated_entries = dict(entries)
            del updated_entries[entry_id]
            await self._store.async_save({STORAGE_DATA_KEY: updated_entries})
