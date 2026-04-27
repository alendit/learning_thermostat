"""Helpers for migrating learning state from versatile_thermostat."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

from homeassistant.util import dt as dt_util

from .state import build_entry_payload
from .storage import (
    STORAGE_DATA_KEY,
    STORAGE_KEY,
    STORAGE_MINOR_VERSION,
    STORAGE_VERSION,
)


class MigrationError(RuntimeError):
    """Raised when migration input or storage is invalid."""


def extract_legacy_learning_payload(legacy_state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the persisted learning payload from a VTherm entity state."""
    attributes = legacy_state.get("attributes")
    if not isinstance(attributes, Mapping):
        raise MigrationError("legacy entity payload has no attributes mapping")

    learning_state = attributes.get("learning_state")
    if not isinstance(learning_state, dict):
        raise MigrationError("legacy entity does not expose a learning_state mapping")

    learning_last_tick = attributes.get("learning_last_tick")
    if isinstance(learning_last_tick, str):
        parsed_tick = dt_util.parse_datetime(learning_last_tick)
        learning_last_tick = (
            parsed_tick.isoformat() if parsed_tick is not None else None
        )
    else:
        learning_last_tick = None

    current_temperature = _coerce_float(attributes.get("current_temperature"))
    specific_states = attributes.get("specific_states")
    outside_temperature = None
    if isinstance(specific_states, Mapping):
        outside_temperature = _coerce_float(
            specific_states.get("ext_current_temperature")
        )

    return build_entry_payload(
        learning_state=learning_state,
        learning_last_tick=learning_last_tick,
        current_temperature=current_temperature,
        outside_temperature=outside_temperature,
    )


def resolve_target_entry_id(config_dir: Path, entity_id: str) -> str:
    """Resolve a Home Assistant entity ID to a config entry ID."""
    registry_path = config_dir / ".storage" / "core.entity_registry"
    registry = json.loads(registry_path.read_text())
    entries = registry.get("data", {}).get("entities", [])
    if not isinstance(entries, list):
        raise MigrationError("entity registry format is invalid")

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("entity_id") == entity_id:
            config_entry_id = entry.get("config_entry_id")
            if not isinstance(config_entry_id, str) or not config_entry_id:
                raise MigrationError(f"entity {entity_id} has no config_entry_id")
            return config_entry_id

    raise MigrationError(f"entity {entity_id} was not found in the entity registry")


def load_storage_document(storage_path: Path) -> dict[str, Any]:
    """Load the integration storage file or return an empty document."""
    if not storage_path.exists():
        return {
            "version": STORAGE_VERSION,
            "minor_version": STORAGE_MINOR_VERSION,
            "key": STORAGE_KEY,
            "data": {STORAGE_DATA_KEY: {}},
        }
    return json.loads(storage_path.read_text())


def build_storage_document(
    existing_document: Mapping[str, Any],
    entry_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one migrated payload to the integration storage document."""
    data = existing_document.get("data")
    if not isinstance(data, Mapping):
        data = {}
    entries = data.get(STORAGE_DATA_KEY)
    if not isinstance(entries, Mapping):
        entries = {}

    updated_entries = dict(entries)
    updated_entries[entry_id] = dict(payload)

    return {
        "version": existing_document.get("version", STORAGE_VERSION),
        "minor_version": existing_document.get("minor_version", STORAGE_MINOR_VERSION),
        "key": existing_document.get("key", STORAGE_KEY),
        "data": {STORAGE_DATA_KEY: updated_entries},
    }


def backup_storage_file(storage_path: Path, backup_dir: Path | None = None) -> Path:
    """Write a timestamped backup of the storage file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    target_dir = backup_dir or storage_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    backup_path = target_dir / f"{storage_path.name}.bak.{timestamp}"
    backup_path.write_text(storage_path.read_text() if storage_path.exists() else "")
    return backup_path


def likely_home_assistant_running(config_dir: Path) -> bool:
    """Return whether HA appears to be live based on common DB sidecars."""
    return any(
        (config_dir / marker).exists()
        for marker in ("home-assistant_v2.db-shm", "home-assistant_v2.db-wal")
    )


def write_storage_document(storage_path: Path, document: Mapping[str, Any]) -> None:
    """Write the updated storage document."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
