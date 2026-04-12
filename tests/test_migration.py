"""Migration helper tests."""

from __future__ import annotations

import json

from custom_components.learning_thermostat.migration import (
    backup_storage_file,
    build_storage_document,
    extract_legacy_learning_payload,
    load_storage_document,
    resolve_target_entry_id,
)


def test_extract_legacy_learning_payload() -> None:
    """The migration helper should extract the learning payload from a legacy state."""
    payload = extract_legacy_learning_payload(
        {
            "attributes": {
                "current_temperature": 20.5,
                "learning_state": {"weights": [1.0, 2.0], "history": {}},
                "learning_last_tick": "2026-04-12T12:00:00+00:00",
                "specific_states": {"ext_current_temperature": 6.5},
            }
        }
    )

    assert payload["current_temperature"] == 20.5
    assert payload["outside_temperature"] == 6.5
    assert payload["learning_state"]["weights"] == [1.0, 2.0]


def test_build_storage_document_and_backup(tmp_path) -> None:
    """Migrated payloads should be written into the integration storage document."""
    config_dir = tmp_path / "config"
    storage_dir = config_dir / ".storage"
    storage_dir.mkdir(parents=True)

    registry_path = storage_dir / "core.entity_registry"
    registry_path.write_text(
        json.dumps(
            {
                "data": {
                    "entities": [
                        {
                            "entity_id": "climate.kitchen_floor",
                            "config_entry_id": "entry-123",
                        }
                    ]
                }
            }
        )
    )

    assert resolve_target_entry_id(config_dir, "climate.kitchen_floor") == "entry-123"

    storage_path = storage_dir / "learning_thermostat_state"
    document = load_storage_document(storage_path)
    updated = build_storage_document(
        document,
        "entry-123",
        {
            "learning_state": {"weights": [1.0]},
            "learning_last_tick": "2026-04-12T12:00:00+00:00",
            "current_temperature": 20.5,
            "outside_temperature": 6.5,
        },
    )
    storage_path.write_text(json.dumps(updated))
    backup_path = backup_storage_file(storage_path)

    assert updated["data"]["entries"]["entry-123"]["outside_temperature"] == 6.5
    assert backup_path.exists()
