#!/usr/bin/env python3
"""Migrate learning state from versatile_thermostat to learning_thermostat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from custom_components.learning_thermostat.migration import (
    MigrationError,
    backup_storage_file,
    build_storage_document,
    extract_legacy_learning_payload,
    likely_home_assistant_running,
    load_storage_document,
    resolve_target_entry_id,
    write_storage_document,
)
from custom_components.learning_thermostat.storage import STORAGE_KEY


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--old-entity-id", required=True)
    parser.add_argument("--new-entity-id", required=True)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fetch_entity_state(base_url: str, token: str, entity_id: str) -> dict[str, Any]:
    """Fetch one Home Assistant entity state over the REST API."""
    request = Request(
        f"{base_url.rstrip('/')}/api/states/{quote(entity_id, safe='')}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as err:
        raise MigrationError(
            f"failed to fetch {entity_id} from Home Assistant: HTTP {err.code}"
        ) from err
    except URLError as err:
        raise MigrationError(f"failed to reach Home Assistant: {err.reason}") from err


def main() -> int:
    """Run the migration CLI."""
    args = parse_args()
    config_dir = args.config_dir.resolve()
    storage_path = config_dir / ".storage" / STORAGE_KEY

    if likely_home_assistant_running(config_dir) and not args.force:
        raise MigrationError(
            "Home Assistant appears to be running; rerun with --force to allow a live storage write"
        )

    legacy_state = fetch_entity_state(args.base_url, args.token, args.old_entity_id)
    migrated_payload = extract_legacy_learning_payload(legacy_state)
    target_entry_id = resolve_target_entry_id(config_dir, args.new_entity_id)

    existing_document = load_storage_document(storage_path)
    updated_document = build_storage_document(
        existing_document,
        target_entry_id,
        migrated_payload,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "target_entry_id": target_entry_id,
                    "storage_path": str(storage_path),
                    "updated_document": updated_document,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    backup_path = backup_storage_file(storage_path, args.backup_dir)
    write_storage_document(storage_path, updated_document)

    print(
        json.dumps(
            {
                "target_entry_id": target_entry_id,
                "storage_path": str(storage_path),
                "backup_path": str(backup_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MigrationError as err:
        print(str(err), file=sys.stderr)
        raise SystemExit(1) from err
