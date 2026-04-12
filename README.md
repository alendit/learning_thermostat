# learning_thermostat

Standalone Home Assistant custom integration for a learning thermostat focused
on hydronic underfloor heating with direct on/off switch control.

This repository contains:

- `custom_components/learning_thermostat/`: the custom integration
- `tests/`: automated tests
- `scripts/`: CI, deploy, and migration utilities
- `.drone.yml`: local Drone pipeline

## Scope

The first version intentionally supports only:

- `climate` entity control
- direct switch `turn_on` / `turn_off`
- room temperature sensor input
- optional outdoor temperature sensor input
- optional weather forecast entity input
- persistent learning weights and migration from `versatile_thermostat`

It intentionally does not include translation support, inverse switch behavior,
custom switch command strings, switch keep-alive logic, or staggered
multi-switch activation.

## Migration

The migration utility reads `learning_state` and `learning_last_tick` from an
existing `versatile_thermostat` learning thermostat entity over the Home
Assistant REST API and writes them into this integration's storage file.

Example:

```bash
uv run python scripts/migrate_learning_state.py \
  --base-url http://homeassistant.local:8123 \
  --token "$HASS_TOKEN" \
  --old-entity-id climate.old_learning_thermostat \
  --new-entity-id climate.new_learning_thermostat \
  --config-dir /path/to/homeassistant
```

By default the script refuses to write while a Home Assistant lock file is
present. It also writes a backup of the affected storage file before changing
it.
