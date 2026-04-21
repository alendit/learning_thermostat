# learning_thermostat

Standalone Home Assistant custom integration for a learning thermostat focused
on hydronic underfloor heating with direct on/off switch control.

This repository contains:

- `custom_components/learning_thermostat/`: the custom integration
- `tests/`: automated tests
- `scripts/`: CI, deploy, and migration utilities
- `.drone.yml`: local Drone pipeline

## Branding Assets

Brand assets use this ownership model:

- `assets/brand/*.svg`: editable source assets
- `custom_components/learning_thermostat/brand/*.png`: tracked runtime assets

Home Assistant reads the PNG files from the integration `brand/` directory.
When branding changes, update the SVG source assets first and then refresh the
PNG files in `custom_components/learning_thermostat/brand/` from those source
files.

## Scope

The first version intentionally supports only:

- `climate` entity control
- direct switch `turn_on` / `turn_off`
- room temperature sensor input
- optional outdoor source via temperature sensor or weather entity
- weather entities supply both current outdoor temperature and forecast temperatures
- optional auto-detected solar input when the selected weather provider exposes linked irradiance sensors
- persistent learning weights and migration from `versatile_thermostat`

It intentionally does not include translation support, inverse switch behavior,
custom switch command strings, switch keep-alive logic, or staggered
multi-switch activation.

Underlying on/off control entities may be either Home Assistant `switch`
entities or `input_boolean` helpers, as long as they support standard
`turn_on` / `turn_off` services.

## Migration

The migration utility reads `learning_state` and `learning_last_tick` from an
existing `versatile_thermostat` learning thermostat entity over the Home
Assistant REST API and writes them into this integration's storage file.

This is enough to preserve learned model state, but it does not migrate the
broader `versatile_thermostat` feature surface such as presets, safety/lock
features, or automations that still point at the old entity IDs.

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
