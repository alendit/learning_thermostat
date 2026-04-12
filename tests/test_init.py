"""Integration setup tests."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.learning_thermostat.const import (
    CONF_CYCLE_MINUTES,
    CONF_LEARNING_ENERGY,
    CONF_LEARNING_ERROR_DECAY,
    CONF_LEARNING_HISTORY_STEPS,
    CONF_LEARNING_L2_REG,
    CONF_LEARNING_LOOKAHEAD_STEPS,
    CONF_LEARNING_RATE,
    CONF_MAX_TEMPERATURE,
    CONF_MIN_TEMPERATURE,
    CONF_TARGET_TEMPERATURE,
    CONF_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_UNDERLYING_ENTITY_IDS,
    DOMAIN,
)

pytestmark = pytest.mark.asyncio


def _entry() -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen Floor",
        data={
            "name": "Kitchen Floor",
            CONF_UNDERLYING_ENTITY_IDS: ["switch.floor_a"],
            CONF_TEMPERATURE_SENSOR_ENTITY_ID: "sensor.kitchen_temperature",
            CONF_CYCLE_MINUTES: 15,
            CONF_TARGET_TEMPERATURE: 21.0,
            CONF_MIN_TEMPERATURE: 15.0,
            CONF_MAX_TEMPERATURE: 28.0,
            CONF_LEARNING_LOOKAHEAD_STEPS: 8,
            CONF_LEARNING_HISTORY_STEPS: 6,
            CONF_LEARNING_L2_REG: 0.001,
            CONF_LEARNING_ERROR_DECAY: 0.9,
            CONF_LEARNING_RATE: 0.005,
            CONF_LEARNING_ENERGY: 0.01,
        },
    )


async def test_setup_entry_creates_climate_entity(hass) -> None:
    """Setting up the entry should expose one climate entity."""
    hass.states.async_set("sensor.kitchen_temperature", "20.0")
    hass.states.async_set("switch.floor_a", "off")

    entry = _entry()
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("climate.kitchen_floor")
    assert state is not None
    assert state.state == "heat"
