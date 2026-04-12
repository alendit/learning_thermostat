"""Climate behavior tests."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

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
    CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_TARGET_TEMPERATURE,
    CONF_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_UNDERLYING_ENTITY_IDS,
    DOMAIN,
    SERVICE_RESET_LEARNING,
    SERVICE_TRIGGER_TICK,
)

pytestmark = pytest.mark.asyncio


def _entry() -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen Floor",
        data={
            "name": "Kitchen Floor",
            CONF_UNDERLYING_ENTITY_IDS: ["switch.floor_a", "switch.floor_b"],
            CONF_TEMPERATURE_SENSOR_ENTITY_ID: "sensor.kitchen_temperature",
            CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID: "sensor.outside_temperature",
            CONF_CYCLE_MINUTES: 15,
            CONF_TARGET_TEMPERATURE: 21.5,
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


async def test_trigger_tick_turns_on_switches_when_heat_is_needed(hass) -> None:
    """A forced tick should turn on all configured switches."""
    hass.states.async_set("sensor.kitchen_temperature", "18.0")
    hass.states.async_set("sensor.outside_temperature", "7.0")
    hass.states.async_set("switch.floor_a", "off")
    hass.states.async_set("switch.floor_b", "off")
    turn_on_calls = async_mock_service(hass, "switch", "turn_on")
    async_mock_service(hass, "switch", "turn_off")

    entry = _entry()
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    await hass.services.async_call(
        DOMAIN,
        SERVICE_TRIGGER_TICK,
        {"entity_id": "climate.kitchen_floor", "force": True},
        blocking=True,
    )
    await hass.async_block_till_done()

    assert [call.data["entity_id"] for call in turn_on_calls] == [
        "switch.floor_a",
        "switch.floor_b",
    ]

    state = hass.states.get("climate.kitchen_floor")
    assert state is not None
    assert state.attributes["learning_state"] is not None


async def test_hvac_off_turns_off_switches(hass) -> None:
    """Turning the thermostat off should turn off all switches."""
    hass.states.async_set("sensor.kitchen_temperature", "20.0")
    hass.states.async_set("sensor.outside_temperature", "8.0")
    hass.states.async_set("switch.floor_a", "on")
    hass.states.async_set("switch.floor_b", "on")
    async_mock_service(hass, "switch", "turn_on")
    turn_off_calls = async_mock_service(hass, "switch", "turn_off")

    entry = _entry()
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    await hass.services.async_call(
        "climate",
        "set_hvac_mode",
        {"entity_id": "climate.kitchen_floor", "hvac_mode": "off"},
        blocking=True,
    )
    await hass.async_block_till_done()

    assert [call.data["entity_id"] for call in turn_off_calls] == [
        "switch.floor_a",
        "switch.floor_b",
    ]


async def test_reset_learning_preserves_state_shape(hass) -> None:
    """Resetting learning should keep a valid persisted state payload."""
    hass.states.async_set("sensor.kitchen_temperature", "18.0")
    hass.states.async_set("sensor.outside_temperature", "7.0")
    hass.states.async_set("switch.floor_a", "off")
    hass.states.async_set("switch.floor_b", "off")
    async_mock_service(hass, "switch", "turn_on")
    async_mock_service(hass, "switch", "turn_off")

    entry = _entry()
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    await hass.services.async_call(
        DOMAIN,
        SERVICE_TRIGGER_TICK,
        {"entity_id": "climate.kitchen_floor", "force": True},
        blocking=True,
    )
    await hass.async_block_till_done()

    await hass.services.async_call(
        DOMAIN,
        SERVICE_RESET_LEARNING,
        {"entity_id": "climate.kitchen_floor", "preserve_history": True},
        blocking=True,
    )
    await hass.async_block_till_done()

    state = hass.states.get("climate.kitchen_floor")
    assert state is not None
    assert "history" in state.attributes["learning_state"]
