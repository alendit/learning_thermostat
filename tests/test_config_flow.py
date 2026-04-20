"""Config flow tests for learning_thermostat."""

from __future__ import annotations

from homeassistant.data_entry_flow import FlowResultType
import pytest
from homeassistant.helpers import selector
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.learning_thermostat.config_flow import _schema
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
)

pytestmark = pytest.mark.asyncio


def _user_input() -> dict:
    return {
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
    }


async def test_user_flow_creates_entry(hass) -> None:
    """The user flow should create a config entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data=_user_input(),
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Kitchen Floor"
    assert result["data"][CONF_UNDERLYING_ENTITY_IDS] == [
        "switch.floor_a",
        "switch.floor_b",
    ]


async def test_user_flow_without_input_shows_form(hass) -> None:
    """Opening the config flow should render the form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"


async def test_schema_accepts_input_boolean_underlyings() -> None:
    """The form should allow switch-like input booleans as actuators."""
    schema = _schema()

    field = next(
        key
        for key in schema.schema
        if getattr(key, "schema", None) == CONF_UNDERLYING_ENTITY_IDS
    )
    entity_selector = schema.schema[field]

    assert isinstance(entity_selector, selector.EntitySelector)
    assert entity_selector.config["domain"] == ["switch", "input_boolean"]


async def test_user_flow_rejects_duplicate_switch_set(hass) -> None:
    """The same switch set should not be configured twice."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen Floor",
        data=_user_input(),
        unique_id="switch_floor_a_switch_floor_b",
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data=_user_input(),
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
