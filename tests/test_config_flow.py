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
    CONF_LEARNING_MODEL,
    CONF_LEARNING_RATE,
    CONF_MAX_TEMPERATURE,
    CONF_MIN_TEMPERATURE,
    CONF_OUTDOOR_ENTITY_ID,
    CONF_TARGET_TEMPERATURE,
    CONF_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_UNDERLYING_ENTITY_IDS,
    DEFAULT_LEARNING_MODEL,
    DOMAIN,
    LEARNING_MODEL_PHYSICAL,
)

pytestmark = pytest.mark.asyncio


def _user_input() -> dict:
    return {
        "name": "Kitchen Floor",
        CONF_UNDERLYING_ENTITY_IDS: ["switch.floor_a", "switch.floor_b"],
        CONF_TEMPERATURE_SENSOR_ENTITY_ID: "sensor.kitchen_temperature",
        CONF_OUTDOOR_ENTITY_ID: "weather.home",
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
    assert result["data"][CONF_LEARNING_MODEL] == DEFAULT_LEARNING_MODEL


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


async def test_schema_allows_weather_or_sensor_for_outdoor_source() -> None:
    """The outdoor source should accept either a weather entity or a sensor."""
    schema = _schema()

    field = next(
        key
        for key in schema.schema
        if getattr(key, "schema", None) == CONF_OUTDOOR_ENTITY_ID
    )
    entity_selector = schema.schema[field]

    assert isinstance(entity_selector, selector.EntitySelector)
    assert entity_selector.config["domain"] == ["weather", "sensor"]


async def test_schema_exposes_learning_model_choices() -> None:
    """The form should expose Linear and Physical learning model choices."""
    schema = _schema()

    field = next(
        key
        for key in schema.schema
        if getattr(key, "schema", None) == CONF_LEARNING_MODEL
    )
    model_selector = schema.schema[field]

    assert isinstance(model_selector, selector.SelectSelector)
    assert model_selector.config["options"] == [
        {"value": "linear", "label": "Linear"},
        {"value": "physical", "label": "Physical"},
    ]


async def test_user_flow_accepts_physical_learning_model(hass) -> None:
    """The selected learning model should be persisted in config entry data."""
    user_input = {**_user_input(), CONF_LEARNING_MODEL: LEARNING_MODEL_PHYSICAL}

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data=user_input,
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_LEARNING_MODEL] == LEARNING_MODEL_PHYSICAL


async def test_options_flow_can_switch_learning_model(hass) -> None:
    """The options flow should update entry data with the selected model."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen Floor",
        data=_user_input(),
        unique_id="switch_floor_a_switch_floor_b",
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(
        entry.entry_id,
        data={**_user_input(), CONF_LEARNING_MODEL: LEARNING_MODEL_PHYSICAL},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_LEARNING_MODEL] == LEARNING_MODEL_PHYSICAL


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
