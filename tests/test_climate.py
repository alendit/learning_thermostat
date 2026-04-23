"""Climate behavior tests."""

from __future__ import annotations

import pytest
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.learning_thermostat import climate as climate_module
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
    CONF_OUTDOOR_ENTITY_ID,
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
            CONF_OUTDOOR_ENTITY_ID: "sensor.outside_temperature",
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


async def test_weather_entity_provides_current_and_forecast_outdoor_temperature(
    hass, monkeypatch
) -> None:
    """A weather outdoor source should drive both current and forecast temperatures."""
    hass.states.async_set("sensor.kitchen_temperature", "18.0")
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "temperature": 7.0,
        },
    )
    hass.states.async_set("switch.floor_a", "off")
    hass.states.async_set("switch.floor_b", "off")
    async_mock_service(hass, "switch", "turn_on")
    async_mock_service(hass, "switch", "turn_off")

    async def fake_weather_forecast(_hass, _entity_id, _forecast_type="hourly"):
        return {
            "forecast": [
                {"temperature": 6.5},
                {"temperature": 6.0},
            ]
        }

    monkeypatch.setattr(
        climate_module, "async_weather_get_forecast", fake_weather_forecast
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen Floor",
        data={
            **_entry().data,
            CONF_OUTDOOR_ENTITY_ID: "weather.home",
        },
    )
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

    state = hass.states.get("climate.kitchen_floor")
    assert state is not None
    assert state.attributes["outside_temperature"] == 7.0
    assert state.attributes["learning_result"]["forecast_temps"][:2] == [6.5, 6.0]


async def test_detect_pirate_weather_solar_entities(hass) -> None:
    """Pirate Weather entries should auto-link the current and hourly solar sensors."""
    pirate_entry = MockConfigEntry(domain="pirateweather", title="Pirate Weather")
    pirate_entry.add_to_hass(hass)

    registry = er.async_get(hass)
    weather_entry = registry.async_get_or_create(
        "weather",
        "pirateweather",
        "weather-unique",
        config_entry=pirate_entry,
        object_id_base="home",
        original_name="Home",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-current",
        config_entry=pirate_entry,
        object_id_base="home_solar",
        original_name="Downward Short-Wave Radiation Flux",
        translation_key="solar",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-1h",
        config_entry=pirate_entry,
        object_id_base="home_solar_1h",
        original_name="Downward Short-Wave Radiation Flux 1h",
        translation_key="solar",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-2h",
        config_entry=pirate_entry,
        object_id_base="home_solar_2h",
        original_name="Downward Short-Wave Radiation Flux 2h",
        translation_key="solar",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-max",
        config_entry=pirate_entry,
        object_id_base="home_solar_max",
        original_name="Downward Short-Wave Radiation Flux Max",
        translation_key="solar_max",
    )

    detected = climate_module._detect_pirate_weather_solar_entities(
        hass, weather_entry.entity_id
    )

    assert detected.current_entity_id == "sensor.home_solar"
    assert list(detected.forecast_entity_ids) == [
        "sensor.home_solar_1h",
        "sensor.home_solar_2h",
    ]


async def test_detect_pirate_weather_live_entity_names(hass) -> None:
    """Pirate Weather auto-detection should recognize the live irradiance entity names."""
    pirate_entry = MockConfigEntry(domain="pirateweather", title="Pirate Weather")
    pirate_entry.add_to_hass(hass)

    registry = er.async_get(hass)
    weather_entry = registry.async_get_or_create(
        "weather",
        "pirateweather",
        "weather-live-unique",
        config_entry=pirate_entry,
        object_id_base="pirateweather",
        original_name="PirateWeather",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-current-live",
        config_entry=pirate_entry,
        object_id_base="pirateweather_downward_short_wave_radiation_flux",
        original_name="PirateWeather Downward Short-Wave Radiation Flux",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-1h-live",
        config_entry=pirate_entry,
        object_id_base="pirateweather_downward_short_wave_radiation_flux_1h",
        original_name="PirateWeather Downward Short-Wave Radiation Flux 1h",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-max-live",
        config_entry=pirate_entry,
        object_id_base="pirateweather_downward_short_wave_radiation_flux_max",
        original_name="PirateWeather Downward Short-Wave Radiation Flux Max",
    )

    detected = climate_module._detect_pirate_weather_solar_entities(
        hass, weather_entry.entity_id
    )

    assert (
        detected.current_entity_id
        == "sensor.pirateweather_downward_short_wave_radiation_flux"
    )
    assert list(detected.forecast_entity_ids) == [
        "sensor.pirateweather_downward_short_wave_radiation_flux_1h",
    ]


async def test_pirate_weather_solar_uses_sun_position_for_profiles(
    hass, monkeypatch
) -> None:
    """Solar flux should be combined with current and forecast sun position."""
    pirate_entry = MockConfigEntry(domain="pirateweather", title="Pirate Weather")
    pirate_entry.add_to_hass(hass)

    registry = er.async_get(hass)
    registry.async_get_or_create(
        "weather",
        "pirateweather",
        "weather-profile-unique",
        config_entry=pirate_entry,
        object_id_base="pirateweather",
        original_name="PirateWeather",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-current-profile",
        config_entry=pirate_entry,
        object_id_base="pirateweather_downward_short_wave_radiation_flux",
        original_name="PirateWeather Downward Short-Wave Radiation Flux",
    )
    registry.async_get_or_create(
        "sensor",
        "pirateweather",
        "solar-1h-profile",
        config_entry=pirate_entry,
        object_id_base="pirateweather_downward_short_wave_radiation_flux_1h",
        original_name="PirateWeather Downward Short-Wave Radiation Flux 1h",
    )

    hass.states.async_set("sensor.kitchen_temperature", "18.0")
    hass.states.async_set("weather.pirateweather", "sunny", {"temperature": 7.0})
    hass.states.async_set("switch.floor_a", "off")
    hass.states.async_set("switch.floor_b", "off")
    hass.states.async_set(
        "sun.sun",
        "above_horizon",
        {"azimuth": 180.0, "elevation": 45.0},
    )
    hass.states.async_set(
        "sensor.pirateweather_downward_short_wave_radiation_flux",
        "200.0",
    )
    hass.states.async_set(
        "sensor.pirateweather_downward_short_wave_radiation_flux_1h",
        "100.0",
    )
    async_mock_service(hass, "switch", "turn_on")
    async_mock_service(hass, "switch", "turn_off")

    monkeypatch.setattr(
        climate_module,
        "_solar_position_at",
        lambda _hass, _when: climate_module.SunPosition(azimuth=180.0, elevation=45.0),
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen Floor",
        data={
            **_entry().data,
            CONF_OUTDOOR_ENTITY_ID: "weather.pirateweather",
        },
    )
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

    state = hass.states.get("climate.kitchen_floor")
    assert state is not None
    current_profile = state.attributes["current_solar_profile"]
    assert current_profile[0] == 200.0
    assert current_profile[5] > current_profile[1]
    assert state.attributes["solar_orientation_azimuth"] is None
    forecast_profile = state.attributes["learning_result"]["forecast_solar_profile"][0]
    assert forecast_profile[0] == 100.0
    assert forecast_profile[5] > forecast_profile[1]
