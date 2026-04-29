"""Home Assistant environmental input collector tests."""

from __future__ import annotations

import pytest

from custom_components.learning_thermostat.config_contract import ThermostatConfig
from custom_components.learning_thermostat.environment import DetectedSolarEntities
from custom_components.learning_thermostat.environment_input import (
    EnvironmentInputCollector,
)

pytestmark = pytest.mark.asyncio


async def test_collector_builds_current_environment_and_cycle_input(hass) -> None:
    """Collector should translate HA state into plain controller input data."""
    config = ThermostatConfig.from_mapping(
        name="Kitchen Floor",
        data={
            "underlying_entity_ids": ["switch.floor_a", "switch.floor_b"],
            "temperature_sensor_entity_id": "sensor.kitchen_temperature",
            "outdoor_entity_id": "weather.home",
            "cycle_minutes": 15,
            "target_temperature": 21.5,
            "min_temperature": 15.0,
            "max_temperature": 28.0,
            "learning_lookahead_steps": 3,
            "learning_history_steps": 6,
            "learning_l2_reg": 0.001,
            "learning_error_decay": 0.9,
            "learning_rate": 0.005,
            "learning_energy": 0.01,
        },
    )
    detected_solar = DetectedSolarEntities(
        current_entity_id="sensor.home_solar",
        forecast_entity_ids=("sensor.home_solar_1h", "sensor.home_solar_2h"),
        forecast_hours=(1, 2),
    )
    hass.states.async_set("sensor.kitchen_temperature", "18.0")
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "temperature": 7.0,
            "forecast": [{"temperature": 6.5}, {"temperature": 6.0}],
        },
    )
    hass.states.async_set("sensor.home_solar", "200.0")
    hass.states.async_set("sensor.home_solar_1h", "100.0")
    hass.states.async_set("sensor.home_solar_2h", "80.0")
    hass.states.async_set(
        "sun.sun",
        "above_horizon",
        {"azimuth": 180.0, "elevation": 45.0},
    )
    hass.states.async_set("switch.floor_a", "off")
    hass.states.async_set("switch.floor_b", "on")

    collector = EnvironmentInputCollector(hass, config)

    current = collector.current_environment(detected_solar)
    cycle_input = await collector.async_control_cycle_input(
        force=True,
        detected_solar=detected_solar,
        current=current,
    )

    assert current.current_temperature == 18.0
    assert current.outside_temperature == 7.0
    assert current.solar == 200.0
    assert current.solar_sample == {
        "flux": 200.0,
        "azimuth": 180.0,
        "elevation": 45.0,
    }
    assert cycle_input.force is True
    assert cycle_input.forecast_temperatures == [6.5, 6.0]
    assert cycle_input.solar == 200.0
    assert cycle_input.solar_forecast == [100.0, 80.0]
    assert cycle_input.solar_sample == current.solar_sample
    assert cycle_input.solar_sample_forecast[0]["flux"] == 100.0
    assert cycle_input.current_switch_active is True
