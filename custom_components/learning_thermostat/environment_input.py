"""Home Assistant input collection for thermostat control cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant, State

from .config_contract import ThermostatConfig
from .controller import ControlCycleInput
from .environment import (
    DetectedSolarEntities,
    async_get_forecast_temperatures,
    build_current_solar_sample,
    coerce_outdoor_temperature,
    coerce_state_temperature,
    forecast_entity_id,
    get_solar_forecast_samples,
    get_solar_sample_forecast,
)
from .switches import any_switch_active


@dataclass(frozen=True, slots=True)
class CurrentEnvironmentInput:
    """Current environmental inputs read from Home Assistant state."""

    current_temperature: float | None
    outside_temperature: float | None
    solar: float | None
    solar_sample: dict[str, Any] | None


class EnvironmentInputCollector:
    """Translate Home Assistant state into controller input contracts."""

    def __init__(self, hass: HomeAssistant, config: ThermostatConfig) -> None:
        self._hass = hass
        self._config = config

    def current_environment(
        self, detected_solar: DetectedSolarEntities
    ) -> CurrentEnvironmentInput:
        """Read current temperature and solar state from Home Assistant."""
        solar = coerce_state_temperature(
            self._hass.states.get(detected_solar.current_entity_id)
            if detected_solar.current_entity_id
            else None
        )
        return CurrentEnvironmentInput(
            current_temperature=coerce_state_temperature(
                self._hass.states.get(self._config.temperature_sensor_entity_id)
            ),
            outside_temperature=self.current_outdoor_temperature(),
            solar=solar,
            solar_sample=build_current_solar_sample(self._hass, solar),
        )

    def current_temperature_from_state(self, state: State | None) -> float | None:
        """Coerce a Home Assistant state into a current room temperature."""
        return coerce_state_temperature(state)

    def outdoor_temperature_from_state(self, state: State | None) -> float | None:
        """Coerce a Home Assistant state into an outdoor temperature."""
        return coerce_outdoor_temperature(state)

    def solar_from_state(
        self, state: State | None
    ) -> tuple[float | None, dict[str, Any] | None]:
        """Coerce a solar state into flux and raw solar sample values."""
        solar = coerce_state_temperature(state)
        return solar, build_current_solar_sample(self._hass, solar)

    def current_outdoor_temperature(self) -> float | None:
        """Return the configured outdoor source's current temperature."""
        if not self._config.outdoor_entity_id:
            return None
        return coerce_outdoor_temperature(
            self._hass.states.get(self._config.outdoor_entity_id)
        )

    async def async_control_cycle_input(
        self,
        *,
        force: bool,
        detected_solar: DetectedSolarEntities,
        current: CurrentEnvironmentInput,
    ) -> ControlCycleInput:
        """Build one controller control-cycle input."""
        solar_samples = get_solar_forecast_samples(
            self._hass, detected_solar, self._config.lookahead_steps
        )
        return ControlCycleInput(
            force=force,
            forecast_temperatures=await async_get_forecast_temperatures(
                self._hass, self._forecast_entity_id, self._config.lookahead_steps
            ),
            solar=current.solar,
            solar_forecast=[solar for _hour, solar in solar_samples],
            solar_sample=current.solar_sample,
            solar_sample_forecast=get_solar_sample_forecast(
                self._hass, self._config.lookahead_steps, solar_samples
            ),
            current_switch_active=any_switch_active(
                self._hass, self._config.switch_entity_ids
            ),
        )

    @property
    def _forecast_entity_id(self) -> str | None:
        """Return the best forecast source for the configured outdoor input."""
        return forecast_entity_id(
            self._config.outdoor_entity_id,
            self._config.legacy_forecast_entity_id,
        )
