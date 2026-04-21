"""Climate platform for the learning thermostat."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import voluptuous as vol

from homeassistant.components.climate import ClimateEntity
from homeassistant.components.climate.const import (
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.components.weather import DOMAIN as WEATHER_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import service as hass_service
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback,
    async_get_current_platform,
)
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DATA_SERVICES_REGISTERED,
    DATA_STORAGE,
    DOMAIN,
    NAME,
    SERVICE_RESET_LEARNING,
    SERVICE_TRIGGER_TICK,
)
from .controller import ThermostatConfig, ThermostatController
from .storage import LearningEntryPayload, LearningStateStore

try:
    from homeassistant.components.weather import (
        async_get_forecast as async_weather_get_forecast,
    )
except ImportError:  # pragma: no cover - older HA versions
    async_weather_get_forecast = None


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the learning thermostat climate entity."""
    _async_ensure_entity_services_registered(hass)
    entity = LearningThermostatEntity(
        hass,
        entry,
        hass.data[DOMAIN][DATA_STORAGE],
    )
    async_add_entities([entity], True)


class LearningThermostatEntity(ClimateEntity, RestoreEntity):
    """A focused learning thermostat climate entity."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
    _attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE
    _attr_translation_key = DOMAIN

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        store: LearningStateStore,
    ) -> None:
        self.hass = hass
        self._entry = entry
        self._store = store
        self._name = entry.title or entry.data.get("name", NAME)
        self._config = ThermostatConfig.from_mapping(
            name=self._name,
            data=dict(entry.data),
        )
        self._controller = ThermostatController(self._config)
        self._attr_name = self._name
        self._attr_unique_id = entry.entry_id
        self._attr_should_poll = False

    @property
    def temperature_unit(self) -> str:
        """Return the configured temperature unit."""
        return self.hass.config.units.temperature_unit

    @property
    def current_temperature(self) -> float | None:
        """Return the current room temperature."""
        return self._controller.current_temperature

    @property
    def target_temperature(self) -> float | None:
        """Return the active target temperature."""
        return self._controller.target_temperature

    @property
    def min_temp(self) -> float:
        """Return the minimum allowed target temperature."""
        return self._config.min_temperature

    @property
    def max_temp(self) -> float:
        """Return the maximum allowed target temperature."""
        return self._config.max_temperature

    @property
    def hvac_mode(self) -> HVACMode:
        """Return the current HVAC mode."""
        return self._controller.hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current HVAC action."""
        if self._controller.hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        return HVACAction.HEATING if self._is_any_switch_active() else HVACAction.IDLE

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose learning details for diagnostics and migration."""
        return {
            "learning_state": self._controller.learning_state,
            "learning_result": self._controller.learning_result,
            "learning_last_tick": (
                self._controller.learning_last_tick.isoformat()
                if self._controller.learning_last_tick is not None
                else None
            ),
            "outside_temperature": self._controller.outside_temperature,
            "on_percent": self._controller.learning_on_percent,
            "underlying_entities": list(self._config.switch_entity_ids),
            "forecast_entity_id": self._config.forecast_entity_id,
        }

    async def async_added_to_hass(self) -> None:
        """Register listeners and restore state."""
        await super().async_added_to_hass()
        await self._async_restore_runtime_state()
        await self._async_refresh_temperature_inputs()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                [self._config.temperature_sensor_entity_id],
                self._async_handle_temperature_change,
            )
        )

        if self._config.outdoor_sensor_entity_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [self._config.outdoor_sensor_entity_id],
                    self._async_handle_outdoor_temperature_change,
                )
            )

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                list(self._config.switch_entity_ids),
                self._async_handle_switch_change,
            )
        )
        self.async_on_remove(
            async_track_time_interval(
                self.hass,
                self._async_handle_scheduled_tick,
                timedelta(minutes=self._config.cycle_minutes),
            )
        )

    async def async_will_remove_from_hass(self) -> None:
        """Clean up any runtime resources."""

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set the HVAC mode."""
        self._controller.set_hvac_mode(hvac_mode)
        await self._async_run_control_cycle(force=True)

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set a new target temperature."""
        if ATTR_TEMPERATURE in kwargs:
            self._controller.set_target_temperature(float(kwargs[ATTR_TEMPERATURE]))
            self.async_write_ha_state()
            await self._async_run_control_cycle(force=True)

    async def async_turn_on(self) -> None:
        """Turn the thermostat on."""
        await self.async_set_hvac_mode(HVACMode.HEAT)

    async def async_turn_off(self) -> None:
        """Turn the thermostat off."""
        await self.async_set_hvac_mode(HVACMode.OFF)

    async def async_reset_learning_service(self, preserve_history: bool = True) -> None:
        """Reset the persisted learning state."""
        self._controller.reset_learning(preserve_history)
        await self._async_persist_learning_state()
        self.async_write_ha_state()

    async def async_trigger_tick_service(self, force: bool = False) -> None:
        """Trigger an immediate control cycle."""
        await self._async_run_control_cycle(force=force)

    @callback
    async def _async_handle_temperature_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the room temperature from a state event."""
        self._controller.update_current_temperature(
            _coerce_state_temperature(event.data.get("new_state"))
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_outdoor_temperature_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the outdoor temperature from a state event."""
        self._controller.update_outside_temperature(
            _coerce_state_temperature(event.data.get("new_state"))
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_switch_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Refresh state after a switch changes."""
        self.async_write_ha_state()

    async def _async_handle_scheduled_tick(self, _now) -> None:
        """Handle the periodic control cycle."""
        await self._async_run_control_cycle(force=False)

    async def _async_refresh_temperature_inputs(self) -> None:
        """Load current sensor values from the state machine."""
        self._controller.update_current_temperature(
            _coerce_state_temperature(
                self.hass.states.get(self._config.temperature_sensor_entity_id)
            )
        )
        if self._config.outdoor_sensor_entity_id:
            self._controller.update_outside_temperature(
                _coerce_state_temperature(
                    self.hass.states.get(self._config.outdoor_sensor_entity_id)
                )
            )

    async def _async_restore_runtime_state(self) -> None:
        """Restore the HA and learning state from storage."""
        hvac_mode = HVACMode.HEAT
        restored_target: float | None = None
        old_state = await self.async_get_last_state()
        if old_state is not None:
            if old_state.state in (HVACMode.HEAT, HVACMode.OFF):
                hvac_mode = HVACMode(old_state.state)
            restored_target = _coerce_float(old_state.attributes.get(ATTR_TEMPERATURE))

        stored = await self._store.async_get_entry(self._entry.entry_id)
        self._controller.restore(
            hvac_mode=hvac_mode,
            target_temperature=restored_target,
            stored_payload=stored,
        )

    async def _async_run_control_cycle(self, force: bool) -> bool:
        """Run one thermostat control cycle."""
        await self._async_refresh_temperature_inputs()
        forecast_temperatures = await self._async_get_forecast_temperatures(
            self._config.lookahead_steps
        )
        ran = await self._controller.run_control_cycle(
            force=force,
            forecast_temperatures=forecast_temperatures,
            current_switch_active=self._is_any_switch_active(),
            turn_switches=self._async_turn_all_switches,
            persist=self._async_persist_learning_state_payload,
        )
        self.async_write_ha_state()
        return ran

    async def _async_persist_learning_state(self) -> None:
        """Persist the learning state contract used by restore and migration."""
        await self._async_persist_learning_state_payload(
            self._controller.build_persisted_payload()
        )

    async def _async_persist_learning_state_payload(
        self, payload: LearningEntryPayload
    ) -> None:
        """Persist one controller payload."""
        await self._store.async_save_entry(self._entry.entry_id, payload)

    async def _async_turn_all_switches(self, turn_on: bool) -> None:
        """Turn all configured switch entities on or off."""
        service = "turn_on" if turn_on else "turn_off"
        for entity_id in self._config.switch_entity_ids:
            await self.hass.services.async_call(
                entity_id.split(".", 1)[0],
                service,
                {ATTR_ENTITY_ID: entity_id},
                blocking=True,
            )

    def _is_any_switch_active(self) -> bool:
        """Return whether any configured switch is currently on."""
        for entity_id in self._config.switch_entity_ids:
            state = self.hass.states.get(entity_id)
            if state is not None and state.state == STATE_ON:
                return True
        return False

    async def _async_get_forecast_temperatures(self, steps: int) -> list[float]:
        """Build forecast temperatures for the lookahead horizon."""
        if not self._config.forecast_entity_id or steps <= 0:
            return []

        state = self.hass.states.get(self._config.forecast_entity_id)
        if state is None:
            return []

        forecast = state.attributes.get("forecast")
        if isinstance(forecast, list):
            return _coerce_forecast_list(forecast, steps)

        if self._config.forecast_entity_id.split(".", 1)[0] == WEATHER_DOMAIN:
            fetched = await _async_weather_forecast(
                self.hass, self._config.forecast_entity_id
            )
            return _coerce_forecast_list(fetched, steps)

        return []


async def _async_weather_forecast(
    hass: HomeAssistant, entity_id: str
) -> list[dict[str, Any]]:
    """Retrieve forecast data from a weather entity."""
    if async_weather_get_forecast is not None:
        try:
            for forecast_type in ("hourly", "daily", "twice_daily"):
                data = await async_weather_get_forecast(hass, entity_id, forecast_type)
                if isinstance(data, dict) and isinstance(data.get("forecast"), list):
                    return data["forecast"]
        except TypeError:
            data = await async_weather_get_forecast(hass, entity_id)
            if isinstance(data, dict) and isinstance(data.get("forecast"), list):
                return data["forecast"]
        except Exception:  # pragma: no cover - depends on HA weather backend
            return []

    return []


def _coerce_forecast_list(values: list[Any], steps: int) -> list[float]:
    """Convert a forecast payload to a list of temperatures."""
    temperatures: list[float] = []
    for value in values:
        if isinstance(value, dict):
            candidate = value.get("temperature", value.get("temp"))
        else:
            candidate = value
        coerced = _coerce_float(candidate)
        if coerced is None:
            continue
        temperatures.append(coerced)
        if len(temperatures) == steps:
            break
    return temperatures


def _coerce_state_temperature(state: State | None) -> float | None:
    if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        return None
    return _coerce_float(state.state)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _async_ensure_entity_services_registered(hass: HomeAssistant) -> None:
    """Register integration services once against HA's entity registry."""
    data = hass.data.setdefault(DOMAIN, {})
    if data.get(DATA_SERVICES_REGISTERED):
        return
    if hass.services.has_service(DOMAIN, SERVICE_RESET_LEARNING):
        data[DATA_SERVICES_REGISTERED] = True
        return

    platform = async_get_current_platform()
    hass_service.async_register_entity_service(
        hass,
        DOMAIN,
        SERVICE_RESET_LEARNING,
        entities=platform.entities,
        func="async_reset_learning_service",
        job_type=None,
        schema={
            vol.Optional("preserve_history", default=True): cv.boolean,
        },
    )
    hass_service.async_register_entity_service(
        hass,
        DOMAIN,
        SERVICE_TRIGGER_TICK,
        entities=platform.entities,
        func="async_trigger_tick_service",
        job_type=None,
        schema={
            vol.Optional("force", default=False): cv.boolean,
        },
    )
    data[DATA_SERVICES_REGISTERED] = True
