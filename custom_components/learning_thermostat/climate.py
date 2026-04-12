"""Climate platform for the learning thermostat."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

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
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CYCLE_MINUTES,
    CONF_FORECAST_ENTITY_ID,
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
    DATA_ENTITIES,
    DATA_STORAGE,
    DOMAIN,
    NAME,
)
from .optimizer import DEFAULT_LEARNING_SWITCH_CONFIG, LearningSwitchOptimizer
from .storage import LearningStateStore, build_entry_payload

_LOGGER = logging.getLogger(__name__)

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
        self._switch_entity_ids = list(entry.data[CONF_UNDERLYING_ENTITY_IDS])
        self._temperature_sensor_entity_id = entry.data[
            CONF_TEMPERATURE_SENSOR_ENTITY_ID
        ]
        self._outdoor_sensor_entity_id = entry.data.get(
            CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID
        )
        self._forecast_entity_id = entry.data.get(CONF_FORECAST_ENTITY_ID)
        self._cycle_minutes = int(entry.data[CONF_CYCLE_MINUTES])
        self._min_temp = float(entry.data[CONF_MIN_TEMPERATURE])
        self._max_temp = float(entry.data[CONF_MAX_TEMPERATURE])
        self._target_temperature = float(entry.data[CONF_TARGET_TEMPERATURE])
        self._lookahead_steps = int(entry.data[CONF_LEARNING_LOOKAHEAD_STEPS])
        self._learning_config = {
            **DEFAULT_LEARNING_SWITCH_CONFIG,
            "lookahead_steps": self._lookahead_steps,
            "history_steps": int(entry.data[CONF_LEARNING_HISTORY_STEPS]),
            "l2_reg": float(entry.data[CONF_LEARNING_L2_REG]),
            "error_decay": float(entry.data[CONF_LEARNING_ERROR_DECAY]),
            "learning_rate": float(entry.data[CONF_LEARNING_RATE]),
            "w_energy": float(entry.data[CONF_LEARNING_ENERGY]),
        }
        self._optimizer: LearningSwitchOptimizer | None = None
        self._learning_state: dict[str, Any] | None = None
        self._learning_result: dict[str, Any] | None = None
        self._learning_last_tick = None
        self._learning_on_percent: float | None = None
        self._current_temperature: float | None = None
        self._outside_temperature: float | None = None
        self._hvac_mode = HVACMode.HEAT
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
        return self._current_temperature

    @property
    def target_temperature(self) -> float | None:
        """Return the active target temperature."""
        return self._target_temperature

    @property
    def min_temp(self) -> float:
        """Return the minimum allowed target temperature."""
        return self._min_temp

    @property
    def max_temp(self) -> float:
        """Return the maximum allowed target temperature."""
        return self._max_temp

    @property
    def hvac_mode(self) -> HVACMode:
        """Return the current HVAC mode."""
        return self._hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current HVAC action."""
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        return HVACAction.HEATING if self._is_any_switch_active() else HVACAction.IDLE

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose learning details for diagnostics and migration."""
        return {
            "learning_state": self._learning_state,
            "learning_result": self._learning_result,
            "learning_last_tick": (
                self._learning_last_tick.isoformat()
                if self._learning_last_tick is not None
                else None
            ),
            "outside_temperature": self._outside_temperature,
            "on_percent": self._learning_on_percent,
            "underlying_entities": list(self._switch_entity_ids),
            "forecast_entity_id": self._forecast_entity_id,
        }

    async def async_added_to_hass(self) -> None:
        """Register listeners and restore state."""
        await super().async_added_to_hass()
        self.hass.data[DOMAIN][DATA_ENTITIES][self.entity_id] = self
        await self._async_restore_runtime_state()
        await self._async_refresh_temperature_inputs()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                [self._temperature_sensor_entity_id],
                self._async_handle_temperature_change,
            )
        )

        if self._outdoor_sensor_entity_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [self._outdoor_sensor_entity_id],
                    self._async_handle_outdoor_temperature_change,
                )
            )

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                self._switch_entity_ids,
                self._async_handle_switch_change,
            )
        )
        self.async_on_remove(
            async_track_time_interval(
                self.hass,
                self._async_handle_scheduled_tick,
                timedelta(minutes=self._cycle_minutes),
            )
        )

    async def async_will_remove_from_hass(self) -> None:
        """Clean up the runtime entity registry."""
        self.hass.data.get(DOMAIN, {}).get(DATA_ENTITIES, {}).pop(self.entity_id, None)

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set the HVAC mode."""
        self._hvac_mode = hvac_mode
        await self._async_run_control_cycle(force=True)

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set a new target temperature."""
        if ATTR_TEMPERATURE in kwargs:
            self._target_temperature = float(kwargs[ATTR_TEMPERATURE])
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
        previous_state = self._learning_state if preserve_history else None
        self._optimizer = LearningSwitchOptimizer(
            self._name, config=self._learning_config
        )
        self._learning_result = None
        base_state = self._optimizer.snapshot_state()
        if preserve_history and isinstance(previous_state, dict):
            history = previous_state.get("history")
            if isinstance(history, dict):
                history_copy = dict(history)
                for key in (
                    "pred_abs_error",
                    "pred_timestamps",
                    "k_step_abs_error",
                    "k_step_pred_improvement",
                ):
                    history_copy.pop(key, None)
                base_state["history"] = history_copy
            for key in ("last_u", "last_plan"):
                if key in previous_state:
                    base_state[key] = previous_state.get(key)
        self._optimizer.restore_state(base_state)
        self._learning_state = self._optimizer.snapshot_state()
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
        self._current_temperature = _coerce_state_temperature(
            event.data.get("new_state")
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_outdoor_temperature_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the outdoor temperature from a state event."""
        self._outside_temperature = _coerce_state_temperature(
            event.data.get("new_state")
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
        self._current_temperature = _coerce_state_temperature(
            self.hass.states.get(self._temperature_sensor_entity_id)
        )
        if self._outdoor_sensor_entity_id:
            self._outside_temperature = _coerce_state_temperature(
                self.hass.states.get(self._outdoor_sensor_entity_id)
            )

    async def _async_restore_runtime_state(self) -> None:
        """Restore the HA and learning state from storage."""
        old_state = await self.async_get_last_state()
        if old_state is not None:
            if old_state.state in (HVACMode.HEAT, HVACMode.OFF):
                self._hvac_mode = HVACMode(old_state.state)
            restored_target = old_state.attributes.get(ATTR_TEMPERATURE)
            if restored_target is not None:
                self._target_temperature = float(restored_target)

        stored = await self._store.async_get_entry(self._entry.entry_id)
        if stored is None:
            return

        learning_state = stored.get("learning_state")
        if isinstance(learning_state, dict):
            self._learning_state = learning_state
        last_tick = stored.get("learning_last_tick")
        if isinstance(last_tick, str):
            self._learning_last_tick = dt_util.parse_datetime(last_tick)
        self._current_temperature = _first_not_none(
            self._current_temperature,
            _coerce_float(stored.get("current_temperature")),
        )
        self._outside_temperature = _first_not_none(
            self._outside_temperature,
            _coerce_float(stored.get("outside_temperature")),
        )

    def _ensure_optimizer(self) -> LearningSwitchOptimizer:
        """Return the optimizer, restoring persisted state when available."""
        if self._optimizer is None:
            self._optimizer = LearningSwitchOptimizer(
                self._name,
                config=self._learning_config,
            )
            if self._learning_state is not None:
                self._optimizer.restore_state(self._learning_state)
        return self._optimizer

    async def _async_run_control_cycle(self, force: bool) -> bool:
        """Run one thermostat control cycle."""
        await self._async_refresh_temperature_inputs()

        if self._hvac_mode == HVACMode.OFF:
            self._learning_on_percent = 0.0
            await self._async_turn_all_switches(False)
            await self._async_persist_learning_state()
            self.async_write_ha_state()
            return True

        if not force and self._learning_last_tick is not None:
            elapsed = dt_util.utcnow() - self._learning_last_tick
            if elapsed < timedelta(minutes=self._cycle_minutes):
                return False

        if self._current_temperature is None or self._target_temperature is None:
            _LOGGER.debug(
                "Skipping learning cycle because room temperature or target temperature is missing"
            )
            return False

        forecast_temperatures = await self._async_get_forecast_temperatures(
            self._lookahead_steps
        )
        outside_temperature = self._outside_temperature
        if outside_temperature is None and forecast_temperatures:
            outside_temperature = forecast_temperatures[0]
        if outside_temperature is None:
            outside_temperature = self._current_temperature

        optimizer = self._ensure_optimizer()
        result = optimizer.tick(
            self._current_temperature,
            outside_temperature,
            self._target_temperature,
            forecast_temperatures,
            current_u=1 if self._is_any_switch_active() else 0,
            step_minutes=self._cycle_minutes,
            timestamp=dt_util.utcnow().timestamp(),
        )
        self._learning_result = result
        self._learning_state = optimizer.snapshot_state()
        self._learning_last_tick = dt_util.utcnow()
        should_heat = bool(result.get("u_planned", 0))
        self._learning_on_percent = 1.0 if should_heat else 0.0

        await self._async_turn_all_switches(should_heat)
        await self._async_persist_learning_state()
        self.async_write_ha_state()
        return True

    async def _async_persist_learning_state(self) -> None:
        """Persist the learning state contract used by restore and migration."""
        await self._store.async_save_entry(
            self._entry.entry_id,
            build_entry_payload(
                learning_state=self._learning_state,
                learning_last_tick=(
                    self._learning_last_tick.isoformat()
                    if self._learning_last_tick is not None
                    else None
                ),
                current_temperature=self._current_temperature,
                outside_temperature=self._outside_temperature,
            ),
        )

    async def _async_turn_all_switches(self, turn_on: bool) -> None:
        """Turn all configured switch entities on or off."""
        service = "turn_on" if turn_on else "turn_off"
        for entity_id in self._switch_entity_ids:
            await self.hass.services.async_call(
                entity_id.split(".", 1)[0],
                service,
                {ATTR_ENTITY_ID: entity_id},
                blocking=True,
            )

    def _is_any_switch_active(self) -> bool:
        """Return whether any configured switch is currently on."""
        for entity_id in self._switch_entity_ids:
            state = self.hass.states.get(entity_id)
            if state is not None and state.state == STATE_ON:
                return True
        return False

    async def _async_get_forecast_temperatures(self, steps: int) -> list[float]:
        """Build forecast temperatures for the lookahead horizon."""
        if not self._forecast_entity_id or steps <= 0:
            return []

        state = self.hass.states.get(self._forecast_entity_id)
        if state is None:
            return []

        forecast = state.attributes.get("forecast")
        if isinstance(forecast, list):
            return _coerce_forecast_list(forecast, steps)

        if self._forecast_entity_id.split(".", 1)[0] == WEATHER_DOMAIN:
            fetched = await _async_weather_forecast(self.hass, self._forecast_entity_id)
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


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None
