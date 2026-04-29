"""Climate platform for the learning thermostat."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components.climate import ClimateEntity
from homeassistant.components.climate.const import (
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_TEMPERATURE,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    callback,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .attributes import build_learning_diagnostic_attributes
from .const import (
    DATA_STORAGE,
    DOMAIN,
    NAME,
)
from .environment import (
    DetectedSolarEntities,
    async_get_forecast_temperatures,
    build_current_solar_sample,
    coerce_float,
    coerce_outdoor_temperature,
    coerce_state_temperature,
    detect_pirate_weather_solar_entities,
    forecast_entity_id,
    get_solar_forecast_samples,
    get_solar_sample_forecast,
)
from .runtime import LearningThermostatRuntime, async_get_or_create_runtime
from .services import async_ensure_entity_services_registered
from .state import LearningEntryPayload, ThermostatMode
from .storage import LearningStateStore
from .switches import any_switch_active, async_turn_switches


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the learning thermostat climate entity."""
    async_ensure_entity_services_registered(hass)
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
        self._runtime: LearningThermostatRuntime = async_get_or_create_runtime(
            hass, entry
        )
        self._config = self._runtime.config
        self._controller = self._runtime.controller
        self._attr_name = self._name
        self._attr_unique_id = entry.entry_id
        self._attr_should_poll = False
        self._detected_solar = DetectedSolarEntities()

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
        return _to_hvac_mode(self._controller.hvac_mode)

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current HVAC action."""
        if self._controller.hvac_mode == ThermostatMode.OFF:
            return HVACAction.OFF
        return (
            HVACAction.HEATING
            if any_switch_active(self.hass, self._config.switch_entity_ids)
            else HVACAction.IDLE
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose learning details for diagnostics and migration."""
        solar_orientation = self._controller.solar_orientation_diagnostics
        return build_learning_diagnostic_attributes(
            learning_state=self._controller.learning_state,
            learning_result=self._controller.learning_result,
            learning_last_tick=(
                self._controller.learning_last_tick.isoformat()
                if self._controller.learning_last_tick is not None
                else None
            ),
            outside_temperature=self._controller.outside_temperature,
            current_solar=self._controller.current_solar,
            current_solar_sample=self._controller.current_solar_sample,
            solar_orientation=solar_orientation,
            on_percent=self._controller.learning_on_percent,
            underlying_entities=list(self._config.switch_entity_ids),
            outdoor_entity_id=self._config.outdoor_entity_id,
            legacy_forecast_entity_id=self._config.legacy_forecast_entity_id,
            solar_entity_id=self._detected_solar.current_entity_id,
            solar_forecast_entity_ids=list(self._detected_solar.forecast_entity_ids),
        )

    async def async_added_to_hass(self) -> None:
        """Register listeners and restore state."""
        await super().async_added_to_hass()
        await self._async_restore_runtime_state()
        self._detected_solar = detect_pirate_weather_solar_entities(
            self.hass, self._config.outdoor_entity_id
        )
        self._controller.set_solar_available(
            self._detected_solar.current_entity_id is not None
        )
        await self._async_refresh_temperature_inputs()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                [self._config.temperature_sensor_entity_id],
                self._async_handle_temperature_change,
            )
        )

        if self._config.outdoor_entity_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [self._config.outdoor_entity_id],
                    self._async_handle_outdoor_temperature_change,
                )
            )
        if self._detected_solar.current_entity_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass,
                    [self._detected_solar.current_entity_id],
                    self._async_handle_solar_change,
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
        self._controller.set_hvac_mode(_to_thermostat_mode(hvac_mode))
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
            coerce_state_temperature(event.data.get("new_state"))
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_outdoor_temperature_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the outdoor temperature from a state event."""
        self._controller.update_outside_temperature(
            coerce_outdoor_temperature(event.data.get("new_state"))
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_solar_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the solar intensity from a state event."""
        solar = coerce_state_temperature(event.data.get("new_state"))
        self._controller.update_current_solar(
            solar, build_current_solar_sample(self.hass, solar)
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
            coerce_state_temperature(
                self.hass.states.get(self._config.temperature_sensor_entity_id)
            )
        )
        if self._config.outdoor_entity_id:
            self._controller.update_outside_temperature(
                coerce_outdoor_temperature(
                    self.hass.states.get(self._config.outdoor_entity_id)
                )
            )
        solar = coerce_state_temperature(
            self.hass.states.get(self._detected_solar.current_entity_id)
            if self._detected_solar.current_entity_id
            else None
        )
        self._controller.update_current_solar(
            solar, build_current_solar_sample(self.hass, solar)
        )

    async def _async_restore_runtime_state(self) -> None:
        """Restore the HA and learning state from storage."""
        hvac_mode = ThermostatMode.HEAT
        restored_target: float | None = None
        old_state = await self.async_get_last_state()
        if old_state is not None:
            if old_state.state in (HVACMode.HEAT, HVACMode.OFF):
                hvac_mode = _to_thermostat_mode(HVACMode(old_state.state))
            restored_target = coerce_float(old_state.attributes.get(ATTR_TEMPERATURE))

        stored = await self._store.async_get_entry(self._entry.entry_id)
        self._controller.restore(
            hvac_mode=hvac_mode,
            target_temperature=restored_target,
            stored_payload=stored,
        )

    async def _async_run_control_cycle(self, force: bool) -> bool:
        """Run one thermostat control cycle."""
        await self._async_refresh_temperature_inputs()
        forecast_temperatures = await async_get_forecast_temperatures(
            self.hass, self._forecast_entity_id, self._config.lookahead_steps
        )
        solar_samples = get_solar_forecast_samples(
            self.hass, self._detected_solar, self._config.lookahead_steps
        )
        solar_forecast = [solar for _hour, solar in solar_samples]
        solar_sample_forecast = get_solar_sample_forecast(
            self.hass, self._config.lookahead_steps, solar_samples
        )
        ran = await self._controller.run_control_cycle(
            force=force,
            forecast_temperatures=forecast_temperatures,
            solar=self._controller.current_solar,
            solar_forecast=solar_forecast,
            solar_sample=self._controller.current_solar_sample,
            solar_sample_forecast=solar_sample_forecast,
            current_switch_active=any_switch_active(
                self.hass, self._config.switch_entity_ids
            ),
            turn_switches=self._async_turn_all_switches,
            persist=self._async_persist_learning_state_payload,
        )
        self.async_write_ha_state()
        async_dispatcher_send(self.hass, self._runtime.update_signal)
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
        await async_turn_switches(self.hass, self._config.switch_entity_ids, turn_on)

    @property
    def _forecast_entity_id(self) -> str | None:
        """Return the best forecast source for the configured outdoor input."""
        return forecast_entity_id(
            self._config.outdoor_entity_id,
            self._config.legacy_forecast_entity_id,
        )


def _to_thermostat_mode(hvac_mode: HVACMode) -> ThermostatMode:
    """Map Home Assistant HVAC modes into the controller mode contract."""
    return ThermostatMode.OFF if hvac_mode == HVACMode.OFF else ThermostatMode.HEAT


def _to_hvac_mode(mode: ThermostatMode) -> HVACMode:
    """Map the controller mode contract into Home Assistant HVAC modes."""
    return HVACMode.OFF if mode == ThermostatMode.OFF else HVACMode.HEAT
