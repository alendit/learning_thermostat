"""Climate platform for the learning thermostat."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
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
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback,
    async_get_current_platform,
)
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util

from .attributes import compact_state_attributes
from .const import (
    DATA_SERVICES_REGISTERED,
    DATA_STORAGE,
    DOMAIN,
    NAME,
    SERVICE_RESET_LEARNING,
    SERVICE_TRIGGER_TICK,
)
from .runtime import LearningThermostatRuntime, async_get_or_create_runtime
from .solar import build_solar_sample
from .state import LearningEntryPayload, ThermostatMode
from .storage import LearningStateStore

try:
    from astral import Observer
    from astral.sun import azimuth as astral_azimuth
    from astral.sun import elevation as astral_elevation
except ImportError:  # pragma: no cover - Home Assistant depends on astral
    Observer = None
    astral_azimuth = None
    astral_elevation = None

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
        return HVACAction.HEATING if self._is_any_switch_active() else HVACAction.IDLE

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose learning details for diagnostics and migration."""
        solar_orientation = self._controller.solar_orientation_diagnostics
        return compact_state_attributes(
            {
                "learning_state": self._controller.learning_state,
                "learning_result": self._controller.learning_result,
                "learning_last_tick": (
                    self._controller.learning_last_tick.isoformat()
                    if self._controller.learning_last_tick is not None
                    else None
                ),
                "outside_temperature": self._controller.outside_temperature,
                "current_solar": self._controller.current_solar,
                "current_solar_sample": self._controller.current_solar_sample,
                "solar_orientation_azimuth": solar_orientation.get("azimuth"),
                "solar_orientation_confidence": solar_orientation.get("confidence"),
                "solar_orientation": solar_orientation,
                "on_percent": self._controller.learning_on_percent,
                "underlying_entities": list(self._config.switch_entity_ids),
                "outdoor_entity_id": self._config.outdoor_entity_id,
                "legacy_forecast_entity_id": self._config.legacy_forecast_entity_id,
                "solar_entity_id": self._detected_solar.current_entity_id,
                "solar_forecast_entity_ids": list(
                    self._detected_solar.forecast_entity_ids
                ),
            }
        )

    async def async_added_to_hass(self) -> None:
        """Register listeners and restore state."""
        await super().async_added_to_hass()
        await self._async_restore_runtime_state()
        self._detected_solar = _detect_pirate_weather_solar_entities(
            self.hass, self._config.outdoor_entity_id
        )
        if self._detected_solar.current_entity_id:
            self._config.learning_config["use_solar_feature"] = True
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
            _coerce_state_temperature(event.data.get("new_state"))
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_outdoor_temperature_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the outdoor temperature from a state event."""
        self._controller.update_outside_temperature(
            _coerce_outdoor_temperature(event.data.get("new_state"))
        )
        self.async_write_ha_state()

    @callback
    async def _async_handle_solar_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Update the solar intensity from a state event."""
        solar = _coerce_state_temperature(event.data.get("new_state"))
        self._controller.update_current_solar(
            solar, self._build_current_solar_sample(solar)
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
        if self._config.outdoor_entity_id:
            self._controller.update_outside_temperature(
                _coerce_outdoor_temperature(
                    self.hass.states.get(self._config.outdoor_entity_id)
                )
            )
        solar = _coerce_state_temperature(
            self.hass.states.get(self._detected_solar.current_entity_id)
            if self._detected_solar.current_entity_id
            else None
        )
        self._controller.update_current_solar(
            solar, self._build_current_solar_sample(solar)
        )

    async def _async_restore_runtime_state(self) -> None:
        """Restore the HA and learning state from storage."""
        hvac_mode = ThermostatMode.HEAT
        restored_target: float | None = None
        old_state = await self.async_get_last_state()
        if old_state is not None:
            if old_state.state in (HVACMode.HEAT, HVACMode.OFF):
                hvac_mode = _to_thermostat_mode(HVACMode(old_state.state))
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
        solar_samples = self._async_get_solar_forecast_samples(
            self._config.lookahead_steps
        )
        solar_forecast = [solar for _hour, solar in solar_samples]
        solar_sample_forecast = self._async_get_solar_sample_forecast(
            self._config.lookahead_steps, solar_samples
        )
        ran = await self._controller.run_control_cycle(
            force=force,
            forecast_temperatures=forecast_temperatures,
            solar=self._controller.current_solar,
            solar_forecast=solar_forecast,
            solar_sample=self._controller.current_solar_sample,
            solar_sample_forecast=solar_sample_forecast,
            current_switch_active=self._is_any_switch_active(),
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
        forecast_entity_id = self._forecast_entity_id
        if not forecast_entity_id or steps <= 0:
            return []

        state = self.hass.states.get(forecast_entity_id)
        if state is None:
            return []

        forecast = state.attributes.get("forecast")
        if isinstance(forecast, list):
            return _coerce_forecast_list(forecast, steps)

        if forecast_entity_id.split(".", 1)[0] == WEATHER_DOMAIN:
            fetched = await _async_weather_forecast(self.hass, forecast_entity_id)
            return _coerce_forecast_list(fetched, steps)

        return []

    @property
    def _forecast_entity_id(self) -> str | None:
        """Return the best forecast source for the configured outdoor input."""
        if (
            self._config.outdoor_entity_id
            and self._config.outdoor_entity_id.startswith(f"{WEATHER_DOMAIN}.")
        ):
            return self._config.outdoor_entity_id
        return self._config.legacy_forecast_entity_id

    def _async_get_solar_forecast(self, steps: int) -> list[float]:
        """Build a solar forecast series from linked forecast sensors."""
        return [solar for _hour, solar in self._async_get_solar_forecast_samples(steps)]

    def _async_get_solar_forecast_samples(self, steps: int) -> list[tuple[int, float]]:
        """Build solar forecast samples paired with their forecast horizons."""
        if steps <= 0 or not self._detected_solar.forecast_entity_ids:
            return []
        forecast: list[tuple[int, float]] = []
        for index, entity_id in enumerate(self._detected_solar.forecast_entity_ids):
            value = _coerce_state_temperature(self.hass.states.get(entity_id))
            if value is None:
                continue
            hour = (
                self._detected_solar.forecast_hours[index]
                if index < len(self._detected_solar.forecast_hours)
                else index + 1
            )
            forecast.append((hour, value))
            if len(forecast) == steps:
                break
        return forecast

    def _async_get_solar_sample_forecast(
        self, steps: int, solar_forecast: list[tuple[int, float]]
    ) -> list[dict[str, float | None]]:
        """Build raw solar samples for the linked solar forecast series."""
        if steps <= 0 or not solar_forecast:
            return []
        samples: list[dict[str, float | None]] = []
        now = dt_util.utcnow()
        for hour, solar in solar_forecast[:steps]:
            position = _solar_position_at(self.hass, now + timedelta(hours=hour))
            samples.append(
                build_solar_sample(
                    solar_flux=solar,
                    sun_azimuth=position.azimuth if position is not None else None,
                    sun_elevation=position.elevation if position is not None else None,
                ).as_dict()
            )
        return samples

    def _build_current_solar_sample(
        self, solar: float | None
    ) -> dict[str, float | None] | None:
        """Build the current raw solar sample from the sun entity."""
        if solar is None:
            return None
        position = _current_sun_position(self.hass)
        return build_solar_sample(
            solar_flux=solar,
            sun_azimuth=position.azimuth if position is not None else None,
            sun_elevation=position.elevation if position is not None else None,
        ).as_dict()


@dataclass(frozen=True, slots=True)
class DetectedSolarEntities:
    """Resolved solar entities associated with one outdoor weather entity."""

    current_entity_id: str | None = None
    forecast_entity_ids: tuple[str, ...] = ()
    forecast_hours: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SunPosition:
    """Sun position in Home Assistant azimuth/elevation degrees."""

    azimuth: float
    elevation: float


def _detect_pirate_weather_solar_entities(
    hass: HomeAssistant, weather_entity_id: str | None
) -> DetectedSolarEntities:
    """Resolve linked Pirate Weather solar entities from the entity registry."""
    if not weather_entity_id or not weather_entity_id.startswith(f"{WEATHER_DOMAIN}."):
        return DetectedSolarEntities()

    registry = er.async_get(hass)
    weather_entry = registry.async_get(weather_entity_id)
    if weather_entry is None:
        return DetectedSolarEntities()

    platform = getattr(weather_entry, "platform", None)
    config_entry_id = getattr(weather_entry, "config_entry_id", None)
    if config_entry_id is None:
        return DetectedSolarEntities()
    if platform != "pirateweather":
        config_entry = hass.config_entries.async_get_entry(config_entry_id)
        if config_entry is None or config_entry.domain != "pirateweather":
            return DetectedSolarEntities()

    current_entity_id: str | None = None
    forecast_candidates: list[tuple[int, str]] = []
    for entry in er.async_entries_for_config_entry(registry, config_entry_id):
        if entry.domain != "sensor" or entry.disabled_by is not None:
            continue
        if _registry_entry_key(entry) == "solar_max":
            continue
        if _registry_entry_key(entry) != "solar":
            continue

        hour = _extract_hourly_forecast_horizon(entry.entity_id, entry.original_name)
        if hour is None:
            current_entity_id = entry.entity_id
        else:
            forecast_candidates.append((hour, entry.entity_id))

    forecast_candidates.sort(key=lambda item: item[0])
    return DetectedSolarEntities(
        current_entity_id=current_entity_id,
        forecast_entity_ids=tuple(
            entity_id for _hour, entity_id in forecast_candidates
        ),
        forecast_hours=tuple(hour for hour, _entity_id in forecast_candidates),
    )


def _registry_entry_key(entry: er.RegistryEntry) -> str | None:
    """Return the best semantic key for a registry entry."""
    for candidate in (
        getattr(entry, "translation_key", None),
        getattr(entry, "original_name", None),
        entry.entity_id,
    ):
        if not candidate:
            continue
        text = str(candidate).lower().replace("-", "_").replace(" ", "_")
        if "solar_max" in text or "downward_short_wave_radiation_flux_max" in text:
            return "solar_max"
        if "solar" in text or "downward_short_wave_radiation_flux" in text:
            return "solar"
    return None


def _extract_hourly_forecast_horizon(
    entity_id: str | None, original_name: str | None
) -> int | None:
    """Extract the hourly forecast horizon from a Pirate Weather solar entity."""
    patterns = []
    if entity_id:
        patterns.append(entity_id)
    if original_name:
        patterns.append(original_name)
    for value in patterns:
        match = re.search(r"[_ ](\d+)h(?:$|[_ ])", value)
        if match is not None:
            return int(match.group(1))
    return None


def _current_sun_position(hass: HomeAssistant) -> SunPosition | None:
    """Return current sun position from Home Assistant state attributes."""
    state = hass.states.get("sun.sun")
    if state is None:
        return None
    azimuth = _coerce_float(state.attributes.get("azimuth"))
    elevation = _coerce_float(state.attributes.get("elevation"))
    if azimuth is None or elevation is None:
        return None
    return SunPosition(azimuth=azimuth, elevation=elevation)


def _solar_position_at(hass: HomeAssistant, when: datetime) -> SunPosition | None:
    """Calculate sun position for a future timestamp using Astral."""
    if Observer is None or astral_azimuth is None or astral_elevation is None:
        return None
    try:
        observer = Observer(
            latitude=hass.config.latitude,
            longitude=hass.config.longitude,
            elevation=hass.config.elevation,
        )
        point_in_time = dt_util.as_utc(when)
        return SunPosition(
            azimuth=float(astral_azimuth(observer, point_in_time)),
            elevation=float(astral_elevation(observer, point_in_time)),
        )
    except Exception:  # pragma: no cover - depends on runtime HA location data
        return None


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


def _coerce_outdoor_temperature(state: State | None) -> float | None:
    """Coerce a sensor or weather state into a temperature value."""
    if state is None:
        return None
    if state.entity_id.startswith(f"{WEATHER_DOMAIN}."):
        for key in ("temperature", "native_temperature", "temp"):
            value = _coerce_float(state.attributes.get(key))
            if value is not None:
                return value
        return None
    return _coerce_state_temperature(state)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_thermostat_mode(hvac_mode: HVACMode) -> ThermostatMode:
    """Map Home Assistant HVAC modes into the controller mode contract."""
    return ThermostatMode.OFF if hvac_mode == HVACMode.OFF else ThermostatMode.HEAT


def _to_hvac_mode(mode: ThermostatMode) -> HVACMode:
    """Map the controller mode contract into Home Assistant HVAC modes."""
    return HVACMode.OFF if mode == ThermostatMode.OFF else HVACMode.HEAT


def _async_ensure_entity_services_registered(hass: HomeAssistant) -> None:
    """Register integration services once against HA's entity registry."""
    data = hass.data.setdefault(DOMAIN, {})
    if data.get(DATA_SERVICES_REGISTERED):
        return
    if hass.services.has_service(DOMAIN, SERVICE_RESET_LEARNING):
        data[DATA_SERVICES_REGISTERED] = True
        return

    platform = async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_RESET_LEARNING,
        schema={
            vol.Optional("preserve_history", default=True): cv.boolean,
        },
        func="async_reset_learning_service",
    )
    platform.async_register_entity_service(
        SERVICE_TRIGGER_TICK,
        schema={
            vol.Optional("force", default=False): cv.boolean,
        },
        func="async_trigger_tick_service",
    )
    data[DATA_SERVICES_REGISTERED] = True
