"""Home Assistant environmental input adapters for thermostat control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Any

from homeassistant.components.weather import DOMAIN as WEATHER_DOMAIN
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .solar import build_solar_sample

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


def forecast_entity_id(
    outdoor_entity_id: str | None, legacy_forecast_entity_id: str | None
) -> str | None:
    """Return the best forecast source for the configured outdoor input."""
    if outdoor_entity_id and outdoor_entity_id.startswith(f"{WEATHER_DOMAIN}."):
        return outdoor_entity_id
    return legacy_forecast_entity_id


async def async_get_forecast_temperatures(
    hass: HomeAssistant, entity_id: str | None, steps: int
) -> list[float]:
    """Build forecast temperatures for the lookahead horizon."""
    if not entity_id or steps <= 0:
        return []

    state = hass.states.get(entity_id)
    if state is None:
        return []

    forecast = state.attributes.get("forecast")
    if isinstance(forecast, list):
        return coerce_forecast_list(forecast, steps)

    if entity_id.split(".", 1)[0] == WEATHER_DOMAIN:
        fetched = await async_weather_forecast(hass, entity_id)
        return coerce_forecast_list(fetched, steps)

    return []


def get_solar_forecast_samples(
    hass: HomeAssistant, detected_solar: DetectedSolarEntities, steps: int
) -> list[tuple[int, float]]:
    """Build solar forecast samples paired with their forecast horizons."""
    if steps <= 0 or not detected_solar.forecast_entity_ids:
        return []
    forecast: list[tuple[int, float]] = []
    for index, entity_id in enumerate(detected_solar.forecast_entity_ids):
        value = coerce_state_temperature(hass.states.get(entity_id))
        if value is None:
            continue
        hour = (
            detected_solar.forecast_hours[index]
            if index < len(detected_solar.forecast_hours)
            else index + 1
        )
        forecast.append((hour, value))
        if len(forecast) == steps:
            break
    return forecast


def get_solar_sample_forecast(
    hass: HomeAssistant, steps: int, solar_forecast: list[tuple[int, float]]
) -> list[dict[str, float | None]]:
    """Build raw solar samples for the linked solar forecast series."""
    if steps <= 0 or not solar_forecast:
        return []
    samples: list[dict[str, float | None]] = []
    now = dt_util.utcnow()
    for hour, solar in solar_forecast[:steps]:
        position = solar_position_at(hass, now + timedelta(hours=hour))
        samples.append(
            build_solar_sample(
                solar_flux=solar,
                sun_azimuth=position.azimuth if position is not None else None,
                sun_elevation=position.elevation if position is not None else None,
            ).as_dict()
        )
    return samples


def build_current_solar_sample(
    hass: HomeAssistant, solar: float | None
) -> dict[str, float | None] | None:
    """Build the current raw solar sample from the sun entity."""
    if solar is None:
        return None
    position = current_sun_position(hass)
    return build_solar_sample(
        solar_flux=solar,
        sun_azimuth=position.azimuth if position is not None else None,
        sun_elevation=position.elevation if position is not None else None,
    ).as_dict()


def detect_pirate_weather_solar_entities(
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
        if registry_entry_key(entry) == "solar_max":
            continue
        if registry_entry_key(entry) != "solar":
            continue

        hour = extract_hourly_forecast_horizon(entry.entity_id, entry.original_name)
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


def registry_entry_key(entry: er.RegistryEntry) -> str | None:
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


def extract_hourly_forecast_horizon(
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


def current_sun_position(hass: HomeAssistant) -> SunPosition | None:
    """Return current sun position from Home Assistant state attributes."""
    state = hass.states.get("sun.sun")
    if state is None:
        return None
    azimuth = coerce_float(state.attributes.get("azimuth"))
    elevation = coerce_float(state.attributes.get("elevation"))
    if azimuth is None or elevation is None:
        return None
    return SunPosition(azimuth=azimuth, elevation=elevation)


def solar_position_at(hass: HomeAssistant, when: datetime) -> SunPosition | None:
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


async def async_weather_forecast(
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


def coerce_forecast_list(values: list[Any], steps: int) -> list[float]:
    """Convert a forecast payload to a list of temperatures."""
    temperatures: list[float] = []
    for value in values:
        if isinstance(value, dict):
            candidate = value.get("temperature", value.get("temp"))
        else:
            candidate = value
        coerced = coerce_float(candidate)
        if coerced is None:
            continue
        temperatures.append(coerced)
        if len(temperatures) == steps:
            break
    return temperatures


def coerce_state_temperature(state: State | None) -> float | None:
    """Coerce a Home Assistant state string into a temperature-like float."""
    if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        return None
    return coerce_float(state.state)


def coerce_outdoor_temperature(state: State | None) -> float | None:
    """Coerce a sensor or weather state into a temperature value."""
    if state is None:
        return None
    if state.entity_id.startswith(f"{WEATHER_DOMAIN}."):
        for key in ("temperature", "native_temperature", "temp"):
            value = coerce_float(state.attributes.get(key))
            if value is not None:
                return value
        return None
    return coerce_state_temperature(state)


def coerce_float(value: Any) -> float | None:
    """Coerce a value into a finite float when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
