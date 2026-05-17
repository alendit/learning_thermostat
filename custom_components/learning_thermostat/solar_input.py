"""Home Assistant solar input discovery and sampling adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .solar import build_solar_sample
from .weather_input import WEATHER_DOMAIN, coerce_float, coerce_state_temperature

try:
    from astral import Observer
    from astral.sun import azimuth as astral_azimuth
    from astral.sun import elevation as astral_elevation
except ImportError:  # pragma: no cover - Home Assistant depends on astral
    Observer = None
    astral_azimuth = None
    astral_elevation = None


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
    return _detect_weather_solar_entities(
        hass, weather_entity_id, supported_domains={"pirateweather"}
    )


def detect_weather_solar_entities(
    hass: HomeAssistant, weather_entity_id: str | None
) -> DetectedSolarEntities:
    """Resolve linked weather-provider solar irradiance entities."""
    return _detect_weather_solar_entities(
        hass, weather_entity_id, supported_domains={"pirateweather", "brightsky"}
    )


def _detect_weather_solar_entities(
    hass: HomeAssistant,
    weather_entity_id: str | None,
    *,
    supported_domains: set[str],
) -> DetectedSolarEntities:
    """Resolve linked solar entities for supported weather providers."""
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

    config_entry = hass.config_entries.async_get_entry(config_entry_id)
    provider_domain = platform or (config_entry.domain if config_entry else None)
    if provider_domain not in supported_domains:
        return DetectedSolarEntities()

    current_entity_id: str | None = None
    forecast_candidates: list[tuple[int, str]] = []
    for entry in er.async_entries_for_config_entry(registry, config_entry_id):
        if entry.domain != "sensor" or entry.disabled_by is not None:
            continue

        if provider_domain == "pirateweather":
            if registry_entry_key(entry) == "solar_max":
                continue
            if registry_entry_key(entry) != "solar":
                continue

            hour = extract_hourly_forecast_horizon(entry.entity_id, entry.original_name)
            if hour is None:
                current_entity_id = entry.entity_id
            else:
                forecast_candidates.append((hour, entry.entity_id))

        if provider_domain == "brightsky":
            if brightsky_registry_entry_key(entry) != "solar_irradiance":
                continue
            if is_brightsky_current_irradiance(entry):
                current_entity_id = entry.entity_id
                continue

            hour = extract_brightsky_forecast_horizon(
                entry.entity_id, entry.original_name
            )
            if hour is not None and hour > 0:
                forecast_candidates.append((hour, entry.entity_id))

    forecast_candidates.sort(key=lambda item: item[0])
    return DetectedSolarEntities(
        current_entity_id=current_entity_id,
        forecast_entity_ids=tuple(
            entity_id for _hour, entity_id in forecast_candidates
        ),
        forecast_hours=tuple(hour for hour, _entity_id in forecast_candidates),
    )


def registry_entry_text(entry: er.RegistryEntry) -> str:
    """Return normalized text from a registry entry's semantic identifiers."""
    parts = []
    for candidate in (
        getattr(entry, "translation_key", None),
        getattr(entry, "original_name", None),
        entry.entity_id,
    ):
        if candidate:
            parts.append(str(candidate).lower().replace("-", "_").replace(" ", "_"))
    return "_".join(parts)


def registry_entry_key(entry: er.RegistryEntry) -> str | None:
    """Return the best semantic key for a registry entry."""
    text = registry_entry_text(entry)
    if "solar_max" in text or "downward_short_wave_radiation_flux_max" in text:
        return "solar_max"
    if "solar" in text or "downward_short_wave_radiation_flux" in text:
        return "solar"
    return None


def brightsky_registry_entry_key(entry: er.RegistryEntry) -> str | None:
    """Return the Bright Sky solar key only for derived W/m² irradiance sensors."""
    text = registry_entry_text(entry)
    if "solar_irradiance" in text or ("solar" in text and "irradiance" in text):
        return "solar_irradiance"
    return None


def is_brightsky_current_irradiance(entry: er.RegistryEntry) -> bool:
    """Return whether a Bright Sky sensor is the current irradiance input."""
    text = registry_entry_text(entry)
    return "solar_irradiance_60" in text or "solar_irradiance_60_min" in text


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


def extract_brightsky_forecast_horizon(
    entity_id: str | None, original_name: str | None
) -> int | None:
    """Extract the hourly forecast horizon from a Bright Sky irradiance entity."""
    patterns = []
    if entity_id:
        patterns.append(entity_id)
    if original_name:
        patterns.append(original_name)
    for value in patterns:
        text = value.lower().replace("-", "_").replace(" ", "_")
        if "current_hour" in text:
            return 0
        if "next_hour" in text:
            return 1
        match = re.search(r"hourly[_ ](?:\+)?(\d+)h", text)
        if match is not None:
            return int(match.group(1))
        match = re.search(r"_hourly_(\d+)_", text)
        if match is not None:
            return int(match.group(1))
        match = re.search(r"[_ ](\d+)h(?:$|[_ ])", text)
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
