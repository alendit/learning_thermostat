"""Home Assistant weather and temperature input adapters."""

from __future__ import annotations

from typing import Any

from homeassistant.components.weather import DOMAIN as WEATHER_DOMAIN
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State

try:
    from homeassistant.components.weather import (
        async_get_forecast as async_weather_get_forecast,
    )
except ImportError:  # pragma: no cover - older HA versions
    async_weather_get_forecast = None


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
