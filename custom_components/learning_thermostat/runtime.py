"""Shared per-entry runtime objects for Home Assistant platforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DATA_RUNTIMES, DOMAIN
from .controller import ThermostatConfig, ThermostatController


@dataclass(slots=True)
class LearningThermostatRuntime:
    """Runtime shared by climate and diagnostic sensor entities."""

    config: ThermostatConfig
    controller: ThermostatController
    update_signal: str


def async_get_or_create_runtime(
    hass: HomeAssistant, entry: ConfigEntry
) -> LearningThermostatRuntime:
    """Return the shared runtime for one config entry."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    runtimes = domain_data.setdefault(DATA_RUNTIMES, {})
    runtime = runtimes.get(entry.entry_id)
    if isinstance(runtime, LearningThermostatRuntime):
        return runtime

    config = ThermostatConfig.from_mapping(
        name=entry.title or entry.data.get("name", ""),
        data=dict(entry.data),
    )
    runtime = LearningThermostatRuntime(
        config=config,
        controller=ThermostatController(config),
        update_signal=f"{DOMAIN}_{entry.entry_id}_updated",
    )
    runtimes[entry.entry_id] = runtime
    return runtime


def async_remove_runtime(hass: HomeAssistant, entry_id: str) -> None:
    """Remove the shared runtime for an unloaded entry."""
    runtimes: dict[str, Any] | None = hass.data.get(DOMAIN, {}).get(DATA_RUNTIMES)
    if isinstance(runtimes, dict):
        runtimes.pop(entry_id, None)
