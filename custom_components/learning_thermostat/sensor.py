"""Diagnostic sensor platform for the learning thermostat."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .runtime import LearningThermostatRuntime, async_get_or_create_runtime


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up diagnostic sensors for one thermostat entry."""
    runtime = async_get_or_create_runtime(hass, entry)
    async_add_entities([SolarOrientationSensor(runtime, entry.entry_id)])


class SolarOrientationSensor(SensorEntity):
    """Human-readable learned solar orientation diagnostic."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:compass-outline"
    _attr_translation_key = "solar_orientation"

    def __init__(self, runtime: LearningThermostatRuntime, entry_id: str) -> None:
        self._runtime = runtime
        self._attr_name = f"{runtime.config.name} Solar Orientation"
        self._attr_unique_id = f"{entry_id}_solar_orientation"
        self._attr_should_poll = False

    @property
    def native_value(self) -> str:
        """Return a compact human-readable orientation summary."""
        state = self._diagnostics().get("state")
        return str(state) if state else STATE_UNKNOWN

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return orientation diagnostics as sensor attributes."""
        diagnostics = self._diagnostics()
        return {
            "orientation_1": diagnostics.get("orientation_1"),
            "orientation_1_azimuth": diagnostics.get("orientation_1_azimuth"),
            "orientation_1_contribution_pct": diagnostics.get(
                "orientation_1_contribution_pct"
            ),
            "orientation_2": diagnostics.get("orientation_2"),
            "orientation_2_azimuth": diagnostics.get("orientation_2_azimuth"),
            "orientation_2_contribution_pct": diagnostics.get(
                "orientation_2_contribution_pct"
            ),
            "confidence": diagnostics.get("confidence"),
            "model": (self._runtime.controller.learning_state or {}).get("model"),
        }

    async def async_added_to_hass(self) -> None:
        """Update the sensor when the climate entity completes a tick."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, self._runtime.update_signal, self._handle_runtime_update
            )
        )

    @callback
    def _handle_runtime_update(self) -> None:
        """Write updated orientation diagnostics."""
        self.async_write_ha_state()

    def _diagnostics(self) -> dict[str, Any]:
        """Return controller diagnostics with a stable fallback."""
        diagnostics = self._runtime.controller.solar_orientation_diagnostics
        return (
            diagnostics if isinstance(diagnostics, dict) else {"state": STATE_UNKNOWN}
        )
