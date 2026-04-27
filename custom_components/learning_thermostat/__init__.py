"""The learning thermostat integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import (
    DATA_RUNTIMES,
    DATA_STORAGE,
    DOMAIN,
)
from .runtime import async_remove_runtime
from .storage import LearningStateStore

PLATFORMS = [Platform.CLIMATE, Platform.SENSOR]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the integration."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(DATA_STORAGE, LearningStateStore(hass))
    hass.data[DOMAIN].setdefault(DATA_RUNTIMES, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a config entry."""
    await async_setup(hass, {})
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        async_remove_runtime(hass, entry.entry_id)
    if unloaded and len(hass.config_entries.async_entries(DOMAIN)) == 1:
        hass.data.pop(DOMAIN, None)
    return unloaded


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Remove stored state for a deleted entry."""
    store = hass.data.get(DOMAIN, {}).get(DATA_STORAGE)
    if store is not None:
        await store.async_delete_entry(entry.entry_id)
