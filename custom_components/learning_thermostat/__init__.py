"""The learning thermostat integration."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import (
    DATA_ENTITIES,
    DATA_SERVICES_REGISTERED,
    DATA_STORAGE,
    DOMAIN,
    PLATFORMS,
    SERVICE_RESET_LEARNING,
    SERVICE_TRIGGER_TICK,
)
from .storage import LearningStateStore


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the integration."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(DATA_STORAGE, LearningStateStore(hass))
    hass.data[DOMAIN].setdefault(DATA_ENTITIES, {})

    if not hass.services.has_service(DOMAIN, SERVICE_RESET_LEARNING):
        hass.services.async_register(
            DOMAIN,
            SERVICE_RESET_LEARNING,
            _async_handle_reset_learning,
            schema=vol.Schema(
                {
                    vol.Required(ATTR_ENTITY_ID): cv.entity_id,
                    vol.Optional("preserve_history", default=True): cv.boolean,
                }
            ),
        )
    if not hass.services.has_service(DOMAIN, SERVICE_TRIGGER_TICK):
        hass.services.async_register(
            DOMAIN,
            SERVICE_TRIGGER_TICK,
            _async_handle_trigger_tick,
            schema=vol.Schema(
                {
                    vol.Required(ATTR_ENTITY_ID): cv.entity_id,
                    vol.Optional("force", default=False): cv.boolean,
                }
            ),
        )
    hass.data[DOMAIN][DATA_SERVICES_REGISTERED] = True
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a config entry."""
    await async_setup(hass, {})
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded and len(hass.config_entries.async_entries(DOMAIN)) == 1:
        hass.data.pop(DOMAIN, None)
    return unloaded


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Remove stored state for a deleted entry."""
    store = hass.data.get(DOMAIN, {}).get(DATA_STORAGE)
    if store is not None:
        await store.async_delete_entry(entry.entry_id)


async def _async_handle_reset_learning(call: ServiceCall) -> None:
    """Handle the integration-level reset service."""
    entity = _entity_from_service_call(call.hass, call)
    await entity.async_reset_learning_service(call.data["preserve_history"])


async def _async_handle_trigger_tick(call: ServiceCall) -> None:
    """Handle the integration-level trigger service."""
    entity = _entity_from_service_call(call.hass, call)
    await entity.async_trigger_tick_service(call.data["force"])


def _entity_from_service_call(hass: HomeAssistant, call: ServiceCall):
    """Resolve an entity object from one service call."""
    entity_id = call.data[ATTR_ENTITY_ID]
    entity = hass.data.get(DOMAIN, {}).get(DATA_ENTITIES, {}).get(entity_id)
    if entity is None:
        raise HomeAssistantError(f"Unknown learning thermostat entity: {entity_id}")
    return entity
