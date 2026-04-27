"""Home Assistant service registration for the learning thermostat."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import async_get_current_platform

from .const import (
    DATA_SERVICES_REGISTERED,
    DOMAIN,
    SERVICE_RESET_LEARNING,
    SERVICE_TRIGGER_TICK,
)


def async_ensure_entity_services_registered(hass: HomeAssistant) -> None:
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
