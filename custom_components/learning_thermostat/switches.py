"""Home Assistant switch adapter helpers for thermostat outputs."""

from __future__ import annotations

from homeassistant.const import ATTR_ENTITY_ID, STATE_ON
from homeassistant.core import HomeAssistant


async def async_turn_switches(
    hass: HomeAssistant, entity_ids: tuple[str, ...], turn_on: bool
) -> None:
    """Turn configured switch entities on or off through HA services."""
    service = "turn_on" if turn_on else "turn_off"
    for entity_id in entity_ids:
        await hass.services.async_call(
            entity_id.split(".", 1)[0],
            service,
            {ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )


def any_switch_active(hass: HomeAssistant, entity_ids: tuple[str, ...]) -> bool:
    """Return whether any configured switch entity is currently on."""
    for entity_id in entity_ids:
        state = hass.states.get(entity_id)
        if state is not None and state.state == STATE_ON:
            return True
    return False
