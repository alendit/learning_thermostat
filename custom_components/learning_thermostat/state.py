"""Home Assistant independent thermostat state contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict


class ThermostatMode(StrEnum):
    """Internal thermostat operating modes."""

    HEAT = "heat"
    OFF = "off"


class LearningEntryPayload(TypedDict):
    """Stored state for one thermostat entry."""

    learning_state: dict[str, Any] | None
    learning_last_tick: str | None
    current_temperature: float | None
    outside_temperature: float | None


def build_entry_payload(
    learning_state: dict[str, Any] | None,
    learning_last_tick: str | None,
    current_temperature: float | None,
    outside_temperature: float | None,
) -> LearningEntryPayload:
    """Build one stored entry payload."""
    return {
        "learning_state": learning_state,
        "learning_last_tick": learning_last_tick,
        "current_temperature": current_temperature,
        "outside_temperature": outside_temperature,
    }
