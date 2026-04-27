"""Home Assistant-free config entry contracts for thermostat runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .const import (
    CONF_CYCLE_MINUTES,
    CONF_FORECAST_ENTITY_ID,
    CONF_LEARNING_ENERGY,
    CONF_LEARNING_ERROR_DECAY,
    CONF_LEARNING_HISTORY_STEPS,
    CONF_LEARNING_L2_REG,
    CONF_LEARNING_LOOKAHEAD_STEPS,
    CONF_LEARNING_MODEL,
    CONF_LEARNING_RATE,
    CONF_MAX_TEMPERATURE,
    CONF_MIN_TEMPERATURE,
    CONF_OUTDOOR_ENTITY_ID,
    CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_TARGET_TEMPERATURE,
    CONF_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_UNDERLYING_ENTITY_IDS,
    DEFAULT_LEARNING_MODEL,
    DEFAULT_NAME,
    LEARNING_MODEL_LABELS,
    LEARNING_MODEL_LINEAR,
)
from .optimizer import DEFAULT_LEARNING_SWITCH_CONFIG


def normalize_config_entry_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize one config-entry data payload."""
    normalized = dict(data)
    normalized["name"] = str(data.get("name", DEFAULT_NAME)).strip() or DEFAULT_NAME
    normalized[CONF_UNDERLYING_ENTITY_IDS] = list(data[CONF_UNDERLYING_ENTITY_IDS])
    normalized[CONF_LEARNING_MODEL] = normalize_learning_model(
        data.get(CONF_LEARNING_MODEL)
    )

    for key in (
        CONF_CYCLE_MINUTES,
        CONF_LEARNING_LOOKAHEAD_STEPS,
        CONF_LEARNING_HISTORY_STEPS,
    ):
        normalized[key] = int(data[key])

    for key in (
        CONF_TARGET_TEMPERATURE,
        CONF_MIN_TEMPERATURE,
        CONF_MAX_TEMPERATURE,
        CONF_LEARNING_L2_REG,
        CONF_LEARNING_ERROR_DECAY,
        CONF_LEARNING_RATE,
        CONF_LEARNING_ENERGY,
    ):
        normalized[key] = float(data[key])

    for optional_key in (
        CONF_OUTDOOR_ENTITY_ID,
        CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID,
        CONF_FORECAST_ENTITY_ID,
    ):
        value = data.get(optional_key)
        if value:
            normalized[optional_key] = str(value)
        else:
            normalized.pop(optional_key, None)

    return normalized


@dataclass(frozen=True, slots=True)
class ThermostatConfig:
    """Stable runtime config for one thermostat entry."""

    name: str
    switch_entity_ids: tuple[str, ...]
    temperature_sensor_entity_id: str
    outdoor_entity_id: str | None
    legacy_forecast_entity_id: str | None
    cycle_minutes: int
    min_temperature: float
    max_temperature: float
    target_temperature: float
    lookahead_steps: int
    learning_config: dict[str, Any]

    @classmethod
    def from_mapping(cls, *, name: str, data: dict[str, Any]) -> ThermostatConfig:
        """Build config from one config-entry payload."""
        normalized = normalize_config_entry_data(
            {**data, "name": name or data.get("name", DEFAULT_NAME)}
        )
        outdoor_entity_id = _coerce_optional_str(normalized.get(CONF_OUTDOOR_ENTITY_ID))
        legacy_forecast_entity_id = _coerce_optional_str(
            normalized.get(CONF_FORECAST_ENTITY_ID)
        )
        if outdoor_entity_id is None:
            outdoor_entity_id = _coerce_optional_str(
                normalized.get(CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID)
            )
        if (
            outdoor_entity_id is None
            and legacy_forecast_entity_id
            and legacy_forecast_entity_id.startswith("weather.")
        ):
            outdoor_entity_id = legacy_forecast_entity_id

        return cls(
            name=normalized["name"],
            switch_entity_ids=tuple(
                str(value) for value in normalized[CONF_UNDERLYING_ENTITY_IDS]
            ),
            temperature_sensor_entity_id=str(
                normalized[CONF_TEMPERATURE_SENSOR_ENTITY_ID]
            ),
            outdoor_entity_id=outdoor_entity_id,
            legacy_forecast_entity_id=legacy_forecast_entity_id,
            cycle_minutes=int(normalized[CONF_CYCLE_MINUTES]),
            min_temperature=float(normalized[CONF_MIN_TEMPERATURE]),
            max_temperature=float(normalized[CONF_MAX_TEMPERATURE]),
            target_temperature=float(normalized[CONF_TARGET_TEMPERATURE]),
            lookahead_steps=int(normalized[CONF_LEARNING_LOOKAHEAD_STEPS]),
            learning_config={
                **DEFAULT_LEARNING_SWITCH_CONFIG,
                "lookahead_steps": int(normalized[CONF_LEARNING_LOOKAHEAD_STEPS]),
                "history_steps": int(normalized[CONF_LEARNING_HISTORY_STEPS]),
                "learning_model": normalized[CONF_LEARNING_MODEL],
                "l2_reg": float(normalized[CONF_LEARNING_L2_REG]),
                "error_decay": float(normalized[CONF_LEARNING_ERROR_DECAY]),
                "learning_rate": float(normalized[CONF_LEARNING_RATE]),
                "w_energy": float(normalized[CONF_LEARNING_ENERGY]),
                "use_solar_feature": bool(data.get("solar_entity_id")),
            },
        )


def normalize_learning_model(value: Any) -> str:
    """Normalize a config-entry learning model identifier."""
    model = str(value or DEFAULT_LEARNING_MODEL)
    if model not in LEARNING_MODEL_LABELS:
        return LEARNING_MODEL_LINEAR
    return model


def _coerce_optional_str(value: Any) -> str | None:
    """Return a trimmed string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
