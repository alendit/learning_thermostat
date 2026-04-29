"""Helpers for Home Assistant state attributes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

RECORDER_ATTRIBUTE_SAFE_BYTES = 16000
_FLOAT_SIGNIFICANT_DIGITS = 6
_SUMMARY_LIST_LIMIT = 8


def build_learning_diagnostic_attributes(
    *,
    learning_state: Mapping[str, Any] | None,
    learning_result: Mapping[str, Any] | None,
    learning_last_tick: str | None,
    outside_temperature: float | None,
    current_solar: float | None,
    current_solar_sample: Mapping[str, Any] | None,
    solar_orientation: Mapping[str, Any],
    on_percent: float | None,
    underlying_entities: list[str],
    outdoor_entity_id: str | None,
    legacy_forecast_entity_id: str | None,
    solar_entity_id: str | None,
    solar_forecast_entity_ids: list[str],
) -> dict[str, Any]:
    """Build the recorder-safe Home Assistant diagnostic attribute surface."""
    state = learning_state if isinstance(learning_state, Mapping) else {}
    result = learning_result if isinstance(learning_result, Mapping) else {}
    attributes = {
        "learning_model": state.get("model"),
        "learning_model_id": state.get("model_id"),
        "learning_model_schema_version": state.get("model_schema_version"),
        "learning_last_tick": learning_last_tick,
        "outside_temperature": outside_temperature,
        "current_solar": current_solar,
        "current_solar_sample": current_solar_sample,
        "solar_orientation_azimuth": solar_orientation.get("azimuth"),
        "solar_orientation_confidence": solar_orientation.get("confidence"),
        "solar_orientation": dict(solar_orientation),
        "on_percent": on_percent,
        "underlying_entities": underlying_entities,
        "outdoor_entity_id": outdoor_entity_id,
        "legacy_forecast_entity_id": legacy_forecast_entity_id,
        "solar_entity_id": solar_entity_id,
        "solar_forecast_entity_ids": solar_forecast_entity_ids,
        "learning_plan": _learning_plan_summary(result),
        "learning_update": _learning_update_summary(result.get("update")),
        "learning_tracking": _learning_tracking_summary(result),
    }
    return compact_state_attributes(attributes)


def compact_state_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return recorder-safe diagnostic attributes.

    The controller/storage payload remains the canonical learning state. The HA
    state attribute surface is compacted so recorder can keep ordinary climate
    attributes instead of dropping the entire attribute blob.
    """
    compacted = _compact_value(attributes)
    if not isinstance(compacted, dict):
        return {}

    learning_result = compacted.get("learning_result")
    if isinstance(learning_result, dict):
        learning_result.pop("weights", None)
        update = learning_result.get("update")
        if isinstance(update, dict):
            update.pop("delta", None)

    return compacted


def learning_model_name(learning_state: Mapping[str, Any] | None) -> Any:
    """Return the diagnostic model name from a persisted optimizer snapshot."""
    if not isinstance(learning_state, Mapping):
        return None
    return learning_state.get("model")


def _learning_plan_summary(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the compact planning and forecast diagnostics from one tick."""
    if not result:
        return None
    return {
        "u_planned": result.get("u_planned"),
        "u_actual": result.get("u_actual"),
        "start_step": result.get("start_step"),
        "stop_step": result.get("stop_step"),
        "lookahead_steps": result.get("lookahead_steps"),
        "lookahead_reason": result.get("lookahead_reason"),
        "plan_cost": result.get("plan_cost"),
        "planned_states": _summary_list(result.get("planned_states")),
        "predicted_temps": _summary_list(result.get("predicted_temps")),
        "forecast_temps": _summary_list(result.get("forecast_temps")),
        "forecast_solar": _summary_list(result.get("forecast_solar")),
        "forecast_solar_samples": _summary_list(result.get("forecast_solar_samples")),
    }


def _learning_update_summary(update: Any) -> dict[str, Any] | None:
    """Return compact learning update diagnostics without raw weight deltas."""
    if not isinstance(update, Mapping):
        return None
    return {
        key: _compact_value(update.get(key))
        for key in (
            "error",
            "errors",
            "eta",
            "base_eta",
            "eta_scale",
            "l2_reg",
            "grad_norm",
            "abs_update_max",
            "steps_used",
        )
        if key in update
    }


def _learning_tracking_summary(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return compact prediction-error diagnostics from one tick."""
    if not result:
        return None
    return {
        "error": result.get("error"),
        "k_step_error": result.get("k_step_error"),
        "k_step_error_mean": result.get("k_step_error_mean"),
        "k_step_pred_improvement": result.get("k_step_pred_improvement"),
        "tick_duration_ms": result.get("tick_duration_ms"),
        "step_count": result.get("step_count"),
        "weights_reset": result.get("weights_reset"),
    }


def _summary_list(value: Any) -> Any:
    """Return a compact prefix of a list-like diagnostic value."""
    if not isinstance(value, list | tuple):
        return value
    return list(value[:_SUMMARY_LIST_LIMIT])


def _compact_value(value: Any) -> Any:
    """Copy ``value`` while trimming verbose float representations."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return _compact_float(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {str(key): _compact_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_compact_value(child) for child in value]
    return value


def _compact_float(value: float) -> float:
    """Round a float to a stable, useful diagnostic precision."""
    if not math.isfinite(value):
        return value
    return float(f"{value:.{_FLOAT_SIGNIFICANT_DIGITS}g}")
