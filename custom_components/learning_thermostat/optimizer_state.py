"""Snapshot and restore helpers for the learning optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .const import (
    DEFAULT_LEARNING_MODEL,
    LEARNING_MODEL_LABELS,
    LEARNING_MODEL_LINEAR,
    LEARNING_MODEL_PHYSICAL,
)
from .solar import (
    SOLAR_ORIENTATION_FEATURE_SIZE,
    SOLAR_PROFILE_BIN_AZIMUTHS,
    SOLAR_PROFILE_SIZE,
    coerce_solar_profile,
    coerce_solar_sample,
    diffuse_only_sample,
)

MODEL_SCHEMA_VERSION = 2
LEGACY_MODEL_TYPES = frozenset(
    {
        "delta_zscore",
        "delta_zscore_solar",
        "delta_zscore_solar_profile",
        "delta_zscore_solar_orientation",
    }
)


@dataclass(frozen=True)
class AdaptedWeights:
    """Weights adapted from a persisted layout plus recovered metadata."""

    weights: np.ndarray
    solar_orientation_azimuth: float | None = None


@dataclass(frozen=True)
class RestoredOrientationState:
    """Persisted solar orientation state."""

    azimuth: float | None
    confidence: float | None
    score: float | None


def optimizer_model_name(learning_model: str) -> str:
    """Return the user-facing learning model label."""
    return LEARNING_MODEL_LABELS[learning_model]


def optimizer_model_id(learning_model: str, use_solar_feature: bool) -> str:
    """Return the compact internal model identifier."""
    if learning_model == LEARNING_MODEL_PHYSICAL:
        return "physical_solar_orientation" if use_solar_feature else "physical"
    return "linear_solar_orientation" if use_solar_feature else "linear"


def normalize_learning_model(value: Any) -> str:
    """Normalize the configured learning model identifier."""
    model = str(value or DEFAULT_LEARNING_MODEL)
    if model not in LEARNING_MODEL_LABELS:
        return LEARNING_MODEL_LINEAR
    return model


def snapshot_model_id(state: dict[str, Any], use_solar_feature: bool) -> str | None:
    """Return the internal model id represented by a snapshot."""
    model_id = state.get("model_id")
    if isinstance(model_id, str) and model_id:
        return model_id
    model = state.get("model")
    if model == "Physical":
        return "physical_solar_orientation" if use_solar_feature else "physical"
    if model == "Linear":
        return "linear_solar_orientation" if use_solar_feature else "linear"
    if model == "delta_zscore":
        return "linear"
    if model in {
        "delta_zscore_solar",
        "delta_zscore_solar_profile",
        "delta_zscore_solar_orientation",
    }:
        return "linear_solar_orientation"
    return None


def legacy_model_type(state: dict[str, Any]) -> str | None:
    """Return a legacy linear model id when one is present."""
    model = state.get("model")
    return model if model in LEGACY_MODEL_TYPES else None


def build_optimizer_snapshot(
    *,
    weights: np.ndarray,
    model_name: str,
    model_id: str,
    model_schema_version: int,
    t_hist: list[float],
    out_hist: list[float],
    solar_hist: list[float],
    solar_sample_hist: list[dict[str, float | None]],
    u_hist: list[int],
    prev_pred_history: list[float],
    prev_abs_error_history: list[float],
    prev_pred_timestamp_history: list[str],
    weight_abs_update_history: list[float],
    weight_delta_abs_latest: list[float],
    k_step_abs_error_history: dict[int, dict[str, float | int | None]],
    k_step_pred_improvement_history: dict[int, dict[str, float | int | None]],
    last_u: int | None,
    last_plan: dict[str, int | None],
    solar_orientation_azimuth: float | None,
    solar_orientation_confidence: float | None,
    solar_orientation_score: float | None,
    eta: float | None,
    base_eta: float | None,
    prev_feature: list[float] | None,
    config: dict[str, Any],
    l2_reg: float,
    history_steps: int,
    history_decay: float,
    error_decay: float,
    default_u_gain_per_hour: float,
    use_solar_feature: bool,
    learning_model: str,
    solar_orientation: dict[str, Any],
) -> dict[str, Any]:
    """Build the persisted optimizer snapshot payload."""
    return {
        "weights": weights.tolist(),
        "model": model_name,
        "model_id": model_id,
        "model_schema_version": model_schema_version,
        "history": {
            "t_room": list(t_hist),
            "t_out": list(out_hist),
            "solar": list(solar_hist),
            "solar_samples": [dict(row) for row in solar_sample_hist],
            "u": list(u_hist),
            "pred_next": list(prev_pred_history),
            "pred_abs_error": list(prev_abs_error_history),
            "pred_timestamps": list(prev_pred_timestamp_history),
            "weight_abs_update": list(weight_abs_update_history),
            "weight_delta_abs_latest": list(weight_delta_abs_latest),
            "k_step_abs_error": {
                str(k): {
                    "mean": values.get("mean"),
                    "count": int(values.get("count", 0)),
                }
                for k, values in k_step_abs_error_history.items()
            },
            "k_step_pred_improvement": {
                str(k): {
                    "mean": values.get("mean"),
                    "count": int(values.get("count", 0)),
                }
                for k, values in k_step_pred_improvement_history.items()
            },
        },
        "last_u": last_u,
        "last_plan": dict(last_plan),
        "solar_orientation_azimuth": solar_orientation_azimuth,
        "solar_orientation_confidence": solar_orientation_confidence,
        "solar_orientation_score": solar_orientation_score,
        "eta": eta,
        "base_eta": base_eta,
        "prev_feature": list(prev_feature) if prev_feature else None,
        "config": {
            "lookahead_steps": config.get("lookahead_steps"),
            "learning_rate": config.get("learning_rate"),
            "l2_reg": l2_reg,
            "w_energy": config.get("w_energy"),
            "history_steps": history_steps,
            "history_decay": history_decay,
            "error_decay": error_decay,
            "default_u_gain_per_hour": default_u_gain_per_hour,
            "use_solar_feature": use_solar_feature,
            "learning_model": learning_model,
        },
        "solar_orientation": solar_orientation,
    }


def adapt_weights(
    restored: np.ndarray,
    model_type: str | None,
    *,
    history_steps: int,
    solar_profile_size: int,
    feature_size: int,
) -> AdaptedWeights | None:
    """Adapt weights saved with a different history length or solar layout."""
    size = restored.size
    if size < 4:
        return None
    old_layout = infer_weight_layout(model_type, size)
    if old_layout is None:
        return None
    old_steps, old_profile_size = old_layout
    if old_steps <= 0:
        return None
    new_steps = history_steps
    new_profile_size = solar_profile_size
    new_weights = np.zeros(feature_size + 1, dtype=float)
    copy_len = min(old_steps, new_steps)
    solar_orientation_azimuth = None

    new_weights[0:copy_len] = restored[0:copy_len]
    old_out_start = old_steps
    new_out_start = new_steps
    new_weights[new_out_start : new_out_start + copy_len] = restored[
        old_out_start : old_out_start + copy_len
    ]

    old_solar_start = old_steps * 2
    new_solar_start = new_steps * 2
    old_u_start = old_solar_start + old_steps * old_profile_size
    new_u_start = new_solar_start + new_steps * new_profile_size
    if old_profile_size and new_profile_size:
        if old_profile_size == SOLAR_PROFILE_SIZE:
            orientation = infer_legacy_orientation_from_weights(
                restored, old_steps, old_solar_start
            )
            if orientation is not None:
                solar_orientation_azimuth = orientation
            for step in range(copy_len):
                old_start = old_solar_start + step * old_profile_size
                new_start = new_solar_start + step * new_profile_size
                old_row = restored[old_start : old_start + old_profile_size]
                new_weights[new_start] = old_row[0]
                if orientation is not None:
                    first_index = legacy_bin_index(orientation)
                    second_index = legacy_bin_index(orientation + 90.0)
                    new_weights[new_start + 1] = old_row[first_index + 1]
                    new_weights[new_start + 2] = old_row[second_index + 1]
        else:
            profile_copy = min(old_profile_size, new_profile_size)
            for step in range(copy_len):
                old_start = old_solar_start + step * old_profile_size
                new_start = new_solar_start + step * new_profile_size
                new_weights[new_start : new_start + profile_copy] = restored[
                    old_start : old_start + profile_copy
                ]
    new_weights[new_u_start : new_u_start + copy_len] = restored[
        old_u_start : old_u_start + copy_len
    ]
    bias = restored[-1]
    if np.isfinite(bias):
        new_weights[-1] = float(bias)
    if np.allclose(new_weights, 0.0) or not np.all(np.isfinite(new_weights)):
        return None
    return AdaptedWeights(
        weights=new_weights,
        solar_orientation_azimuth=solar_orientation_azimuth,
    )


def infer_weight_layout(model_type: str | None, size: int) -> tuple[int, int] | None:
    """Infer (history steps, solar profile size) from saved model metadata."""
    if model_type == "delta_zscore":
        return ((size - 1) // 3, 0) if (size - 1) % 3 == 0 else None
    if model_type == "delta_zscore_solar":
        return ((size - 1) // 4, 1) if (size - 1) % 4 == 0 else None
    if model_type == "delta_zscore_solar_profile":
        group_count = 3 + SOLAR_PROFILE_SIZE
        return (
            ((size - 1) // group_count, SOLAR_PROFILE_SIZE)
            if (size - 1) % group_count == 0
            else None
        )
    if model_type == "delta_zscore_solar_orientation":
        group_count = 3 + SOLAR_ORIENTATION_FEATURE_SIZE
        return (
            ((size - 1) // group_count, SOLAR_ORIENTATION_FEATURE_SIZE)
            if (size - 1) % group_count == 0
            else None
        )
    group_count = 3 + SOLAR_ORIENTATION_FEATURE_SIZE
    if (size - 1) % group_count == 0:
        return ((size - 1) // group_count, SOLAR_ORIENTATION_FEATURE_SIZE)
    group_count = 3 + SOLAR_PROFILE_SIZE
    if (size - 1) % group_count == 0:
        return ((size - 1) // group_count, SOLAR_PROFILE_SIZE)
    if (size - 1) % 3 == 0:
        return ((size - 1) // 3, 0)
    if (size - 1) % 4 == 0:
        return ((size - 1) // 4, 1)
    return None


def coerce_int(value: Any, default: int) -> int:
    """Coerce a positive integer, falling back to default."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    if value <= 0:
        value = default
    return value


def coerce_float(value: Any, default: float) -> float:
    """Coerce a float, falling back to default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_float_list(values: Any) -> list[float]:
    """Coerce a sequence into finite and non-finite float values."""
    if not isinstance(values, (list, tuple)):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def coerce_int_list(values: Any) -> list[int]:
    """Coerce a sequence into integer values."""
    if not isinstance(values, (list, tuple)):
        return []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def coerce_optional_float(value: Any) -> float | None:
    """Coerce a finite optional float."""
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def coerce_solar_profile_history(values: Any) -> list[list[float]]:
    """Coerce persisted solar profile history rows."""
    if not isinstance(values, (list, tuple)):
        return []
    out: list[list[float]] = []
    for value in values:
        if not isinstance(value, (list, tuple)):
            continue
        profile = coerce_solar_profile(value)
        out.append(list(profile))
    return out


def coerce_solar_sample_history(values: Any) -> list[dict[str, float | None]]:
    """Coerce persisted solar sample history rows."""
    if not isinstance(values, (list, tuple)):
        return []
    out: list[dict[str, float | None]] = []
    for value in values:
        out.append(coerce_solar_sample(value).as_dict())
    return out


def restore_solar_sample_history(
    history: dict[str, Any],
    solar_hist: list[float],
    *,
    use_solar_feature: bool,
    history_steps: int,
) -> list[dict[str, float | None]]:
    """Restore solar samples with legacy profile and scalar fallbacks."""
    samples = coerce_solar_sample_history(history.get("solar_samples"))[:history_steps]
    if not use_solar_feature or samples:
        return samples

    legacy_profiles = coerce_solar_profile_history(history.get("solar_profile"))
    if legacy_profiles:
        return [coerce_solar_sample(row).as_dict() for row in legacy_profiles][
            :history_steps
        ]
    return [diffuse_only_sample(value).as_dict() for value in solar_hist][
        :history_steps
    ]


def restore_orientation_state(state: dict[str, Any]) -> RestoredOrientationState:
    """Restore persisted orientation metadata defensively."""
    azimuth = coerce_optional_float(state.get("solar_orientation_azimuth"))
    if azimuth is None:
        diagnostics = state.get("solar_orientation")
        if isinstance(diagnostics, dict):
            azimuth = coerce_optional_float(diagnostics.get("azimuth"))
    confidence = coerce_optional_float(state.get("solar_orientation_confidence"))
    if confidence is None:
        diagnostics = state.get("solar_orientation")
        if isinstance(diagnostics, dict):
            confidence = coerce_optional_float(diagnostics.get("confidence"))
    return RestoredOrientationState(
        azimuth=azimuth % 360.0 if azimuth is not None else None,
        confidence=confidence,
        score=coerce_optional_float(state.get("solar_orientation_score")),
    )


def restore_k_step_stats(
    payload: Any,
) -> dict[int, dict[str, float | int | None]]:
    """Restore k-step stats, supporting legacy list values."""
    if payload is None:
        return {}
    restored: dict[int, dict[str, float | int | None]] = {}
    for key, values in payload.items():
        horizon = int(key)
        if isinstance(values, dict):
            mean = values.get("mean")
            count = values.get("count", 0)
            mean_val = float(mean) if mean is not None else None
            count_val = int(count)
        else:
            items = coerce_float_list(values)
            mean_val = float(np.mean(items)) if items else None
            count_val = len(items)
        restored[horizon] = {"mean": mean_val, "count": count_val}
    return restored


def infer_legacy_orientation_from_weights(
    weights: np.ndarray, old_steps: int, old_solar_start: int
) -> float | None:
    """Infer a two-facade orientation from old compass-bin solar weights."""
    if weights.size < old_solar_start + SOLAR_PROFILE_SIZE:
        return None
    exposure_bins = np.zeros(len(SOLAR_PROFILE_BIN_AZIMUTHS), dtype=float)
    for step in range(old_steps):
        row_start = old_solar_start + step * SOLAR_PROFILE_SIZE
        row = weights[row_start : row_start + SOLAR_PROFILE_SIZE]
        if row.size < SOLAR_PROFILE_SIZE:
            continue
        for index in range(len(SOLAR_PROFILE_BIN_AZIMUTHS)):
            weight = float(row[index + 1])
            if np.isfinite(weight) and weight > 0.0:
                exposure_bins[index] += weight
    if not np.any(exposure_bins > 0.0):
        return None
    best_score = -1.0
    best_azimuth = None
    for azimuth in SOLAR_PROFILE_BIN_AZIMUTHS:
        first = legacy_bin_index(azimuth)
        second = legacy_bin_index(azimuth + 90.0)
        score = exposure_bins[first] + exposure_bins[second]
        if score > best_score:
            best_score = float(score)
            best_azimuth = float(azimuth)
    return best_azimuth


def legacy_bin_index(azimuth: float) -> int:
    """Return the nearest legacy eight-way bin index for an azimuth."""
    return int(round((azimuth % 360.0) / 45.0)) % len(SOLAR_PROFILE_BIN_AZIMUTHS)
