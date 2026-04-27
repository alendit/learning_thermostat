"""Feature-vector and weight-layout helpers for the learning optimizer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .const import LEARNING_MODEL_PHYSICAL
from .solar import SOLAR_ORIENTATION_FEATURE_SIZE

Z_SCORE_MIN_STD = 0.2
PHYSICAL_HALF_LIFE_MINUTES = (60.0, 180.0, 360.0)
PHYSICAL_SOLAR_SCALE = 500.0


def solar_profile_size(use_solar_feature: bool) -> int:
    """Return the number of solar features per history step."""
    return SOLAR_ORIENTATION_FEATURE_SIZE if use_solar_feature else 0


def feature_size(
    *,
    learning_model: str,
    history_steps: int,
    use_solar_feature: bool,
) -> int:
    """Return the feature vector size without the bias term."""
    if learning_model == LEARNING_MODEL_PHYSICAL:
        solar_size = 9 if use_solar_feature else 0
        return 6 + solar_size + 3
    return history_steps * (3 + solar_profile_size(use_solar_feature))


def default_weights(
    *,
    learning_model: str,
    history_steps: int,
    use_solar_feature: bool,
) -> np.ndarray:
    """Return default model weights."""
    size = feature_size(
        learning_model=learning_model,
        history_steps=history_steps,
        use_solar_feature=use_solar_feature,
    )
    return np.zeros(size + 1, dtype=float)


def init_weights(
    weights_init: Any,
    *,
    learning_model: str,
    history_steps: int,
    use_solar_feature: bool,
) -> np.ndarray:
    """Build initial weights, optionally seeded by configuration."""
    weights = default_weights(
        learning_model=learning_model,
        history_steps=history_steps,
        use_solar_feature=use_solar_feature,
    )
    size = weights.size
    if isinstance(weights_init, (list, tuple)) and len(weights_init) == size:
        try:
            weights = np.array([float(v) for v in weights_init], dtype=float)
        except (TypeError, ValueError):
            pass
    return weights


def u_weight_start(
    *,
    learning_model: str,
    history_steps: int,
    use_solar_feature: bool,
) -> int:
    """Return the start index of the heating-control weights."""
    if learning_model == LEARNING_MODEL_PHYSICAL:
        return 6 + (9 if use_solar_feature else 0)
    return history_steps * 2 + history_steps * solar_profile_size(use_solar_feature)


def heat_feature_count(*, learning_model: str, history_steps: int) -> int:
    """Return the number of heating-control features."""
    return 3 if learning_model == LEARNING_MODEL_PHYSICAL else history_steps


def sanitize_array(values: np.ndarray) -> np.ndarray:
    """Replace non-finite values with a reasonable fill."""
    if np.all(np.isfinite(values)):
        return values
    finite_vals = values[np.isfinite(values)]
    fill_value = float(finite_vals[0]) if finite_vals.size > 0 else 0.0
    return np.where(np.isfinite(values), values, fill_value)


def pad_sequence(
    values: list[float],
    pad_value: float,
    *,
    history_steps: int,
) -> np.ndarray:
    """Pad history to a fixed length using the most recent value."""
    if not values:
        return np.full(history_steps, pad_value, dtype=float)
    if len(values) >= history_steps:
        return np.array(values[:history_steps], dtype=float)
    padding = [pad_value] * (history_steps - len(values))
    return np.array(values + padding, dtype=float)


def transform_histories(
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Transform raw histories into model inputs (z-scored temps + raw u)."""
    t_mean = float(np.mean(t_hist)) if t_hist.size else 0.0
    t_std = float(np.std(t_hist)) if t_hist.size else 1.0
    if not np.isfinite(t_mean):
        t_mean = 0.0
    if not np.isfinite(t_std) or t_std < Z_SCORE_MIN_STD:
        t_std = Z_SCORE_MIN_STD
    t_norm = (t_hist - t_mean) / t_std

    out_mean = float(np.mean(out_hist)) if out_hist.size else 0.0
    out_std = float(np.std(out_hist)) if out_hist.size else 1.0
    if not np.isfinite(out_mean):
        out_mean = 0.0
    if not np.isfinite(out_std) or out_std < Z_SCORE_MIN_STD:
        out_std = Z_SCORE_MIN_STD
    out_norm = (out_hist - out_mean) / out_std
    solar_norm: np.ndarray | None = None
    if solar_hist is not None:
        solar_mean = (
            np.mean(solar_hist, axis=0)
            if solar_hist.size
            else np.zeros(SOLAR_ORIENTATION_FEATURE_SIZE, dtype=float)
        )
        solar_std = (
            np.std(solar_hist, axis=0)
            if solar_hist.size
            else np.ones(SOLAR_ORIENTATION_FEATURE_SIZE, dtype=float)
        )
        solar_mean = np.where(np.isfinite(solar_mean), solar_mean, 0.0)
        solar_std = np.where(
            np.isfinite(solar_std) & (solar_std >= Z_SCORE_MIN_STD),
            solar_std,
            Z_SCORE_MIN_STD,
        )
        solar_norm = (solar_hist - solar_mean) / solar_std
    return (
        sanitize_array(t_norm),
        sanitize_array(out_norm),
        sanitize_array(solar_norm) if solar_norm is not None else None,
        u_hist,
    )


def build_model_feature(
    *,
    learning_model: str,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
) -> np.ndarray:
    """Build a feature vector for the selected learning model."""
    if learning_model == LEARNING_MODEL_PHYSICAL:
        return build_physical_feature(
            feature_step_minutes=feature_step_minutes,
            t_hist=t_hist,
            out_hist=out_hist,
            solar_hist=solar_hist,
            u_hist=u_hist,
        )
    t_feat, out_feat, solar_feat, u_feat = transform_histories(
        t_hist, out_hist, solar_hist, u_hist
    )
    return build_linear_feature(
        history_decay_vector=history_decay_vector,
        t_hist=t_feat,
        out_hist=out_feat,
        solar_hist=solar_feat,
        u_hist=u_feat,
    )


def build_physical_feature(
    *,
    feature_step_minutes: float,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
) -> np.ndarray:
    """Build fixed-scale physical thermal features plus bias."""
    room_now = float(t_hist[0]) if t_hist.size else 20.0
    outside_now = float(out_hist[0]) if out_hist.size else 10.0
    room_1h = history_value_at(t_hist, 60.0, room_now, feature_step_minutes)
    room_3h = history_value_at(t_hist, 180.0, room_now, feature_step_minutes)
    outside_3h = history_value_at(out_hist, 180.0, outside_now, feature_step_minutes)
    parts = [
        np.array(
            [
                (room_now - 20.0) / 10.0,
                (outside_now - 10.0) / 20.0,
                (room_now - outside_now) / 20.0,
                (room_now - room_1h) / 2.0,
                (room_now - room_3h) / 4.0,
                (outside_now - outside_3h) / 10.0,
            ],
            dtype=float,
        )
    ]
    if solar_hist is not None:
        parts.append(
            physical_solar_features(
                solar_hist,
                feature_step_minutes=feature_step_minutes,
            )
        )
    parts.append(
        physical_filter_features(
            u_hist,
            scale=1.0,
            feature_step_minutes=feature_step_minutes,
        )
    )
    parts.append(np.array([1.0], dtype=float))
    return sanitize_array(np.concatenate(parts))


def history_value_at(
    values: np.ndarray,
    minutes_ago: float,
    default: float,
    feature_step_minutes: float,
) -> float:
    """Return a latest-first history value at a time offset with interpolation."""
    if values.size == 0:
        return float(default)
    step_minutes = max(float(feature_step_minutes or 60.0), 1.0)
    position = max(float(minutes_ago), 0.0) / step_minutes
    if position <= 0.0:
        return float(values[0])
    if position >= values.size - 1:
        return float(values[-1])
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def physical_filter_delta(half_life_minutes: float, step_minutes: float) -> float:
    """Return the newest-sample response of one exponential filter."""
    minutes = max(float(step_minutes or 60.0), 1.0)
    a = 0.5 ** (minutes / float(half_life_minutes))
    return 1.0 - a


def physical_filter_features(
    values: np.ndarray,
    *,
    scale: float,
    feature_step_minutes: float,
) -> np.ndarray:
    """Return fast/medium/slow exposure filters for latest-first values."""
    if values.size == 0:
        values = np.zeros(1, dtype=float)
    ordered = list(reversed([float(v) for v in values]))
    step_minutes = max(float(feature_step_minutes or 60.0), 1.0)
    filtered: list[float] = []
    for half_life in PHYSICAL_HALF_LIFE_MINUTES:
        state = 0.0
        a = 0.5 ** (step_minutes / half_life)
        for value in ordered:
            state = a * state + (1.0 - a) * value
        filtered.append(state / scale)
    return np.array(filtered, dtype=float)


def physical_solar_features(
    solar_hist: np.ndarray,
    *,
    feature_step_minutes: float,
) -> np.ndarray:
    """Return filtered diffuse and facade solar exposure features."""
    rows = np.array(solar_hist, dtype=float)
    if rows.ndim != 2 or rows.shape[1] < SOLAR_ORIENTATION_FEATURE_SIZE:
        rows = np.zeros((1, SOLAR_ORIENTATION_FEATURE_SIZE), dtype=float)
    diffuse = physical_filter_features(
        rows[:, 0],
        scale=PHYSICAL_SOLAR_SCALE,
        feature_step_minutes=feature_step_minutes,
    )
    facade_1 = physical_filter_features(
        rows[:, 1],
        scale=PHYSICAL_SOLAR_SCALE,
        feature_step_minutes=feature_step_minutes,
    )
    facade_2 = physical_filter_features(
        rows[:, 2],
        scale=PHYSICAL_SOLAR_SCALE,
        feature_step_minutes=feature_step_minutes,
    )
    return np.concatenate((diffuse, facade_1, facade_2))


def build_linear_feature(
    *,
    history_decay_vector: np.ndarray,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
) -> np.ndarray:
    """Assemble the z-scored linear feature vector plus bias."""
    decay = history_decay_vector
    t_hist = t_hist * decay
    out_hist = out_hist * decay
    parts = [t_hist, out_hist]
    if solar_hist is not None:
        solar_decay = np.repeat(decay, SOLAR_ORIENTATION_FEATURE_SIZE)
        parts.append(solar_hist.reshape(-1) * solar_decay)
    u_hist = u_hist * decay
    parts.extend((u_hist, np.array([1.0], dtype=float)))
    return np.concatenate(parts)
