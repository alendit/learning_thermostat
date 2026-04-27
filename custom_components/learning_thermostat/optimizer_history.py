"""History buffer helpers for the learning optimizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from .optimizer_features import pad_sequence, sanitize_array
from .solar import (
    SOLAR_ORIENTATION_FEATURE_SIZE,
    build_solar_orientation_features,
    coerce_solar_sample,
    diffuse_only_sample,
)


def update_latest_first_histories(
    *,
    t_hist: list[float],
    out_hist: list[float],
    solar_hist: list[float],
    solar_sample_hist: list[dict[str, float | None]],
    u_hist: list[int],
    history_steps: int,
    use_solar_feature: bool,
    t_room: float,
    t_out: float,
    u_actual: int,
    solar: float | None,
    solar_sample: dict[str, Any] | None,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[dict[str, float | None]],
    list[int],
]:
    """Return updated latest-first history buffers."""
    updated_t_hist = [float(t_room), *t_hist][:history_steps]
    updated_out_hist = [float(t_out), *out_hist][:history_steps]
    updated_solar_hist = solar_hist[:history_steps]
    updated_solar_sample_hist = solar_sample_hist[:history_steps]
    if use_solar_feature:
        if solar is None:
            solar_value = solar_hist[0] if solar_hist else 0.0
        else:
            solar_value = float(solar)
        updated_solar_hist = [solar_value, *solar_hist][:history_steps]
        sample_value = coerce_solar_sample(solar_sample, solar_value)
        updated_solar_sample_hist = [
            sample_value.as_dict(),
            *solar_sample_hist,
        ][:history_steps]
    updated_u_hist = [int(u_actual), *u_hist][:history_steps]
    return (
        updated_t_hist,
        updated_out_hist,
        updated_solar_hist,
        updated_solar_sample_hist,
        updated_u_hist,
    )


def build_history_arrays(
    *,
    t_hist: list[float],
    out_hist: list[float],
    solar_hist: list[float],
    solar_sample_hist: list[dict[str, float | None]],
    u_hist: list[int],
    history_steps: int,
    use_solar_feature: bool,
    solar_orientation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Build sanitized fixed-length history arrays for the model."""
    t_pad = pad_sequence(
        t_hist,
        t_hist[-1] if t_hist else 0.0,
        history_steps=history_steps,
    )
    out_pad = pad_sequence(
        out_hist,
        out_hist[-1] if out_hist else 0.0,
        history_steps=history_steps,
    )
    solar_pad: np.ndarray | None = None
    if use_solar_feature:
        sample_rows = solar_sample_hist
        if not sample_rows and solar_hist:
            sample_rows = [diffuse_only_sample(value).as_dict() for value in solar_hist]
        if sample_rows:
            rows = [
                list(
                    build_solar_orientation_features(
                        coerce_solar_sample(row),
                        solar_orientation,
                    )
                )
                for row in sample_rows
            ]
            if len(rows) < history_steps:
                rows.extend([rows[-1]] * (history_steps - len(rows)))
            solar_pad = np.array(rows[:history_steps], dtype=float)
        else:
            solar_pad = np.zeros(
                (history_steps, SOLAR_ORIENTATION_FEATURE_SIZE), dtype=float
            )
    u_pad = pad_sequence(
        [float(v) for v in u_hist],
        0.0,
        history_steps=history_steps,
    )
    return (
        sanitize_array(t_pad),
        sanitize_array(out_pad),
        sanitize_array(solar_pad) if solar_pad is not None else None,
        sanitize_array(u_pad),
    )
