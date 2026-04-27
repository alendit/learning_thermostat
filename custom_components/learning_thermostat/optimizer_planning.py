"""Planning and forecast helpers for the learning optimizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from .optimizer_features import build_model_feature
from .solar import (
    SOLAR_ORIENTATION_FEATURE_SIZE,
    build_solar_orientation_features,
    coerce_solar_sample,
    diffuse_only_sample,
)


def effective_solar_orientation(azimuth: float | None) -> float:
    """Return the active orientation angle for feature construction."""
    if azimuth is None:
        return 0.0
    return azimuth % 360.0


def normalize_forecast(
    t_out: float,
    t_out_forecast: list[float] | None,
    steps: int,
    last_forecast_temp: float | None,
) -> tuple[list[float], float | None]:
    """Normalize forecast input to the lookahead length with fallbacks."""
    temps: list[float] = []
    if isinstance(t_out_forecast, list) and t_out_forecast:
        for value in t_out_forecast:
            try:
                temp_val = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(temp_val):
                continue
            temps.append(temp_val)

    if not temps:
        if last_forecast_temp is not None:
            temps = [t_out] + [last_forecast_temp] * (steps - 1)
        else:
            temps = [t_out] * steps
    elif len(temps) < steps:
        temps.extend([temps[-1]] * (steps - len(temps)))
    elif len(temps) > steps:
        temps = temps[:steps]

    return temps, temps[-1] if temps else last_forecast_temp


def normalize_solar_forecast(
    *,
    use_solar_feature: bool,
    solar: float | None,
    solar_forecast: list[float] | None,
    solar_sample: dict[str, Any] | None,
    solar_sample_forecast: list[dict[str, Any]] | None,
    steps: int,
    last_forecast_solar: float | None,
) -> tuple[list[dict[str, float | None]], float | None]:
    """Normalize solar forecast input to the lookahead length with fallbacks."""
    if not use_solar_feature or steps <= 0:
        return [], last_forecast_solar

    values: list[dict[str, float | None]] = []
    if isinstance(solar_sample_forecast, list) and solar_sample_forecast:
        for value in solar_sample_forecast:
            values.append(coerce_solar_sample(value).as_dict())

    if not values and isinstance(solar_forecast, list) and solar_forecast:
        for value in solar_forecast:
            values.append(diffuse_only_sample(value).as_dict())

    if not values:
        current_sample = coerce_solar_sample(solar_sample, solar)
        if current_sample.flux > 0.0:
            values = [current_sample.as_dict()] * steps
        elif last_forecast_solar is not None:
            values = [diffuse_only_sample(last_forecast_solar).as_dict()] * steps
        else:
            values = [diffuse_only_sample(0.0).as_dict()] * steps
    elif len(values) < steps:
        values.extend([dict(values[-1])] * (steps - len(values)))
    elif len(values) > steps:
        values = values[:steps]

    next_last = float(values[-1]["flux"] or 0.0) if values else last_forecast_solar
    return values, next_last


def predict_next(
    *,
    learning_model: str,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
    weights: np.ndarray,
    solar_orientation_azimuth: float | None,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
    u_value: int,
    solar_value: dict[str, Any] | float | None,
) -> tuple[float, np.ndarray]:
    """Predict next room temperature for a single step and return features."""
    u_hist_step = np.concatenate((np.array([float(u_value)], dtype=float), u_hist[:-1]))
    solar_hist_step: np.ndarray | None = None
    if solar_hist is not None:
        solar_step = np.array(
            build_solar_orientation_features(
                coerce_solar_sample(
                    solar_value,
                    solar_hist[0][0] if solar_hist.size else 0.0,
                ),
                effective_solar_orientation(solar_orientation_azimuth),
            ),
            dtype=float,
        )
        solar_hist_step = np.concatenate(
            (
                solar_step.reshape(1, SOLAR_ORIENTATION_FEATURE_SIZE),
                solar_hist[:-1],
            )
        )
    x = build_model_feature(
        learning_model=learning_model,
        history_decay_vector=history_decay_vector,
        feature_step_minutes=feature_step_minutes,
        t_hist=t_hist,
        out_hist=out_hist,
        solar_hist=solar_hist_step,
        u_hist=u_hist_step,
    )
    delta = float(x @ weights)
    t_next = float(t_hist[0] + delta)
    return t_next, x


def rollout(
    *,
    learning_model: str,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
    weights: np.ndarray,
    solar_orientation_azimuth: float | None,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
    t_outs: list[float],
    solar_forecast: list[dict[str, float | None]],
    start: int,
    stop: int,
    steps: int,
    capture_features: bool,
) -> tuple[list[float], list[np.ndarray]]:
    """Roll the model forward for a planned ON interval."""
    preds: list[float] = []
    features: list[np.ndarray] = []
    t_hist_cur = np.array(t_hist, dtype=float)
    out_hist_cur = np.array(out_hist, dtype=float)
    solar_hist_cur = (
        np.array(solar_hist, dtype=float) if solar_hist is not None else None
    )
    u_hist_cur = np.array(u_hist, dtype=float)

    for idx in range(steps):
        u = 1 if start <= idx < stop else 0
        solar_value = solar_forecast[idx] if idx < len(solar_forecast) else None
        t_next, x = predict_next(
            learning_model=learning_model,
            history_decay_vector=history_decay_vector,
            feature_step_minutes=feature_step_minutes,
            weights=weights,
            solar_orientation_azimuth=solar_orientation_azimuth,
            t_hist=t_hist_cur,
            out_hist=out_hist_cur,
            solar_hist=solar_hist_cur,
            u_hist=u_hist_cur,
            u_value=u,
            solar_value=solar_value,
        )
        preds.append(t_next)
        if capture_features:
            features.append(x)

        t_hist_cur = np.concatenate((np.array([t_next], dtype=float), t_hist_cur[:-1]))
        out_hist_cur = np.concatenate(
            (np.array([float(t_outs[idx])], dtype=float), out_hist_cur[:-1])
        )
        if solar_hist_cur is not None:
            next_solar = np.array(
                build_solar_orientation_features(
                    coerce_solar_sample(
                        solar_value,
                        solar_hist_cur[0][0] if solar_hist_cur.size else 0.0,
                    ),
                    effective_solar_orientation(solar_orientation_azimuth),
                ),
                dtype=float,
            )
            solar_hist_cur = np.concatenate(
                (
                    next_solar.reshape(1, SOLAR_ORIENTATION_FEATURE_SIZE),
                    solar_hist_cur[:-1],
                )
            )
        u_hist_cur = np.concatenate(
            (np.array([float(u)], dtype=float), u_hist_cur[:-1])
        )

    return preds, features


def simulate_schedule(
    *,
    learning_model: str,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
    weights: np.ndarray,
    solar_orientation_azimuth: float | None,
    error_decay: float,
    w_energy: Any,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
    t_set: float,
    t_outs: list[float],
    solar_forecast: list[dict[str, float | None]],
    start: int,
    stop: int,
) -> tuple[float, list[float]]:
    """Score a plan by simulating its temperature error and energy cost."""
    steps = len(t_outs)
    preds, _ = rollout(
        learning_model=learning_model,
        history_decay_vector=history_decay_vector,
        feature_step_minutes=feature_step_minutes,
        weights=weights,
        solar_orientation_azimuth=solar_orientation_azimuth,
        t_hist=t_hist,
        out_hist=out_hist,
        solar_hist=solar_hist,
        u_hist=u_hist,
        t_outs=t_outs,
        solar_forecast=solar_forecast,
        start=start,
        stop=stop,
        steps=steps,
        capture_features=False,
    )
    try:
        energy_weight = float(w_energy)
    except (TypeError, ValueError):
        energy_weight = 0.0
    if not np.isfinite(error_decay) or error_decay < 0.0 or error_decay > 1.0:
        error_decay = 1.0
    error_weights = None
    if error_decay < 1.0:
        error_weights = error_decay ** np.arange(steps, dtype=float)
    cost = 0.0
    for idx, t_pred in enumerate(preds):
        u = 1 if start <= idx < stop else 0
        err = t_pred - t_set
        err_weight = error_weights[idx] if error_weights is not None else 1.0
        cost += err_weight * err * err + energy_weight * u
    return cost, preds


def find_start_stop(
    *,
    learning_model: str,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
    weights: np.ndarray,
    solar_orientation_azimuth: float | None,
    error_decay: float,
    w_energy: Any,
    t_hist: np.ndarray,
    out_hist: np.ndarray,
    solar_hist: np.ndarray | None,
    u_hist: np.ndarray,
    t_set: float,
    t_outs: list[float],
    solar_forecast: list[dict[str, float | None]],
) -> tuple[int, int, float]:
    """Search all start/stop pairs for a single ON interval to minimize cost."""
    steps = len(t_outs)
    best_start = steps
    best_stop = steps
    best_cost, _ = simulate_schedule(
        learning_model=learning_model,
        history_decay_vector=history_decay_vector,
        feature_step_minutes=feature_step_minutes,
        weights=weights,
        solar_orientation_azimuth=solar_orientation_azimuth,
        error_decay=error_decay,
        w_energy=w_energy,
        t_hist=t_hist,
        out_hist=out_hist,
        solar_hist=solar_hist,
        u_hist=u_hist,
        t_set=t_set,
        t_outs=t_outs,
        solar_forecast=solar_forecast,
        start=steps,
        stop=steps,
    )
    for start in range(steps + 1):
        for stop in range(start, steps + 1):
            cost, _ = simulate_schedule(
                learning_model=learning_model,
                history_decay_vector=history_decay_vector,
                feature_step_minutes=feature_step_minutes,
                weights=weights,
                solar_orientation_azimuth=solar_orientation_azimuth,
                error_decay=error_decay,
                w_energy=w_energy,
                t_hist=t_hist,
                out_hist=out_hist,
                solar_hist=solar_hist,
                u_hist=u_hist,
                t_set=t_set,
                t_outs=t_outs,
                solar_forecast=solar_forecast,
                start=start,
                stop=stop,
            )
            if cost < best_cost:
                best_cost = cost
                best_start = start
                best_stop = stop
    return best_start, best_stop, best_cost
