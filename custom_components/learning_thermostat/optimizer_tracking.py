"""K-step prediction tracking helpers for the learning optimizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from .optimizer_types import KStepUpdateInput

DYNAMIC_LOOKAHEAD_TOLERANCE = 0.2
MULTI_STEP_UPDATE_HORIZON = 3

TrackingStats = dict[int, dict[str, float | int | None]]
PendingPredictions = dict[int, dict[int, float]]
PendingFeatures = dict[int, dict[int, list[float]]]
PendingBaselines = dict[int, dict[int, float]]


def clear_pending_predictions(
    *,
    pending_preds: PendingPredictions,
    pending_features: PendingFeatures,
    pending_baseline: PendingBaselines,
) -> None:
    """Drop pending learning rows whose feature semantics may be stale."""
    pending_preds.clear()
    pending_features.clear()
    pending_baseline.clear()


def record_k_step_errors(
    *,
    actual_temp: float,
    step_index: int,
    pending_preds: PendingPredictions,
    pending_features: PendingFeatures,
    pending_baseline: PendingBaselines,
    abs_error_history: TrackingStats,
    pred_improvement_history: TrackingStats,
    multi_step_update_horizon: int = MULTI_STEP_UPDATE_HORIZON,
) -> tuple[dict[int, float], list[KStepUpdateInput]]:
    """Update k-step error stats and return absolute errors plus update inputs."""
    errors: dict[int, float] = {}
    updates: list[KStepUpdateInput] = []
    _drop_stale(pending_preds, step_index)
    _drop_stale(pending_features, step_index)
    _drop_stale(pending_baseline, step_index)

    matured = pending_preds.pop(step_index, None)
    if not matured:
        return errors, updates
    matured_features = pending_features.pop(step_index, None) or {}
    baseline_matured = pending_baseline.pop(step_index, None) or {}
    for horizon, pred in matured.items():
        try:
            pred_val = float(pred)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pred_val):
            continue
        signed_error = actual_temp - pred_val
        err = abs(signed_error)
        if not np.isfinite(err):
            continue
        horizon_int = int(horizon)
        errors[horizon_int] = float(err)
        if horizon_int <= multi_step_update_horizon:
            feature = matured_features.get(horizon_int)
            if feature is not None:
                updates.append(
                    {
                        "horizon": horizon_int,
                        "error": float(signed_error),
                        "feature": feature,
                    }
                )
        _update_mean_stat(abs_error_history, horizon_int, float(err))

        baseline_pred = baseline_matured.get(horizon)
        if baseline_pred is not None:
            try:
                baseline_val = float(baseline_pred)
            except (TypeError, ValueError):
                baseline_val = None
            if baseline_val is not None and np.isfinite(baseline_val):
                baseline_err = abs(actual_temp - baseline_val)
                if np.isfinite(baseline_err):
                    improvement = float(baseline_err - err)
                    _update_mean_stat(
                        pred_improvement_history, horizon_int, improvement
                    )
    return errors, updates


def queue_k_step_predictions(
    *,
    pending_preds: PendingPredictions,
    preds: list[float],
    step_index: int,
    max_steps: int,
) -> None:
    """Queue k-step predictions keyed by target step index."""
    horizon_max = min(max_steps, len(preds))
    if horizon_max <= 0:
        return
    for horizon in range(1, horizon_max + 1):
        try:
            pred_val = float(preds[horizon - 1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pred_val):
            continue
        target_step = step_index + horizon
        bucket = pending_preds.setdefault(target_step, {})
        bucket[horizon] = pred_val


def queue_k_step_features(
    *,
    pending_features: PendingFeatures,
    features: list[np.ndarray],
    step_index: int,
    max_steps: int,
    multi_step_update_horizon: int = MULTI_STEP_UPDATE_HORIZON,
) -> None:
    """Queue k-step feature vectors keyed by target step index."""
    horizon_max = min(multi_step_update_horizon, max_steps, len(features))
    if horizon_max <= 0:
        return
    for horizon in range(1, horizon_max + 1):
        feature = features[horizon - 1]
        if feature is None:
            continue
        target_step = step_index + horizon
        bucket = pending_features.setdefault(target_step, {})
        bucket[horizon] = feature.tolist()


def queue_k_step_baseline(
    *,
    pending_baseline: PendingBaselines,
    baseline_temp: float,
    step_index: int,
    max_steps: int,
) -> None:
    """Queue trivial baseline predictions keyed by target step index."""
    try:
        baseline_val = float(baseline_temp)
    except (TypeError, ValueError):
        return
    if not np.isfinite(baseline_val):
        return
    for horizon in range(1, max_steps + 1):
        target_step = step_index + horizon
        bucket = pending_baseline.setdefault(target_step, {})
        bucket[horizon] = baseline_val


def k_step_error_stats(
    abs_error_history: TrackingStats, max_steps: int
) -> TrackingStats:
    """Return mean absolute error and count per horizon."""
    stats: TrackingStats = {}
    for horizon in range(1, max_steps + 1):
        values = abs_error_history.get(horizon)
        if isinstance(values, dict):
            mean = values.get("mean")
            count = values.get("count", 0)
            try:
                mean_val = float(mean) if mean is not None else None
            except (TypeError, ValueError):
                mean_val = None
            try:
                count_val = int(count)
            except (TypeError, ValueError):
                count_val = 0
            stats[horizon] = {"mean": mean_val, "count": count_val}
        else:
            stats[horizon] = {"mean": None, "count": 0}
    return stats


def select_dynamic_lookahead(
    *,
    stats: TrackingStats,
    min_steps: int,
    max_steps: int,
    tolerance: float = DYNAMIC_LOOKAHEAD_TOLERANCE,
) -> tuple[int, str]:
    """Pick the largest horizon whose mean k-step error stays within tolerance."""
    reason = "insufficient_data"
    for horizon in range(max_steps, min_steps - 1, -1):
        ok = True
        for step in range(1, horizon + 1):
            mean = stats.get(step, {}).get("mean")
            if mean is None:
                ok = False
                break
            if mean > tolerance:
                ok = False
                reason = "above_tolerance"
                break
        if ok:
            return horizon, "within_tolerance"
    return min_steps, reason


def _drop_stale(payload: dict[int, Any], step_index: int) -> None:
    """Remove target rows older than the current step index."""
    if payload:
        stale_steps = [key for key in payload if key < step_index]
        for key in stale_steps:
            payload.pop(key, None)


def _update_mean_stat(stats: TrackingStats, horizon: int, value: float) -> None:
    """Update one running mean/count stat."""
    row = stats.setdefault(horizon, {"mean": None, "count": 0})
    count = row.get("count", 0)
    mean = row.get("mean")
    try:
        count_val = int(count)
    except (TypeError, ValueError):
        count_val = 0
    try:
        mean_val = float(mean) if mean is not None else None
    except (TypeError, ValueError):
        mean_val = None
    if mean_val is None or count_val <= 0:
        mean_val = float(value)
        count_val = 1
    else:
        mean_val = (mean_val * count_val + float(value)) / float(count_val + 1)
        count_val += 1
    row["mean"] = mean_val
    row["count"] = count_val
