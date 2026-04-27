"""Gradient update helpers for the learning optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimizer_types import KStepUpdateInput, LearningUpdatePayload


@dataclass(frozen=True, slots=True)
class LearningUpdate:
    """Candidate optimizer weights plus the stable diagnostic payload."""

    weights: np.ndarray
    payload: LearningUpdatePayload


def compute_single_step_update(
    *,
    weights: np.ndarray,
    l2_reg: float,
    actual_temp: float,
    prev_pred: float | None,
    prev_feature: list[float] | None,
    base_eta: float,
    eta: float,
    fallback_learning_rate: float,
) -> LearningUpdate | None:
    """Compute a normalized gradient update from the previous rollout."""
    if prev_pred is None or prev_feature is None:
        return None

    try:
        base_eta = float(base_eta)
    except (TypeError, ValueError):
        base_eta = float(fallback_learning_rate)
    if not np.isfinite(base_eta) or base_eta < 0.0:
        base_eta = float(fallback_learning_rate)
    try:
        eta = float(eta)
    except (TypeError, ValueError):
        eta = base_eta
    if not np.isfinite(eta) or eta < 0.0:
        eta = base_eta

    try:
        error = actual_temp - float(prev_pred)
        feature = np.array(prev_feature, dtype=float)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(error) or not np.all(np.isfinite(feature)):
        return None

    denom = float(np.dot(feature, feature)) / float(feature.size)
    if not np.isfinite(denom) or denom <= 0:
        return None

    current_weights = np.array(weights, dtype=float, copy=True)
    gradient = error * (feature / (denom + 1e-6))
    reg = 0.0
    if l2_reg:
        reg_weights = current_weights.copy()
        if reg_weights.size:
            reg_weights[-1] = 0.0
        reg = l2_reg * reg_weights
    delta = eta * (gradient - reg)
    updated_weights = current_weights + delta
    return LearningUpdate(
        weights=updated_weights,
        payload={
            "error": error,
            "errors": [{"step": 0, "error": error, "weight": 1.0}],
            "eta": eta,
            "base_eta": base_eta,
            "eta_scale": eta / base_eta if base_eta else None,
            "l2_reg": l2_reg,
            "grad_norm": (
                float(np.linalg.norm(gradient))
                if np.all(np.isfinite(gradient))
                else None
            ),
            "abs_update_max": (
                float(np.max(np.abs(delta))) if np.all(np.isfinite(delta)) else None
            ),
            "delta": delta.tolist() if np.all(np.isfinite(delta)) else None,
            "steps_used": 1,
        },
    )


def compute_multi_step_update(
    *,
    weights: np.ndarray,
    l2_reg: float,
    updates: list[KStepUpdateInput],
    base_eta: float,
    eta: float,
) -> LearningUpdate | None:
    """Compute a normalized gradient update from multiple matured predictions."""
    if not updates:
        return None

    gradients = None
    total_weight = 0.0
    errors: list[dict[str, float | int]] = []
    for update in updates:
        error = update.get("error")
        feature = update.get("feature")
        step = update.get("horizon")
        if error is None or feature is None:
            continue
        try:
            error_val = float(error)
            feature_vec = np.array(feature, dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(error_val) or not np.all(np.isfinite(feature_vec)):
            continue
        denom = float(np.dot(feature_vec, feature_vec)) / float(feature_vec.size)
        if not np.isfinite(denom) or denom <= 0:
            continue
        weight = 1.0
        gradient = weight * error_val * (feature_vec / (denom + 1e-6))
        gradients = gradient if gradients is None else gradients + gradient
        total_weight += weight
        errors.append(
            {
                "step": int(step) if isinstance(step, (int, float)) else 0,
                "error": error_val,
                "weight": weight,
            }
        )

    if gradients is None or total_weight <= 0:
        return None

    current_weights = np.array(weights, dtype=float, copy=True)
    gradients = gradients / total_weight
    reg = 0.0
    if l2_reg:
        reg_weights = current_weights.copy()
        if reg_weights.size:
            reg_weights[-1] = 0.0
        reg = l2_reg * reg_weights
    delta = eta * (gradients - reg)
    updated_weights = current_weights + delta
    mean_error = (
        float(np.mean([entry["error"] for entry in errors])) if errors else None
    )
    return LearningUpdate(
        weights=updated_weights,
        payload={
            "error": mean_error,
            "errors": errors,
            "eta": eta,
            "base_eta": base_eta,
            "eta_scale": eta / base_eta if base_eta else None,
            "l2_reg": l2_reg,
            "grad_norm": (
                float(np.linalg.norm(gradients))
                if np.all(np.isfinite(gradients))
                else None
            ),
            "abs_update_max": (
                float(np.max(np.abs(delta))) if np.all(np.isfinite(delta)) else None
            ),
            "delta": delta.tolist() if np.all(np.isfinite(delta)) else None,
            "steps_used": len(errors),
        },
    )
