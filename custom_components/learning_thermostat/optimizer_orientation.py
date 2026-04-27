"""Solar orientation diagnostics and scoring helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .const import LEARNING_MODEL_PHYSICAL
from .optimizer_features import (
    PHYSICAL_SOLAR_SCALE,
    build_model_feature,
    sanitize_array,
)
from .optimizer_state import coerce_optional_float
from .optimizer_types import SolarOrientationDiagnostics
from .solar import (
    SOLAR_ORIENTATION_FEATURE_SIZE,
    build_solar_orientation_features,
    compass_label,
    coerce_solar_sample,
)

SOLAR_ACTIVE_FLUX_MIN = 10.0


@dataclass(frozen=True)
class BestSolarOrientation:
    """Best candidate returned by the solar orientation grid search."""

    orientation: float
    weights: np.ndarray
    score: float
    baseline_score: float


def solar_weight_start(
    *,
    use_solar_feature: bool,
    learning_model: str,
    history_steps: int,
    weight_count: int,
) -> int:
    """Return the start index of solar weights for the active model."""
    if not use_solar_feature:
        return weight_count
    if learning_model == LEARNING_MODEL_PHYSICAL:
        return 6
    return history_steps * 2


def directional_solar_weight_indices(
    *,
    facade_index: int,
    use_solar_feature: bool,
    learning_model: str,
    history_steps: int,
    weight_count: int,
) -> list[int]:
    """Return positive-directional solar weight indices for one facade."""
    if not use_solar_feature:
        return []
    start_index = solar_weight_start(
        use_solar_feature=use_solar_feature,
        learning_model=learning_model,
        history_steps=history_steps,
        weight_count=weight_count,
    )
    if learning_model == LEARNING_MODEL_PHYSICAL:
        start = start_index + 3 + int(facade_index) * 3
        return [start, start + 1, start + 2]
    offset = 1 + int(facade_index)
    return [
        start_index + step * SOLAR_ORIENTATION_FEATURE_SIZE + offset
        for step in range(history_steps)
    ]


def unknown_solar_orientation_diagnostics() -> SolarOrientationDiagnostics:
    """Return the unknown orientation diagnostics payload."""
    return {
        "azimuth": None,
        "confidence": None,
        "orientation_1": None,
        "orientation_1_azimuth": None,
        "orientation_1_contribution_pct": None,
        "orientation_2": None,
        "orientation_2_azimuth": None,
        "orientation_2_contribution_pct": None,
        "state": "unknown",
    }


def solar_orientation_diagnostics(
    *,
    use_solar_feature: bool,
    weights: np.ndarray,
    learning_model: str,
    history_steps: int,
    solar_orientation_azimuth: float | None,
    solar_orientation_confidence: float | None,
) -> SolarOrientationDiagnostics:
    """Return the dominant learned solar exposure direction."""
    if not use_solar_feature:
        return unknown_solar_orientation_diagnostics()

    facade_weights = [0.0, 0.0]
    for facade_index in (0, 1):
        for weight_index in directional_solar_weight_indices(
            facade_index=facade_index,
            use_solar_feature=use_solar_feature,
            learning_model=learning_model,
            history_steps=history_steps,
            weight_count=weights.size,
        ):
            if weight_index >= weights.size:
                continue
            weight = float(weights[weight_index])
            if np.isfinite(weight) and weight > 0.0:
                facade_weights[facade_index] += weight

    total = sum(facade_weights)
    if total <= 0.0 or solar_orientation_azimuth is None:
        return unknown_solar_orientation_diagnostics()
    first = {
        "azimuth": solar_orientation_azimuth % 360.0,
        "contribution": facade_weights[0],
    }
    second = {
        "azimuth": (solar_orientation_azimuth + 90.0) % 360.0,
        "contribution": facade_weights[1],
    }
    ranked = sorted(
        [first, second], key=lambda item: item["contribution"], reverse=True
    )
    orientation_1_pct = 100.0 * ranked[0]["contribution"] / total
    orientation_2_pct = 100.0 * ranked[1]["contribution"] / total
    return {
        "azimuth": ranked[0]["azimuth"],
        "confidence": solar_orientation_confidence,
        "orientation_1": compass_label(ranked[0]["azimuth"]),
        "orientation_1_azimuth": ranked[0]["azimuth"],
        "orientation_1_contribution_pct": round(orientation_1_pct, 3),
        "orientation_2": compass_label(ranked[1]["azimuth"]),
        "orientation_2_azimuth": ranked[1]["azimuth"],
        "orientation_2_contribution_pct": round(orientation_2_pct, 3),
        "state": (
            f"{compass_label(ranked[0]['azimuth'])} {orientation_1_pct:.0f}%, "
            f"{compass_label(ranked[1]['azimuth'])} {orientation_2_pct:.0f}%"
        ),
    }


def find_best_solar_orientation(
    *,
    examples: list[dict[str, Any]],
    current_orientation: float | None,
    current_weights: np.ndarray,
    l2_reg: float,
    learning_model: str,
    history_steps: int,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
    grid_degrees: int,
    min_examples: int,
    min_active_examples: int,
    improvement_min: float,
) -> BestSolarOrientation | None:
    """Find the best solar orientation candidate from recorded examples."""
    if len(examples) < min_examples:
        return None
    active_count = sum(1 for example in examples if example_solar_active(example))
    if active_count < min_active_examples:
        return None

    baseline_score = fit_orientation_score(
        examples=examples,
        orientation=None,
        current_weights=current_weights,
        l2_reg=l2_reg,
        learning_model=learning_model,
        history_steps=history_steps,
        history_decay_vector=history_decay_vector,
        feature_step_minutes=feature_step_minutes,
    )[1]
    best_orientation = None
    best_weights = None
    best_score = math.inf
    best_tie_score = -math.inf
    for azimuth in range(0, 360, grid_degrees):
        weights, score = fit_orientation_score(
            examples=examples,
            orientation=float(azimuth),
            current_weights=current_weights,
            l2_reg=l2_reg,
            learning_model=learning_model,
            history_steps=history_steps,
            history_decay_vector=history_decay_vector,
            feature_step_minutes=feature_step_minutes,
        )
        tie_score = orientation_tie_score(
            weights=weights,
            examples=examples,
            orientation=float(azimuth),
            use_solar_feature=True,
            learning_model=learning_model,
            history_steps=history_steps,
        )
        if score < best_score - 1e-6 or (
            abs(score - best_score) <= 1e-6 and tie_score > best_tie_score
        ):
            best_orientation = float(azimuth)
            best_weights = weights
            best_score = score
            best_tie_score = tie_score

    if best_orientation is None or best_weights is None or not np.isfinite(best_score):
        return None

    _, current_score = fit_orientation_score(
        examples=examples,
        orientation=current_orientation,
        current_weights=current_weights,
        l2_reg=l2_reg,
        learning_model=learning_model,
        history_steps=history_steps,
        history_decay_vector=history_decay_vector,
        feature_step_minutes=feature_step_minutes,
    )
    if current_orientation is not None and np.isfinite(current_score):
        improvement = (current_score - best_score) / max(abs(current_score), 1e-9)
        if improvement < improvement_min:
            return None

    return BestSolarOrientation(
        orientation=best_orientation,
        weights=best_weights,
        score=best_score,
        baseline_score=baseline_score,
    )


def fit_orientation_score(
    *,
    examples: list[dict[str, Any]],
    orientation: float | None,
    current_weights: np.ndarray,
    l2_reg: float,
    learning_model: str,
    history_steps: int,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
) -> tuple[np.ndarray, float]:
    """Fit ridge weights for one candidate orientation and return MSE."""
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for example in examples:
        feature = feature_from_orientation_example(
            example=example,
            orientation=orientation,
            learning_model=learning_model,
            history_steps=history_steps,
            history_decay_vector=history_decay_vector,
            feature_step_minutes=feature_step_minutes,
        )
        target = coerce_optional_float(example.get("target_delta"))
        if feature is None or target is None:
            continue
        rows.append(feature)
        targets.append(target)
    if not rows:
        return current_weights.copy(), math.inf
    x = np.vstack(rows)
    y = np.array(targets, dtype=float)
    reg = max(float(l2_reg or 0.0), 1e-6)
    ridge = reg * np.eye(x.shape[1], dtype=float)
    ridge[-1, -1] = 0.0
    try:
        weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x.T @ x + ridge) @ x.T @ y
    if not np.all(np.isfinite(weights)):
        return current_weights.copy(), math.inf
    residual = x @ weights - y
    score = float(np.mean(residual * residual))
    return weights, score


def feature_from_orientation_example(
    *,
    example: dict[str, Any],
    orientation: float | None,
    learning_model: str,
    history_steps: int,
    history_decay_vector: np.ndarray,
    feature_step_minutes: float,
) -> np.ndarray | None:
    """Build a feature row for a raw orientation-training example."""
    try:
        t_hist = np.array(example["t_hist"], dtype=float)
        out_hist = np.array(example["out_hist"], dtype=float)
        u_hist = np.array(example["u_hist"], dtype=float)
        samples = example["solar_samples"]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        t_hist.size != history_steps
        or out_hist.size != history_steps
        or u_hist.size != history_steps
        or not isinstance(samples, list)
    ):
        return None
    if orientation is None:
        solar_rows = [
            [coerce_solar_sample(sample).flux, 0.0, 0.0]
            for sample in samples[:history_steps]
        ]
    else:
        solar_rows = [
            list(
                build_solar_orientation_features(
                    coerce_solar_sample(sample), orientation
                )
            )
            for sample in samples[:history_steps]
        ]
    if len(solar_rows) != history_steps:
        return None
    solar_hist = np.array(solar_rows, dtype=float)
    return build_model_feature(
        learning_model=learning_model,
        history_decay_vector=history_decay_vector,
        feature_step_minutes=feature_step_minutes,
        t_hist=sanitize_array(t_hist),
        out_hist=sanitize_array(out_hist),
        solar_hist=sanitize_array(solar_hist),
        u_hist=sanitize_array(u_hist),
    )


def example_solar_active(example: dict[str, Any]) -> bool:
    """Return whether an orientation example contains usable directional sun."""
    samples = example.get("solar_samples")
    if not isinstance(samples, list):
        return False
    for sample in samples:
        value = coerce_solar_sample(sample)
        if (
            value.flux > SOLAR_ACTIVE_FLUX_MIN
            and value.azimuth is not None
            and value.elevation is not None
            and value.elevation > 0.0
        ):
            return True
    return False


def clamp_directional_solar_weights(
    weights: np.ndarray,
    *,
    use_solar_feature: bool,
    learning_model: str,
    history_steps: int,
) -> np.ndarray:
    """Clamp learned directional solar weights to non-negative values."""
    clamped = np.array(weights, dtype=float)
    if not use_solar_feature:
        return clamped
    for facade_index in (0, 1):
        for index in directional_solar_weight_indices(
            facade_index=facade_index,
            use_solar_feature=use_solar_feature,
            learning_model=learning_model,
            history_steps=history_steps,
            weight_count=clamped.size,
        ):
            if index < clamped.size and clamped[index] < 0.0:
                clamped[index] = 0.0
    return clamped


def apply_empirical_solar_gains(
    weights: np.ndarray,
    *,
    examples: list[dict[str, Any]],
    orientation: float,
    learning_model: str,
    history_steps: int,
) -> np.ndarray:
    """Stabilize directional gains with non-negative one-step solar evidence."""
    adjusted = np.array(weights, dtype=float)
    numerators = [0.0, 0.0]
    denominators = [0.0, 0.0]
    for example in examples:
        samples = example.get("solar_samples")
        if not isinstance(samples, list) or not samples:
            continue
        target = coerce_optional_float(example.get("target_delta"))
        if target is None or target <= 0.0:
            continue
        features = build_solar_orientation_features(
            coerce_solar_sample(samples[0]), orientation
        )
        for idx, exposure in enumerate((features[1], features[2])):
            if exposure <= 0.0:
                continue
            numerators[idx] += target * exposure
            denominators[idx] += exposure * exposure
    gains = [
        (
            max(0.0, numerators[idx] / denominators[idx])
            if denominators[idx] > 0.0
            else 0.0
        )
        for idx in range(2)
    ]
    if sum(gains) <= 0.0:
        return adjusted
    if learning_model == LEARNING_MODEL_PHYSICAL:
        for facade_index, gain in enumerate(gains):
            for index in directional_solar_weight_indices(
                facade_index=facade_index,
                use_solar_feature=True,
                learning_model=learning_model,
                history_steps=history_steps,
                weight_count=adjusted.size,
            ):
                if index < adjusted.size:
                    adjusted[index] = gain * PHYSICAL_SOLAR_SCALE
        return adjusted
    solar_start = history_steps * 2
    for step in range(history_steps):
        row_start = solar_start + step * SOLAR_ORIENTATION_FEATURE_SIZE
        if row_start + 2 >= adjusted.size:
            break
        decay = 1.0 if step == 0 else 0.5**step
        adjusted[row_start + 1] = gains[0] * decay
        adjusted[row_start + 2] = gains[1] * decay
    return adjusted


def orientation_tie_score(
    *,
    weights: np.ndarray,
    examples: list[dict[str, Any]],
    orientation: float,
    use_solar_feature: bool,
    learning_model: str,
    history_steps: int,
) -> float:
    """Prefer tied orientations that use both facade channels positively."""
    if not use_solar_feature:
        return 0.0
    facade_weights = [0.0, 0.0]
    for facade_index in (0, 1):
        for index in directional_solar_weight_indices(
            facade_index=facade_index,
            use_solar_feature=use_solar_feature,
            learning_model=learning_model,
            history_steps=history_steps,
            weight_count=weights.size,
        ):
            if index < weights.size:
                value = float(weights[index])
                if np.isfinite(value) and value > 0.0:
                    facade_weights[facade_index] += value
    assigned_targets = [0.0, 0.0]
    for example in examples:
        samples = example.get("solar_samples")
        if not isinstance(samples, list) or not samples:
            continue
        features = build_solar_orientation_features(
            coerce_solar_sample(samples[0]), orientation
        )
        if features[1] <= 0.0 and features[2] <= 0.0:
            continue
        target = abs(float(example.get("target_delta") or 0.0))
        sharpness = max(features[1], features[2]) / max(features[0], 1e-9)
        if features[1] >= features[2]:
            assigned_targets[0] += target * sharpness
        else:
            assigned_targets[1] += target * sharpness
    return (
        min(assigned_targets)
        + 0.001 * sum(assigned_targets)
        + 0.000001 * min(facade_weights)
    )
