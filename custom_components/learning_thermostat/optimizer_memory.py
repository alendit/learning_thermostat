"""Mutable optimizer learning memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .optimizer_features import feature_size, init_weights, solar_profile_size
from .optimizer_state import (
    MODEL_SCHEMA_VERSION,
    adapt_weights,
    build_optimizer_snapshot,
    coerce_float_list,
    coerce_int_list,
    legacy_model_type,
    optimizer_model_id,
    optimizer_model_name,
    restore_k_step_stats,
    restore_orientation_state,
    restore_solar_sample_history,
    snapshot_model_id,
)
from .optimizer_tracking import clear_pending_predictions


@dataclass(slots=True)
class OptimizerLearningMemory:
    """Own learned histories, pending predictions, and last plan state."""

    t_hist: list[float] = field(default_factory=list)
    out_hist: list[float] = field(default_factory=list)
    solar_hist: list[float] = field(default_factory=list)
    solar_sample_hist: list[dict[str, float | None]] = field(default_factory=list)
    solar_orientation_azimuth: float | None = None
    solar_orientation_confidence: float | None = None
    solar_orientation_score: float | None = None
    solar_training_examples: list[dict[str, Any]] = field(default_factory=list)
    prev_orientation_input: dict[str, Any] | None = None
    u_hist: list[int] = field(default_factory=list)
    last_u: int | None = None
    prev_pred_history: list[float] = field(default_factory=list)
    prev_abs_error_history: list[float] = field(default_factory=list)
    prev_pred_timestamp_history: list[str] = field(default_factory=list)
    weight_abs_update_history: list[float] = field(default_factory=list)
    weight_delta_abs_latest: list[float] = field(default_factory=list)
    pending_k_step_preds: dict[int, dict[int, float]] = field(default_factory=dict)
    pending_k_step_features: dict[int, dict[int, list[float]]] = field(
        default_factory=dict
    )
    pending_k_step_baseline: dict[int, dict[int, float]] = field(default_factory=dict)
    k_step_abs_error_history: dict[int, dict[str, float | int | None]] = field(
        default_factory=dict
    )
    k_step_pred_improvement_history: dict[int, dict[str, float | int | None]] = field(
        default_factory=dict
    )
    last_dynamic_lookahead: int | None = None
    last_eta: float | None = None
    base_eta: float | None = None
    prev_feature: list[float] | None = None
    last_forecast_temp: float | None = None
    last_forecast_solar: float | None = None
    step_count: int = 0
    last_plan: dict[str, int | None] = field(
        default_factory=lambda: {"start_step": None, "stop_step": None}
    )

    def snapshot(
        self,
        *,
        weights: np.ndarray,
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
        """Serialize optimizer learning memory and model metadata."""
        return build_optimizer_snapshot(
            weights=weights,
            model_name=optimizer_model_name(learning_model),
            model_id=optimizer_model_id(learning_model, use_solar_feature),
            model_schema_version=MODEL_SCHEMA_VERSION,
            t_hist=self.t_hist,
            out_hist=self.out_hist,
            solar_hist=self.solar_hist,
            solar_sample_hist=self.solar_sample_hist,
            u_hist=self.u_hist,
            prev_pred_history=self.prev_pred_history,
            prev_abs_error_history=self.prev_abs_error_history,
            prev_pred_timestamp_history=self.prev_pred_timestamp_history,
            weight_abs_update_history=self.weight_abs_update_history,
            weight_delta_abs_latest=self.weight_delta_abs_latest,
            k_step_abs_error_history=self.k_step_abs_error_history,
            k_step_pred_improvement_history=self.k_step_pred_improvement_history,
            last_u=self.last_u,
            last_plan=self.last_plan,
            solar_orientation_azimuth=self.solar_orientation_azimuth,
            solar_orientation_confidence=self.solar_orientation_confidence,
            solar_orientation_score=self.solar_orientation_score,
            eta=self.last_eta,
            base_eta=self.base_eta,
            prev_feature=self.prev_feature,
            config=config,
            l2_reg=l2_reg,
            history_steps=history_steps,
            history_decay=history_decay,
            error_decay=error_decay,
            default_u_gain_per_hour=default_u_gain_per_hour,
            use_solar_feature=use_solar_feature,
            learning_model=learning_model,
            solar_orientation=solar_orientation,
        )

    def restore(
        self,
        state: dict[str, Any],
        *,
        current_weights: np.ndarray,
        weights_init: Any,
        learning_model: str,
        use_solar_feature: bool,
        history_steps: int,
    ) -> np.ndarray:
        """Restore memory and return the restored model weights."""
        restored_weights = current_weights
        model_type = legacy_model_type(state)
        restored_model_id = snapshot_model_id(state, use_solar_feature)
        model_matches = restored_model_id == optimizer_model_id(
            learning_model, use_solar_feature
        )
        weights = state.get("weights")
        if model_matches and weights is not None:
            restored = np.array([float(v) for v in weights], dtype=float)
            if restored.size == current_weights.size:
                if np.allclose(restored, 0.0) or not np.all(np.isfinite(restored)):
                    restored = init_weights(
                        weights_init,
                        learning_model=learning_model,
                        history_steps=history_steps,
                        use_solar_feature=use_solar_feature,
                    )
                restored_weights = restored
            elif _can_adapt_linear(learning_model):
                adapted = adapt_weights(
                    restored,
                    model_type,
                    history_steps=history_steps,
                    solar_profile_size=solar_profile_size(use_solar_feature),
                    feature_size=feature_size(
                        learning_model=learning_model,
                        history_steps=history_steps,
                        use_solar_feature=use_solar_feature,
                    ),
                )
                if adapted is not None:
                    restored_weights = adapted.weights
                    if adapted.solar_orientation_azimuth is not None:
                        self.solar_orientation_azimuth = (
                            adapted.solar_orientation_azimuth
                        )
        elif weights is not None and _can_adapt_linear(learning_model) and model_type:
            restored = np.array([float(v) for v in weights], dtype=float)
            adapted = adapt_weights(
                restored,
                model_type,
                history_steps=history_steps,
                solar_profile_size=solar_profile_size(use_solar_feature),
                feature_size=feature_size(
                    learning_model=learning_model,
                    history_steps=history_steps,
                    use_solar_feature=use_solar_feature,
                ),
            )
            if adapted is not None:
                restored_weights = adapted.weights
                if adapted.solar_orientation_azimuth is not None:
                    self.solar_orientation_azimuth = adapted.solar_orientation_azimuth
        elif weights is not None:
            restored_weights = init_weights(
                weights_init,
                learning_model=learning_model,
                history_steps=history_steps,
                use_solar_feature=use_solar_feature,
            )

        history = state.get("history") or {}
        self.t_hist = coerce_float_list(history.get("t_room"))[:history_steps]
        self.out_hist = coerce_float_list(history.get("t_out"))[:history_steps]
        self.solar_hist = coerce_float_list(history.get("solar"))[:history_steps]
        self.solar_sample_hist = restore_solar_sample_history(
            history,
            self.solar_hist,
            use_solar_feature=use_solar_feature,
            history_steps=history_steps,
        )
        self.u_hist = coerce_int_list(history.get("u"))[:history_steps]
        self.prev_pred_history = coerce_float_list(history.get("pred_next"))[
            :history_steps
        ]
        self.prev_abs_error_history = coerce_float_list(history.get("pred_abs_error"))[
            :history_steps
        ]
        pred_timestamps = history.get("pred_timestamps") or []
        self.prev_pred_timestamp_history = [str(value) for value in pred_timestamps][
            :history_steps
        ]
        self.weight_abs_update_history = coerce_float_list(
            history.get("weight_abs_update")
        )[:history_steps]
        self.weight_delta_abs_latest = coerce_float_list(
            history.get("weight_delta_abs_latest")
        )[: restored_weights.size]
        self.k_step_abs_error_history = restore_k_step_stats(
            history.get("k_step_abs_error")
        )
        self.k_step_pred_improvement_history = restore_k_step_stats(
            history.get("k_step_pred_improvement")
        )

        last_u = state.get("last_u")
        if last_u is not None:
            self.last_u = int(last_u)

        step_count = state.get("step_count")
        if step_count is not None:
            self.step_count = int(step_count)

        last_plan = state.get("last_plan") or {}
        self.last_plan = {
            "start_step": last_plan.get("start_step"),
            "stop_step": last_plan.get("stop_step"),
        }
        orientation = restore_orientation_state(state)
        if orientation.azimuth is not None:
            self.solar_orientation_azimuth = orientation.azimuth
        self.solar_orientation_confidence = orientation.confidence
        self.solar_orientation_score = orientation.score
        eta = state.get("eta")
        if eta is not None:
            self.last_eta = float(eta)
        base_eta = state.get("base_eta")
        if base_eta is not None:
            self.base_eta = float(base_eta)

        if model_matches:
            prev_feature = state.get("prev_feature") or []
            prev_vals = [float(v) for v in prev_feature]
            self.prev_feature = (
                prev_vals if len(prev_vals) == restored_weights.size else None
            )
        else:
            self.prev_feature = None
            self.clear_pending_predictions()
            self.prev_orientation_input = None
        return restored_weights

    def clear_pending_predictions(self) -> None:
        """Clear pending multi-step predictions after a model-layout change."""
        clear_pending_predictions(
            pending_preds=self.pending_k_step_preds,
            pending_features=self.pending_k_step_features,
            pending_baseline=self.pending_k_step_baseline,
        )


def _can_adapt_linear(learning_model: str) -> bool:
    """Return whether legacy weights may be adapted into this model."""
    return learning_model == "linear"
