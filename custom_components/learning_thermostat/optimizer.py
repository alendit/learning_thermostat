"""Learning optimizer for switch-based thermostats with lookahead planning.

Algorithm overview:
- Model: linear one-step delta predictor ΔT(t+1) = w * [T_hist, Tout_hist, U_hist, bias],
  then T(t+1) = T(t) + ΔT(t+1). Temperature inputs are z-score normalized
  absolute temps.
- History: uses the last N room temps, outdoor temps, and on/off states; values
  are padded and sanitized to keep the model stable with missing data.
- Learning: updates weights from the previous rollout using the immediate
  prediction error only, with optional L2 regularization.
- Planning: simulates a single contiguous ON interval over a lookahead window,
  greedily choosing start then stop to minimize squared error plus energy cost.
- Forecast: uses a weather forecast when available, otherwise repeats the last
  known or previous forecast temperature.
- Initialization: default weights predict no change and add a small positive
  heating gain so ON predicts slightly warmer than OFF.
"""

from __future__ import annotations

import logging
import time
from typing import Any
import numpy as np

from .const import LEARNING_MODEL_LINEAR, LEARNING_MODEL_PHYSICAL
from .learning_defaults import DEFAULT_LEARNING_SWITCH_CONFIG
from .optimizer_features import (
    PHYSICAL_HALF_LIFE_MINUTES,
    build_model_feature,
    default_weights,
    feature_size,
    heat_feature_count,
    init_weights,
    physical_filter_delta,
    solar_profile_size,
    u_weight_start,
)
from .optimizer_history import build_history_arrays, update_latest_first_histories
from .optimizer_learning import compute_multi_step_update, compute_single_step_update
from .optimizer_planning import (
    effective_solar_orientation,
    find_start_stop,
    normalize_forecast,
    normalize_solar_forecast,
    predict_next,
    rollout,
)
from .optimizer_orientation import (
    apply_empirical_solar_gains,
    clamp_directional_solar_weights,
    find_best_solar_orientation,
    solar_orientation_diagnostics,
)
from .optimizer_state import (
    MODEL_SCHEMA_VERSION,
    adapt_weights,
    build_optimizer_snapshot,
    coerce_float,
    coerce_float_list,
    coerce_int,
    coerce_int_list,
    coerce_optional_float,
    legacy_model_type,
    normalize_learning_model,
    optimizer_model_id,
    optimizer_model_name,
    restore_k_step_stats,
    restore_orientation_state,
    restore_solar_sample_history,
    snapshot_model_id,
)
from .optimizer_tracking import (
    DYNAMIC_LOOKAHEAD_TOLERANCE,
    clear_pending_predictions,
    k_step_error_stats,
    queue_k_step_baseline,
    queue_k_step_features,
    queue_k_step_predictions,
    record_k_step_errors,
    select_dynamic_lookahead,
)
from .optimizer_types import OptimizerTickResult, SolarOrientationDiagnostics
from .solar import (
    SOLAR_ORIENTATION_FEATURE_SIZE,
    coerce_solar_sample,
    diffuse_only_sample,
)

DYNAMIC_LOOKAHEAD_MIN = 12
DYNAMIC_LOOKAHEAD_MAX = 24
LEARNING_RATE_TARGET_ERROR = 0.2
LEARNING_RATE_SCALE_MIN = 0.2
LEARNING_RATE_SCALE_MAX = 2.0
U_WEIGHT_MIN = 1e-6
SOLAR_ORIENTATION_GRID_DEGREES = 5
SOLAR_ORIENTATION_MIN_EXAMPLES = 8
SOLAR_ORIENTATION_MIN_ACTIVE_EXAMPLES = 4
SOLAR_ORIENTATION_TRAINING_MAX = 96
SOLAR_ORIENTATION_IMPROVEMENT_MIN = 0.02

_LOGGER = logging.getLogger(__name__)


class LearningSwitchOptimizer:
    """Online learner that plans a single on/off interval with lookahead."""

    def __init__(self, name: str, config: dict | None = None):
        """Initialize config, weights, and history buffers."""
        # Used for logging/identification.
        self._name = name
        # Working config snapshot.
        self._config = dict(DEFAULT_LEARNING_SWITCH_CONFIG)
        if config:
            self._config.update({k: v for k, v in config.items() if v is not None})

        # Length of history window.
        self._history_steps = coerce_int(
            self._config.get("history_steps"),
            DEFAULT_LEARNING_SWITCH_CONFIG["history_steps"],
        )
        # History discounting is disabled (no decay).
        self._history_decay = 1.0
        self._history_decay_vector = np.ones(self._history_steps, dtype=float)
        # Decay applied to future prediction errors in planning (0 <= decay <= 1).
        self._error_decay = coerce_float(
            self._config.get("error_decay"),
            DEFAULT_LEARNING_SWITCH_CONFIG["error_decay"],
        )
        if not (0.0 <= self._error_decay <= 1.0):
            self._error_decay = DEFAULT_LEARNING_SWITCH_CONFIG["error_decay"]
        # L2 regularization coefficient for weight decay (>= 0).
        self._l2_reg = coerce_float(
            self._config.get("l2_reg"), DEFAULT_LEARNING_SWITCH_CONFIG["l2_reg"]
        )
        if not np.isfinite(self._l2_reg) or self._l2_reg < 0.0:
            self._l2_reg = DEFAULT_LEARNING_SWITCH_CONFIG["l2_reg"]
        # Default heating effect when weights are fresh.
        self._default_u_gain_per_hour = coerce_float(
            self._config.get("default_u_gain_per_hour"),
            DEFAULT_LEARNING_SWITCH_CONFIG["default_u_gain_per_hour"],
        )
        self._learning_model = normalize_learning_model(
            self._config.get("learning_model")
        )
        self._model_schema_version = MODEL_SCHEMA_VERSION
        self._feature_step_minutes = 60.0

        # Optional starting weights.
        self._weights_init = self._config.get("weights_init")
        # Whether the model includes a solar irradiance feature channel.
        self._use_solar_feature = bool(self._config.get("use_solar_feature"))
        # Current model weights, shape: [t_hist, out_hist, (solar_orientation_hist), u_hist, bias].
        self._weights = init_weights(
            self._weights_init,
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
        )
        # Recent room temperatures (latest first).
        self._t_hist: list[float] = []
        # Recent outdoor temperatures (latest first).
        self._out_hist: list[float] = []
        # Recent solar irradiance values (latest first).
        self._solar_hist: list[float] = []
        # Recent raw solar samples (latest first).
        self._solar_sample_hist: list[dict[str, float | None]] = []
        # Learned dominant facade orientation in HA azimuth degrees.
        self._solar_orientation_azimuth: float | None = None
        # Latest orientation evidence/confidence.
        self._solar_orientation_confidence: float | None = None
        # Latest orientation fit score; lower is better.
        self._solar_orientation_score: float | None = None
        # Rolling one-step examples for orientation grid search.
        self._solar_training_examples: list[dict[str, Any]] = []
        # Raw previous one-step input waiting for the next actual temperature.
        self._prev_orientation_input: dict[str, Any] | None = None
        # Recent on/off states (latest first).
        self._u_hist: list[int] = []
        # Last planned on/off state.
        self._last_u: int | None = None
        # History of one-step predictions for debugging.
        self._prev_pred_history: list[float] = []
        # History of absolute one-step prediction errors for debugging.
        self._prev_abs_error_history: list[float] = []
        # History of timestamps when prediction/error entries were added.
        self._prev_pred_timestamp_history: list[str] = []
        # History of absolute weight update magnitudes (max |delta_w| per tick).
        self._weight_abs_update_history: list[float] = []
        # Latest per-weight absolute delta vector for debugging.
        self._weight_delta_abs_latest: list[float] = []
        # Pending k-step predictions keyed by target step index.
        self._pending_k_step_preds: dict[int, dict[int, float]] = {}
        # Pending k-step feature vectors keyed by target step index.
        self._pending_k_step_features: dict[int, dict[int, list[float]]] = {}
        # Pending trivial baseline predictions keyed by target step index.
        self._pending_k_step_baseline: dict[int, dict[int, float]] = {}
        # Rolling k-step prediction error stats keyed by horizon.
        self._k_step_abs_error_history: dict[int, dict[str, float | int | None]] = {}
        # Rolling k-step prediction improvement stats keyed by horizon.
        self._k_step_pred_improvement_history: dict[
            int, dict[str, float | int | None]
        ] = {}
        # Last dynamic lookahead decision.
        self._last_dynamic_lookahead: int | None = None
        # Latest effective learning rate (eta) and base learning rate.
        self._last_eta: float | None = None
        self._base_eta: float | None = None
        # One-step feature from last tick for learning.
        self._prev_feature: list[float] | None = None
        # Sticky fallback forecast value.
        self._last_forecast_temp: float | None = None
        # Sticky fallback solar forecast value.
        self._last_forecast_solar: float | None = None
        # Number of ticks processed.
        self._step_count = 0
        # Last plan interval.
        self._last_plan: dict[str, int | None] = {"start_step": None, "stop_step": None}
        self._warned_invalid_weights = False

    def snapshot_state(self) -> dict:
        """Serialize weights and history for persistence/debugging."""
        return build_optimizer_snapshot(
            weights=self._weights,
            model_name=optimizer_model_name(self._learning_model),
            model_id=optimizer_model_id(self._learning_model, self._use_solar_feature),
            model_schema_version=self._model_schema_version,
            t_hist=self._t_hist,
            out_hist=self._out_hist,
            solar_hist=self._solar_hist,
            solar_sample_hist=self._solar_sample_hist,
            u_hist=self._u_hist,
            prev_pred_history=self._prev_pred_history,
            prev_abs_error_history=self._prev_abs_error_history,
            prev_pred_timestamp_history=self._prev_pred_timestamp_history,
            weight_abs_update_history=self._weight_abs_update_history,
            weight_delta_abs_latest=self._weight_delta_abs_latest,
            k_step_abs_error_history=self._k_step_abs_error_history,
            k_step_pred_improvement_history=self._k_step_pred_improvement_history,
            last_u=self._last_u,
            last_plan=self._last_plan,
            solar_orientation_azimuth=self._solar_orientation_azimuth,
            solar_orientation_confidence=self._solar_orientation_confidence,
            solar_orientation_score=self._solar_orientation_score,
            eta=self._last_eta,
            base_eta=self._base_eta,
            prev_feature=self._prev_feature,
            config=self._config,
            l2_reg=self._l2_reg,
            history_steps=self._history_steps,
            history_decay=self._history_decay,
            error_decay=self._error_decay,
            default_u_gain_per_hour=self._default_u_gain_per_hour,
            use_solar_feature=self._use_solar_feature,
            learning_model=self._learning_model,
            solar_orientation=self.solar_orientation_diagnostics(),
        )

    def restore_state(self, state: dict | None) -> None:
        """Restore the internal state defensively from a snapshot."""
        try:
            if not isinstance(state, dict):
                return

            model_type = legacy_model_type(state)
            restored_model_id = snapshot_model_id(state, self._use_solar_feature)
            model_matches = restored_model_id == optimizer_model_id(
                self._learning_model, self._use_solar_feature
            )
            weights = state.get("weights")
            if model_matches and weights is not None:
                restored = np.array([float(v) for v in weights], dtype=float)
                if restored.size == self._weights.size:
                    if np.allclose(restored, 0.0) or not np.all(np.isfinite(restored)):
                        restored = init_weights(
                            self._weights_init,
                            learning_model=self._learning_model,
                            history_steps=self._history_steps,
                            use_solar_feature=self._use_solar_feature,
                        )
                    self._weights = restored
                elif self._learning_model == LEARNING_MODEL_LINEAR:
                    adapted = adapt_weights(
                        restored,
                        model_type,
                        history_steps=self._history_steps,
                        solar_profile_size=solar_profile_size(self._use_solar_feature),
                        feature_size=feature_size(
                            learning_model=self._learning_model,
                            history_steps=self._history_steps,
                            use_solar_feature=self._use_solar_feature,
                        ),
                    )
                    if adapted is not None:
                        self._weights = adapted.weights
                        if adapted.solar_orientation_azimuth is not None:
                            self._solar_orientation_azimuth = (
                                adapted.solar_orientation_azimuth
                            )
            elif (
                weights is not None
                and self._learning_model == LEARNING_MODEL_LINEAR
                and model_type is not None
            ):
                restored = np.array([float(v) for v in weights], dtype=float)
                adapted = adapt_weights(
                    restored,
                    model_type,
                    history_steps=self._history_steps,
                    solar_profile_size=solar_profile_size(self._use_solar_feature),
                    feature_size=feature_size(
                        learning_model=self._learning_model,
                        history_steps=self._history_steps,
                        use_solar_feature=self._use_solar_feature,
                    ),
                )
                if adapted is not None:
                    self._weights = adapted.weights
                    if adapted.solar_orientation_azimuth is not None:
                        self._solar_orientation_azimuth = (
                            adapted.solar_orientation_azimuth
                        )
            elif weights is not None:
                self._weights = init_weights(
                    self._weights_init,
                    learning_model=self._learning_model,
                    history_steps=self._history_steps,
                    use_solar_feature=self._use_solar_feature,
                )

            history = state.get("history") or {}
            self._t_hist = coerce_float_list(history.get("t_room"))[
                : self._history_steps
            ]
            self._out_hist = coerce_float_list(history.get("t_out"))[
                : self._history_steps
            ]
            self._solar_hist = coerce_float_list(history.get("solar"))[
                : self._history_steps
            ]
            self._solar_sample_hist = restore_solar_sample_history(
                history,
                self._solar_hist,
                use_solar_feature=self._use_solar_feature,
                history_steps=self._history_steps,
            )
            self._u_hist = coerce_int_list(history.get("u"))[: self._history_steps]
            self._prev_pred_history = coerce_float_list(history.get("pred_next"))[
                : self._history_steps
            ]
            self._prev_abs_error_history = coerce_float_list(
                history.get("pred_abs_error")
            )[: self._history_steps]
            pred_timestamps = history.get("pred_timestamps") or []
            self._prev_pred_timestamp_history = [
                str(value) for value in pred_timestamps
            ][: self._history_steps]
            self._weight_abs_update_history = coerce_float_list(
                history.get("weight_abs_update")
            )[: self._history_steps]
            self._weight_delta_abs_latest = coerce_float_list(
                history.get("weight_delta_abs_latest")
            )[: self._weights.size]
            self._k_step_abs_error_history = restore_k_step_stats(
                history.get("k_step_abs_error")
            )
            self._k_step_pred_improvement_history = restore_k_step_stats(
                history.get("k_step_pred_improvement")
            )

            last_u = state.get("last_u")
            if last_u is not None:
                self._last_u = int(last_u)

            step_count = state.get("step_count")
            if step_count is not None:
                self._step_count = int(step_count)

            last_plan = state.get("last_plan") or {}
            self._last_plan = {
                "start_step": last_plan.get("start_step"),
                "stop_step": last_plan.get("stop_step"),
            }
            orientation = restore_orientation_state(state)
            if orientation.azimuth is not None:
                self._solar_orientation_azimuth = orientation.azimuth
            self._solar_orientation_confidence = orientation.confidence
            self._solar_orientation_score = orientation.score
            eta = state.get("eta")
            if eta is not None:
                self._last_eta = float(eta)
            base_eta = state.get("base_eta")
            if base_eta is not None:
                self._base_eta = float(base_eta)

            if model_matches:
                prev_feature = state.get("prev_feature") or []
                prev_vals = [float(v) for v in prev_feature]
                self._prev_feature = (
                    prev_vals if len(prev_vals) == self._weights.size else None
                )
            else:
                self._prev_feature = None
                clear_pending_predictions(
                    pending_preds=self._pending_k_step_preds,
                    pending_features=self._pending_k_step_features,
                    pending_baseline=self._pending_k_step_baseline,
                )
                self._prev_orientation_input = None
        except Exception as exc:  # noqa: BLE001 - keep restore resilient
            _LOGGER.warning("%s - Learning state restore failed: %s", self._name, exc)

    def _ensure_weights_finite(self) -> bool:
        """Reset weights if any non-finite values appear."""
        if np.all(np.isfinite(self._weights)):
            self._warned_invalid_weights = False
            return False
        self._weights = init_weights(
            self._weights_init,
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
        )
        if not self._warned_invalid_weights:
            _LOGGER.warning(
                "%s - Reset learning weights due to non-finite values", self._name
            )
            self._warned_invalid_weights = True
        return True

    def _apply_default_u_gain(self, step_minutes: int | None) -> bool:
        """Seed a small heating gain so ON predicts a higher next temperature."""
        baseline_weights = default_weights(
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
        )
        if not np.allclose(self._weights, baseline_weights, rtol=0.0, atol=1e-9):
            return False
        try:
            minutes = float(step_minutes) if step_minutes is not None else 60.0
        except (TypeError, ValueError):
            minutes = 60.0
        if minutes <= 0:
            minutes = 60.0
        delta = self._default_u_gain_per_hour * (minutes / 60.0)
        if not np.isfinite(delta):
            return False
        u_index = u_weight_start(
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
        )
        if u_index < self._weights.size:
            if self._learning_model == LEARNING_MODEL_PHYSICAL:
                feature_delta = physical_filter_delta(
                    PHYSICAL_HALF_LIFE_MINUTES[0], minutes
                )
                if feature_delta <= 0.0:
                    return False
                self._weights[u_index] = float(delta / feature_delta)
            else:
                self._weights[u_index] = float(delta)
            return True
        return False

    def _cap_u_weights(self) -> bool:
        """Clamp on/off weights to a small positive minimum."""
        start = u_weight_start(
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
        )
        end = start + heat_feature_count(
            learning_model=self._learning_model,
            history_steps=self._history_steps,
        )
        if start >= self._weights.size:
            return False
        floor = U_WEIGHT_MIN
        if not np.isfinite(floor) or floor <= 0.0:
            return False
        before = self._weights[start:end].copy()
        self._weights[start:end] = np.maximum(self._weights[start:end], floor)
        return not np.allclose(before, self._weights[start:end], rtol=0.0, atol=1e-12)

    def weight_contributions(self, u_value: int | None = None) -> np.ndarray:
        """Return absolute per-weight contributions for the next-step prediction."""
        t_hist, out_hist, solar_hist, u_hist = build_history_arrays(
            t_hist=self._t_hist,
            out_hist=self._out_hist,
            solar_hist=self._solar_hist,
            solar_sample_hist=self._solar_sample_hist,
            u_hist=self._u_hist,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
            solar_orientation=self._effective_solar_orientation(),
        )
        if u_value is None:
            if self._u_hist:
                u_value = int(self._u_hist[0])
            elif self._last_u is not None:
                u_value = int(self._last_u)
            else:
                u_value = 0
        else:
            u_value = int(u_value)
        u_hist_step = np.concatenate(
            (np.array([float(u_value)], dtype=float), u_hist[:-1])
        )
        solar_hist_step: np.ndarray | None = None
        if solar_hist is not None:
            solar_hist_step = np.concatenate(
                (
                    solar_hist[0].reshape(1, SOLAR_ORIENTATION_FEATURE_SIZE),
                    solar_hist[:-1],
                )
            )
        x = build_model_feature(
            learning_model=self._learning_model,
            history_decay_vector=self._history_decay_vector,
            feature_step_minutes=self._feature_step_minutes,
            t_hist=t_hist,
            out_hist=out_hist,
            solar_hist=solar_hist_step,
            u_hist=u_hist_step,
        )
        if x.size != self._weights.size:
            return np.zeros_like(self._weights)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(self._weights)):
            return np.zeros_like(self._weights)
        return np.abs(x * self._weights)

    def solar_orientation_diagnostics(self) -> SolarOrientationDiagnostics:
        """Return the dominant learned solar exposure direction."""
        return solar_orientation_diagnostics(
            use_solar_feature=self._use_solar_feature,
            weights=self._weights,
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            solar_orientation_azimuth=self._solar_orientation_azimuth,
            solar_orientation_confidence=self._solar_orientation_confidence,
        )

    def _effective_solar_orientation(self) -> float:
        """Return the active orientation angle for feature construction."""
        return effective_solar_orientation(self._solar_orientation_azimuth)

    def _record_orientation_example(self, actual_temp: float) -> None:
        """Record a matured one-step training example for orientation search."""
        if not self._use_solar_feature or self._prev_orientation_input is None:
            return
        previous = self._prev_orientation_input
        base_temp = coerce_optional_float(previous.get("base_temp"))
        if base_temp is None:
            self._prev_orientation_input = None
            return
        target_delta = float(actual_temp - base_temp)
        if not np.isfinite(target_delta):
            self._prev_orientation_input = None
            return
        example = dict(previous)
        example["target_delta"] = target_delta
        self._solar_training_examples.insert(0, example)
        if len(self._solar_training_examples) > SOLAR_ORIENTATION_TRAINING_MAX:
            self._solar_training_examples = self._solar_training_examples[
                :SOLAR_ORIENTATION_TRAINING_MAX
            ]
        self._prev_orientation_input = None
        self._maybe_update_solar_orientation()

    def _store_orientation_input(
        self,
        *,
        base_temp: float,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_sample: dict[str, float | None] | None,
        u_hist: np.ndarray,
        u_value: int,
    ) -> None:
        """Store raw one-step inputs until the next actual temperature arrives."""
        if not self._use_solar_feature:
            self._prev_orientation_input = None
            return
        sample_rows = [solar_sample or diffuse_only_sample(0.0).as_dict()]
        sample_rows.extend(self._solar_sample_hist[: max(0, self._history_steps - 1)])
        if len(sample_rows) < self._history_steps:
            fallback = (
                sample_rows[-1] if sample_rows else diffuse_only_sample(0.0).as_dict()
            )
            sample_rows.extend(
                [dict(fallback)] * (self._history_steps - len(sample_rows))
            )
        u_hist_step = np.concatenate(
            (np.array([float(u_value)], dtype=float), u_hist[:-1])
        )
        self._prev_orientation_input = {
            "base_temp": float(base_temp),
            "t_hist": t_hist.tolist(),
            "out_hist": out_hist.tolist(),
            "solar_samples": [dict(row) for row in sample_rows[: self._history_steps]],
            "u_hist": u_hist_step.tolist(),
        }

    def _maybe_update_solar_orientation(self) -> None:
        """Grid-search the compact solar orientation when enough evidence exists."""
        examples = list(self._solar_training_examples)
        best = find_best_solar_orientation(
            examples=examples,
            current_orientation=self._solar_orientation_azimuth,
            current_weights=self._weights,
            l2_reg=self._l2_reg,
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            history_decay_vector=self._history_decay_vector,
            feature_step_minutes=self._feature_step_minutes,
            grid_degrees=SOLAR_ORIENTATION_GRID_DEGREES,
            min_examples=SOLAR_ORIENTATION_MIN_EXAMPLES,
            min_active_examples=SOLAR_ORIENTATION_MIN_ACTIVE_EXAMPLES,
            improvement_min=SOLAR_ORIENTATION_IMPROVEMENT_MIN,
        )
        if best is None:
            return

        changed = (
            self._solar_orientation_azimuth is None
            or abs(
                ((self._solar_orientation_azimuth - best.orientation + 180.0) % 360.0)
                - 180.0
            )
            > 1e-9
        )
        self._solar_orientation_azimuth = best.orientation
        self._solar_orientation_score = best.score
        if np.isfinite(best.baseline_score) and best.baseline_score > 0.0:
            self._solar_orientation_confidence = max(
                0.0, min(1.0, 1.0 - best.score / best.baseline_score)
            )
        else:
            self._solar_orientation_confidence = None
        best_weights = apply_empirical_solar_gains(
            best.weights,
            examples=examples,
            orientation=best.orientation,
            learning_model=self._learning_model,
            history_steps=self._history_steps,
        )
        self._weights = clamp_directional_solar_weights(
            best_weights,
            use_solar_feature=self._use_solar_feature,
            learning_model=self._learning_model,
            history_steps=self._history_steps,
        )
        self._ensure_weights_finite()
        self._cap_u_weights()
        if changed:
            clear_pending_predictions(
                pending_preds=self._pending_k_step_preds,
                pending_features=self._pending_k_step_features,
                pending_baseline=self._pending_k_step_baseline,
            )
            self._prev_feature = None
            self._prev_orientation_input = None

    def tick(
        self,
        t_room: float,
        t_out: float,
        t_set: float,
        t_out_forecast: list[float] | None,
        solar: float | None = None,
        solar_forecast: list[float] | None = None,
        solar_sample: dict[str, Any] | None = None,
        solar_sample_forecast: list[dict[str, Any]] | None = None,
        solar_profile: list[float] | tuple[float, ...] | None = None,
        solar_profile_forecast: list[list[float]] | None = None,
        current_u: int | None = None,
        step_minutes: int | None = None,
        timestamp: float | None = None,
    ) -> OptimizerTickResult:
        """Advance one tick: update history, learn weights, plan the next action."""
        tick_start = time.perf_counter()
        try:
            t_room = float(t_room)
            t_out = float(t_out)
            t_set = float(t_set)
        except (TypeError, ValueError):
            return {}
        if not np.isfinite(t_room) or not np.isfinite(t_out) or not np.isfinite(t_set):
            return {}

        if isinstance(current_u, (int, float)):
            u_actual = int(current_u)
        elif self._last_u is not None:
            u_actual = int(self._last_u)
        else:
            u_actual = 0
        try:
            self._feature_step_minutes = float(step_minutes or 60.0)
        except (TypeError, ValueError):
            self._feature_step_minutes = 60.0
        if (
            not np.isfinite(self._feature_step_minutes)
            or self._feature_step_minutes <= 0
        ):
            self._feature_step_minutes = 60.0

        weights_reset = self._ensure_weights_finite()
        if self._apply_default_u_gain(step_minutes):
            weights_reset = True
        self._cap_u_weights()

        step_index = self._step_count
        k_step_errors, k_step_updates = record_k_step_errors(
            actual_temp=t_room,
            step_index=step_index,
            pending_preds=self._pending_k_step_preds,
            pending_features=self._pending_k_step_features,
            pending_baseline=self._pending_k_step_baseline,
            abs_error_history=self._k_step_abs_error_history,
            pred_improvement_history=self._k_step_pred_improvement_history,
        )
        self._record_orientation_example(t_room)

        solar_current: float | None = None
        if solar is not None:
            try:
                solar_current = float(solar)
            except (TypeError, ValueError):
                solar_current = None
            if solar_current is not None and not np.isfinite(solar_current):
                solar_current = None

        current_solar_sample = None
        if self._use_solar_feature:
            current_solar_sample = coerce_solar_sample(
                solar_sample if solar_sample is not None else solar_profile,
                solar_current,
            ).as_dict()
        (
            self._t_hist,
            self._out_hist,
            self._solar_hist,
            self._solar_sample_hist,
            self._u_hist,
        ) = update_latest_first_histories(
            t_hist=self._t_hist,
            out_hist=self._out_hist,
            solar_hist=self._solar_hist,
            solar_sample_hist=self._solar_sample_hist,
            u_hist=self._u_hist,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
            t_room=t_room,
            t_out=t_out,
            u_actual=u_actual,
            solar=solar_current,
            solar_sample=current_solar_sample,
        )
        t_hist, out_hist, solar_hist, u_hist = build_history_arrays(
            t_hist=self._t_hist,
            out_hist=self._out_hist,
            solar_hist=self._solar_hist,
            solar_sample_hist=self._solar_sample_hist,
            u_hist=self._u_hist,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
            solar_orientation=self._effective_solar_orientation(),
        )

        configured_steps = coerce_int(
            self._config.get("lookahead_steps"),
            DEFAULT_LEARNING_SWITCH_CONFIG["lookahead_steps"],
        )
        max_steps = min(
            DYNAMIC_LOOKAHEAD_MAX, max(DYNAMIC_LOOKAHEAD_MIN, configured_steps)
        )
        min_steps = min(DYNAMIC_LOOKAHEAD_MIN, max_steps)
        k_step_stats = k_step_error_stats(self._k_step_abs_error_history, max_steps)
        steps, decision_reason = select_dynamic_lookahead(
            stats=k_step_stats,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        mean_summary = {
            horizon: (
                round(values["mean"], 3) if values.get("mean") is not None else None
            )
            for horizon, values in k_step_stats.items()
        }
        missing = [
            horizon for horizon, values in k_step_stats.items() if values["count"] == 0
        ]
        _LOGGER.info(
            "%s - Dynamic lookahead=%s (min=%s max=%s tol=%s reason=%s) k_step_mae=%s missing=%s",
            self._name,
            steps,
            min_steps,
            max_steps,
            DYNAMIC_LOOKAHEAD_TOLERANCE,
            decision_reason,
            mean_summary,
            missing,
        )
        self._last_dynamic_lookahead = steps

        t_outs_max, self._last_forecast_temp = normalize_forecast(
            t_out,
            t_out_forecast,
            max_steps,
            self._last_forecast_temp,
        )
        t_outs = t_outs_max[:steps]
        solar_outs_max, self._last_forecast_solar = normalize_solar_forecast(
            use_solar_feature=self._use_solar_feature,
            solar=solar_current,
            solar_forecast=solar_forecast,
            solar_sample=current_solar_sample,
            solar_sample_forecast=(
                solar_sample_forecast
                if solar_sample_forecast is not None
                else [
                    coerce_solar_sample(value).as_dict()
                    for value in solar_profile_forecast or []
                ]
            ),
            steps=max_steps,
            last_forecast_solar=self._last_forecast_solar,
        )
        solar_outs = solar_outs_max[:steps]
        solar_outs_raw = [float(value.get("flux") or 0.0) for value in solar_outs]

        prev_pred = self._prev_pred_history[0] if self._prev_pred_history else None
        abs_error = None
        if prev_pred is not None and np.isfinite(prev_pred):
            abs_error = abs(t_room - prev_pred)
            if np.isfinite(abs_error):
                self._prev_abs_error_history.insert(0, float(abs_error))
                if len(self._prev_abs_error_history) > self._history_steps:
                    self._prev_abs_error_history = self._prev_abs_error_history[
                        : self._history_steps
                    ]
                if timestamp is not None:
                    entry_ts = time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)
                    )
                else:
                    entry_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                self._prev_pred_timestamp_history.insert(0, entry_ts)
                if len(self._prev_pred_timestamp_history) > self._history_steps:
                    self._prev_pred_timestamp_history = (
                        self._prev_pred_timestamp_history[: self._history_steps]
                    )

        base_eta = coerce_float(
            self._config.get("learning_rate"),
            DEFAULT_LEARNING_SWITCH_CONFIG["learning_rate"],
        )
        if not np.isfinite(base_eta) or base_eta < 0.0:
            base_eta = DEFAULT_LEARNING_SWITCH_CONFIG["learning_rate"]
        eta = base_eta
        mean_error = k_step_stats.get(1, {}).get("mean")
        if mean_error is None or not np.isfinite(mean_error):
            mean_error = (
                abs_error if abs_error is not None and np.isfinite(abs_error) else None
            )
        if mean_error is not None:
            target = LEARNING_RATE_TARGET_ERROR
            if not np.isfinite(target) or target <= 0.0:
                target = DYNAMIC_LOOKAHEAD_TOLERANCE
            if np.isfinite(target) and target > 0.0:
                scale = mean_error / target
                if np.isfinite(scale):
                    scale = min(
                        LEARNING_RATE_SCALE_MAX, max(LEARNING_RATE_SCALE_MIN, scale)
                    )
                    eta = base_eta * scale
        self._base_eta = base_eta
        self._last_eta = eta

        learning_update = compute_multi_step_update(
            weights=self._weights,
            l2_reg=self._l2_reg,
            updates=k_step_updates,
            base_eta=base_eta,
            eta=eta,
        )
        if learning_update is None:
            learning_update = compute_single_step_update(
                weights=self._weights,
                l2_reg=self._l2_reg,
                actual_temp=t_room,
                prev_pred=prev_pred,
                prev_feature=self._prev_feature,
                base_eta=base_eta,
                eta=eta,
                fallback_learning_rate=DEFAULT_LEARNING_SWITCH_CONFIG["learning_rate"],
            )
        update = learning_update.payload if learning_update is not None else None
        if learning_update is not None:
            self._weights = learning_update.weights
        if self._ensure_weights_finite():
            weights_reset = True
        self._cap_u_weights()
        abs_update = 0.0
        if isinstance(update, dict):
            abs_update = update.get("abs_update_max")
        if abs_update is None or not np.isfinite(abs_update):
            abs_update = 0.0
        self._weight_abs_update_history.insert(0, float(abs_update))
        if len(self._weight_abs_update_history) > self._history_steps:
            self._weight_abs_update_history = self._weight_abs_update_history[
                : self._history_steps
            ]
        if isinstance(update, dict) and isinstance(update.get("delta"), list):
            self._weight_delta_abs_latest = [
                abs(value) for value in update.get("delta")
            ]
        else:
            self._weight_delta_abs_latest = []
        if update is None:
            _LOGGER.warning(
                "%s - Learning update skipped (no valid errors/features)",
                self._name,
            )
        else:
            _LOGGER.warning(
                "%s - Learning update: error=%s steps=%s grad_norm=%s eta=%s base_eta=%s l2_reg=%s abs_update_max=%s",
                self._name,
                update.get("error"),
                update.get("steps_used"),
                update.get("grad_norm"),
                update.get("eta"),
                update.get("base_eta"),
                update.get("l2_reg"),
                update.get("abs_update_max"),
            )
            if update.get("error") is not None:
                eta = update.get("eta") or 0.0
                base_eta = update.get("base_eta") or 0.0
                abs_update = update.get("abs_update_max") or 0.0
                _LOGGER.warning(
                    "%s - Learning update scale: error=%s eta=%s base_eta=%s l2_reg=%s abs_update_max=%s",
                    self._name,
                    update.get("error"),
                    eta,
                    base_eta,
                    update.get("l2_reg"),
                    abs_update,
                )

        current_solar_for_prediction = solar_outs[0] if solar_outs else solar_current
        t_pred_off, _ = predict_next(
            learning_model=self._learning_model,
            history_decay_vector=self._history_decay_vector,
            feature_step_minutes=self._feature_step_minutes,
            weights=self._weights,
            solar_orientation_azimuth=self._solar_orientation_azimuth,
            t_hist=t_hist,
            out_hist=out_hist,
            solar_hist=solar_hist,
            u_hist=u_hist,
            u_value=0,
            solar_value=current_solar_for_prediction,
        )
        t_pred_on, _ = predict_next(
            learning_model=self._learning_model,
            history_decay_vector=self._history_decay_vector,
            feature_step_minutes=self._feature_step_minutes,
            weights=self._weights,
            solar_orientation_azimuth=self._solar_orientation_azimuth,
            t_hist=t_hist,
            out_hist=out_hist,
            solar_hist=solar_hist,
            u_hist=u_hist,
            u_value=1,
            solar_value=current_solar_for_prediction,
        )
        t_pred_actual = t_pred_on if u_actual == 1 else t_pred_off

        planning_args = {
            "learning_model": self._learning_model,
            "history_decay_vector": self._history_decay_vector,
            "feature_step_minutes": self._feature_step_minutes,
            "weights": self._weights,
            "solar_orientation_azimuth": self._solar_orientation_azimuth,
            "t_hist": t_hist,
            "out_hist": out_hist,
            "solar_hist": solar_hist,
            "u_hist": u_hist,
        }
        start_step, stop_step, plan_cost = find_start_stop(
            **planning_args,
            error_decay=self._error_decay,
            w_energy=self._config.get("w_energy", 0.0),
            t_set=t_set,
            t_outs=t_outs,
            solar_forecast=solar_outs,
        )
        u_planned = 1 if start_step == 0 and stop_step > 0 else 0
        planned_states = [
            1 if start_step <= idx < stop_step else 0 for idx in range(len(t_outs))
        ]

        rollout_steps = max(1, len(t_outs))
        preds_next, feats_next = rollout(
            **planning_args,
            t_outs=t_outs,
            solar_forecast=solar_outs,
            start=start_step,
            stop=stop_step,
            steps=rollout_steps,
            capture_features=True,
        )
        preds_for_error = preds_next
        if max_steps != rollout_steps:
            preds_for_error, _ = rollout(
                **planning_args,
                t_outs=t_outs_max,
                solar_forecast=solar_outs_max,
                start=start_step,
                stop=stop_step,
                steps=max_steps,
                capture_features=False,
            )
        if u_planned == 1:
            _LOGGER.warning(
                "%s - Planned heating ON: cost=%s schedule=(start=%s stop=%s) "
                "planned_states=%s predicted_temps=%s",
                self._name,
                plan_cost,
                start_step,
                stop_step,
                planned_states,
                preds_next,
            )
        queue_k_step_predictions(
            pending_preds=self._pending_k_step_preds,
            preds=preds_for_error,
            step_index=step_index,
            max_steps=max_steps,
        )
        queue_k_step_features(
            pending_features=self._pending_k_step_features,
            features=feats_next,
            step_index=step_index,
            max_steps=max_steps,
        )
        queue_k_step_baseline(
            pending_baseline=self._pending_k_step_baseline,
            baseline_temp=t_room,
            step_index=step_index,
            max_steps=max_steps,
        )
        if preds_next:
            self._prev_pred_history.insert(0, float(preds_next[0]))
            if len(self._prev_pred_history) > self._history_steps:
                self._prev_pred_history = self._prev_pred_history[: self._history_steps]
        self._prev_feature = feats_next[0].tolist() if feats_next else None
        self._store_orientation_input(
            base_temp=float(t_hist[0]),
            t_hist=t_hist,
            out_hist=out_hist,
            solar_sample=current_solar_for_prediction,
            u_hist=u_hist,
            u_value=u_planned,
        )

        self._last_u = u_planned
        self._step_count += 1
        self._last_plan = {"start_step": start_step, "stop_step": stop_step}

        theta_outside = None
        theta_inside = None
        u_group_index = u_weight_start(
            learning_model=self._learning_model,
            history_steps=self._history_steps,
            use_solar_feature=self._use_solar_feature,
        )
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            if self._weights.size >= u_group_index + 3:
                theta_outside = float(self._weights[1])
                theta_inside = float(self._weights[u_group_index])
        elif self._weights.size >= u_group_index + self._history_steps:
            theta_outside = float(self._weights[self._history_steps])
            theta_inside = float(self._weights[u_group_index])

        error = update.get("error") if isinstance(update, dict) else None

        tick_duration_ms = (time.perf_counter() - tick_start) * 1000.0

        k_step_means = {
            horizon: values.get("mean") for horizon, values in k_step_stats.items()
        }
        k_step_improvement = {}
        for horizon in range(1, max_steps + 1):
            values = self._k_step_pred_improvement_history.get(horizon)
            mean_val = values.get("mean") if isinstance(values, dict) else None
            k_step_improvement[horizon] = mean_val

        return {
            "t_room": t_room,
            "t_out": t_out,
            "t_set": t_set,
            "theta1": theta_outside,
            "theta2": theta_inside,
            "weights": self._weights.tolist(),
            "history_steps": self._history_steps,
            "history_len": len(self._t_hist),
            "t_pred_on": t_pred_on,
            "t_pred_off": t_pred_off,
            "t_pred_actual": t_pred_actual,
            "u_planned": u_planned,
            "u_actual": u_actual,
            "start_step": start_step,
            "stop_step": stop_step,
            "lookahead_steps": steps,
            "step_minutes": step_minutes,
            "forecast_temps": t_outs,
            "forecast_solar": solar_outs_raw,
            "forecast_solar_samples": solar_outs,
            "predicted_temps": preds_next,
            "planned_states": planned_states,
            "plan_cost": plan_cost,
            "tick_duration_ms": tick_duration_ms,
            "error": error,
            "update": update,
            "eta": eta,
            "base_eta": base_eta,
            "k_step_error": k_step_errors,
            "k_step_error_mean": k_step_means,
            "k_step_pred_improvement": k_step_improvement,
            "lookahead_selected": steps,
            "lookahead_bounds": {"min": min_steps, "max": max_steps},
            "lookahead_reason": decision_reason,
            "step_count": self._step_count,
            "weights_reset": weights_reset,
            "solar_orientation": self.solar_orientation_diagnostics(),
        }
