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
import math
import time
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
    build_solar_orientation_features,
    compass_label,
    coerce_solar_sample,
    coerce_solar_profile,
    diffuse_only_sample,
)

DYNAMIC_LOOKAHEAD_MIN = 12
DYNAMIC_LOOKAHEAD_MAX = 24
DYNAMIC_LOOKAHEAD_TOLERANCE = 0.2
LEARNING_RATE_TARGET_ERROR = 0.2
LEARNING_RATE_SCALE_MIN = 0.2
LEARNING_RATE_SCALE_MAX = 2.0
MULTI_STEP_UPDATE_HORIZON = 3
Z_SCORE_MIN_STD = 0.2
U_WEIGHT_MIN = 1e-6
SOLAR_ORIENTATION_GRID_DEGREES = 5
SOLAR_ORIENTATION_MIN_EXAMPLES = 8
SOLAR_ORIENTATION_MIN_ACTIVE_EXAMPLES = 4
SOLAR_ORIENTATION_TRAINING_MAX = 96
SOLAR_ACTIVE_FLUX_MIN = 10.0
SOLAR_ORIENTATION_IMPROVEMENT_MIN = 0.02
MODEL_SCHEMA_VERSION = 2
PHYSICAL_HALF_LIFE_MINUTES = (60.0, 180.0, 360.0)
PHYSICAL_SOLAR_SCALE = 500.0

DEFAULT_LEARNING_SWITCH_CONFIG = {
    "lookahead_steps": 24,
    "learning_rate": 0.005,
    "l2_reg": 0.001,
    "w_energy": 0.01,
    "history_steps": 12,
    "history_decay": 1.0,
    "error_decay": 0.9,
    "default_u_gain_per_hour": 0.1,
    "learning_model": DEFAULT_LEARNING_MODEL,
}


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
        self._history_steps = self._coerce_int(
            self._config.get("history_steps"),
            DEFAULT_LEARNING_SWITCH_CONFIG["history_steps"],
        )
        # History discounting is disabled (no decay).
        self._history_decay = 1.0
        self._history_decay_vector = np.ones(self._history_steps, dtype=float)
        # Decay applied to future prediction errors in planning (0 <= decay <= 1).
        self._error_decay = self._coerce_float(
            self._config.get("error_decay"),
            DEFAULT_LEARNING_SWITCH_CONFIG["error_decay"],
        )
        if not (0.0 <= self._error_decay <= 1.0):
            self._error_decay = DEFAULT_LEARNING_SWITCH_CONFIG["error_decay"]
        # L2 regularization coefficient for weight decay (>= 0).
        self._l2_reg = self._coerce_float(
            self._config.get("l2_reg"), DEFAULT_LEARNING_SWITCH_CONFIG["l2_reg"]
        )
        if not np.isfinite(self._l2_reg) or self._l2_reg < 0.0:
            self._l2_reg = DEFAULT_LEARNING_SWITCH_CONFIG["l2_reg"]
        # Default heating effect when weights are fresh.
        self._default_u_gain_per_hour = self._coerce_float(
            self._config.get("default_u_gain_per_hour"),
            DEFAULT_LEARNING_SWITCH_CONFIG["default_u_gain_per_hour"],
        )
        self._learning_model = self._normalize_learning_model(
            self._config.get("learning_model")
        )
        self._model_schema_version = MODEL_SCHEMA_VERSION
        self._feature_step_minutes = 60.0

        # Optional starting weights.
        self._weights_init = self._config.get("weights_init")
        # Whether the model includes a solar irradiance feature channel.
        self._use_solar_feature = bool(self._config.get("use_solar_feature"))
        # Current model weights, shape: [t_hist, out_hist, (solar_orientation_hist), u_hist, bias].
        self._weights = self._init_weights(self._weights_init)
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

    def _model_name(self) -> str:
        """Return the user-facing learning model label."""
        return LEARNING_MODEL_LABELS[self._learning_model]

    def _model_id(self) -> str:
        """Return the compact internal model identifier."""
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            return (
                "physical_solar_orientation" if self._use_solar_feature else "physical"
            )
        return "linear_solar_orientation" if self._use_solar_feature else "linear"

    def _normalize_learning_model(self, value: Any) -> str:
        """Normalize the configured learning model identifier."""
        model = str(value or DEFAULT_LEARNING_MODEL)
        if model not in LEARNING_MODEL_LABELS:
            return LEARNING_MODEL_LINEAR
        return model

    def _snapshot_model_id(self, state: dict) -> str | None:
        """Return the internal model id represented by a snapshot."""
        model_id = state.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
        model = state.get("model")
        if model == "Physical":
            return (
                "physical_solar_orientation" if self._use_solar_feature else "physical"
            )
        if model == "Linear":
            return "linear_solar_orientation" if self._use_solar_feature else "linear"
        if model == "delta_zscore":
            return "linear"
        if model in {
            "delta_zscore_solar",
            "delta_zscore_solar_profile",
            "delta_zscore_solar_orientation",
        }:
            return "linear_solar_orientation"
        return None

    def _legacy_model_type(self, state: dict) -> str | None:
        """Return a legacy linear model id when one is present."""
        model = state.get("model")
        return (
            model
            if model
            in {
                "delta_zscore",
                "delta_zscore_solar",
                "delta_zscore_solar_profile",
                "delta_zscore_solar_orientation",
            }
            else None
        )

    def snapshot_state(self) -> dict:
        """Serialize weights and history for persistence/debugging."""
        return {
            "weights": self._weights.tolist(),
            "model": self._model_name(),
            "model_id": self._model_id(),
            "model_schema_version": self._model_schema_version,
            "history": {
                "t_room": list(self._t_hist),
                "t_out": list(self._out_hist),
                "solar": list(self._solar_hist),
                "solar_samples": [dict(row) for row in self._solar_sample_hist],
                "u": list(self._u_hist),
                "pred_next": list(self._prev_pred_history),
                "pred_abs_error": list(self._prev_abs_error_history),
                "pred_timestamps": list(self._prev_pred_timestamp_history),
                "weight_abs_update": list(self._weight_abs_update_history),
                "weight_delta_abs_latest": list(self._weight_delta_abs_latest),
                "k_step_abs_error": {
                    str(k): {
                        "mean": values.get("mean"),
                        "count": int(values.get("count", 0)),
                    }
                    for k, values in self._k_step_abs_error_history.items()
                },
                "k_step_pred_improvement": {
                    str(k): {
                        "mean": values.get("mean"),
                        "count": int(values.get("count", 0)),
                    }
                    for k, values in self._k_step_pred_improvement_history.items()
                },
            },
            "last_u": self._last_u,
            "last_plan": dict(self._last_plan),
            "solar_orientation_azimuth": self._solar_orientation_azimuth,
            "solar_orientation_confidence": self._solar_orientation_confidence,
            "solar_orientation_score": self._solar_orientation_score,
            "eta": self._last_eta,
            "base_eta": self._base_eta,
            "prev_feature": list(self._prev_feature) if self._prev_feature else None,
            "config": {
                "lookahead_steps": self._config.get("lookahead_steps"),
                "learning_rate": self._config.get("learning_rate"),
                "l2_reg": self._l2_reg,
                "w_energy": self._config.get("w_energy"),
                "history_steps": self._history_steps,
                "history_decay": self._history_decay,
                "error_decay": self._error_decay,
                "default_u_gain_per_hour": self._default_u_gain_per_hour,
                "use_solar_feature": self._use_solar_feature,
                "learning_model": self._learning_model,
            },
            "solar_orientation": self.solar_orientation_diagnostics(),
        }

    def restore_state(self, state: dict | None) -> None:
        """Restore the internal state defensively from a snapshot."""
        try:
            if not isinstance(state, dict):
                return

            model_type = self._legacy_model_type(state)
            snapshot_model_id = self._snapshot_model_id(state)
            model_matches = snapshot_model_id == self._model_id()
            weights = state.get("weights")
            if model_matches and weights is not None:
                restored = np.array([float(v) for v in weights], dtype=float)
                if restored.size == self._weights.size:
                    if np.allclose(restored, 0.0) or not np.all(np.isfinite(restored)):
                        restored = self._init_weights(self._weights_init)
                    self._weights = restored
                elif self._learning_model == LEARNING_MODEL_LINEAR:
                    adapted = self._adapt_weights(restored, model_type)
                    if adapted is not None:
                        self._weights = adapted
            elif (
                weights is not None
                and self._learning_model == LEARNING_MODEL_LINEAR
                and model_type is not None
            ):
                restored = np.array([float(v) for v in weights], dtype=float)
                adapted = self._adapt_weights(restored, model_type)
                if adapted is not None:
                    self._weights = adapted
            elif weights is not None:
                self._weights = self._init_weights(self._weights_init)

            history = state.get("history") or {}
            self._t_hist = self._coerce_float_list(history.get("t_room"))[
                : self._history_steps
            ]
            self._out_hist = self._coerce_float_list(history.get("t_out"))[
                : self._history_steps
            ]
            self._solar_hist = self._coerce_float_list(history.get("solar"))[
                : self._history_steps
            ]
            self._solar_sample_hist = self._coerce_solar_sample_history(
                history.get("solar_samples")
            )[: self._history_steps]
            if self._use_solar_feature and not self._solar_sample_hist:
                legacy_profiles = self._coerce_solar_profile_history(
                    history.get("solar_profile")
                )
                if legacy_profiles:
                    self._solar_sample_hist = [
                        coerce_solar_sample(row).as_dict() for row in legacy_profiles
                    ][: self._history_steps]
                else:
                    self._solar_sample_hist = [
                        diffuse_only_sample(value).as_dict()
                        for value in self._solar_hist
                    ][: self._history_steps]
            self._u_hist = self._coerce_int_list(history.get("u"))[
                : self._history_steps
            ]
            self._prev_pred_history = self._coerce_float_list(history.get("pred_next"))[
                : self._history_steps
            ]
            self._prev_abs_error_history = self._coerce_float_list(
                history.get("pred_abs_error")
            )[: self._history_steps]
            pred_timestamps = history.get("pred_timestamps") or []
            self._prev_pred_timestamp_history = [
                str(value) for value in pred_timestamps
            ][: self._history_steps]
            self._weight_abs_update_history = self._coerce_float_list(
                history.get("weight_abs_update")
            )[: self._history_steps]
            self._weight_delta_abs_latest = self._coerce_float_list(
                history.get("weight_delta_abs_latest")
            )[: self._weights.size]
            self._k_step_abs_error_history = self._restore_k_step_stats(
                history.get("k_step_abs_error")
            )
            self._k_step_pred_improvement_history = self._restore_k_step_stats(
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
            self._restore_orientation_state(state)
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
                self._clear_pending_predictions()
        except Exception as exc:  # noqa: BLE001 - keep restore resilient
            _LOGGER.warning("%s - Learning state restore failed: %s", self._name, exc)

    def _adapt_weights(
        self, restored: np.ndarray, model_type: str | None
    ) -> np.ndarray | None:
        """Adapt weights saved with a different history length."""
        size = restored.size
        if size < 4:
            return None
        old_layout = self._infer_weight_layout(model_type, size)
        if old_layout is None:
            return None
        old_steps, old_profile_size = old_layout
        if old_steps <= 0:
            return None
        new_steps = self._history_steps
        new_profile_size = self._solar_profile_size()
        new_weights = np.zeros(self._feature_size() + 1, dtype=float)
        copy_len = min(old_steps, new_steps)

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
                orientation = self._infer_legacy_orientation_from_weights(
                    restored, old_steps, old_solar_start
                )
                if orientation is not None:
                    self._solar_orientation_azimuth = orientation
                for step in range(copy_len):
                    old_start = old_solar_start + step * old_profile_size
                    new_start = new_solar_start + step * new_profile_size
                    old_row = restored[old_start : old_start + old_profile_size]
                    new_weights[new_start] = old_row[0]
                    if orientation is not None:
                        first_index = self._legacy_bin_index(orientation)
                        second_index = self._legacy_bin_index(orientation + 90.0)
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
        return new_weights

    def _infer_weight_layout(
        self, model_type: str | None, size: int
    ) -> tuple[int, int] | None:
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

    def _solar_profile_size(self) -> int:
        """Return the number of solar features per history step."""
        return SOLAR_ORIENTATION_FEATURE_SIZE if self._use_solar_feature else 0

    def _feature_size(self) -> int:
        """Return the feature vector size without the bias term."""
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            solar_size = 9 if self._use_solar_feature else 0
            return 6 + solar_size + 3
        return self._history_steps * (3 + self._solar_profile_size())

    def _init_weights(self, weights_init) -> np.ndarray:
        """Build initial weights, optionally seeded by configuration."""
        weights = self._default_weights()
        size = weights.size
        if isinstance(weights_init, (list, tuple)) and len(weights_init) == size:
            try:
                weights = np.array([float(v) for v in weights_init], dtype=float)
            except (TypeError, ValueError):
                pass
        return weights

    def _default_weights(self) -> np.ndarray:
        """Return default linear model weights."""
        size = self._feature_size() + 1
        return np.zeros(size, dtype=float)

    def _ensure_weights_finite(self) -> bool:
        """Reset weights if any non-finite values appear."""
        if np.all(np.isfinite(self._weights)):
            self._warned_invalid_weights = False
            return False
        self._weights = self._init_weights(self._weights_init)
        if not self._warned_invalid_weights:
            _LOGGER.warning(
                "%s - Reset learning weights due to non-finite values", self._name
            )
            self._warned_invalid_weights = True
        return True

    def _apply_default_u_gain(self, step_minutes: int | None) -> bool:
        """Seed a small heating gain so ON predicts a higher next temperature."""
        default_weights = self._default_weights()
        if not np.allclose(self._weights, default_weights, rtol=0.0, atol=1e-9):
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
        u_index = self._u_weight_start()
        if u_index < self._weights.size:
            if self._learning_model == LEARNING_MODEL_PHYSICAL:
                feature_delta = self._physical_filter_delta(
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
        start = self._u_weight_start()
        end = start + self._heat_feature_count()
        if start >= self._weights.size:
            return False
        floor = U_WEIGHT_MIN
        if not np.isfinite(floor) or floor <= 0.0:
            return False
        before = self._weights[start:end].copy()
        self._weights[start:end] = np.maximum(self._weights[start:end], floor)
        return not np.allclose(before, self._weights[start:end], rtol=0.0, atol=1e-12)

    def _u_weight_start(self) -> int:
        """Return the start index of the heating-control weights."""
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            return 6 + (9 if self._use_solar_feature else 0)
        return (
            self._history_steps * 2 + self._history_steps * self._solar_profile_size()
        )

    def _heat_feature_count(self) -> int:
        """Return the number of heating-control features."""
        return (
            3
            if self._learning_model == LEARNING_MODEL_PHYSICAL
            else self._history_steps
        )

    def _coerce_int(self, value, default: int) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if value <= 0:
            value = default
        return value

    def _coerce_float(self, value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float_list(self, values) -> list[float]:
        if not isinstance(values, (list, tuple)):
            return []
        out: list[float] = []
        for value in values:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

    def _coerce_solar_profile_history(self, values) -> list[list[float]]:
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

    def _coerce_solar_sample_history(self, values) -> list[dict[str, float | None]]:
        """Coerce persisted solar sample history rows."""
        if not isinstance(values, (list, tuple)):
            return []
        out: list[dict[str, float | None]] = []
        for value in values:
            out.append(coerce_solar_sample(value).as_dict())
        return out

    def _restore_orientation_state(self, state: dict) -> None:
        """Restore persisted orientation metadata defensively."""
        azimuth = self._coerce_optional_float(state.get("solar_orientation_azimuth"))
        if azimuth is None:
            diagnostics = state.get("solar_orientation")
            if isinstance(diagnostics, dict):
                azimuth = self._coerce_optional_float(diagnostics.get("azimuth"))
        self._solar_orientation_azimuth = (
            azimuth % 360.0 if azimuth is not None else self._solar_orientation_azimuth
        )
        confidence = self._coerce_optional_float(
            state.get("solar_orientation_confidence")
        )
        if confidence is None:
            diagnostics = state.get("solar_orientation")
            if isinstance(diagnostics, dict):
                confidence = self._coerce_optional_float(diagnostics.get("confidence"))
        self._solar_orientation_confidence = confidence
        self._solar_orientation_score = self._coerce_optional_float(
            state.get("solar_orientation_score")
        )

    def _coerce_optional_float(self, value) -> float | None:
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

    def _infer_legacy_orientation_from_weights(
        self, weights: np.ndarray, old_steps: int, old_solar_start: int
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
            first = self._legacy_bin_index(azimuth)
            second = self._legacy_bin_index(azimuth + 90.0)
            score = exposure_bins[first] + exposure_bins[second]
            if score > best_score:
                best_score = float(score)
                best_azimuth = float(azimuth)
        return best_azimuth

    def _legacy_bin_index(self, azimuth: float) -> int:
        """Return the nearest legacy eight-way bin index for an azimuth."""
        return int(round((azimuth % 360.0) / 45.0)) % len(SOLAR_PROFILE_BIN_AZIMUTHS)

    def _coerce_int_list(self, values) -> list[int]:
        if not isinstance(values, (list, tuple)):
            return []
        out: list[int] = []
        for value in values:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
        return out

    def _restore_k_step_stats(
        self, payload
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
                items = self._coerce_float_list(values)
                mean_val = float(np.mean(items)) if items else None
                count_val = len(items)
            restored[horizon] = {"mean": mean_val, "count": count_val}
        return restored

    def _clear_pending_predictions(self) -> None:
        """Drop pending learning rows whose feature semantics may be stale."""
        self._pending_k_step_preds.clear()
        self._pending_k_step_features.clear()
        self._pending_k_step_baseline.clear()
        self._prev_feature = None
        self._prev_orientation_input = None

    def _record_k_step_errors(
        self, actual_temp: float, step_index: int
    ) -> tuple[dict[int, float], list[dict[str, float | int | list[float]]]]:
        """Update k-step error stats and return absolute errors plus update inputs."""
        errors: dict[int, float] = {}
        updates: list[dict[str, float | int | list[float]]] = []
        if self._pending_k_step_preds:
            stale_steps = [
                key for key in self._pending_k_step_preds if key < step_index
            ]
            for key in stale_steps:
                self._pending_k_step_preds.pop(key, None)
        if self._pending_k_step_features:
            stale_steps = [
                key for key in self._pending_k_step_features if key < step_index
            ]
            for key in stale_steps:
                self._pending_k_step_features.pop(key, None)
        if self._pending_k_step_baseline:
            stale_steps = [
                key for key in self._pending_k_step_baseline if key < step_index
            ]
            for key in stale_steps:
                self._pending_k_step_baseline.pop(key, None)
        matured = self._pending_k_step_preds.pop(step_index, None)
        if not matured:
            return errors, updates
        matured_features = self._pending_k_step_features.pop(step_index, None) or {}
        baseline_matured = self._pending_k_step_baseline.pop(step_index, None) or {}
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
            if horizon_int <= MULTI_STEP_UPDATE_HORIZON:
                feature = matured_features.get(horizon_int)
                if feature is not None:
                    updates.append(
                        {
                            "horizon": horizon_int,
                            "error": float(signed_error),
                            "feature": feature,
                        }
                    )
            stats = self._k_step_abs_error_history.setdefault(
                horizon_int, {"mean": None, "count": 0}
            )
            count = stats.get("count", 0)
            mean = stats.get("mean")
            try:
                count_val = int(count)
            except (TypeError, ValueError):
                count_val = 0
            try:
                mean_val = float(mean) if mean is not None else None
            except (TypeError, ValueError):
                mean_val = None
            if mean_val is None or count_val <= 0:
                mean_val = float(err)
                count_val = 1
            else:
                mean_val = (mean_val * count_val + float(err)) / float(count_val + 1)
                count_val += 1
            stats["mean"] = mean_val
            stats["count"] = count_val
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
                        imp_stats = self._k_step_pred_improvement_history.setdefault(
                            horizon_int, {"mean": None, "count": 0}
                        )
                        imp_count = imp_stats.get("count", 0)
                        imp_mean = imp_stats.get("mean")
                        try:
                            imp_count_val = int(imp_count)
                        except (TypeError, ValueError):
                            imp_count_val = 0
                        try:
                            imp_mean_val = (
                                float(imp_mean) if imp_mean is not None else None
                            )
                        except (TypeError, ValueError):
                            imp_mean_val = None
                        if imp_mean_val is None or imp_count_val <= 0:
                            imp_mean_val = improvement
                            imp_count_val = 1
                        else:
                            imp_mean_val = (
                                imp_mean_val * imp_count_val + improvement
                            ) / float(imp_count_val + 1)
                            imp_count_val += 1
                        imp_stats["mean"] = imp_mean_val
                        imp_stats["count"] = imp_count_val
        return errors, updates

    def _queue_k_step_predictions(
        self, preds: list[float], step_index: int, max_steps: int
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
            bucket = self._pending_k_step_preds.setdefault(target_step, {})
            bucket[horizon] = pred_val

    def _queue_k_step_features(
        self, features: list[np.ndarray], step_index: int, max_steps: int
    ) -> None:
        """Queue k-step feature vectors keyed by target step index."""
        horizon_max = min(MULTI_STEP_UPDATE_HORIZON, max_steps, len(features))
        if horizon_max <= 0:
            return
        for horizon in range(1, horizon_max + 1):
            feature = features[horizon - 1]
            if feature is None:
                continue
            target_step = step_index + horizon
            bucket = self._pending_k_step_features.setdefault(target_step, {})
            bucket[horizon] = feature.tolist()

    def _queue_k_step_baseline(
        self, baseline_temp: float, step_index: int, max_steps: int
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
            bucket = self._pending_k_step_baseline.setdefault(target_step, {})
            bucket[horizon] = baseline_val

    def _k_step_error_stats(
        self, max_steps: int
    ) -> dict[int, dict[str, float | int | None]]:
        """Return mean absolute error and count per horizon."""
        stats: dict[int, dict[str, float | int | None]] = {}
        for horizon in range(1, max_steps + 1):
            values = self._k_step_abs_error_history.get(horizon)
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

    def _select_dynamic_lookahead(
        self,
        stats: dict[int, dict[str, float | int | None]],
        min_steps: int,
        max_steps: int,
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
                if mean > DYNAMIC_LOOKAHEAD_TOLERANCE:
                    ok = False
                    reason = "above_tolerance"
                    break
            if ok:
                return horizon, "within_tolerance"
        return min_steps, reason

    def _update_history(
        self,
        t_room: float,
        t_out: float,
        u_actual: int,
        solar: float | None,
        solar_sample: dict[str, Any] | None,
    ) -> None:
        """Maintain fixed-length history buffers for model inputs."""
        self._t_hist.insert(0, float(t_room))
        self._out_hist.insert(0, float(t_out))
        if self._use_solar_feature:
            if solar is None:
                solar_value = self._solar_hist[0] if self._solar_hist else 0.0
            else:
                solar_value = float(solar)
            self._solar_hist.insert(0, solar_value)
            sample_value = coerce_solar_sample(solar_sample, solar_value)
            self._solar_sample_hist.insert(0, sample_value.as_dict())
        self._u_hist.insert(0, int(u_actual))
        if len(self._t_hist) > self._history_steps:
            self._t_hist = self._t_hist[: self._history_steps]
        if len(self._out_hist) > self._history_steps:
            self._out_hist = self._out_hist[: self._history_steps]
        if len(self._solar_hist) > self._history_steps:
            self._solar_hist = self._solar_hist[: self._history_steps]
        if len(self._solar_sample_hist) > self._history_steps:
            self._solar_sample_hist = self._solar_sample_hist[: self._history_steps]
        if len(self._u_hist) > self._history_steps:
            self._u_hist = self._u_hist[: self._history_steps]

    def _pad_sequence(self, values: list[float], pad_value: float) -> np.ndarray:
        """Pad history to a fixed length using the most recent value."""
        if not values:
            return np.full(self._history_steps, pad_value, dtype=float)
        if len(values) >= self._history_steps:
            return np.array(values[: self._history_steps], dtype=float)
        padding = [pad_value] * (self._history_steps - len(values))
        return np.array(values + padding, dtype=float)

    def _history_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """Build sanitized fixed-length history arrays for the model."""
        t_pad = self._pad_sequence(
            self._t_hist, self._t_hist[-1] if self._t_hist else 0.0
        )
        out_pad = self._pad_sequence(
            self._out_hist, self._out_hist[-1] if self._out_hist else 0.0
        )
        solar_pad: np.ndarray | None = None
        if self._use_solar_feature:
            sample_rows = self._solar_sample_hist
            if not sample_rows and self._solar_hist:
                sample_rows = [
                    diffuse_only_sample(value).as_dict() for value in self._solar_hist
                ]
            if sample_rows:
                rows = [
                    list(
                        build_solar_orientation_features(
                            coerce_solar_sample(row),
                            self._effective_solar_orientation(),
                        )
                    )
                    for row in sample_rows
                ]
                if len(rows) < self._history_steps:
                    rows.extend([rows[-1]] * (self._history_steps - len(rows)))
                solar_pad = np.array(rows[: self._history_steps], dtype=float)
            else:
                solar_pad = np.zeros(
                    (self._history_steps, SOLAR_ORIENTATION_FEATURE_SIZE), dtype=float
                )
        u_pad = self._pad_sequence([float(v) for v in self._u_hist], 0.0)
        return (
            self._sanitize_array(t_pad),
            self._sanitize_array(out_pad),
            self._sanitize_array(solar_pad) if solar_pad is not None else None,
            self._sanitize_array(u_pad),
        )

    def _transform_histories(
        self,
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
            self._sanitize_array(t_norm),
            self._sanitize_array(out_norm),
            self._sanitize_array(solar_norm) if solar_norm is not None else None,
            u_hist,
        )

    def _build_model_feature(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
    ) -> np.ndarray:
        """Build a feature vector for the active learning model."""
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            return self._build_physical_feature(t_hist, out_hist, solar_hist, u_hist)
        t_feat, out_feat, solar_feat, u_feat = self._transform_histories(
            t_hist, out_hist, solar_hist, u_hist
        )
        return self._build_linear_feature(t_feat, out_feat, solar_feat, u_feat)

    def _build_physical_feature(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
    ) -> np.ndarray:
        """Build fixed-scale physical thermal features plus bias."""
        room_now = float(t_hist[0]) if t_hist.size else 20.0
        outside_now = float(out_hist[0]) if out_hist.size else 10.0
        room_1h = self._history_value_at(t_hist, 60.0, room_now)
        room_3h = self._history_value_at(t_hist, 180.0, room_now)
        outside_3h = self._history_value_at(out_hist, 180.0, outside_now)
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
            parts.append(self._physical_solar_features(solar_hist))
        parts.append(self._physical_filter_features(u_hist, scale=1.0))
        parts.append(np.array([1.0], dtype=float))
        return self._sanitize_array(np.concatenate(parts))

    def _history_value_at(
        self, values: np.ndarray, minutes_ago: float, default: float
    ) -> float:
        """Return a latest-first history value at a time offset with interpolation."""
        if values.size == 0:
            return float(default)
        step_minutes = max(float(self._feature_step_minutes or 60.0), 1.0)
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

    def _physical_filter_delta(
        self, half_life_minutes: float, step_minutes: float
    ) -> float:
        """Return the newest-sample response of one exponential filter."""
        minutes = max(float(step_minutes or 60.0), 1.0)
        a = 0.5 ** (minutes / float(half_life_minutes))
        return 1.0 - a

    def _physical_filter_features(
        self, values: np.ndarray, *, scale: float
    ) -> np.ndarray:
        """Return fast/medium/slow exposure filters for latest-first values."""
        if values.size == 0:
            values = np.zeros(1, dtype=float)
        ordered = list(reversed([float(v) for v in values]))
        step_minutes = max(float(self._feature_step_minutes or 60.0), 1.0)
        filtered: list[float] = []
        for half_life in PHYSICAL_HALF_LIFE_MINUTES:
            state = 0.0
            a = 0.5 ** (step_minutes / half_life)
            for value in ordered:
                state = a * state + (1.0 - a) * value
            filtered.append(state / scale)
        return np.array(filtered, dtype=float)

    def _physical_solar_features(self, solar_hist: np.ndarray) -> np.ndarray:
        """Return filtered diffuse and facade solar exposure features."""
        rows = np.array(solar_hist, dtype=float)
        if rows.ndim != 2 or rows.shape[1] < SOLAR_ORIENTATION_FEATURE_SIZE:
            rows = np.zeros((1, SOLAR_ORIENTATION_FEATURE_SIZE), dtype=float)
        diffuse = self._physical_filter_features(rows[:, 0], scale=PHYSICAL_SOLAR_SCALE)
        facade_1 = self._physical_filter_features(
            rows[:, 1], scale=PHYSICAL_SOLAR_SCALE
        )
        facade_2 = self._physical_filter_features(
            rows[:, 2], scale=PHYSICAL_SOLAR_SCALE
        )
        return np.concatenate((diffuse, facade_1, facade_2))

    def _sanitize_array(self, values: np.ndarray) -> np.ndarray:
        """Replace non-finite values with a reasonable fill."""
        if np.all(np.isfinite(values)):
            return values
        finite_vals = values[np.isfinite(values)]
        fill_value = float(finite_vals[0]) if finite_vals.size > 0 else 0.0
        return np.where(np.isfinite(values), values, fill_value)

    def _build_feature(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
    ) -> np.ndarray:
        """Assemble the linear feature vector from transformed histories plus bias."""
        return self._build_linear_feature(t_hist, out_hist, solar_hist, u_hist)

    def _build_linear_feature(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
    ) -> np.ndarray:
        """Assemble the z-scored linear feature vector plus bias."""
        decay = self._history_decay_vector
        t_hist = t_hist * decay
        out_hist = out_hist * decay
        parts = [t_hist, out_hist]
        if solar_hist is not None:
            solar_decay = np.repeat(decay, SOLAR_ORIENTATION_FEATURE_SIZE)
            parts.append(solar_hist.reshape(-1) * solar_decay)
        u_hist = u_hist * decay
        parts.extend((u_hist, np.array([1.0], dtype=float)))
        return np.concatenate(parts)

    def _predict_next(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
        u_value: int,
        solar_value: dict[str, Any] | float | None,
    ) -> tuple[float, np.ndarray]:
        """Predict next room temperature for a single step and return features."""
        u_hist_step = np.concatenate(
            (np.array([float(u_value)], dtype=float), u_hist[:-1])
        )
        solar_hist_step: np.ndarray | None = None
        if solar_hist is not None:
            solar_step = np.array(
                build_solar_orientation_features(
                    coerce_solar_sample(
                        solar_value,
                        solar_hist[0][0] if solar_hist.size else 0.0,
                    ),
                    self._effective_solar_orientation(),
                ),
                dtype=float,
            )
            solar_hist_step = np.concatenate(
                (
                    solar_step.reshape(1, SOLAR_ORIENTATION_FEATURE_SIZE),
                    solar_hist[:-1],
                )
            )
        x = self._build_model_feature(t_hist, out_hist, solar_hist_step, u_hist_step)
        delta = float(x @ self._weights)
        t_next = float(t_hist[0] + delta)
        return t_next, x

    def weight_contributions(self, u_value: int | None = None) -> np.ndarray:
        """Return absolute per-weight contributions for the next-step prediction."""
        t_hist, out_hist, solar_hist, u_hist = self._history_arrays()
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
        x = self._build_model_feature(t_hist, out_hist, solar_hist_step, u_hist_step)
        if x.size != self._weights.size:
            return np.zeros_like(self._weights)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(self._weights)):
            return np.zeros_like(self._weights)
        return np.abs(x * self._weights)

    def _solar_weight_start(self) -> int:
        """Return the start index of solar weights for the active model."""
        if not self._use_solar_feature:
            return self._weights.size
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            return 6
        return self._history_steps * 2

    def _directional_solar_weight_indices(self, facade_index: int) -> list[int]:
        """Return positive-directional solar weight indices for one facade."""
        if not self._use_solar_feature:
            return []
        solar_start = self._solar_weight_start()
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            start = solar_start + 3 + int(facade_index) * 3
            return [start, start + 1, start + 2]
        offset = 1 + int(facade_index)
        return [
            solar_start + step * SOLAR_ORIENTATION_FEATURE_SIZE + offset
            for step in range(self._history_steps)
        ]

    def solar_orientation_diagnostics(self) -> dict:
        """Return the dominant learned solar exposure direction."""
        if not self._use_solar_feature:
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

        facade_weights = [0.0, 0.0]
        for facade_index in (0, 1):
            for weight_index in self._directional_solar_weight_indices(facade_index):
                if weight_index >= self._weights.size:
                    continue
                weight = float(self._weights[weight_index])
                if np.isfinite(weight) and weight > 0.0:
                    facade_weights[facade_index] += weight

        total = sum(facade_weights)
        if total <= 0.0 or self._solar_orientation_azimuth is None:
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
        first = {
            "azimuth": self._solar_orientation_azimuth % 360.0,
            "contribution": facade_weights[0],
        }
        second = {
            "azimuth": (self._solar_orientation_azimuth + 90.0) % 360.0,
            "contribution": facade_weights[1],
        }
        ranked = sorted(
            [first, second], key=lambda item: item["contribution"], reverse=True
        )
        orientation_1_pct = 100.0 * ranked[0]["contribution"] / total
        orientation_2_pct = 100.0 * ranked[1]["contribution"] / total
        return {
            "azimuth": ranked[0]["azimuth"],
            "confidence": self._solar_orientation_confidence,
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

    def _rollout(
        self,
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
        """Roll the model forward for a planned ON interval, optionally capturing features."""
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
            t_next, x = self._predict_next(
                t_hist_cur,
                out_hist_cur,
                solar_hist_cur,
                u_hist_cur,
                u,
                solar_value,
            )
            preds.append(t_next)
            if capture_features:
                features.append(x)

            t_hist_cur = np.concatenate(
                (np.array([t_next], dtype=float), t_hist_cur[:-1])
            )
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
                        self._effective_solar_orientation(),
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

    def _simulate_schedule(
        self,
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
        preds, _ = self._rollout(
            t_hist,
            out_hist,
            solar_hist,
            u_hist,
            t_outs,
            solar_forecast,
            start,
            stop,
            steps,
            capture_features=False,
        )
        try:
            w_energy = float(self._config.get("w_energy", 0.0))
        except (TypeError, ValueError):
            w_energy = 0.0
        error_decay = self._error_decay
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
            cost += err_weight * err * err + w_energy * u
        return cost, preds

    def _find_start_stop(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
        t_set: float,
        t_outs: list[float],
        solar_forecast: list[dict[str, float | None]],
    ) -> tuple[int, int, float]:
        """Search all start/stop pairs for a single ON interval to minimize cost."""
        # Optimization idea (not implemented): for each start, batch all stop candidates
        # in a single vectorized rollout (S stops x T steps) to remove the inner Python
        # loop and speed up evaluation while keeping the same O(T^3) arithmetic.
        steps = len(t_outs)
        best_start = steps
        best_stop = steps
        best_cost, _ = self._simulate_schedule(
            t_hist,
            out_hist,
            solar_hist,
            u_hist,
            t_set,
            t_outs,
            solar_forecast,
            steps,
            steps,
        )
        for start in range(steps + 1):
            for stop in range(start, steps + 1):
                cost, _ = self._simulate_schedule(
                    t_hist,
                    out_hist,
                    solar_hist,
                    u_hist,
                    t_set,
                    t_outs,
                    solar_forecast,
                    start,
                    stop,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
                    best_stop = stop
        return best_start, best_stop, best_cost

    def _build_forecast(
        self, t_out: float, t_out_forecast: list[float] | None, steps: int
    ) -> list[float]:
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
            if self._last_forecast_temp is not None:
                temps = [t_out] + [self._last_forecast_temp] * (steps - 1)
            else:
                temps = [t_out] * steps
        elif len(temps) < steps:
            temps.extend([temps[-1]] * (steps - len(temps)))
        elif len(temps) > steps:
            temps = temps[:steps]

        if temps:
            self._last_forecast_temp = temps[-1]
        return temps

    def _build_solar_forecast(
        self,
        solar: float | None,
        solar_forecast: list[float] | None,
        solar_sample: dict[str, Any] | None,
        solar_sample_forecast: list[dict[str, Any]] | None,
        steps: int,
    ) -> list[dict[str, float | None]]:
        """Normalize solar forecast input to the lookahead length with fallbacks."""
        if not self._use_solar_feature or steps <= 0:
            return []

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
            elif self._last_forecast_solar is not None:
                values = [
                    diffuse_only_sample(self._last_forecast_solar).as_dict()
                ] * steps
            else:
                values = [diffuse_only_sample(0.0).as_dict()] * steps
        elif len(values) < steps:
            values.extend([dict(values[-1])] * (steps - len(values)))
        elif len(values) > steps:
            values = values[:steps]

        if values:
            self._last_forecast_solar = float(values[-1]["flux"] or 0.0)
        return values

    def _effective_solar_orientation(self) -> float:
        """Return the active orientation angle for feature construction."""
        if self._solar_orientation_azimuth is None:
            return 0.0
        return self._solar_orientation_azimuth % 360.0

    def _record_orientation_example(self, actual_temp: float) -> None:
        """Record a matured one-step training example for orientation search."""
        if not self._use_solar_feature or self._prev_orientation_input is None:
            return
        previous = self._prev_orientation_input
        base_temp = self._coerce_optional_float(previous.get("base_temp"))
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
        if len(examples) < SOLAR_ORIENTATION_MIN_EXAMPLES:
            return
        active_count = sum(
            1 for example in examples if self._example_solar_active(example)
        )
        if active_count < SOLAR_ORIENTATION_MIN_ACTIVE_EXAMPLES:
            return

        baseline_score = self._fit_orientation_score(examples, None)[1]
        best_orientation = None
        best_weights = None
        best_score = math.inf
        best_tie_score = -math.inf
        for azimuth in range(0, 360, SOLAR_ORIENTATION_GRID_DEGREES):
            weights, score = self._fit_orientation_score(examples, float(azimuth))
            tie_score = self._orientation_tie_score(weights, examples, float(azimuth))
            if score < best_score - 1e-6 or (
                abs(score - best_score) <= 1e-6 and tie_score > best_tie_score
            ):
                best_orientation = float(azimuth)
                best_weights = weights
                best_score = score
                best_tie_score = tie_score

        if (
            best_orientation is None
            or best_weights is None
            or not np.isfinite(best_score)
        ):
            return

        _, current_score = self._fit_orientation_score(
            examples, self._solar_orientation_azimuth
        )
        if self._solar_orientation_azimuth is not None and np.isfinite(current_score):
            improvement = (current_score - best_score) / max(abs(current_score), 1e-9)
            if improvement < SOLAR_ORIENTATION_IMPROVEMENT_MIN:
                return

        changed = (
            self._solar_orientation_azimuth is None
            or abs(
                ((self._solar_orientation_azimuth - best_orientation + 180.0) % 360.0)
                - 180.0
            )
            > 1e-9
        )
        self._solar_orientation_azimuth = best_orientation
        self._solar_orientation_score = best_score
        if np.isfinite(baseline_score) and baseline_score > 0.0:
            self._solar_orientation_confidence = max(
                0.0, min(1.0, 1.0 - best_score / baseline_score)
            )
        else:
            self._solar_orientation_confidence = None
        best_weights = self._apply_empirical_solar_gains(
            best_weights, examples, best_orientation
        )
        self._weights = self._clamp_directional_solar_weights(best_weights)
        self._ensure_weights_finite()
        self._cap_u_weights()
        if changed:
            self._clear_pending_predictions()

    def _fit_orientation_score(
        self, examples: list[dict[str, Any]], orientation: float | None
    ) -> tuple[np.ndarray, float]:
        """Fit ridge weights for one candidate orientation and return MSE."""
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for example in examples:
            feature = self._feature_from_orientation_example(example, orientation)
            target = self._coerce_optional_float(example.get("target_delta"))
            if feature is None or target is None:
                continue
            rows.append(feature)
            targets.append(target)
        if not rows:
            return self._weights.copy(), math.inf
        x = np.vstack(rows)
        y = np.array(targets, dtype=float)
        reg = max(float(self._l2_reg or 0.0), 1e-6)
        ridge = reg * np.eye(x.shape[1], dtype=float)
        ridge[-1, -1] = 0.0
        try:
            weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(x.T @ x + ridge) @ x.T @ y
        if not np.all(np.isfinite(weights)):
            return self._weights.copy(), math.inf
        residual = x @ weights - y
        score = float(np.mean(residual * residual))
        return weights, score

    def _feature_from_orientation_example(
        self, example: dict[str, Any], orientation: float | None
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
            t_hist.size != self._history_steps
            or out_hist.size != self._history_steps
            or u_hist.size != self._history_steps
            or not isinstance(samples, list)
        ):
            return None
        if orientation is None:
            solar_rows = [
                [coerce_solar_sample(sample).flux, 0.0, 0.0]
                for sample in samples[: self._history_steps]
            ]
        else:
            solar_rows = [
                list(
                    build_solar_orientation_features(
                        coerce_solar_sample(sample), orientation
                    )
                )
                for sample in samples[: self._history_steps]
            ]
        if len(solar_rows) != self._history_steps:
            return None
        solar_hist = np.array(solar_rows, dtype=float)
        return self._build_model_feature(
            self._sanitize_array(t_hist),
            self._sanitize_array(out_hist),
            self._sanitize_array(solar_hist),
            self._sanitize_array(u_hist),
        )

    def _example_solar_active(self, example: dict[str, Any]) -> bool:
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

    def _clamp_directional_solar_weights(self, weights: np.ndarray) -> np.ndarray:
        """Clamp learned directional solar weights to non-negative values."""
        clamped = np.array(weights, dtype=float)
        if not self._use_solar_feature:
            return clamped
        for facade_index in (0, 1):
            for index in self._directional_solar_weight_indices(facade_index):
                if index < clamped.size and clamped[index] < 0.0:
                    clamped[index] = 0.0
        return clamped

    def _apply_empirical_solar_gains(
        self,
        weights: np.ndarray,
        examples: list[dict[str, Any]],
        orientation: float,
    ) -> np.ndarray:
        """Stabilize directional gains with non-negative one-step solar evidence."""
        adjusted = np.array(weights, dtype=float)
        numerators = [0.0, 0.0]
        denominators = [0.0, 0.0]
        for example in examples:
            samples = example.get("solar_samples")
            if not isinstance(samples, list) or not samples:
                continue
            target = self._coerce_optional_float(example.get("target_delta"))
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
        if self._learning_model == LEARNING_MODEL_PHYSICAL:
            for facade_index, gain in enumerate(gains):
                for index in self._directional_solar_weight_indices(facade_index):
                    if index < adjusted.size:
                        adjusted[index] = gain * PHYSICAL_SOLAR_SCALE
            return adjusted
        solar_start = self._history_steps * 2
        for step in range(self._history_steps):
            row_start = solar_start + step * SOLAR_ORIENTATION_FEATURE_SIZE
            if row_start + 2 >= adjusted.size:
                break
            decay = 1.0 if step == 0 else 0.5**step
            adjusted[row_start + 1] = gains[0] * decay
            adjusted[row_start + 2] = gains[1] * decay
        return adjusted

    def _orientation_tie_score(
        self, weights: np.ndarray, examples: list[dict[str, Any]], orientation: float
    ) -> float:
        """Prefer tied orientations that use both facade channels positively."""
        if not self._use_solar_feature:
            return 0.0
        facade_weights = [0.0, 0.0]
        for facade_index in (0, 1):
            for index in self._directional_solar_weight_indices(facade_index):
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

    def _compute_update(
        self,
        actual_temp: float,
        prev_pred: float | None,
        prev_feature: list[float] | None,
        base_eta: float,
        eta: float,
    ) -> dict | None:
        """Compute a normalized gradient update from the previous rollout."""
        if prev_pred is None or prev_feature is None:
            _LOGGER.warning(
                "%s - Learning update skipped: missing previous prediction/feature",
                self._name,
            )
            return None

        try:
            base_eta = float(base_eta)
        except (TypeError, ValueError):
            base_eta = float(self._config.get("learning_rate", 0.01))
        if not np.isfinite(base_eta) or base_eta < 0.0:
            base_eta = float(self._config.get("learning_rate", 0.01))
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
            _LOGGER.warning(
                "%s - Learning update skipped: invalid previous prediction/feature",
                self._name,
            )
            return None

        if not np.isfinite(error) or not np.all(np.isfinite(feature)):
            _LOGGER.warning(
                "%s - Learning update skipped: non-finite error/features", self._name
            )
            return None

        denom = float(np.dot(feature, feature)) / float(feature.size)
        if not np.isfinite(denom) or denom <= 0:
            _LOGGER.warning(
                "%s - Learning update skipped: invalid feature norm", self._name
            )
            return None

        gradient = error * (feature / (denom + 1e-6))
        reg = 0.0
        if self._l2_reg:
            reg_weights = self._weights.copy()
            if reg_weights.size:
                reg_weights[-1] = 0.0
            reg = self._l2_reg * reg_weights
        delta = eta * (gradient - reg)
        self._weights = self._weights + delta
        self._ensure_weights_finite()
        return {
            "error": error,
            "errors": [{"step": 0, "error": error, "weight": 1.0}],
            "eta": eta,
            "base_eta": base_eta,
            "eta_scale": eta / base_eta if base_eta else None,
            "l2_reg": self._l2_reg,
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
        }

    def _compute_update_multi(
        self,
        updates: list[dict[str, float | int | list[float]]],
        base_eta: float,
        eta: float,
    ) -> dict | None:
        """Compute a normalized gradient update from multiple steps."""
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

        gradients = gradients / total_weight
        reg = 0.0
        if self._l2_reg:
            reg_weights = self._weights.copy()
            if reg_weights.size:
                reg_weights[-1] = 0.0
            reg = self._l2_reg * reg_weights
        delta = eta * (gradients - reg)
        self._weights = self._weights + delta
        self._ensure_weights_finite()
        mean_error = (
            float(np.mean([entry["error"] for entry in errors])) if errors else None
        )
        return {
            "error": mean_error,
            "errors": errors,
            "eta": eta,
            "base_eta": base_eta,
            "eta_scale": eta / base_eta if base_eta else None,
            "l2_reg": self._l2_reg,
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
        }

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
    ) -> dict:
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
        k_step_errors, k_step_updates = self._record_k_step_errors(t_room, step_index)
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
        self._update_history(
            t_room, t_out, u_actual, solar_current, current_solar_sample
        )
        t_hist, out_hist, solar_hist, u_hist = self._history_arrays()

        configured_steps = self._coerce_int(
            self._config.get("lookahead_steps"),
            DEFAULT_LEARNING_SWITCH_CONFIG["lookahead_steps"],
        )
        max_steps = min(
            DYNAMIC_LOOKAHEAD_MAX, max(DYNAMIC_LOOKAHEAD_MIN, configured_steps)
        )
        min_steps = min(DYNAMIC_LOOKAHEAD_MIN, max_steps)
        k_step_stats = self._k_step_error_stats(max_steps)
        steps, decision_reason = self._select_dynamic_lookahead(
            k_step_stats, min_steps, max_steps
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

        t_outs_max = self._build_forecast(t_out, t_out_forecast, max_steps)
        t_outs = t_outs_max[:steps]
        solar_outs_max = self._build_solar_forecast(
            solar_current,
            solar_forecast,
            current_solar_sample,
            (
                solar_sample_forecast
                if solar_sample_forecast is not None
                else [
                    coerce_solar_sample(value).as_dict()
                    for value in solar_profile_forecast or []
                ]
            ),
            max_steps,
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

        base_eta = self._coerce_float(
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

        update = self._compute_update_multi(k_step_updates, base_eta, eta)
        if update is None:
            update = self._compute_update(
                t_room, prev_pred, self._prev_feature, base_eta, eta
            )
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
        t_pred_off, _ = self._predict_next(
            t_hist, out_hist, solar_hist, u_hist, 0, current_solar_for_prediction
        )
        t_pred_on, _ = self._predict_next(
            t_hist, out_hist, solar_hist, u_hist, 1, current_solar_for_prediction
        )
        t_pred_actual = t_pred_on if u_actual == 1 else t_pred_off

        start_step, stop_step, plan_cost = self._find_start_stop(
            t_hist, out_hist, solar_hist, u_hist, t_set, t_outs, solar_outs
        )
        u_planned = 1 if start_step == 0 and stop_step > 0 else 0
        planned_states = [
            1 if start_step <= idx < stop_step else 0 for idx in range(len(t_outs))
        ]

        rollout_steps = max(1, len(t_outs))
        preds_next, feats_next = self._rollout(
            t_hist,
            out_hist,
            solar_hist,
            u_hist,
            t_outs,
            solar_outs,
            start_step,
            stop_step,
            rollout_steps,
            capture_features=True,
        )
        preds_for_error = preds_next
        if max_steps != rollout_steps:
            preds_for_error, _ = self._rollout(
                t_hist,
                out_hist,
                solar_hist,
                u_hist,
                t_outs_max,
                solar_outs_max,
                start_step,
                stop_step,
                max_steps,
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
        self._queue_k_step_predictions(preds_for_error, step_index, max_steps)
        self._queue_k_step_features(feats_next, step_index, max_steps)
        self._queue_k_step_baseline(t_room, step_index, max_steps)
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
        u_group_index = self._u_weight_start()
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
