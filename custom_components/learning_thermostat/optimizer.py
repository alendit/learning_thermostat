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
import numpy as np

DYNAMIC_LOOKAHEAD_MIN = 12
DYNAMIC_LOOKAHEAD_MAX = 24
DYNAMIC_LOOKAHEAD_TOLERANCE = 0.2
LEARNING_RATE_TARGET_ERROR = 0.2
LEARNING_RATE_SCALE_MIN = 0.2
LEARNING_RATE_SCALE_MAX = 2.0
MULTI_STEP_UPDATE_HORIZON = 3
Z_SCORE_MIN_STD = 0.2
U_WEIGHT_MIN = 1e-6

DEFAULT_LEARNING_SWITCH_CONFIG = {
    "lookahead_steps": 24,
    "learning_rate": 0.005,
    "l2_reg": 0.001,
    "w_energy": 0.01,
    "history_steps": 12,
    "history_decay": 1.0,
    "error_decay": 0.9,
    "default_u_gain_per_hour": 0.1,
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

        # Optional starting weights.
        self._weights_init = self._config.get("weights_init")
        # Whether the model includes a solar irradiance feature channel.
        self._use_solar_feature = bool(self._config.get("use_solar_feature"))
        # Current model weights, shape: (groups * history_steps + 1,) = [t_hist, out_hist, (solar_hist), u_hist, bias].
        self._weights = self._init_weights(self._weights_init)
        # Recent room temperatures (latest first).
        self._t_hist: list[float] = []
        # Recent outdoor temperatures (latest first).
        self._out_hist: list[float] = []
        # Recent solar irradiance values (latest first).
        self._solar_hist: list[float] = []
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
        return {
            "weights": self._weights.tolist(),
            "model": (
                "delta_zscore_solar" if self._use_solar_feature else "delta_zscore"
            ),
            "history": {
                "t_room": list(self._t_hist),
                "t_out": list(self._out_hist),
                "solar": list(self._solar_hist),
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
            },
        }

    def restore_state(self, state: dict | None) -> None:
        """Restore the internal state defensively from a snapshot."""
        try:
            if not isinstance(state, dict):
                return

            model_type = state.get("model")
            model_matches = model_type in {"delta_zscore", "delta_zscore_solar"}
            weights = state.get("weights")
            if model_matches and weights is not None:
                restored = np.array([float(v) for v in weights], dtype=float)
                if restored.size == self._weights.size:
                    if np.allclose(restored, 0.0) or not np.all(np.isfinite(restored)):
                        restored = self._init_weights(self._weights_init)
                    self._weights = restored
                else:
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
        except Exception as exc:  # noqa: BLE001 - keep restore resilient
            _LOGGER.warning("%s - Learning state restore failed: %s", self._name, exc)

    def _adapt_weights(
        self, restored: np.ndarray, model_type: str | None
    ) -> np.ndarray | None:
        """Adapt weights saved with a different history length."""
        size = restored.size
        if size < 4:
            return None
        old_group_count = self._infer_group_count(model_type, size)
        if old_group_count is None:
            return None
        old_steps = int((size - 1) / old_group_count)
        if old_steps <= 0:
            return None
        new_steps = self._history_steps
        new_group_count = self._feature_group_count()
        new_weights = np.zeros(new_group_count * new_steps + 1, dtype=float)
        copy_len = min(old_steps, new_steps)
        semantic_order_old = ["t_room", "t_out"]
        if old_group_count == 4:
            semantic_order_old.append("solar")
        semantic_order_old.append("u")
        semantic_order_new = ["t_room", "t_out"]
        if new_group_count == 4:
            semantic_order_new.append("solar")
        semantic_order_new.append("u")

        old_group_index = {key: idx for idx, key in enumerate(semantic_order_old)}
        new_group_index = {key: idx for idx, key in enumerate(semantic_order_new)}
        for key in ("t_room", "t_out", "solar", "u"):
            if key not in old_group_index or key not in new_group_index:
                continue
            start_old = old_group_index[key] * old_steps
            end_old = start_old + copy_len
            start_new = new_group_index[key] * new_steps
            end_new = start_new + copy_len
            new_weights[start_new:end_new] = restored[start_old:end_old]
        bias = restored[-1]
        if np.isfinite(bias):
            new_weights[-1] = float(bias)
        if np.allclose(new_weights, 0.0) or not np.all(np.isfinite(new_weights)):
            return None
        return new_weights

    def _infer_group_count(self, model_type: str | None, size: int) -> int | None:
        """Infer feature-group count from the saved model metadata."""
        if model_type == "delta_zscore":
            return 3 if (size - 1) % 3 == 0 else None
        if model_type == "delta_zscore_solar":
            return 4 if (size - 1) % 4 == 0 else None
        if (size - 1) % 4 == 0:
            return 4
        if (size - 1) % 3 == 0:
            return 3
        return None

    def _feature_group_count(self) -> int:
        """Return the number of feature history groups in the model."""
        return 4 if self._use_solar_feature else 3

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
        size = self._feature_group_count() * self._history_steps + 1
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
        u_index = self._history_steps * (3 if self._use_solar_feature else 2)
        if u_index < self._weights.size:
            self._weights[u_index] = float(delta)
            return True
        return False

    def _cap_u_weights(self) -> bool:
        """Clamp on/off weights to a small positive minimum."""
        start = self._history_steps * (3 if self._use_solar_feature else 2)
        end = start + self._history_steps
        if start >= self._weights.size:
            return False
        floor = U_WEIGHT_MIN
        if not np.isfinite(floor) or floor <= 0.0:
            return False
        before = self._weights[start:end].copy()
        self._weights[start:end] = np.maximum(self._weights[start:end], floor)
        return not np.allclose(before, self._weights[start:end], rtol=0.0, atol=1e-12)

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
        self, t_room: float, t_out: float, u_actual: int, solar: float | None
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
        self._u_hist.insert(0, int(u_actual))
        if len(self._t_hist) > self._history_steps:
            self._t_hist = self._t_hist[: self._history_steps]
        if len(self._out_hist) > self._history_steps:
            self._out_hist = self._out_hist[: self._history_steps]
        if len(self._solar_hist) > self._history_steps:
            self._solar_hist = self._solar_hist[: self._history_steps]
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
            solar_pad = self._pad_sequence(
                self._solar_hist, self._solar_hist[-1] if self._solar_hist else 0.0
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
            solar_mean = float(np.mean(solar_hist)) if solar_hist.size else 0.0
            solar_std = float(np.std(solar_hist)) if solar_hist.size else 1.0
            if not np.isfinite(solar_mean):
                solar_mean = 0.0
            if not np.isfinite(solar_std) or solar_std < Z_SCORE_MIN_STD:
                solar_std = Z_SCORE_MIN_STD
            solar_norm = (solar_hist - solar_mean) / solar_std
        return (
            self._sanitize_array(t_norm),
            self._sanitize_array(out_norm),
            self._sanitize_array(solar_norm) if solar_norm is not None else None,
            u_hist,
        )

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
        decay = self._history_decay_vector
        t_hist = t_hist * decay
        out_hist = out_hist * decay
        parts = [t_hist, out_hist]
        if solar_hist is not None:
            parts.append(solar_hist * decay)
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
        solar_value: float | None,
    ) -> tuple[float, np.ndarray]:
        """Predict next room temperature for a single step and return features."""
        u_hist_step = np.concatenate(
            (np.array([float(u_value)], dtype=float), u_hist[:-1])
        )
        solar_hist_step: np.ndarray | None = None
        if solar_hist is not None:
            solar_step = (
                float(solar_value)
                if solar_value is not None and np.isfinite(float(solar_value))
                else float(solar_hist[0])
            )
            solar_hist_step = np.concatenate(
                (np.array([solar_step], dtype=float), solar_hist[:-1])
            )
        t_feat, out_feat, solar_feat, u_feat = self._transform_histories(
            t_hist, out_hist, solar_hist_step, u_hist_step
        )
        x = self._build_feature(t_feat, out_feat, solar_feat, u_feat)
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
                (np.array([float(solar_hist[0])], dtype=float), solar_hist[:-1])
            )
        t_feat, out_feat, solar_feat, u_feat = self._transform_histories(
            t_hist, out_hist, solar_hist_step, u_hist_step
        )
        x = self._build_feature(t_feat, out_feat, solar_feat, u_feat)
        if x.size != self._weights.size:
            return np.zeros_like(self._weights)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(self._weights)):
            return np.zeros_like(self._weights)
        return np.abs(x * self._weights)

    def _rollout(
        self,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
        t_outs: list[float],
        solar_forecast: list[float],
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
                next_solar = (
                    float(solar_value)
                    if solar_value is not None
                    else float(solar_hist_cur[0])
                )
                solar_hist_cur = np.concatenate(
                    (np.array([next_solar], dtype=float), solar_hist_cur[:-1])
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
        solar_forecast: list[float],
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
        solar_forecast: list[float],
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
        self, solar: float | None, solar_forecast: list[float] | None, steps: int
    ) -> list[float]:
        """Normalize solar forecast input to the lookahead length with fallbacks."""
        if not self._use_solar_feature or steps <= 0:
            return []

        values: list[float] = []
        if isinstance(solar_forecast, list) and solar_forecast:
            for value in solar_forecast:
                try:
                    solar_val = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(solar_val):
                    continue
                values.append(solar_val)

        if not values:
            if solar is not None and np.isfinite(float(solar)):
                solar_current = float(solar)
                values = [solar_current] * steps
            elif self._last_forecast_solar is not None:
                values = [self._last_forecast_solar] * steps
            else:
                values = [0.0] * steps
        elif len(values) < steps:
            values.extend([values[-1]] * (steps - len(values)))
        elif len(values) > steps:
            values = values[:steps]

        if values:
            self._last_forecast_solar = values[-1]
        return values

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

        weights_reset = self._ensure_weights_finite()
        if self._apply_default_u_gain(step_minutes):
            weights_reset = True
        self._cap_u_weights()

        step_index = self._step_count
        k_step_errors, k_step_updates = self._record_k_step_errors(t_room, step_index)

        solar_current: float | None = None
        if solar is not None:
            try:
                solar_current = float(solar)
            except (TypeError, ValueError):
                solar_current = None
            if solar_current is not None and not np.isfinite(solar_current):
                solar_current = None

        self._update_history(t_room, t_out, u_actual, solar_current)
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
            solar_current, solar_forecast, max_steps
        )
        solar_outs = solar_outs_max[:steps]

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

        self._last_u = u_planned
        self._step_count += 1
        self._last_plan = {"start_step": start_step, "stop_step": stop_step}

        theta_outside = None
        theta_inside = None
        u_group_index = 3 if self._use_solar_feature else 2
        if self._weights.size >= self._history_steps * (u_group_index + 1):
            theta_outside = float(self._weights[self._history_steps])
            theta_inside = float(self._weights[self._history_steps * u_group_index])

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
            "forecast_solar": solar_outs,
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
        }
