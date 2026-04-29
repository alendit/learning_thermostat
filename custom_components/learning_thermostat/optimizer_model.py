"""Concrete optimizer model component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .const import LEARNING_MODEL_PHYSICAL
from .optimizer_features import (
    build_model_feature,
    default_weights,
    heat_feature_count,
    init_weights,
    u_weight_start,
)


@dataclass(frozen=True, slots=True)
class OptimizerModelComponent:
    """Own feature layout and weight interpretation for one optimizer model."""

    learning_model: str
    use_solar_feature: bool

    @property
    def is_physical(self) -> bool:
        """Return whether this is the physical model family."""
        return self.learning_model == LEARNING_MODEL_PHYSICAL

    def default_weights(self, *, history_steps: int) -> np.ndarray:
        """Return the zero baseline for this model layout."""
        return default_weights(
            learning_model=self.learning_model,
            history_steps=history_steps,
            use_solar_feature=self.use_solar_feature,
        )

    def init_weights(self, weights_init: Any, *, history_steps: int) -> np.ndarray:
        """Build initial weights for this model layout."""
        return init_weights(
            weights_init,
            learning_model=self.learning_model,
            history_steps=history_steps,
            use_solar_feature=self.use_solar_feature,
        )

    def u_weight_start(self, *, history_steps: int) -> int:
        """Return the start index of heating-control weights."""
        return u_weight_start(
            learning_model=self.learning_model,
            history_steps=history_steps,
            use_solar_feature=self.use_solar_feature,
        )

    def heat_feature_count(self, *, history_steps: int) -> int:
        """Return the number of heating-control features."""
        return heat_feature_count(
            learning_model=self.learning_model,
            history_steps=history_steps,
        )

    def build_feature(
        self,
        *,
        history_decay_vector: np.ndarray,
        feature_step_minutes: float,
        t_hist: np.ndarray,
        out_hist: np.ndarray,
        solar_hist: np.ndarray | None,
        u_hist: np.ndarray,
    ) -> np.ndarray:
        """Build a prediction feature vector for this model."""
        return build_model_feature(
            learning_model=self.learning_model,
            history_decay_vector=history_decay_vector,
            feature_step_minutes=feature_step_minutes,
            t_hist=t_hist,
            out_hist=out_hist,
            solar_hist=solar_hist,
            u_hist=u_hist,
        )
