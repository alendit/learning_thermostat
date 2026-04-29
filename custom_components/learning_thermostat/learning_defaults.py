"""Stable learning configuration defaults."""

from __future__ import annotations

from .const import DEFAULT_LEARNING_MODEL

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
