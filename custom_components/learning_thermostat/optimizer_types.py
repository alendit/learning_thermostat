"""Typed internal optimizer payload contracts."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class SolarOrientationDiagnostics(TypedDict):
    """Diagnostic payload for the learned solar orientation."""

    azimuth: float | None
    confidence: float | None
    orientation_1: str | None
    orientation_1_azimuth: float | None
    orientation_1_contribution_pct: float | None
    orientation_2: str | None
    orientation_2_azimuth: float | None
    orientation_2_contribution_pct: float | None
    state: str


class LearningErrorEntry(TypedDict):
    """One error contribution in a learning update payload."""

    step: int
    error: float
    weight: float


class LearningUpdatePayload(TypedDict):
    """Stable diagnostic payload emitted after a weight update."""

    error: float | None
    errors: list[LearningErrorEntry]
    eta: float
    base_eta: float
    eta_scale: float | None
    l2_reg: float
    grad_norm: float | None
    abs_update_max: float | None
    delta: list[float] | None
    steps_used: int


class KStepUpdateInput(TypedDict):
    """Matured k-step prediction input for a learning update."""

    horizon: int
    error: float
    feature: list[float]


class LookaheadBounds(TypedDict):
    """Dynamic lookahead bounds included in optimizer tick diagnostics."""

    min: int
    max: int


class OptimizerTickResult(TypedDict, total=False):
    """Stable optimizer tick result payload.

    The empty dict is valid for invalid tick inputs, so all keys are optional.
    """

    t_room: float
    t_out: float
    t_set: float
    theta1: float | None
    theta2: float | None
    weights: list[float]
    history_steps: int
    history_len: int
    t_pred_on: float
    t_pred_off: float
    t_pred_actual: float
    u_planned: int
    u_actual: int
    start_step: int
    stop_step: int
    lookahead_steps: int
    step_minutes: int | None
    forecast_temps: list[float]
    forecast_solar: list[float]
    forecast_solar_samples: list[dict[str, Any]]
    predicted_temps: list[float]
    planned_states: list[int]
    plan_cost: float
    tick_duration_ms: float
    error: float | None
    update: LearningUpdatePayload | None
    eta: float
    base_eta: float
    k_step_error: dict[int, float]
    k_step_error_mean: dict[int, float | None]
    k_step_pred_improvement: dict[int, float | None]
    lookahead_selected: int
    lookahead_bounds: LookaheadBounds
    lookahead_reason: str
    step_count: int
    weights_reset: bool
    solar_orientation: SolarOrientationDiagnostics


OptimizerSnapshot = dict[str, Any]
SolarSamplePayload = dict[str, Any]
SolarSampleForecast = list[dict[str, Any]]
