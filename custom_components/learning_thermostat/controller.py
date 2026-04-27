"""Application service for thermostat control orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .const import (
    CONF_LEARNING_MODEL,
    DEFAULT_LEARNING_MODEL,
    LEARNING_MODEL_LABELS,
    LEARNING_MODEL_LINEAR,
)
from .optimizer import DEFAULT_LEARNING_SWITCH_CONFIG, LearningSwitchOptimizer
from .state import LearningEntryPayload, ThermostatMode, build_entry_payload


class Optimizer(Protocol):
    """Protocol for the learning optimizer dependency."""

    def restore_state(self, state: dict[str, Any] | None) -> None:
        """Restore one persisted optimizer snapshot."""

    def snapshot_state(self) -> dict[str, Any]:
        """Return the current optimizer snapshot."""

    def tick(
        self,
        t_room: float,
        t_out: float,
        t_set: float,
        t_out_forecast: list[float],
        solar: float | None,
        solar_forecast: list[float],
        solar_sample: dict[str, Any] | None,
        solar_sample_forecast: list[dict[str, Any]],
        current_u: int,
        step_minutes: int,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Advance the optimizer by one control interval."""


OptimizerFactory = Callable[[str, dict[str, Any]], Optimizer]
SwitchWriter = Callable[[bool], Awaitable[None]]
StatePersister = Callable[[LearningEntryPayload], Awaitable[None]]


def _utcnow() -> datetime:
    """Return the current UTC time without Home Assistant helpers."""
    return datetime.now(UTC)


def _default_optimizer_factory(name: str, config: dict[str, Any]) -> Optimizer:
    """Build the concrete optimizer dependency."""
    return LearningSwitchOptimizer(name, config=config)


@dataclass(frozen=True, slots=True)
class ThermostatConfig:
    """Stable runtime config for one thermostat entry."""

    name: str
    switch_entity_ids: tuple[str, ...]
    temperature_sensor_entity_id: str
    outdoor_entity_id: str | None
    legacy_forecast_entity_id: str | None
    cycle_minutes: int
    min_temperature: float
    max_temperature: float
    target_temperature: float
    lookahead_steps: int
    learning_config: dict[str, Any]

    @classmethod
    def from_mapping(cls, *, name: str, data: dict[str, Any]) -> ThermostatConfig:
        """Build config from one config-entry payload."""
        outdoor_entity_id = _coerce_optional_str(data.get("outdoor_entity_id"))
        legacy_forecast_entity_id = _coerce_optional_str(data.get("forecast_entity_id"))
        if outdoor_entity_id is None:
            outdoor_entity_id = _coerce_optional_str(
                data.get("outdoor_temperature_sensor_entity_id")
            )
        if (
            outdoor_entity_id is None
            and legacy_forecast_entity_id
            and legacy_forecast_entity_id.startswith("weather.")
        ):
            outdoor_entity_id = legacy_forecast_entity_id

        return cls(
            name=name,
            switch_entity_ids=tuple(
                str(value) for value in data["underlying_entity_ids"]
            ),
            temperature_sensor_entity_id=str(data["temperature_sensor_entity_id"]),
            outdoor_entity_id=outdoor_entity_id,
            legacy_forecast_entity_id=legacy_forecast_entity_id,
            cycle_minutes=int(data["cycle_minutes"]),
            min_temperature=float(data["min_temperature"]),
            max_temperature=float(data["max_temperature"]),
            target_temperature=float(data["target_temperature"]),
            lookahead_steps=int(data["learning_lookahead_steps"]),
            learning_config={
                **DEFAULT_LEARNING_SWITCH_CONFIG,
                "lookahead_steps": int(data["learning_lookahead_steps"]),
                "history_steps": int(data["learning_history_steps"]),
                "learning_model": _normalize_learning_model(
                    data.get(CONF_LEARNING_MODEL)
                ),
                "l2_reg": float(data["learning_l2_reg"]),
                "error_decay": float(data["learning_error_decay"]),
                "learning_rate": float(data["learning_rate"]),
                "w_energy": float(data["learning_energy"]),
                "use_solar_feature": bool(data.get("solar_entity_id")),
            },
        )


class ThermostatController:
    """Own thermostat behavior while adapters handle HA I/O."""

    def __init__(
        self,
        config: ThermostatConfig,
        *,
        optimizer_factory: OptimizerFactory = _default_optimizer_factory,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.config = config
        self._learning_config = dict(config.learning_config)
        self._optimizer_factory = optimizer_factory
        self._now = now
        self._optimizer: Optimizer | None = None
        self._learning_state: dict[str, Any] | None = None
        self._learning_result: dict[str, Any] | None = None
        self._learning_last_tick = None
        self._learning_on_percent: float | None = None
        self._current_temperature: float | None = None
        self._outside_temperature: float | None = None
        self._current_solar: float | None = None
        self._current_solar_sample: dict[str, Any] | None = None
        self._hvac_mode = ThermostatMode.HEAT
        self._target_temperature = config.target_temperature

    @property
    def current_temperature(self) -> float | None:
        """Return the current room temperature."""
        return self._current_temperature

    @property
    def target_temperature(self) -> float:
        """Return the target temperature."""
        return self._target_temperature

    @property
    def hvac_mode(self) -> ThermostatMode:
        """Return the current thermostat mode."""
        return self._hvac_mode

    @property
    def learning_state(self) -> dict[str, Any] | None:
        """Return the persisted learning snapshot."""
        return self._learning_state

    @property
    def learning_result(self) -> dict[str, Any] | None:
        """Return the latest optimizer result."""
        return self._learning_result

    @property
    def learning_last_tick(self):
        """Return when the controller last ran a learning cycle."""
        return self._learning_last_tick

    @property
    def outside_temperature(self) -> float | None:
        """Return the current outdoor temperature."""
        return self._outside_temperature

    @property
    def learning_on_percent(self) -> float | None:
        """Return the current learning heating decision."""
        return self._learning_on_percent

    @property
    def current_solar(self) -> float | None:
        """Return the current solar intensity."""
        return self._current_solar

    @property
    def current_solar_sample(self) -> dict[str, Any] | None:
        """Return the current raw solar sample."""
        return self._current_solar_sample

    @property
    def solar_orientation_diagnostics(self) -> dict[str, Any]:
        """Return optimizer-provided solar orientation diagnostics."""
        if self._optimizer is not None and hasattr(
            self._optimizer, "solar_orientation_diagnostics"
        ):
            return self._optimizer.solar_orientation_diagnostics()
        state = self._learning_state or {}
        diagnostics = state.get("solar_orientation")
        if isinstance(diagnostics, dict):
            return diagnostics
        return {"azimuth": None, "confidence": None, "state": "unknown"}

    def restore(
        self,
        *,
        hvac_mode: ThermostatMode,
        target_temperature: float | None,
        stored_payload: LearningEntryPayload | None,
    ) -> None:
        """Restore controller state from HA and persistence."""
        self._hvac_mode = hvac_mode
        if target_temperature is not None:
            self._target_temperature = float(target_temperature)
        if stored_payload is None:
            return

        learning_state = stored_payload.get("learning_state")
        if isinstance(learning_state, dict):
            self._learning_state = learning_state
        learning_last_tick = stored_payload.get("learning_last_tick")
        if isinstance(learning_last_tick, str):
            self._learning_last_tick = _parse_datetime(learning_last_tick)
        self._current_temperature = _coerce_float(
            stored_payload.get("current_temperature")
        )
        self._outside_temperature = _coerce_float(
            stored_payload.get("outside_temperature")
        )

    def set_hvac_mode(self, hvac_mode: ThermostatMode) -> None:
        """Set the active thermostat mode."""
        self._hvac_mode = hvac_mode

    def set_target_temperature(self, temperature: float) -> None:
        """Set the active target temperature."""
        self._target_temperature = float(temperature)

    def update_current_temperature(self, temperature: float | None) -> None:
        """Update the current room temperature."""
        self._current_temperature = temperature

    def update_outside_temperature(self, temperature: float | None) -> None:
        """Update the current outdoor temperature."""
        self._outside_temperature = temperature

    def update_current_solar(
        self, solar: float | None, solar_sample: dict[str, Any] | None = None
    ) -> None:
        """Update the current solar intensity."""
        self._current_solar = solar
        self._current_solar_sample = solar_sample

    def set_solar_available(self, enabled: bool) -> None:
        """Update solar feature availability for optimizer construction."""
        enabled = bool(enabled)
        if bool(self._learning_config.get("use_solar_feature")) == enabled:
            return

        snapshot = (
            self._optimizer.snapshot_state()
            if self._optimizer is not None
            else self._learning_state
        )
        self._learning_config = {
            **self._learning_config,
            "use_solar_feature": enabled,
        }
        if self._optimizer is None:
            return

        self._optimizer = self._optimizer_factory(
            self.config.name, self._learning_config
        )
        if snapshot is not None:
            self._optimizer.restore_state(snapshot)
        self._learning_state = self._optimizer.snapshot_state()

    def reset_learning(self, preserve_history: bool = True) -> None:
        """Reset persisted learning while keeping durable history when requested."""
        previous_state = self._learning_state if preserve_history else None
        optimizer = self._optimizer_factory(
            self.config.name, self._learning_config
        )
        self._optimizer = optimizer
        self._learning_result = None
        base_state = optimizer.snapshot_state()

        if preserve_history and isinstance(previous_state, dict):
            history = previous_state.get("history")
            if isinstance(history, dict):
                history_copy = dict(history)
                for key in (
                    "pred_abs_error",
                    "pred_timestamps",
                    "k_step_abs_error",
                    "k_step_pred_improvement",
                ):
                    history_copy.pop(key, None)
                base_state["history"] = history_copy
            for key in ("last_u", "last_plan"):
                if key in previous_state:
                    base_state[key] = previous_state.get(key)

        optimizer.restore_state(base_state)
        self._learning_state = optimizer.snapshot_state()

    def build_persisted_payload(self) -> LearningEntryPayload:
        """Return the persistence document for the current runtime state."""
        return build_entry_payload(
            learning_state=self._learning_state,
            learning_last_tick=(
                self._learning_last_tick.isoformat()
                if self._learning_last_tick is not None
                else None
            ),
            current_temperature=self._current_temperature,
            outside_temperature=self._outside_temperature,
        )

    async def run_control_cycle(
        self,
        *,
        force: bool,
        forecast_temperatures: list[float],
        solar: float | None,
        solar_forecast: list[float],
        solar_sample: dict[str, Any] | None,
        solar_sample_forecast: list[dict[str, Any]],
        current_switch_active: bool,
        turn_switches: SwitchWriter,
        persist: StatePersister,
    ) -> bool:
        """Run one thermostat control cycle."""
        if self._hvac_mode == ThermostatMode.OFF:
            self._learning_on_percent = 0.0
            await turn_switches(False)
            await persist(self.build_persisted_payload())
            return True

        if not force and self._learning_last_tick is not None:
            elapsed = self._now() - self._learning_last_tick
            if elapsed < timedelta(minutes=self.config.cycle_minutes):
                return False

        if self._current_temperature is None:
            return False

        outside_temperature = self._outside_temperature
        if outside_temperature is None and forecast_temperatures:
            outside_temperature = forecast_temperatures[0]
        if outside_temperature is None:
            outside_temperature = self._current_temperature

        optimizer = self._ensure_optimizer()
        now = self._now()
        result = optimizer.tick(
            self._current_temperature,
            outside_temperature,
            self._target_temperature,
            forecast_temperatures,
            solar,
            solar_forecast,
            solar_sample,
            solar_sample_forecast,
            current_u=1 if current_switch_active else 0,
            step_minutes=self.config.cycle_minutes,
            timestamp=now.timestamp(),
        )
        self._learning_result = result
        self._learning_state = optimizer.snapshot_state()
        self._learning_last_tick = now
        should_heat = bool(result.get("u_planned", 0))
        self._learning_on_percent = 1.0 if should_heat else 0.0

        await turn_switches(should_heat)
        await persist(self.build_persisted_payload())
        return True

    def _ensure_optimizer(self) -> Optimizer:
        """Build or restore the optimizer on demand."""
        if self._optimizer is None:
            self._optimizer = self._optimizer_factory(
                self.config.name, self._learning_config
            )
            if self._learning_state is not None:
                self._optimizer.restore_state(self._learning_state)
        return self._optimizer


def _coerce_optional_str(value: Any) -> str | None:
    """Return a trimmed string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_learning_model(value: Any) -> str:
    """Normalize a config-entry learning model identifier."""
    model = str(value or DEFAULT_LEARNING_MODEL)
    if model not in LEARNING_MODEL_LABELS:
        return LEARNING_MODEL_LINEAR
    return model


def _coerce_float(value: Any) -> float | None:
    """Coerce a float-like value from persisted payloads."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: str) -> datetime | None:
    """Parse persisted ISO datetimes without depending on Home Assistant."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
