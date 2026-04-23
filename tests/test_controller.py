"""Tests for thermostat control orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from homeassistant.components.climate.const import HVACMode

from custom_components.learning_thermostat.controller import (
    ThermostatConfig,
    ThermostatController,
)


class StubOptimizer:
    """Small optimizer test double for controller tests."""

    def __init__(
        self,
        *,
        tick_result: dict | None = None,
        snapshot: dict | None = None,
    ) -> None:
        self.tick_result = tick_result or {"u_planned": 0}
        self.snapshot = snapshot or {"history": {}, "weights": [1.0]}
        self.restore_calls: list[dict] = []
        self.tick_calls: list[dict] = []

    def restore_state(self, state: dict | None) -> None:
        if state is not None:
            self.restore_calls.append(state)

    def snapshot_state(self) -> dict:
        return self.snapshot

    def tick(
        self,
        t_room: float,
        t_out: float,
        t_set: float,
        t_out_forecast: list[float],
        solar: float | None,
        solar_forecast: list[float],
        solar_profile: list[float] | None,
        solar_profile_forecast: list[list[float]],
        current_u: int,
        step_minutes: int,
        timestamp: float | None = None,
    ) -> dict:
        self.tick_calls.append(
            {
                "t_room": t_room,
                "t_out": t_out,
                "t_set": t_set,
                "t_out_forecast": t_out_forecast,
                "solar": solar,
                "solar_forecast": solar_forecast,
                "solar_profile": solar_profile,
                "solar_profile_forecast": solar_profile_forecast,
                "current_u": current_u,
                "step_minutes": step_minutes,
                "timestamp": timestamp,
            }
        )
        return dict(self.tick_result)


def _config() -> ThermostatConfig:
    return ThermostatConfig(
        name="Kitchen Floor",
        switch_entity_ids=("switch.floor_a", "switch.floor_b"),
        temperature_sensor_entity_id="sensor.kitchen_temperature",
        outdoor_entity_id="weather.home",
        legacy_forecast_entity_id=None,
        cycle_minutes=15,
        min_temperature=15.0,
        max_temperature=28.0,
        target_temperature=21.5,
        lookahead_steps=8,
        learning_config={"history_steps": 6, "lookahead_steps": 8},
    )


@pytest.mark.asyncio
async def test_run_control_cycle_turns_switches_off_when_hvac_is_off() -> None:
    """Off mode should bypass the optimizer and switch everything off."""
    fixed_now = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    optimizer = StubOptimizer()
    controller = ThermostatController(
        _config(),
        optimizer_factory=lambda _name, _config: optimizer,
        now=lambda: fixed_now,
    )
    controller.set_hvac_mode(HVACMode.OFF)

    switch_calls: list[bool] = []
    persisted_payloads: list[dict] = []

    async def turn_switches(turn_on: bool) -> None:
        switch_calls.append(turn_on)

    async def persist(payload: dict) -> None:
        persisted_payloads.append(payload)

    ran = await controller.run_control_cycle(
        force=False,
        forecast_temperatures=[],
        solar=None,
        solar_forecast=[],
        solar_profile=None,
        solar_profile_forecast=[],
        current_switch_active=True,
        turn_switches=turn_switches,
        persist=persist,
    )

    assert ran is True
    assert switch_calls == [False]
    assert persisted_payloads == [
        {
            "learning_state": None,
            "learning_last_tick": None,
            "current_temperature": None,
            "outside_temperature": None,
        }
    ]
    assert optimizer.tick_calls == []
    assert controller.learning_on_percent == 0.0


@pytest.mark.asyncio
async def test_run_control_cycle_delegates_to_optimizer_and_persists_state() -> None:
    """Controller should own orchestration around the optimizer."""
    fixed_now = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    optimizer = StubOptimizer(
        tick_result={"u_planned": 1, "predicted_temps": [20.0, 20.5]},
        snapshot={"history": {"t_room": [18.0]}, "weights": [1.0, 2.0]},
    )
    controller = ThermostatController(
        _config(),
        optimizer_factory=lambda _name, _config: optimizer,
        now=lambda: fixed_now,
    )
    controller.update_current_temperature(18.0)
    controller.update_outside_temperature(7.0)

    switch_calls: list[bool] = []
    persisted_payloads: list[dict] = []

    async def turn_switches(turn_on: bool) -> None:
        switch_calls.append(turn_on)

    async def persist(payload: dict) -> None:
        persisted_payloads.append(payload)

    ran = await controller.run_control_cycle(
        force=True,
        forecast_temperatures=[6.5, 6.0],
        solar=120.0,
        solar_forecast=[100.0, 80.0],
        solar_profile=[120.0, 0.0, 0.0, 0.0, 60.0, 90.0, 60.0, 0.0, 0.0],
        solar_profile_forecast=[
            [100.0, 0.0, 0.0, 0.0, 50.0, 75.0, 50.0, 0.0, 0.0],
            [80.0, 0.0, 0.0, 0.0, 40.0, 60.0, 40.0, 0.0, 0.0],
        ],
        current_switch_active=False,
        turn_switches=turn_switches,
        persist=persist,
    )

    assert ran is True
    assert optimizer.tick_calls == [
        {
            "t_room": 18.0,
            "t_out": 7.0,
            "t_set": 21.5,
            "t_out_forecast": [6.5, 6.0],
            "solar": 120.0,
            "solar_forecast": [100.0, 80.0],
            "solar_profile": [
                120.0,
                0.0,
                0.0,
                0.0,
                60.0,
                90.0,
                60.0,
                0.0,
                0.0,
            ],
            "solar_profile_forecast": [
                [100.0, 0.0, 0.0, 0.0, 50.0, 75.0, 50.0, 0.0, 0.0],
                [80.0, 0.0, 0.0, 0.0, 40.0, 60.0, 40.0, 0.0, 0.0],
            ],
            "current_u": 0,
            "step_minutes": 15,
            "timestamp": fixed_now.timestamp(),
        }
    ]
    assert switch_calls == [True]
    assert persisted_payloads == [
        {
            "learning_state": {"history": {"t_room": [18.0]}, "weights": [1.0, 2.0]},
            "learning_last_tick": fixed_now.isoformat(),
            "current_temperature": 18.0,
            "outside_temperature": 7.0,
        }
    ]
    assert controller.learning_result == {
        "u_planned": 1,
        "predicted_temps": [20.0, 20.5],
    }
    assert controller.learning_last_tick == fixed_now
    assert controller.learning_on_percent == 1.0


@pytest.mark.asyncio
async def test_run_control_cycle_falls_back_to_forecast_outside_temperature_for_weather_source() -> (
    None
):
    """Weather-backed entries should use the first forecast when no current outdoor value exists."""
    fixed_now = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    optimizer = StubOptimizer()
    controller = ThermostatController(
        _config(),
        optimizer_factory=lambda _name, _config: optimizer,
        now=lambda: fixed_now,
    )
    controller.update_current_temperature(18.0)

    async def turn_switches(_turn_on: bool) -> None:
        return None

    async def persist(_payload: dict) -> None:
        return None

    ran = await controller.run_control_cycle(
        force=True,
        forecast_temperatures=[6.5, 6.0],
        solar=None,
        solar_forecast=[],
        solar_profile=None,
        solar_profile_forecast=[],
        current_switch_active=False,
        turn_switches=turn_switches,
        persist=persist,
    )

    assert ran is True
    assert optimizer.tick_calls[0]["t_out"] == 6.5


def test_reset_learning_preserves_history_without_debug_noise() -> None:
    """Reset should keep durable history and drop transient diagnostics."""
    base_snapshot = {
        "history": {
            "t_room": [],
            "pred_abs_error": [1.2],
            "pred_timestamps": ["2026-04-21T12:00:00"],
            "k_step_abs_error": {"1": {"mean": 0.3, "count": 4}},
            "k_step_pred_improvement": {"1": {"mean": 0.1, "count": 4}},
        },
        "last_u": None,
        "last_plan": {"start_step": None, "stop_step": None},
        "weights": [1.0],
    }
    optimizer = StubOptimizer(snapshot=base_snapshot)
    controller = ThermostatController(
        _config(),
        optimizer_factory=lambda _name, _config: optimizer,
    )
    controller.restore(
        hvac_mode=HVACMode.HEAT,
        target_temperature=21.5,
        stored_payload={
            "learning_state": {
                "history": {
                    "t_room": [20.0],
                    "t_out": [7.0],
                    "pred_abs_error": [9.9],
                    "pred_timestamps": ["2026-04-20T12:00:00"],
                    "k_step_abs_error": {"1": {"mean": 0.9, "count": 2}},
                    "k_step_pred_improvement": {"1": {"mean": 0.4, "count": 2}},
                },
                "last_u": 1,
                "last_plan": {"start_step": 0, "stop_step": 2},
            },
            "learning_last_tick": None,
            "current_temperature": None,
            "outside_temperature": None,
        },
    )

    controller.reset_learning(preserve_history=True)

    assert controller.learning_state == {
        "history": {
            "t_room": [20.0],
            "t_out": [7.0],
        },
        "last_u": 1,
        "last_plan": {"start_step": 0, "stop_step": 2},
        "weights": [1.0],
    }
