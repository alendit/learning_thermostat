"""Tests for the learning optimizer extraction."""

from __future__ import annotations

from custom_components.learning_thermostat.optimizer import LearningSwitchOptimizer


def test_snapshot_restore_round_trip() -> None:
    """The optimizer should restore a captured state snapshot."""
    optimizer = LearningSwitchOptimizer(
        "zone_a",
        config={"history_steps": 6, "lookahead_steps": 8},
    )

    for _ in range(4):
        optimizer.tick(
            t_room=19.0,
            t_out=8.0,
            t_set=21.0,
            t_out_forecast=[8.0] * 8,
            current_u=0,
            step_minutes=15,
        )

    snapshot = optimizer.snapshot_state()

    restored = LearningSwitchOptimizer(
        "zone_a",
        config={"history_steps": 6, "lookahead_steps": 8},
    )
    restored.restore_state(snapshot)

    assert (
        restored.snapshot_state()["history"]["t_room"] == snapshot["history"]["t_room"]
    )
    assert restored.snapshot_state()["weights"] == snapshot["weights"]


def test_restore_adapts_history_length() -> None:
    """The optimizer should adapt persisted weights to a new history length."""
    source = LearningSwitchOptimizer(
        "zone_b",
        config={"history_steps": 12, "lookahead_steps": 12},
    )
    source.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 12,
        current_u=0,
        step_minutes=15,
    )
    snapshot = source.snapshot_state()

    target = LearningSwitchOptimizer(
        "zone_b",
        config={"history_steps": 6, "lookahead_steps": 12},
    )
    target.restore_state(snapshot)
    result = target.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 12,
        current_u=0,
        step_minutes=15,
    )

    assert result["history_steps"] == 6
    assert len(target.snapshot_state()["weights"]) == 19


def test_solar_enabled_snapshot_tracks_solar_history_and_expands_weights() -> None:
    """Solar-enabled mode should add a directional solar profile channel."""
    optimizer = LearningSwitchOptimizer(
        "zone_solar",
        config={"history_steps": 6, "lookahead_steps": 8, "use_solar_feature": True},
    )

    for _ in range(3):
        optimizer.tick(
            t_room=19.0,
            t_out=8.0,
            t_set=21.0,
            t_out_forecast=[8.0] * 8,
            solar=120.0,
            solar_forecast=[120.0] * 8,
            solar_profile=[120.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 30.0, 20.0],
            solar_profile_forecast=[
                [120.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 30.0, 20.0]
            ]
            * 8,
            current_u=0,
            step_minutes=15,
        )

    snapshot = optimizer.snapshot_state()

    assert snapshot["model"] == "delta_zscore_solar_profile"
    assert snapshot["history"]["solar"] == [120.0, 120.0, 120.0]
    assert snapshot["history"]["solar_profile"][0] == [
        120.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        40.0,
        30.0,
        20.0,
    ]
    assert len(snapshot["weights"]) == 73


def test_restore_adapts_non_solar_snapshot_into_solar_model() -> None:
    """Old non-solar snapshots should restore into the larger solar-aware model."""
    source = LearningSwitchOptimizer(
        "zone_base",
        config={"history_steps": 6, "lookahead_steps": 8},
    )
    source.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 8,
        current_u=0,
        step_minutes=15,
    )
    snapshot = source.snapshot_state()

    target = LearningSwitchOptimizer(
        "zone_solar",
        config={"history_steps": 6, "lookahead_steps": 8, "use_solar_feature": True},
    )
    target.restore_state(snapshot)

    result = target.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 8,
        solar=100.0,
        solar_forecast=[100.0] * 8,
        current_u=0,
        step_minutes=15,
    )

    assert result["history_steps"] == 6
    assert len(target.snapshot_state()["weights"]) == 73


def test_restore_adapts_scalar_solar_snapshot_into_solar_profile_model() -> None:
    """Old scalar solar snapshots should map solar weights into the diffuse channel."""
    legacy_snapshot = {
        "model": "delta_zscore_solar",
        "weights": [0.0] * 25,
        "history": {
            "t_room": [20.0],
            "t_out": [8.0],
            "solar": [150.0],
            "u": [0],
        },
    }
    legacy_snapshot["weights"][12] = 0.25

    target = LearningSwitchOptimizer(
        "zone_solar",
        config={"history_steps": 6, "lookahead_steps": 8, "use_solar_feature": True},
    )
    target.restore_state(legacy_snapshot)
    snapshot = target.snapshot_state()

    assert snapshot["model"] == "delta_zscore_solar_profile"
    assert snapshot["history"]["solar"] == [150.0]
    assert snapshot["history"]["solar_profile"][0] == [
        150.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert snapshot["weights"][12] == 0.25


def test_solar_orientation_diagnostics_select_strongest_positive_bin() -> None:
    """Diagnostics should expose the dominant learned solar direction."""
    optimizer = LearningSwitchOptimizer(
        "zone_solar",
        config={"history_steps": 3, "lookahead_steps": 8, "use_solar_feature": True},
    )
    snapshot = optimizer.snapshot_state()
    weights = snapshot["weights"]
    # Solar profile weights start after t_room and t_out histories. The first
    # solar profile row is [diffuse, N, NE, E, SE, S, SW, W, NW].
    weights[3 * 2 + 5] = 0.8
    snapshot["weights"] = weights
    optimizer.restore_state(snapshot)

    diagnostics = optimizer.solar_orientation_diagnostics()

    assert diagnostics["azimuth"] == 180.0
    assert diagnostics["confidence"] == 1.0
    assert diagnostics["exposure_bins"][180.0] == 0.8
