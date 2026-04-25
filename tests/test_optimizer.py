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
    """Solar-enabled mode should add compact orientation solar channels."""
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
            solar_sample={"flux": 120.0, "azimuth": 270.0, "elevation": 30.0},
            solar_sample_forecast=[{"flux": 120.0, "azimuth": 270.0, "elevation": 30.0}]
            * 8,
            current_u=0,
            step_minutes=15,
        )

    snapshot = optimizer.snapshot_state()

    assert snapshot["model"] == "delta_zscore_solar_orientation"
    assert snapshot["history"]["solar"] == [120.0, 120.0, 120.0]
    assert snapshot["history"]["solar_samples"][0] == {
        "flux": 120.0,
        "azimuth": 270.0,
        "elevation": 30.0,
    }
    assert len(snapshot["weights"]) == 37


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
    assert len(target.snapshot_state()["weights"]) == 37


def test_restore_adapts_scalar_solar_snapshot_into_orientation_model() -> None:
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

    assert snapshot["model"] == "delta_zscore_solar_orientation"
    assert snapshot["history"]["solar"] == [150.0]
    assert snapshot["history"]["solar_samples"][0] == {
        "flux": 150.0,
        "azimuth": None,
        "elevation": None,
    }
    assert snapshot["weights"][12] == 0.25


def test_restore_adapts_solar_profile_snapshot_into_orientation_model() -> None:
    """Old 8-bin snapshots should infer an initial two-facade orientation."""
    legacy_snapshot = {
        "model": "delta_zscore_solar_profile",
        "weights": [0.0] * 73,
        "history": {
            "t_room": [20.0],
            "t_out": [8.0],
            "solar": [150.0],
            "solar_profile": [[150.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.0, 0.2, 0.0]],
            "u": [0],
        },
    }
    legacy_snapshot["weights"][12 + 5] = 0.7
    legacy_snapshot["weights"][12 + 7] = 0.2

    optimizer = LearningSwitchOptimizer(
        "zone_solar",
        config={"history_steps": 6, "lookahead_steps": 8, "use_solar_feature": True},
    )
    optimizer.restore_state(legacy_snapshot)
    snapshot = optimizer.snapshot_state()

    diagnostics = optimizer.solar_orientation_diagnostics()

    assert snapshot["model"] == "delta_zscore_solar_orientation"
    assert diagnostics["orientation_1"] == "south"
    assert diagnostics["orientation_2"] == "west"
    assert diagnostics["orientation_1_contribution_pct"] > 70.0


def test_solar_orientation_diagnostics_sort_positive_facade_contributions() -> None:
    """Diagnostics should expose the strongest learned facade first."""
    optimizer = LearningSwitchOptimizer(
        "zone_solar",
        config={"history_steps": 3, "lookahead_steps": 8, "use_solar_feature": True},
    )
    snapshot = optimizer.snapshot_state()
    weights = snapshot["weights"]
    # Solar features start after t_room and t_out histories. Per row:
    # [diffuse, learned facade, perpendicular facade].
    weights[3 * 2 + 1] = 0.1
    weights[3 * 2 + 2] = 0.9
    snapshot["weights"] = weights
    snapshot["solar_orientation_azimuth"] = 180.0
    snapshot["solar_orientation_confidence"] = 0.8
    optimizer.restore_state(snapshot)

    diagnostics = optimizer.solar_orientation_diagnostics()

    assert diagnostics["orientation_1"] == "west"
    assert diagnostics["orientation_1_contribution_pct"] == 90.0
    assert diagnostics["orientation_2"] == "south"
    assert diagnostics["orientation_2_contribution_pct"] == 10.0
    assert diagnostics["confidence"] == 0.8


def test_orientation_grid_search_selects_west_south_facades() -> None:
    """Synthetic solar-active data should learn the compact facade orientation."""
    optimizer = LearningSwitchOptimizer(
        "zone_solar",
        config={
            "history_steps": 3,
            "lookahead_steps": 8,
            "use_solar_feature": True,
            "learning_rate": 0.0,
        },
    )
    samples = [
        (20.00, 270.0, 1.00),
        (20.20, 180.0, 0.25),
        (20.25, 270.0, 1.00),
        (20.45, 180.0, 0.25),
        (20.50, 270.0, 1.00),
        (20.70, 180.0, 0.25),
        (20.75, 270.0, 1.00),
        (20.95, 180.0, 0.25),
        (21.00, 270.0, 1.00),
    ]
    for temp, azimuth, _effect in samples:
        optimizer.tick(
            t_room=temp,
            t_out=8.0,
            t_set=20.0,
            t_out_forecast=[8.0] * 8,
            solar=500.0,
            solar_forecast=[500.0] * 8,
            solar_sample={"flux": 500.0, "azimuth": azimuth, "elevation": 30.0},
            solar_sample_forecast=[
                {"flux": 500.0, "azimuth": azimuth, "elevation": 30.0}
            ]
            * 8,
            current_u=0,
            step_minutes=15,
        )

    diagnostics = optimizer.solar_orientation_diagnostics()

    assert diagnostics["orientation_1"] == "west"
    assert diagnostics["orientation_2"] == "south"
    assert diagnostics["orientation_1_contribution_pct"] > 70.0
