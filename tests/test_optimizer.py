"""Tests for the learning optimizer extraction."""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.learning_thermostat.const import LEARNING_MODEL_PHYSICAL
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

    assert snapshot["model"] == "Linear"
    assert snapshot["model_id"] == "linear_solar_orientation"
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

    assert snapshot["model"] == "Linear"
    assert snapshot["model_id"] == "linear_solar_orientation"
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

    assert snapshot["model"] == "Linear"
    assert snapshot["model_id"] == "linear_solar_orientation"
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


def test_physical_no_solar_snapshot_has_fixed_weight_count() -> None:
    """Physical no-solar mode should use compact fixed-size features."""
    optimizer = LearningSwitchOptimizer(
        "zone_physical",
        config={
            "history_steps": 12,
            "lookahead_steps": 8,
            "learning_model": LEARNING_MODEL_PHYSICAL,
        },
    )

    optimizer.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 8,
        current_u=0,
        step_minutes=15,
    )
    snapshot = optimizer.snapshot_state()

    assert snapshot["model"] == "Physical"
    assert snapshot["model_id"] == "physical"
    assert len(snapshot["weights"]) == 10


def test_physical_solar_snapshot_has_fixed_weight_count() -> None:
    """Physical solar mode should add nine filtered solar features."""
    optimizer = LearningSwitchOptimizer(
        "zone_physical_solar",
        config={
            "history_steps": 12,
            "lookahead_steps": 8,
            "learning_model": LEARNING_MODEL_PHYSICAL,
            "use_solar_feature": True,
        },
    )

    optimizer.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 8,
        solar=200.0,
        solar_forecast=[200.0] * 8,
        solar_sample={"flux": 200.0, "azimuth": 270.0, "elevation": 30.0},
        solar_sample_forecast=[{"flux": 200.0, "azimuth": 270.0, "elevation": 30.0}]
        * 8,
        current_u=0,
        step_minutes=15,
    )
    snapshot = optimizer.snapshot_state()

    assert snapshot["model"] == "Physical"
    assert snapshot["model_id"] == "physical_solar_orientation"
    assert len(snapshot["weights"]) == 19


def test_physical_features_use_fixed_scaling_trends_and_filters() -> None:
    """Physical features should avoid local z-scoring."""
    optimizer = LearningSwitchOptimizer(
        "zone_physical",
        config={"history_steps": 4, "learning_model": LEARNING_MODEL_PHYSICAL},
    )
    optimizer._feature_step_minutes = 60.0

    features = optimizer._build_model_feature(
        t_hist=optimizer._sanitize_array(np.array([22.0, 21.0, 20.0, 19.0])),
        out_hist=optimizer._sanitize_array(np.array([12.0, 10.0, 8.0, 6.0])),
        solar_hist=None,
        u_hist=optimizer._sanitize_array(np.array([1.0, 0.0, 0.0, 0.0])),
    )

    assert features[:6].tolist() == pytest.approx([0.2, 0.1, 0.5, 0.5, 0.75, 0.6])
    assert features[6:9].tolist() == pytest.approx(
        [0.5, 1.0 - 0.5 ** (1.0 / 3.0), 1.0 - 0.5 ** (1.0 / 6.0)]
    )
    assert features[-1] == 1.0


def test_switching_linear_snapshot_to_physical_resets_weights_and_preserves_history() -> (
    None
):
    """Switching model families should keep raw history but avoid weight migration."""
    linear = LearningSwitchOptimizer(
        "zone_switch",
        config={"history_steps": 6, "lookahead_steps": 8},
    )
    linear.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 8,
        current_u=0,
        step_minutes=15,
    )
    snapshot = linear.snapshot_state()
    snapshot["weights"] = [0.2] * len(snapshot["weights"])
    snapshot["prev_feature"] = [0.1] * len(snapshot["weights"])

    physical = LearningSwitchOptimizer(
        "zone_switch",
        config={
            "history_steps": 6,
            "lookahead_steps": 8,
            "learning_model": LEARNING_MODEL_PHYSICAL,
        },
    )
    physical.restore_state(snapshot)
    restored = physical.snapshot_state()

    assert restored["history"]["t_room"] == snapshot["history"]["t_room"]
    assert restored["weights"] == [0.0] * 10
    assert restored["prev_feature"] is None


def test_physical_default_heating_seed_matches_existing_cold_start_delta() -> None:
    """The Physical fast heat seed should preserve the old ON/OFF delta."""
    optimizer = LearningSwitchOptimizer(
        "zone_physical",
        config={
            "history_steps": 6,
            "lookahead_steps": 8,
            "learning_model": LEARNING_MODEL_PHYSICAL,
            "default_u_gain_per_hour": 0.1,
            "learning_rate": 0.0,
        },
    )

    result = optimizer.tick(
        t_room=20.0,
        t_out=5.0,
        t_set=22.0,
        t_out_forecast=[5.0] * 8,
        current_u=0,
        step_minutes=15,
    )

    assert result["t_pred_on"] - result["t_pred_off"] == pytest.approx(
        0.1 * 15.0 / 60.0,
        abs=1e-5,
    )


def test_physical_solar_orientation_diagnostics_rank_filter_weights() -> None:
    """Physical diagnostics should sum positive facade weights across half-lives."""
    optimizer = LearningSwitchOptimizer(
        "zone_physical_solar",
        config={
            "history_steps": 6,
            "lookahead_steps": 8,
            "learning_model": LEARNING_MODEL_PHYSICAL,
            "use_solar_feature": True,
        },
    )
    snapshot = optimizer.snapshot_state()
    weights = snapshot["weights"]
    weights[9] = 0.2
    weights[10] = 0.3
    weights[12] = 1.0
    snapshot["weights"] = weights
    snapshot["solar_orientation_azimuth"] = 180.0
    snapshot["solar_orientation_confidence"] = 0.6
    optimizer.restore_state(snapshot)

    diagnostics = optimizer.solar_orientation_diagnostics()

    assert diagnostics["orientation_1"] == "west"
    assert diagnostics["orientation_1_contribution_pct"] == pytest.approx(66.667)
    assert diagnostics["orientation_2"] == "south"
    assert diagnostics["confidence"] == 0.6
