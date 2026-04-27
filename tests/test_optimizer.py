"""Tests for the learning optimizer extraction."""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.learning_thermostat.const import LEARNING_MODEL_PHYSICAL
from custom_components.learning_thermostat.optimizer_features import (
    build_model_feature,
    sanitize_array,
)
from custom_components.learning_thermostat.optimizer import (
    MODEL_SCHEMA_VERSION,
    LearningSwitchOptimizer,
)
from custom_components.learning_thermostat.optimizer_history import (
    build_history_arrays,
    update_latest_first_histories,
)
from custom_components.learning_thermostat.optimizer_learning import (
    compute_multi_step_update,
    compute_single_step_update,
)
from custom_components.learning_thermostat.optimizer_tracking import (
    queue_k_step_baseline,
    queue_k_step_features,
    queue_k_step_predictions,
    record_k_step_errors,
    select_dynamic_lookahead,
)


def test_snapshot_contract_keys_and_model_metadata_for_supported_models() -> None:
    """The persisted optimizer snapshot contract should stay stable."""
    cases = [
        ({}, "Linear", "linear", 19),
        ({"use_solar_feature": True}, "Linear", "linear_solar_orientation", 37),
        ({"learning_model": LEARNING_MODEL_PHYSICAL}, "Physical", "physical", 10),
        (
            {"learning_model": LEARNING_MODEL_PHYSICAL, "use_solar_feature": True},
            "Physical",
            "physical_solar_orientation",
            19,
        ),
    ]

    for config, model_label, model_id, weight_count in cases:
        optimizer = LearningSwitchOptimizer(
            "zone_contract",
            config={"history_steps": 6, "lookahead_steps": 8, **config},
        )

        snapshot = optimizer.snapshot_state()

        assert set(snapshot) == {
            "weights",
            "model",
            "model_id",
            "model_schema_version",
            "history",
            "last_u",
            "last_plan",
            "solar_orientation_azimuth",
            "solar_orientation_confidence",
            "solar_orientation_score",
            "eta",
            "base_eta",
            "prev_feature",
            "config",
            "solar_orientation",
        }
        assert set(snapshot["history"]) == {
            "t_room",
            "t_out",
            "solar",
            "solar_samples",
            "u",
            "pred_next",
            "pred_abs_error",
            "pred_timestamps",
            "weight_abs_update",
            "weight_delta_abs_latest",
            "k_step_abs_error",
            "k_step_pred_improvement",
        }
        assert set(snapshot["config"]) == {
            "lookahead_steps",
            "learning_rate",
            "l2_reg",
            "w_energy",
            "history_steps",
            "history_decay",
            "error_decay",
            "default_u_gain_per_hour",
            "use_solar_feature",
            "learning_model",
        }
        assert snapshot["model"] == model_label
        assert snapshot["model_id"] == model_id
        assert snapshot["model_schema_version"] == MODEL_SCHEMA_VERSION
        assert len(snapshot["weights"]) == weight_count


def test_deterministic_linear_tick_sequence_is_preserved() -> None:
    """A fixed linear sequence should keep the same planning and learning outputs."""
    optimizer = LearningSwitchOptimizer(
        "zone_characterization",
        config={
            "history_steps": 4,
            "lookahead_steps": 12,
            "learning_rate": 0.01,
            "w_energy": 0.02,
        },
    )
    rows = [
        (19.0, 6.0, 21.0, 0),
        (19.1, 6.5, 21.0, 1),
        (19.2, 7.0, 21.0, 1),
    ]

    results = [
        optimizer.tick(
            t_room=t_room,
            t_out=t_out,
            t_set=t_set,
            t_out_forecast=[t_out] * 12,
            current_u=current_u,
            step_minutes=15,
            timestamp=1_700_000_000 + index * 900,
        )
        for index, (t_room, t_out, t_set, current_u) in enumerate(rows)
    ]

    assert [result["u_planned"] for result in results] == [1, 1, 1]
    assert [result["start_step"] for result in results] == [0, 0, 0]
    assert [result["stop_step"] for result in results] == [12, 12, 12]
    assert [result["lookahead_steps"] for result in results] == [12, 12, 12]
    assert [result["history_len"] for result in results] == [1, 2, 3]
    assert [result["weights_reset"] for result in results] == [True, False, False]
    assert [result["planned_states"] for result in results] == [[1] * 12] * 3
    assert [result["forecast_temps"][0] for result in results] == [6.0, 6.5, 7.0]
    assert [result["t_pred_off"] for result in results] == pytest.approx(
        [19.0, 19.101829113117265, 19.205659100136977]
    )
    assert [result["t_pred_on"] for result in results] == pytest.approx(
        [19.025, 19.12865713248453, 19.233907463651537]
    )

    snapshot = optimizer.snapshot_state()

    assert snapshot["history"]["t_room"] == [19.2, 19.1, 19.0]
    assert snapshot["history"]["t_out"] == [7.0, 6.5, 6.0]
    assert snapshot["history"]["u"] == [1, 1, 0]
    assert snapshot["history"]["k_step_abs_error"] == {
        "1": {"count": 2, "mean": pytest.approx(0.07317143375773583)},
        "2": {"count": 1, "mean": pytest.approx(0.1499990000000011)},
    }
    assert snapshot["history"]["k_step_pred_improvement"] == {
        "1": {"count": 2, "mean": pytest.approx(0.026828566242263818)},
        "2": {"count": 1, "mean": pytest.approx(0.050000999999998186)},
    }
    assert snapshot["weights"][:4] == pytest.approx(
        [
            0.0001995544334208881,
            -0.0000665181444736406,
            -0.0000665181444736406,
            -0.0000665181444736406,
        ]
    )
    assert snapshot["last_plan"] == {"start_step": 0, "stop_step": 12}


def test_k_step_tracking_records_matured_errors_and_drops_stale_predictions() -> None:
    """K-step tracking should mature only current predictions and prune stale rows."""
    pending_preds = {0: {1: 99.0}}
    pending_features = {0: {1: [99.0]}}
    pending_baseline = {0: {1: 99.0}}
    abs_error_history = {}
    pred_improvement_history = {}

    queue_k_step_predictions(
        pending_preds=pending_preds,
        preds=[20.1, 20.4, float("nan")],
        step_index=1,
        max_steps=3,
    )
    queue_k_step_features(
        pending_features=pending_features,
        features=[
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ],
        step_index=1,
        max_steps=3,
    )
    queue_k_step_baseline(
        pending_baseline=pending_baseline,
        baseline_temp=20.0,
        step_index=1,
        max_steps=3,
    )

    errors, updates = record_k_step_errors(
        actual_temp=20.7,
        step_index=3,
        pending_preds=pending_preds,
        pending_features=pending_features,
        pending_baseline=pending_baseline,
        abs_error_history=abs_error_history,
        pred_improvement_history=pred_improvement_history,
    )

    assert 0 not in pending_preds
    assert 0 not in pending_features
    assert 0 not in pending_baseline
    assert errors == {2: pytest.approx(0.3)}
    assert updates == [
        {"horizon": 2, "error": pytest.approx(0.3), "feature": [3.0, 4.0]}
    ]
    assert abs_error_history == {2: {"mean": pytest.approx(0.3), "count": 1}}
    assert pred_improvement_history == {2: {"mean": pytest.approx(0.4), "count": 1}}


def test_dynamic_lookahead_selection_contract() -> None:
    """Dynamic lookahead should report the same bounds and reason contract."""
    assert select_dynamic_lookahead(
        stats={1: {"mean": 0.1, "count": 1}},
        min_steps=2,
        max_steps=4,
    ) == (2, "insufficient_data")

    assert select_dynamic_lookahead(
        stats={
            1: {"mean": 0.1, "count": 1},
            2: {"mean": 0.1, "count": 1},
            3: {"mean": 0.25, "count": 1},
            4: {"mean": 0.1, "count": 1},
        },
        min_steps=2,
        max_steps=4,
    ) == (2, "within_tolerance")

    assert select_dynamic_lookahead(
        stats={
            1: {"mean": 0.1, "count": 1},
            2: {"mean": 0.1, "count": 1},
            3: {"mean": 0.15, "count": 1},
            4: {"mean": 0.2, "count": 1},
        },
        min_steps=2,
        max_steps=4,
    ) == (4, "within_tolerance")


def test_learning_update_payloads_preserve_shape_and_weight_effects() -> None:
    """Single and multi-step learning updates should keep payload compatibility."""
    weights = np.zeros(4, dtype=float)

    update = compute_single_step_update(
        weights=weights,
        l2_reg=0.0,
        actual_temp=21.0,
        prev_pred=20.0,
        prev_feature=[1.0, 0.0, 0.0, 1.0],
        base_eta=0.1,
        eta=0.1,
        fallback_learning_rate=0.01,
    )

    assert update is not None
    assert update.payload["error"] == pytest.approx(1.0)
    assert update.payload["errors"] == [{"step": 0, "error": 1.0, "weight": 1.0}]
    assert update.payload["eta_scale"] == pytest.approx(1.0)
    assert update.payload["steps_used"] == 1
    assert update.payload["delta"] == pytest.approx([0.1999996, 0.0, 0.0, 0.1999996])
    assert update.weights.tolist() == pytest.approx(update.payload["delta"])
    assert weights.tolist() == [0.0, 0.0, 0.0, 0.0]

    multi_update = compute_multi_step_update(
        weights=weights,
        l2_reg=0.0,
        updates=[
            {"horizon": 1, "error": 1.0, "feature": [1.0, 0.0, 0.0, 1.0]},
            {"horizon": 2, "error": -0.5, "feature": [0.0, 1.0, 0.0, 1.0]},
        ],
        base_eta=0.2,
        eta=0.1,
    )

    assert multi_update is not None
    assert multi_update.payload["error"] == pytest.approx(0.25)
    assert multi_update.payload["errors"] == [
        {"step": 1, "error": 1.0, "weight": 1.0},
        {"step": 2, "error": -0.5, "weight": 1.0},
    ]
    assert multi_update.payload["eta_scale"] == pytest.approx(0.5)
    assert multi_update.payload["steps_used"] == 2
    assert multi_update.payload["delta"] == pytest.approx(
        [0.0999998, -0.0499999, 0.0, 0.0499999]
    )
    assert multi_update.weights.tolist() == pytest.approx(multi_update.payload["delta"])
    assert weights.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_history_helpers_update_latest_first_and_build_model_arrays() -> None:
    """History helpers should preserve latest-first buffers and sanitized arrays."""
    t_hist, out_hist, solar_hist, solar_sample_hist, u_hist = (
        update_latest_first_histories(
            t_hist=[19.0, 18.0],
            out_hist=[5.0],
            solar_hist=[100.0],
            solar_sample_hist=[{"flux": 100.0, "azimuth": None, "elevation": None}],
            u_hist=[0],
            history_steps=2,
            use_solar_feature=True,
            t_room=20.0,
            t_out=7.0,
            u_actual=1,
            solar=None,
            solar_sample=None,
        )
    )

    assert t_hist == [20.0, 19.0]
    assert out_hist == [7.0, 5.0]
    assert solar_hist == [100.0, 100.0]
    assert solar_sample_hist == [
        {"flux": 100.0, "azimuth": None, "elevation": None},
        {"flux": 100.0, "azimuth": None, "elevation": None},
    ]
    assert u_hist == [1, 0]

    t_array, out_array, solar_array, u_array = build_history_arrays(
        t_hist=t_hist,
        out_hist=out_hist,
        solar_hist=solar_hist,
        solar_sample_hist=solar_sample_hist,
        u_hist=u_hist,
        history_steps=2,
        use_solar_feature=True,
        solar_orientation=180.0,
    )

    assert t_array.tolist() == [20.0, 19.0]
    assert out_array.tolist() == [7.0, 5.0]
    assert solar_array is not None
    assert solar_array.tolist() == [[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    assert u_array.tolist() == [1.0, 0.0]


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
    features = build_model_feature(
        learning_model=LEARNING_MODEL_PHYSICAL,
        history_decay_vector=np.ones(4, dtype=float),
        feature_step_minutes=60.0,
        t_hist=sanitize_array(np.array([22.0, 21.0, 20.0, 19.0])),
        out_hist=sanitize_array(np.array([12.0, 10.0, 8.0, 6.0])),
        solar_hist=None,
        u_hist=sanitize_array(np.array([1.0, 0.0, 0.0, 0.0])),
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
