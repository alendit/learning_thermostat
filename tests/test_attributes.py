"""Tests for recorder-safe climate attributes."""

from __future__ import annotations

import json

from custom_components.learning_thermostat.attributes import (
    RECORDER_ATTRIBUTE_SAFE_BYTES,
    build_learning_diagnostic_attributes,
    compact_state_attributes,
)


def _oversized_attributes() -> dict:
    """Build an attribute payload shaped like a learned solar-orientation thermostat."""
    weights = [(-0.175 + idx * 0.00234123456789) for idx in range(67)]
    solar_samples = [
        {
            "flux": 316.67123456789 - idx * 19.123456789,
            "azimuth": 270.123456789,
            "elevation": 34.987654321,
        }
        for idx in range(12)
    ]
    history = {
        "t_room": [22.23456789 + idx * 0.01 for idx in range(12)],
        "t_out": [8.923456789 - idx * 0.57 for idx in range(12)],
        "solar": [316.67123456789 - idx * 19.123456789 for idx in range(12)],
        "solar_samples": solar_samples,
        "u": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "pred_next": [22.501245129640964 + idx * 0.01 for idx in range(12)],
        "pred_abs_error": [0.1012451296409651 + idx * 0.01 for idx in range(12)],
        "pred_timestamps": [f"2026-04-24T0{idx % 8}:18:55" for idx in range(12)],
        "weight_abs_update": [0.007989287809642816 + idx * 0.001 for idx in range(12)],
        "weight_delta_abs_latest": [
            abs(-0.00013231718261295694 + idx * 0.000013456789) for idx in range(139)
        ],
        "k_step_abs_error": {
            str(idx): {"mean": 0.10389987833859779 + idx * 0.031, "count": idx}
            for idx in range(1, 25)
        },
        "k_step_pred_improvement": {
            str(idx): {"mean": -0.04477899921771842 + idx * 0.004, "count": idx}
            for idx in range(1, 25)
        },
    }
    update = {
        "error": -0.25213939138917735,
        "errors": [0.1804654045233569, 0.16648640049127295],
        "grad_norm": 3.8124965298293265,
        "eta": 0.005194993916929889,
        "base_eta": 0.01,
        "eta_scale": 0.5194993916929889,
        "l2_reg": 0.001,
        "delta": [weight * 0.123456789 for weight in weights],
        "abs_update_max": 0.006580280067529406,
        "steps_used": 3,
    }
    return {
        "learning_state": {
            "weights": weights,
            "model": "delta_zscore_solar_orientation",
            "history": history,
            "last_u": 1,
            "last_plan": {"start_step": 0, "stop_step": 4},
            "solar_orientation_azimuth": 180.0,
            "solar_orientation_confidence": 0.7348204209906434,
            "solar_orientation_score": 0.00234123456789,
            "eta": 0.005194993916929889,
            "base_eta": 0.01,
            "prev_feature": [idx * 0.123456789 for idx in range(139)],
            "config": {
                "lookahead_steps": 24,
                "learning_rate": 0.01,
                "l2_reg": 0.001,
                "w_energy": 0.01,
                "history_steps": 12,
                "history_decay": 1.0,
                "error_decay": 0.9,
                "default_u_gain_per_hour": 0.2,
                "use_solar_feature": True,
            },
            "solar_orientation": {
                "azimuth": 270.0,
                "confidence": 0.7348204209906434,
                "orientation_1": "west",
                "orientation_1_azimuth": 270.0,
                "orientation_1_contribution_pct": 90.123456789,
                "orientation_2": "south",
                "orientation_2_azimuth": 180.0,
                "orientation_2_contribution_pct": 9.876543211,
                "state": "west 90%, south 10%",
            },
        },
        "learning_result": {
            "weights": weights,
            "forecast_solar_samples": solar_samples[:8],
            "predicted_temps": [22.14617716929864 + idx * 0.03 for idx in range(12)],
            "planned_states": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "update": update,
            "k_step_error_mean": {
                str(idx): 0.10389987833859779 + idx * 0.031 for idx in range(1, 25)
            },
            "k_step_pred_improvement": {
                str(idx): -0.04477899921771842 + idx * 0.004 for idx in range(1, 25)
            },
        },
        "solar_forecast_entity_ids": [
            f"sensor.pirateweather_downward_short_wave_radiation_flux_{idx}h"
            for idx in range(1, 25)
        ],
        "outside_temperature": 9.8,
        "current_solar": 378.24,
        "current_solar_sample": solar_samples[0],
    }


def _json_size(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":")))


def test_compact_state_attributes_fit_recorder_limit_without_dropping_model() -> None:
    """Exposed climate attributes should stay below HA recorder's attribute limit."""
    oversized = _oversized_attributes()

    assert _json_size(oversized) > 16384

    compacted = compact_state_attributes(oversized)

    assert _json_size(compacted) < RECORDER_ATTRIBUTE_SAFE_BYTES
    assert RECORDER_ATTRIBUTE_SAFE_BYTES == 16000
    assert compacted["learning_state"]["weights"]
    assert compacted["learning_state"]["history"]["solar_samples"]
    assert compacted["learning_state"]["prev_feature"]
    assert "solar_training_examples" not in compacted["learning_state"]
    assert "weights" not in compacted["learning_result"]
    assert "delta" not in compacted["learning_result"]["update"]


def test_learning_diagnostics_do_not_expose_raw_optimizer_snapshots() -> None:
    """HA attributes should expose diagnostics, not the raw persistence payload."""
    oversized = _oversized_attributes()

    diagnostics = build_learning_diagnostic_attributes(
        learning_state=oversized["learning_state"],
        learning_result=oversized["learning_result"],
        learning_last_tick="2026-04-24T08:18:55+00:00",
        outside_temperature=9.8,
        current_solar=378.24,
        current_solar_sample=oversized["current_solar_sample"],
        solar_orientation=oversized["learning_state"]["solar_orientation"],
        on_percent=1.0,
        underlying_entities=["switch.floor_a"],
        outdoor_entity_id="weather.home",
        legacy_forecast_entity_id=None,
        solar_entity_id="sensor.home_solar",
        solar_forecast_entity_ids=["sensor.home_solar_1h"],
    )

    assert "learning_state" not in diagnostics
    assert "learning_result" not in diagnostics
    assert diagnostics["learning_model"] == "delta_zscore_solar_orientation"
    assert diagnostics["learning_last_tick"] == "2026-04-24T08:18:55+00:00"
    assert diagnostics["learning_plan"]["planned_states"] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert diagnostics["learning_plan"]["predicted_temps"] == [
        22.1462,
        22.1762,
        22.2062,
        22.2362,
        22.2662,
        22.2962,
        22.3262,
        22.3562,
    ]
    assert diagnostics["learning_update"]["steps_used"] == 3
    assert "delta" not in diagnostics["learning_update"]
    assert _json_size(diagnostics) < RECORDER_ATTRIBUTE_SAFE_BYTES
