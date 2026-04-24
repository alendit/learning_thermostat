"""Tests for recorder-safe climate attributes."""

from __future__ import annotations

import json

from custom_components.learning_thermostat.attributes import (
    RECORDER_ATTRIBUTE_SAFE_BYTES,
    compact_state_attributes,
)


def _oversized_attributes() -> dict:
    """Build an attribute payload shaped like a learned solar-profile thermostat."""
    weights = [(-0.175 + idx * 0.00234123456789) for idx in range(139)]
    solar_profile = [
        [316.67123456789, 0.0, 134.17893181267667, 262.98711266187144]
        + [237.74100964308306, 73.22944750764938, 0.0, 0.0, 0.0]
        for _ in range(12)
    ]
    history = {
        "t_room": [22.23456789 + idx * 0.01 for idx in range(12)],
        "t_out": [8.923456789 - idx * 0.57 for idx in range(12)],
        "solar": [316.67123456789 - idx * 19.123456789 for idx in range(12)],
        "solar_profile": solar_profile,
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
            "model": "delta_zscore_solar_profile",
            "history": history,
            "last_u": 1,
            "last_plan": {"start_step": 0, "stop_step": 4},
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
                "azimuth": 45.0,
                "confidence": 0.23482042099064335,
                "exposure_bins": {
                    str(idx * 45.0): idx * 0.00432123456789 for idx in range(8)
                },
            },
        },
        "learning_result": {
            "weights": weights,
            "forecast_solar_profile": solar_profile[:8],
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
        "current_solar_profile": solar_profile[0],
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
    assert compacted["learning_state"]["history"]["solar_profile"]
    assert compacted["learning_state"]["prev_feature"]
    assert "weights" not in compacted["learning_result"]
    assert "delta" not in compacted["learning_result"]["update"]
