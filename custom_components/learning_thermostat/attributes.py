"""Helpers for Home Assistant state attributes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

RECORDER_ATTRIBUTE_SAFE_BYTES = 16000
_FLOAT_SIGNIFICANT_DIGITS = 6


def compact_state_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return recorder-safe diagnostic attributes.

    The controller/storage payload remains the canonical learning state. The HA
    state attribute surface is compacted so recorder can keep ordinary climate
    attributes instead of dropping the entire attribute blob.
    """
    compacted = _compact_value(attributes)
    if not isinstance(compacted, dict):
        return {}

    learning_result = compacted.get("learning_result")
    if isinstance(learning_result, dict):
        learning_result.pop("weights", None)
        update = learning_result.get("update")
        if isinstance(update, dict):
            update.pop("delta", None)

    return compacted


def _compact_value(value: Any) -> Any:
    """Copy ``value`` while trimming verbose float representations."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return _compact_float(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {str(key): _compact_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_compact_value(child) for child in value]
    return value


def _compact_float(value: float) -> float:
    """Round a float to a stable, useful diagnostic precision."""
    if not math.isfinite(value):
        return value
    return float(f"{value:.{_FLOAT_SIGNIFICANT_DIGITS}g}")
