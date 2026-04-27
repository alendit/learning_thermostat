"""Architecture boundary tests for the learning thermostat."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = [
    "attributes.py",
    "config_contract.py",
    "controller.py",
    "optimizer.py",
    "optimizer_features.py",
    "optimizer_history.py",
    "optimizer_learning.py",
    "optimizer_orientation.py",
    "optimizer_planning.py",
    "optimizer_state.py",
    "optimizer_tracking.py",
    "optimizer_types.py",
    "solar.py",
    "state.py",
]


def test_core_modules_do_not_import_home_assistant() -> None:
    """Domain and optimizer core modules should stay Home Assistant-free."""
    for module in CORE_MODULES:
        text = _component_path(module).read_text(encoding="utf-8")

        assert "from homeassistant" not in text
        assert "import homeassistant" not in text


def test_core_modules_do_not_define_broad_fake_interfaces() -> None:
    """Core code should use concrete helpers instead of speculative interfaces."""
    forbidden_tokens = ("Protocol", "ABC", "abstractmethod", "NotImplementedError")
    for module in CORE_MODULES:
        text = _component_path(module).read_text(encoding="utf-8")

        for token in forbidden_tokens:
            assert token not in text


def _component_path(module: str) -> Path:
    return PROJECT_ROOT / "custom_components" / "learning_thermostat" / module
