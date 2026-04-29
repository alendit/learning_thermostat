"""Architecture boundary tests for the learning thermostat."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = [
    "attributes.py",
    "config_contract.py",
    "controller.py",
    "learning_defaults.py",
    "optimizer.py",
    "optimizer_features.py",
    "optimizer_history.py",
    "optimizer_learning.py",
    "optimizer_memory.py",
    "optimizer_orientation.py",
    "optimizer_planning.py",
    "optimizer_state.py",
    "optimizer_tracking.py",
    "optimizer_types.py",
    "solar.py",
    "state.py",
]
HA_ADAPTER_MODULES = {
    "__init__.py",
    "climate.py",
    "config_flow.py",
    "environment.py",
    "migration.py",
    "runtime.py",
    "sensor.py",
    "services.py",
    "storage.py",
    "switches.py",
}


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


def test_home_assistant_imports_stay_in_adapter_modules() -> None:
    """Home Assistant imports should stay out of core/domain modules."""
    component_dir = PROJECT_ROOT / "custom_components" / "learning_thermostat"
    for path in component_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        imports_home_assistant = (
            "from homeassistant" in text or "import homeassistant" in text
        )

        if path.name in HA_ADAPTER_MODULES:
            continue
        assert not imports_home_assistant, path.name


def test_config_modules_do_not_import_optimizer() -> None:
    """Config contracts and flows should not depend on optimizer internals."""
    for module in ("config_contract.py", "config_flow.py"):
        text = _component_path(module).read_text(encoding="utf-8")

        assert "from .optimizer import" not in text
        assert "import .optimizer" not in text


def _component_path(module: str) -> Path:
    return PROJECT_ROOT / "custom_components" / "learning_thermostat" / module
