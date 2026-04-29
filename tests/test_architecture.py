"""Architecture boundary tests for the learning thermostat."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_type_hints

from custom_components.learning_thermostat.environment_input import (
    CurrentEnvironmentInput,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = PROJECT_ROOT / "custom_components" / "learning_thermostat"
CORE_MODULES = [
    "attributes.py",
    "config_contract.py",
    "const.py",
    "controller.py",
    "learning_defaults.py",
    "optimizer.py",
    "optimizer_features.py",
    "optimizer_history.py",
    "optimizer_learning.py",
    "optimizer_memory.py",
    "optimizer_model.py",
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
    "environment_input.py",
    "migration.py",
    "runtime.py",
    "sensor.py",
    "services.py",
    "solar_input.py",
    "storage.py",
    "switches.py",
    "weather_input.py",
}


def test_all_component_modules_are_classified() -> None:
    """Every integration module should be classified as core or adapter."""
    classified = set(CORE_MODULES) | HA_ADAPTER_MODULES
    actual = {path.name for path in COMPONENT_DIR.glob("*.py")}

    assert actual == classified


def test_core_modules_do_not_import_home_assistant() -> None:
    """Domain and optimizer core modules should stay Home Assistant-free."""
    for module in CORE_MODULES:
        imports = _imports_for(module)

        assert "homeassistant" not in imports, module


def test_core_modules_do_not_import_adapter_modules() -> None:
    """Core modules should not depend on Home Assistant adapter modules."""
    adapter_imports = {
        module.removesuffix(".py")
        for module in HA_ADAPTER_MODULES
        if module != "__init__.py"
    }
    for module in CORE_MODULES:
        imports = _imports_for(module)

        assert adapter_imports.isdisjoint(imports), module


def test_core_modules_do_not_define_broad_fake_interfaces() -> None:
    """Core code should use concrete helpers instead of speculative interfaces."""
    forbidden_tokens = ("Protocol", "ABC", "abstractmethod", "NotImplementedError")
    for module in CORE_MODULES:
        text = _component_path(module).read_text(encoding="utf-8")

        for token in forbidden_tokens:
            assert token not in text


def test_home_assistant_imports_stay_in_adapter_modules() -> None:
    """Home Assistant imports should stay out of core/domain modules."""
    for path in COMPONENT_DIR.glob("*.py"):
        if path.name in HA_ADAPTER_MODULES:
            continue

        assert "homeassistant" not in _imports_for(path.name), path.name


def test_config_modules_do_not_import_optimizer() -> None:
    """Config contracts and flows should not depend on optimizer internals."""
    for module in ("config_contract.py", "config_flow.py"):
        imports = _imports_for(module)

        assert "optimizer" not in imports, module


def test_climate_entity_does_not_instantiate_optimizer() -> None:
    """The climate adapter should go through runtime/controller boundaries."""
    imports = _imports_for("climate.py")
    tree = _parse_module("climate.py")

    assert "optimizer" not in imports
    assert not any(
        isinstance(node, ast.Name) and node.id == "LearningSwitchOptimizer"
        for node in ast.walk(tree)
    )


def test_environment_current_input_contract_is_plain_data() -> None:
    """Collected current inputs should not expose Home Assistant objects."""
    hints = get_type_hints(CurrentEnvironmentInput)

    for field_type in hints.values():
        assert not _type_uses_homeassistant(field_type)


def _component_path(module: str) -> Path:
    return COMPONENT_DIR / module


def _parse_module(module: str) -> ast.Module:
    return ast.parse(_component_path(module).read_text(encoding="utf-8"))


def _imports_for(module: str) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_parse_module(module)):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                imports.add(node.module.split(".", 1)[0])
            elif node.module:
                imports.add(node.module.split(".", 1)[0])
    return imports


def _type_uses_homeassistant(field_type: object) -> bool:
    module = getattr(field_type, "__module__", "")
    if module.startswith("homeassistant"):
        return True
    for arg in getattr(field_type, "__args__", ()):
        if _type_uses_homeassistant(arg):
            return True
    return False
