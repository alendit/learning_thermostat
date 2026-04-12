"""Shared pytest fixtures for the learning thermostat integration."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading custom integrations in tests."""


def pytest_configure(config: pytest.Config) -> None:
    """Use asyncio auto mode for HA fixtures."""
    config.option.asyncio_mode = "auto"
