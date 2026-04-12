#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev --no-install-project
.venv/bin/black --check custom_components tests scripts
.venv/bin/python -m pytest -q
