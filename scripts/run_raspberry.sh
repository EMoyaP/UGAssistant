#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PROJECT_ROOT}/.venv/bin/python"

if [ ! -x "${PYTHON_EXE}" ]; then
    PYTHON_EXE="python3"
fi

export PYTHONPATH="${PROJECT_ROOT}/src"
exec "${PYTHON_EXE}" -m ugassistant
