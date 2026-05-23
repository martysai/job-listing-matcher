#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_env.sh
#   PYTHON_BIN=python3.11 bash scripts/setup_env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

# For VPS, install CPU-only torch first to avoid pip pulling the 2 GB CUDA wheel
# that sentence-transformers would otherwise trigger via PyPI's default index.
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e .[server,rerank,dev]

python scripts/check_env.py
