#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python 3.10-3.12 is required; found {sys.version.split()[0]}"
    )
PY

if [[ -e "${VENV_DIR}" ]]; then
    echo "Environment already exists: ${VENV_DIR}" >&2
    echo "Remove it explicitly or set VENV_DIR to a new path." >&2
    exit 2
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install --pre \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    -r "${ROOT_DIR}/requirements.bootstrap.txt"

mkdir -p "${ROOT_DIR}/runtime"
"${VENV_DIR}/bin/python" -m pip freeze \
    > "${ROOT_DIR}/runtime/environment.freeze.txt"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import vllm

print(f"vLLM={vllm.__version__}")
print(f"PyTorch={torch.__version__}")
print("Environment installation complete. CUDA is verified inside the GPU job.")
PY
