#!/usr/bin/env bash
# Load the cluster runtime and submit both Gemma models from this checkout.
set -euo pipefail

SERVER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SERVER_VENV_DIR:-${SERVER_ROOT}/.venv}"

if ! command -v module >/dev/null 2>&1; then
    for init_file in "${MODULESHOME:-/nonexistent}/init/bash" \
                     /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        if [[ -r "${init_file}" ]]; then
            # shellcheck disable=SC1090
            source "${init_file}"
            command -v module >/dev/null 2>&1 && break
        fi
    done
fi
command -v module >/dev/null 2>&1 || {
    echo "Module initialization unavailable; run from a cluster terminal with Environment Modules or Lmod." >&2
    exit 2
}

module load "${SERVER_GCC_MODULE:-gcc/15.2.0}"
module load "${SERVER_PYTHON_MODULE:-python/cpu/3.10.6}"
unset PYTHONPATH GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH

[[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/bin/vllm" ]] || {
    echo "Missing server environment: ${VENV_DIR}; install it with scripts/setup_env.sh." >&2
    exit 2
}
export SERVER_ROOT VENV_DIR
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
# Check interpreter portability without detecting GPUs on the login node.
"${VENV_DIR}/bin/python" -c 'import ssl, sys; print("Using Python: " + sys.executable)'
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is unavailable in this terminal." >&2; exit 2; }

PARTITION="${PARTITION:-superpod}"
WALLTIME="${WALLTIME:-3-00:00:00}"
scheduling=(--partition="${PARTITION}" --time="${WALLTIME}")
if [[ -n "${NODE:-}" ]]; then
    scheduling+=(--nodelist="${NODE}")
fi
# Each server advertises the actual allocated hostname unless overridden.
cd "${SERVER_ROOT}"
mkdir -p logs
exec sbatch "${scheduling[@]}" --export=ALL \
    "$@" "${SERVER_ROOT}/slurm/serve_gemma_medgemma.sbatch"
