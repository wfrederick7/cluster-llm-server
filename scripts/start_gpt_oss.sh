#!/usr/bin/env bash
# Load the cluster runtime and submit GPT-OSS from this checkout.
set -euo pipefail

SERVER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SERVER_VENV_DIR:-${SERVER_ROOT}/.venv}"
PROFILE="${SERVER_ROOT}/profiles/gpt-oss-120b.env"

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
# Select this checkout directly; activation scripts can retain paths after a move.
export SERVER_ROOT VENV_DIR PROFILE
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

# Check shared-library availability without asking vLLM to detect a GPU here.
"${VENV_DIR}/bin/python" -c 'import sys; print("Using Python: " + sys.executable)'
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is unavailable in this terminal." >&2; exit 2; }

PARTITION="${PARTITION:-superpod}"
NODE="${NODE:-sp-0003}"
WALLTIME="${WALLTIME:-5-00:00:00}"
ADVERTISE_HOST="${ADVERTISE_HOST:-${NODE}}"
export ADVERTISE_HOST

cd "${SERVER_ROOT}"
mkdir -p logs
exec sbatch \
    --partition="${PARTITION}" \
    --nodelist="${NODE}" \
    --time="${WALLTIME}" \
    --export="ALL,SERVER_ROOT=${SERVER_ROOT},VENV_DIR=${VENV_DIR},PROFILE=${PROFILE},ADVERTISE_HOST=${ADVERTISE_HOST}" \
    "$@" "${SERVER_ROOT}/slurm/serve.sbatch"
