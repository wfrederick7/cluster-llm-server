#!/usr/bin/env bash
#SBATCH --job-name=gpt-oss-120b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SERVER_ROOT="$(cd "${SERVER_ROOT}" && pwd)"
export SERVER_ROOT
PROFILE="${PROFILE:-${SERVER_ROOT}/profiles/gpt-oss-120b.env}"
SECRETS_FILE="${SECRETS_FILE:-${HOME}/.config/cluster-llm-server/server.env}"
VENV_DIR="${VENV_DIR:-${SERVER_ROOT}/.venv}"

[[ -f "${PROFILE}" ]] || { echo "Missing profile: ${PROFILE}" >&2; exit 2; }
[[ -f "${SECRETS_FILE}" ]] || { echo "Missing secrets file: ${SECRETS_FILE}" >&2; exit 2; }
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "Missing environment: ${VENV_DIR}" >&2; exit 2; }

permissions="$(stat -c '%a' "${SECRETS_FILE}")"
[[ "${permissions}" == "600" ]] || {
    echo "Secrets file must have mode 600; found ${permissions}" >&2
    exit 2
}

set -a
# The profile path is selected at submission time.
# shellcheck disable=SC1090
source "${PROFILE}"
# Secrets remain outside the repository.
# shellcheck disable=SC1090
source "${SECRETS_FILE}"
set +a

: "${MODEL_ID:?MODEL_ID is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${VLLM_API_KEY:?VLLM_API_KEY is required}"

export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${SERVER_ROOT}/runtime/${SLURM_JOB_ID}" "${HF_HOME}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

"${VENV_DIR}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
name = torch.cuda.get_device_name(0)
memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if "H100" not in name or memory_gb < 75:
    raise SystemExit(f"Expected an H100 80 GB GPU; found {name} ({memory_gb:.1f} GB)")
print(f"CUDA preflight passed: {name} ({memory_gb:.1f} GB)")
PY

"${VENV_DIR}/bin/python" "${SERVER_ROOT}/scripts/prefetch_model.py" \
    --model "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --min-free-gb "${MIN_FREE_GB}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT TERM INT

"${VENV_DIR}/bin/vllm" serve "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-requests &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}"
        exit $?
    fi
    if curl --silent --show-error --fail \
        "http://127.0.0.1:${PORT}/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 5
done
[[ "${ready}" == "1" ]] || { echo "Server did not become ready" >&2; exit 1; }

"${VENV_DIR}/bin/python" "${SERVER_ROOT}/scripts/verify_server.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --revision "${MODEL_REVISION}"

ADVERTISE_HOST="${ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
export ADVERTISE_HOST
"${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "hostname": platform.node(),
    "advertise_host": os.environ["ADVERTISE_HOST"],
    "base_url": f"http://{os.environ['ADVERTISE_HOST']}:{os.environ['PORT']}/v1",
    "model_id": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "tensor_parallel_size": int(os.environ["TENSOR_PARALLEL_SIZE"]),
    "max_num_seqs": int(os.environ["MAX_NUM_SEQS"]),
    "max_num_batched_tokens": int(os.environ["MAX_NUM_BATCHED_TOKENS"]),
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
    "dtype": os.environ["DTYPE"],
    "kv_cache_dtype": os.environ["KV_CACHE_DTYPE"],
    "python": platform.python_version(),
    "vllm": importlib.metadata.version("vllm"),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}
path = (
    Path(os.environ["SERVER_ROOT"])
    / "runtime"
    / os.environ["SLURM_JOB_ID"]
    / "manifest.json"
)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "OPENAI_BASE_URL=http://${ADVERTISE_HOST}:${PORT}/v1"
echo "OPENAI_MODEL=${SERVED_MODEL_NAME}"
echo "Server ready; API key omitted from logs"

wait "${SERVER_PID}"
