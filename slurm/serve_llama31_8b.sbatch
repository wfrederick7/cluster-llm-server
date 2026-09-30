#!/usr/bin/env bash
#SBATCH --job-name=llama31-8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=superpod
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
unset GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH PYTHONPATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

SERVER_ROOT="${SERVER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SERVER_ROOT="$(cd "${SERVER_ROOT}" && pwd)"
export SERVER_ROOT
PROFILE="${PROFILE:-${SERVER_ROOT}/profiles/llama31-8b.env}"
SECRETS_FILE="${SECRETS_FILE:-${HOME}/.config/cluster-llm-server/server.env}"
VENV_DIR="${VENV_DIR:-${SERVER_ROOT}/.venv}"
[[ -f "${PROFILE}" && -f "${SECRETS_FILE}" && -x "${VENV_DIR}/bin/python" ]] || {
  echo "Missing profile, secrets file, or installed environment" >&2
  exit 2
}
[[ "$(stat -c '%a' "${SECRETS_FILE}")" == "600" ]] || {
  echo "Secrets file must have mode 600" >&2
  exit 2
}
set -a
# shellcheck disable=SC1090
source "${PROFILE}"
# shellcheck disable=SC1090
source "${SECRETS_FILE}"
set +a
: "${MODEL_REVISION:?Set MODEL_REVISION to a tested Hugging Face commit SHA}"
: "${HF_HOME:?HF_HOME is required in the secrets file}"
: "${HF_TOKEN:?HF_TOKEN is required for the gated Llama model}"
: "${VLLM_API_KEY:?VLLM_API_KEY is required in the secrets file}"
[[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "MODEL_REVISION must be a 40-character commit SHA" >&2
  exit 2
}
[[ "${REPLICA_COUNT}" == "8" ]] || { echo "This launcher requires eight replicas" >&2; exit 2; }
mkdir -p "${SERVER_ROOT}/runtime/${SLURM_JOB_ID}" "${HF_HOME}"
RUNTIME_DIR="${SERVER_ROOT}/runtime/${SLURM_JOB_ID}"

"${VENV_DIR}/bin/python" - <<'PY'
import subprocess
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("Expected eight visible CUDA devices")
usage = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
).splitlines()
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capacity = torch.cuda.get_device_properties(index).total_memory / 1024**3
    used = int(usage[index].strip())
    if "H100" not in name or capacity < 75 or used > 2048:
        raise SystemExit(f"GPU {index} unavailable for a replica: {name}, {capacity:.1f} GiB, {used} MiB used")
print("CUDA preflight passed: eight unoccupied H100 80 GB GPUs", flush=True)
PY

"${VENV_DIR}/bin/python" "${SERVER_ROOT}/scripts/prefetch_model.py" \
  --model "${MODEL_ID}" --revision "${MODEL_REVISION}" --min-free-gb "${MIN_FREE_GB}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

BACKENDS=()
for index in $(seq 0 7); do
  port=$((REPLICA_BASE_PORT + index))
  BACKENDS+=("http://127.0.0.1:${port}")
  echo "Starting replica ${index} on GPU ${index}, port ${port}"
  CUDA_VISIBLE_DEVICES="${index}" "${VENV_DIR}/bin/vllm" serve "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 --port "${port}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-chunked-prefill --enable-prefix-caching --enforce-eager \
    >"${RUNTIME_DIR}/replica_${index}.log" 2>&1 &
  replica_pid="$!"
  PIDS+=("${replica_pid}")
  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${replica_pid}" 2>/dev/null; then
      echo "Replica ${index} exited during startup; see ${RUNTIME_DIR}/replica_${index}.log" >&2
      exit 1
    fi
    if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" == "1" ]] || { echo "Replica ${index} did not become ready" >&2; exit 1; }
done

PROXY_ARGS=()
for backend in "${BACKENDS[@]}"; do PROXY_ARGS+=(--backend "${backend}"); done
"${VENV_DIR}/bin/python" "${SERVER_ROOT}/scripts/replica_proxy.py" \
  --port "${PORT}" \
  --max-in-flight "${PROXY_MAX_IN_FLIGHT_PER_BACKEND}" \
  "${PROXY_ARGS[@]}" >"${RUNTIME_DIR}/proxy.log" 2>&1 &
PIDS+=("$!")

ready=0
for _ in $(seq 1 30); do
  if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then ready=1; break; fi
  sleep 1
done
[[ "${ready}" == "1" ]] || { echo "Proxy did not become ready" >&2; exit 1; }
"${VENV_DIR}/bin/python" "${SERVER_ROOT}/scripts/verify_server.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED_MODEL_NAME}" \
  --revision "${MODEL_REVISION}" --mode chat

ADVERTISE_HOST="${ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
export ADVERTISE_HOST
"${VENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "base_url": f"http://{os.environ['ADVERTISE_HOST']}:{os.environ['PORT']}/v1",
    "model_id": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "replica_count": 8,
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "max_num_seqs_per_replica": int(os.environ["MAX_NUM_SEQS"]),
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
}
path = Path(os.environ["SERVER_ROOT"]) / "runtime" / os.environ["SLURM_JOB_ID"] / "manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
echo "OPENAI_BASE_URL=http://${ADVERTISE_HOST}:${PORT}/v1"
echo "OPENAI_MODEL=${SERVED_MODEL_NAME}"
echo "Eight-replica server ready; API key omitted from logs"
wait -n "${PIDS[@]}"
echo "One server process exited; shutting down" >&2
exit 1
