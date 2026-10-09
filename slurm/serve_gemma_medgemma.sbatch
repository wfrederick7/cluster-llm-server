#!/usr/bin/env bash
#SBATCH --job-name=fairsumeval-gemma-medgemma
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One allocation, two exclusive four-GPU steps. Slurm assigns the devices.
set -euo pipefail
SERVER_ROOT="${SERVER_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the server repository}}"
export SERVER_ROOT
mkdir -p "${SERVER_ROOT}/runtime/${SLURM_JOB_ID}"
echo "Starting Gemma/MedGemma job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-unknown node}"
echo "Model logs are mirrored here and saved under ${SERVER_ROOT}/runtime/${SLURM_JOB_ID}/"
pids=()
cleanup() {
    trap - EXIT TERM INT
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
start() {
    local name="$1"
    local log_root="${SERVER_ROOT}/runtime/${SLURM_JOB_ID}/${name}"
    echo "Starting ${name} with four GPUs"
    INSTANCE_NAME="$name" PROFILE="${SERVER_ROOT}/profiles/${name}.env" \
      srun --unbuffered --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 \
      --gpus-per-task=4 --gpu-bind=closest --mem=120G \
      bash "${SERVER_ROOT}/slurm/serve.sbatch" \
      > >(tee "${log_root}.out" | while IFS= read -r line || [[ -n "$line" ]]; do
          printf '[%s] %s\n' "$name" "$line"
      done) \
      2> >(tee "${log_root}.err" | while IFS= read -r line || [[ -n "$line" ]]; do
          printf '[%s] %s\n' "$name" "$line" >&2
      done) &
    pids+=("$!")
}
start gemma3-27b
start medgemma-27b-text
# If either server exits, stop its companion rather than leave a partial pair.
while true; do
    for pid in "${pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            set +e
            wait "$pid"
            status=$?
            set -e
            echo "A Gemma/MedGemma step exited (status ${status}); stopping the pair." >&2
            exit 1
        fi
    done
    sleep 1
done
