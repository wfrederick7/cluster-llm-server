#!/usr/bin/env bash
#SBATCH --job-name=fairsumeval-three-models
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One allocation, three exclusive steps. Slurm assigns disjoint GPU sets;
# do not hard-code physical CUDA device numbers on a shared cluster.
set -euo pipefail
SERVER_ROOT="${SERVER_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the server repository}}"
export SERVER_ROOT
mkdir -p "${SERVER_ROOT}/runtime/${SLURM_JOB_ID}"
pids=()
cleanup() {
    trap - EXIT TERM INT
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT
start() {
    local name="$1" gpus="$2"
    INSTANCE_NAME="$name" PROFILE="${SERVER_ROOT}/profiles/${name}.env" \
      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 \
      --gpus-per-task="$gpus" --gpu-bind=closest --mem=160G \
      --output="${SERVER_ROOT}/runtime/${SLURM_JOB_ID}/${name}.out" \
      --error="${SERVER_ROOT}/runtime/${SLURM_JOB_ID}/${name}.err" \
      bash "${SERVER_ROOT}/slurm/serve.sbatch" &
    pids+=("$!")
}
start llama33-70b 2
start med42-v2-70b 2
start gpt-oss-120b-tp4 4
# Any server exit terminates the allocation; supervise all three steps.
while true; do
    for pid in "${pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            set +e
            wait "$pid"
            status=$?
            set -e
            echo "A server step exited (status ${status}); stopping the other servers." >&2
            exit 1
        fi
    done
    sleep 1
done
