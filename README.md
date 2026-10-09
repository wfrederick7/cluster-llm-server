# Cluster LLM server

Minimal Slurm deployment for an authenticated, OpenAI-compatible vLLM server.
The default profile serves `openai/gpt-oss-120b` across eight H100 80 GB GPUs with a
131,072-token context limit.

The server exposes both `/v1/chat/completions` and `/v1/responses`. It is meant
for jobs on the private cluster network, not public internet access.

## Llama 3.1 8B for discharge generation

`slurm/serve_llama31_8b.sbatch` starts eight independent Llama-3.1-8B-Instruct
replicas on eight H100 80 GB GPUs and one authenticated least-in-flight proxy.
This profile supports non-streaming Chat Completions at `/v1/chat/completions`;
it does not expose the Responses API. It uses the same external mode-600
`server.env` secrets file and `.venv` as the other server profiles. The gated
Llama model requires `HF_TOKEN` in that file.

The profile pins the [model repository's published commit SHA](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct). To use another
tested revision, set `MODEL_REVISION` to its 40-character SHA. Submit from the
repository root:

```bash
cd /path/to/cluster-llm-server
mkdir -p logs
sbatch --export=ALL slurm/serve_llama31_8b.sbatch
```

The launcher checks that all eight GPUs are visible and have at most 2 GiB
already used, prefetches the selected revision, starts replicas one at a time,
and runs an authenticated smoke test. It writes a non-secret
`runtime/<job-id>/manifest.json` containing the base URL and served model name.
Use those values and `VLLM_API_KEY` from the external secrets file in the
Discharge_summary generation jobs; no node address is baked into that repo.

Status: the repository is locally tested, but the full install and serving path
must be validated on the target GPU cluster.

## Requirements

- Linux x86-64 Slurm cluster
- One node with eight H100 GPUs with at least 80 GB VRAM each
- Python 3.10-3.12 with `venv` support
- A shared cache location with at least 150 GB free
- Network access to the Python package indexes and Hugging Face, or equivalent
  pre-populated caches

## One-time setup

Choose a shared filesystem location with at least 150 GB free, then create a
secret file outside the repository:

```bash
CACHE_ROOT=/absolute/path/to/shared/model-cache
mkdir -p ~/.config/cluster-llm-server
umask 077
API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
{
  printf 'VLLM_API_KEY=%s\n' "$API_KEY"
  printf 'HF_TOKEN=\n'
  printf 'HF_HOME=%s/huggingface\n' "$CACHE_ROOT"
} > ~/.config/cluster-llm-server/server.env
unset API_KEY CACHE_ROOT
chmod 600 ~/.config/cluster-llm-server/server.env
```

`HF_TOKEN` can remain blank because the pinned GPT-OSS model is public.

Create the isolated environment with a Python 3.10-3.12 interpreter:

```bash
cd /path/to/cluster-llm-server
PYTHON_BIN=/path/to/python3.10 ./scripts/setup_env.sh
```

This installs the pinned official
[vLLM release documented to support GPT-OSS](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html)
and records the resolved packages in the ignored
`runtime/environment.freeze.txt`. The script does not modify an existing Conda
environment.

The profile uses tensor parallelism across all eight GPUs, initially allowing
eight concurrent sequences and 4,096 batched tokens. These values can be
overridden at submission time and should be benchmarked against the target
workload before increasing them.

Before submission, adapt the `#SBATCH` resource directives in
`slurm/serve.sbatch` to the local cluster, especially the partition, account,
QoS, and time limit.

## Start the server

On the module-based superpod cluster, use the startup helper from a cluster
terminal. It loads `gcc/15.2.0` and `python/cpu/3.10.6`, clears conflicting
Python/compiler paths, selects this checkout's `.venv`, creates the log directory,
and submits GPT-OSS on eight GPUs on `sp-0003` for five days:

```bash
./scripts/start_gpt_oss.sh
```

The interpreter check runs without GPU detection; CUDA is checked inside the
GPU job. Existing credentials remain in `~/.config/cluster-llm-server/server.env`.
Override the scheduling settings or pass additional Slurm options as needed:

```bash
NODE=sp-0004 WALLTIME=2-00:00:00 ./scripts/start_gpt_oss.sh --account=your-account
```

`PARTITION`, `NODE` and `WALLTIME` control scheduling. `SERVER_GCC_MODULE` and
`SERVER_PYTHON_MODULE` select modules; `SERVER_VENV_DIR` selects another installed
server environment. Executing the helper sets the job environment without
changing the calling terminal.

For a cluster with its environment already configured, submit the launcher
directly. The Slurm output directory must exist before submission:

```bash
cd /path/to/cluster-llm-server
mkdir -p logs
sbatch slurm/serve.sbatch
```

The job checks the GPU, downloads the pinned model revision, starts vLLM, and
runs authenticated Chat Completions and Responses API smoke tests. When ready,
the log prints the OpenAI-compatible base URL and served model name. The API key
is never printed. The non-secret runtime configuration is written under
`runtime/<job-id>/manifest.json`. If the default compute-node hostname is not
reachable from client jobs, submit with `ADVERTISE_HOST` set to the appropriate
private hostname or address.

## Connect a client

From a cluster host that can reach the compute node, load the key and call the
OpenAI-compatible API:

```bash
source ~/.config/cluster-llm-server/server.env
export OPENAI_BASE_URL="http://<node-host>:8000/v1"
export OPENAI_API_KEY="$VLLM_API_KEY"

curl --fail-with-body "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 128
  }'
```

Clients must run inside the private network unless the cluster provides an
approved tunnel or gateway. Do not expose the server directly to the public
internet.

`VLLM_API_KEY` authenticates the OpenAI-compatible API routes; it is not a
network security boundary for the whole HTTP service. Restrict access with the
cluster network or an approved authenticated gateway.

## Validate long context

After the normal smoke test passes, run the near-limit check from a cluster
host that can reach the server:

```bash
source ~/.config/cluster-llm-server/server.env
source .venv/bin/activate
python scripts/verify_server.py \
  --base-url "http://<node-host>:8000/v1" \
  --model openai/gpt-oss-120b \
  --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --long-context-tokens 120000 \
  --timeout 3600
```

If startup or this request fails, keep the failure visible. Do not silently
enable FP8 KV cache or reduce the requested context length.

## Boundaries

- No model weights, prompts, responses, secrets, logs, or runtime artifacts are
  committed.
- No fixed compute node, personal path, or default credential is stored.
- The deployment is one tensor-parallel replica across eight H100s.
  Multi-replica serving and TensorRT-LLM remain deferred.

## Three models on one node: 2 + 2 + 4 GPUs

`slurm/serve_three.sbatch` allocates eight H100 80 GB GPUs and starts three
exclusive Slurm steps: Llama-3.3-70B on two GPUs (port 8001), Med42-v2-70B on
two (8002), and GPT-OSS-120B on four (8000). Each step sees only its assigned
GPUs. Profiles pin model revisions and keep separate logs and manifests under
`runtime/<job-id>/<profile>/manifest.json`. The original TP8 launcher remains
available. The allocation stops if any server exits.

Prerequisites: the existing environment and `server.env`, accepted access to
Meta's gated Llama weights and an `HF_TOKEN` with access, sufficient shared
cache space (allow at least 500 GB for all three uncached models), a Slurm configuration supporting GPU allocation per job step. Prefetch
models before reserving the GPU node when possible. The three-model job requests
512 GB host RAM; adapt partition/account/time to the cluster.

```bash
mkdir -p logs
sbatch slurm/serve_three.sbatch
```

Check all three instance logs for `Server ready` and all three manifests before
starting clients. Base URLs share the compute-node hostname but use the ports
above; request model names are `llama33-70b`, `med42-v2-70b`, and
`gpt-oss-120b-tp4`. Names are aliases; manifests retain actual checkpoints and
revisions. The ordinary Llama/Med42 smoke test uses Chat Completions without
GPT-OSS reasoning or Responses API requirements.

Both generation profiles use 8,192 total context tokens, one concurrent
sequence and unquantized model defaults. Proposed client output cap: 2,048
with the same input evidence for both models. This is a deployment candidate,
not a demonstrated fit: two-GPU 70B startup and near-limit requests must pass
on the HPC. If memory is insufficient, do not silently quantize or truncate;
use the four-GPU GPT-OSS plus one four-GPU generator arrangement sequentially.
The GPT-OSS four-GPU profile also requires long-context load validation.

## Gemma and MedGemma on one SP node: 4 + 4 GPUs

`slurm/serve_gemma_medgemma.sbatch` starts the following pair on one node with
eight H100 80 GB GPUs, using two exclusive four-GPU Slurm steps:

| Profile / request model name | Hugging Face checkpoint | Port |
| --- | --- | --- |
| `gemma3-27b` | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 8003 |
| `medgemma-27b-text` | [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) | 8004 |

The profiles pin exact checkpoint revisions. Both use BF16 weights without
quantization, an 8,192-token total context limit, one concurrent sequence, and
2,048 batched tokens, matching the existing generation pair's context and
batching limits. Gemma's image inputs are disabled; MedGemma is the text-only
27B checkpoint. The existing vLLM 0.10.2 environment supports both architectures
([supported models](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html)).
Use an identical client output cap (initially 2,048 tokens) and input evidence
for both generators. The context limit includes the prompt, chat template and
output; check token counts rather than silently truncating evidence.

Before starting, accept the Gemma and Health AI Developer Foundations terms on
the two Hugging Face pages using the account associated with `HF_TOKEN`. Put
that read token in the existing mode-600 external `server.env` file, alongside
`VLLM_API_KEY` and `HF_HOME`. Allow at least 150 GB free shared cache space for
the uncached pair; the job requests 256 GB host RAM. Prefetch on the login node
before reserving GPUs when possible:

```bash
cd /path/to/cluster-llm-server
source .venv/bin/activate
set -a
source ~/.config/cluster-llm-server/server.env
set +a
for profile in gemma3-27b medgemma-27b-text; do
    (
        set -a
        source "profiles/${profile}.env"
        set +a
        python scripts/prefetch_model.py
    )
done
```

The startup helper loads the same cluster modules as the GPT-OSS helper and
lets Slurm select an available node in `superpod`:

```bash
./scripts/start_gemma_medgemma.sh
```

To select a particular SP node, set `NODE`; partition, walltime, environment and
module overrides work as for the GPT-OSS helper. Additional Slurm options are
passed through:

```bash
NODE=sp-0004 WALLTIME=2-00:00:00 ./scripts/start_gemma_medgemma.sh --account=your-account
```

Alternatively, with the environment already configured, run
`mkdir -p logs` followed by `sbatch --partition=superpod slurm/serve_gemma_medgemma.sbatch`.
The main Slurm `.out` and `.err` files show live model output, prefixed with
`[gemma3-27b]` or `[medgemma-27b-text]`. Unprefixed per-model copies remain in
`runtime/<job-id>/<profile>.out` and `.err`. Startup messages appear immediately;
an already running job keeps its original logging behavior.

Both servers use the existing authenticated Chat Completions smoke test and
write separate non-secret manifests to
`runtime/<job-id>/<profile>/manifest.json`. Wait for `Server ready` in both
instance logs before using their manifest URLs:
`http://<allocated-host>:8003/v1` and `http://<allocated-host>:8004/v1`.
An exited server stops the whole pair. This job serves only the two generators;
use the separately deployed GPT-OSS endpoint for evaluation.

Local tests validate orchestration and configuration, not model loading. Before
running the study, verify each endpoint on the cluster, including a near-limit
request with the same input/output budget. For example, source the secrets and
the corresponding profile, then run this for each port and served model name:

```bash
python scripts/verify_server.py --mode chat \
    --base-url "http://<allocated-host>:${PORT}/v1" \
    --model "$SERVED_MODEL_NAME" --tokenizer-model "$MODEL_ID" \
    --revision "$MODEL_REVISION" --long-context-tokens 7500 --timeout 3600
```
