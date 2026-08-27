# Cluster LLM server

Minimal Slurm deployment for an authenticated, OpenAI-compatible vLLM server.
The default profile serves `openai/gpt-oss-120b` across eight H100 80 GB GPUs with a
131,072-token context limit.

The server exposes both `/v1/chat/completions` and `/v1/responses`. It is meant
for jobs on the private cluster network, not public internet access.

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

The Slurm output directory must exist before submission:

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
