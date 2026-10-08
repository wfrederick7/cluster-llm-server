from __future__ import annotations

import re
import json
import os
import sys
import tempfile
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class RepositoryTests(unittest.TestCase):
    def test_example_secrets_are_blank(self) -> None:
        values = {}
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual(values["VLLM_API_KEY"], "")
        self.assertEqual(values["HF_TOKEN"], "")
        self.assertEqual(values["HF_HOME"], "")

    def test_no_personal_cluster_values_are_committed(self) -> None:
        forbidden = (
            re.compile(r"/Users/[^/\s]+/"),
            re.compile(r"/gpfs/data/[^/\s]+/users/[A-Za-z0-9._-]+/"),
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        )
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        for relative_path in result.stdout.decode("utf-8").split("\0"):
            if not relative_path:
                continue
            path = ROOT / relative_path
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in forbidden:
                self.assertIsNone(pattern.search(text), str(path))

    def test_slurm_launcher_uses_submission_directory(self) -> None:
        launcher = (ROOT / "slurm" / "serve.sbatch").read_text(encoding="utf-8")
        self.assertIn("SLURM_SUBMIT_DIR", launcher)
        self.assertNotIn('dirname "${BASH_SOURCE[0]}"', launcher)

    def test_server_uses_selected_python_with_stale_console_shebang(self) -> None:
        launcher = (ROOT / "slurm" / "serve.sbatch").read_text(encoding="utf-8")
        start = launcher.index('model_args=()')
        end = launcher.index('SERVER_PID="$!"', start) + len('SERVER_PID="$!"')
        command = 'set -euo pipefail\n' + launcher[start:end] + '\nwait "$SERVER_PID"\n'
        with tempfile.TemporaryDirectory() as directory:
            env_dir = Path(directory) / "selected-env"
            binaries = env_dir / "bin"
            binaries.mkdir(parents=True)
            (binaries / "python").symlink_to(sys.executable)
            console = binaries / "vllm"
            console.write_text(
                "#!/nonexistent-original-env/bin/python\n"
                "import json, sys\n"
                "print(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            console.chmod(0o755)
            environment = {
                **os.environ,
                "VENV_DIR": str(env_dir),
                "MODEL_ID": "synthetic-model",
                "MODEL_REVISION": "synthetic-revision",
                "SERVED_MODEL_NAME": "synthetic-served-name",
                "PORT": "8000",
                "TENSOR_PARALLEL_SIZE": "8",
                "MAX_MODEL_LEN": "131072",
                "MAX_NUM_SEQS": "8",
                "MAX_NUM_BATCHED_TOKENS": "4096",
                "GPU_MEMORY_UTILIZATION": "0.95",
                "DTYPE": "auto",
                "KV_CACHE_DTYPE": "auto",
            }
            result = subprocess.run(
                ["bash", "-c", command], env=environment,
                text=True, capture_output=True, check=True,
            )
            arguments = json.loads(result.stdout)
            self.assertEqual(arguments[:2], ["serve", "synthetic-model"])
            self.assertIn("--tensor-parallel-size", arguments)
            self.assertIn("--enable-prefix-caching", arguments)

            environment["LIMIT_MM_PER_PROMPT"] = '{"image":0}'
            result = subprocess.run(
                ["bash", "-c", command], env=environment,
                text=True, capture_output=True, check=True,
            )
            arguments = json.loads(result.stdout)
            index = arguments.index("--limit-mm-per-prompt")
            self.assertEqual(json.loads(arguments[index + 1]), {"image": 0})

    def test_bootstrap_uses_stable_release_dependencies(self) -> None:
        requirements = (ROOT / "requirements.bootstrap.txt").read_text(
            encoding="utf-8"
        )
        setup = (ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")
        self.assertIn("vllm==0.10.2", requirements)
        self.assertIn("transformers==4.55.2", requirements)
        self.assertNotIn("+gptoss", requirements)
        self.assertNotIn("--pre", setup)
        self.assertNotIn("wheels.vllm.ai/gpt-oss", setup)
        self.assertNotIn("download.pytorch.org/whl/nightly", setup)

    def test_eight_h100_profile_uses_conservative_batching(self) -> None:
        profile = (ROOT / "profiles" / "gpt-oss-120b.env").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"', profile
        )
        self.assertIn('MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"', profile)
        self.assertIn(
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"', profile
        )
        self.assertIn(
            'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"', profile
        )

    def test_launcher_requests_and_checks_all_eight_gpus(self) -> None:
        launcher = (ROOT / "slurm" / "serve.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:8", launcher)
        self.assertIn('torch.cuda.device_count()', launcher)
        self.assertIn("--enforce-eager", launcher)
        self.assertIn("export CC=/usr/bin/gcc", launcher)


if __name__ == "__main__":
    unittest.main()
