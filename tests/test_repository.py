from __future__ import annotations

import re
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

    def test_single_h100_profile_uses_conservative_memory_settings(self) -> None:
        profile = (ROOT / "profiles" / "gpt-oss-120b.env").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"', profile
        )
        self.assertIn(
            'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"', profile
        )


if __name__ == "__main__":
    unittest.main()
