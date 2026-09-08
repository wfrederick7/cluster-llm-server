"""Exercise Slurm orchestration without a GPU or a running Slurm controller."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ThreeServerTests(unittest.TestCase):
    def test_three_disjoint_steps_and_failure_propagation(self):
        # macOS /bin/bash lacks wait -n; use the HPC-compatible installed bash.
        bash = "/opt/homebrew/bin/bash"
        if not Path(bash).exists():
            bash = "bash"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "srun").write_text(
                '#!/bin/sh\nprintf "%s %s %s\\n" "$INSTANCE_NAME" "$PROFILE" "$*" >> "$CALL_LOG"\nsleep 0.2\nexit 7\n'
            )
            (root / "srun").chmod(0o755)
            env = os.environ | {
                "PATH": str(root) + ":" + os.environ["PATH"],
                "SERVER_ROOT": str(root),
                "SLURM_JOB_ID": "123",
                "CALL_LOG": str(root / "calls"),
            }
            result = subprocess.run(
                [bash, str(ROOT / "slurm/serve_three.sbatch")],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            lines = (root / "calls").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(sum("--gpus-per-task=2" in s for s in lines), 2)
            self.assertEqual(sum("--gpus-per-task=4" in s for s in lines), 1)
            self.assertTrue(all("--exclusive" in s for s in lines))

    def test_all_shell_files_parse(self):
        for path in list((ROOT / "slurm").glob("*.sbatch")) + list(
            (ROOT / "profiles").glob("*.env")
        ):
            subprocess.run(["bash", "-n", str(path)], check=True)
