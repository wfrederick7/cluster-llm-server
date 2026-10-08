"""Check paired GPU allocation, failure cleanup and login-node submission."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import test_start_gpt_oss as gpt_start

ROOT = Path(__file__).resolve().parents[1]


class GemmaStartupTests(gpt_start.StartupTests):
    def setUp(self):
        super().setUp()
        self.script = self.checkout / "scripts" / "start_gemma_medgemma.sh"
        shutil.copy2(ROOT / "scripts" / "start_gemma_medgemma.sh", self.script)

    def test_loads_modules_and_submits_from_selected_checkout(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(self.submission.read_text())
        self.assertEqual(self.module_log.read_text().splitlines(),
                         ["load gcc/15.2.0", "load python/cpu/3.10.6"])
        self.assertEqual(payload["args"][:3],
                         ["--partition=superpod", "--time=3-00:00:00", "--export=ALL"])
        self.assertFalse(any(arg.startswith("--nodelist") for arg in payload["args"]))
        self.assertEqual(Path(payload["cwd"]).resolve(), self.checkout.resolve())
        self.assertEqual(payload["env"]["VENV_DIR"], str(self.checkout / ".venv"))
        self.assertEqual(payload["env"]["VIRTUAL_ENV"], str(self.checkout / ".venv"))
        for key in ("PYTHONPATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH"):
            self.assertIsNone(payload["env"][key])
        self.assertIn("loaded-module-libs", payload["env"]["LD_LIBRARY_PATH"])
        self.assertEqual(payload["args"][-1],
                         str(self.checkout / "slurm" / "serve_gemma_medgemma.sbatch"))
        self.assertTrue((self.checkout / "logs").is_dir())

    def test_overrides_and_additional_slurm_options(self):
        self.environment.update(NODE="sp-0004", WALLTIME="2-00:00:00",
                                PARTITION="other-partition")
        result = self.run_helper("--account=research-account")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.submission.read_text())["args"]
        self.assertEqual(args[:4], ["--partition=other-partition", "--time=2-00:00:00",
                                   "--nodelist=sp-0004", "--export=ALL"])
        self.assertIn("--account=research-account", args)


class GemmaPairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        stub = self.root / "srun.py"
        stub.write_text(
            'import json, os, signal, sys, time\n'
            'from pathlib import Path\n'
            'name = os.environ["INSTANCE_NAME"]\n'
            'def stopped(*_):\n'
            '    Path(os.environ["STUB_ROOT"], name + ".stopped").touch()\n'
            '    sys.exit(0)\n'
            'signal.signal(signal.SIGTERM, stopped)\n'
            'with open(os.environ["CALL_LOG"], "a") as log:\n'
            '    log.write(json.dumps({"name": name, "profile": os.environ["PROFILE"], '
            '"args": sys.argv[1:]}) + "\\n")\n'
            'if name == os.environ.get("FAIL_INSTANCE"):\n'
            '    time.sleep(0.3)\n'
            '    sys.exit(int(os.environ["STEP_EXIT"]))\n'
            'while True:\n'
            '    time.sleep(0.05)\n'
        )
        binary = self.root / "srun"
        binary.write_text('#!/bin/sh\nexec "$STUB_PYTHON" "$STUB_SRUN_IMPL" "$@"\n')
        binary.chmod(0o755)
        self.environment = os.environ | {
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "SERVER_ROOT": str(self.root), "SLURM_JOB_ID": "123",
            "STUB_ROOT": str(self.root), "CALL_LOG": str(self.root / "calls"),
            "STUB_PYTHON": sys.executable, "STUB_SRUN_IMPL": str(stub),
        }

    def start_pair(self):
        process = subprocess.Popen(
            ["bash", str(ROOT / "slurm/serve_gemma_medgemma.sbatch")],
            env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.addCleanup(self.stop_pair, process)
        return process

    @staticmethod
    def stop_pair(process):
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)

    def calls(self):
        path = self.root / "calls"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_disjoint_four_gpu_steps_and_companion_cleanup(self):
        for step_exit in (0, 7):
            with self.subTest(step_exit=step_exit):
                self.environment.update(FAIL_INSTANCE="gemma3-27b", STEP_EXIT=str(step_exit))
                (self.root / "calls").unlink(missing_ok=True)
                stopped = self.root / "medgemma-27b-text.stopped"
                stopped.unlink(missing_ok=True)
                process = self.start_pair()
                _, error = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 1, error)
                calls = self.calls()
                self.assertEqual({call["name"] for call in calls},
                                 {"gemma3-27b", "medgemma-27b-text"})
                self.assertEqual(len(calls), 2)
                for call in calls:
                    self.assertIn("--gpus-per-task=4", call["args"])
                    self.assertIn("--exclusive", call["args"])
                    self.assertIn("--nodes=1", call["args"])
                    self.assertEqual(call["profile"],
                                     str(self.root / "profiles" / (call["name"] + ".env")))
                self.assertTrue(stopped.exists(), "The surviving server must be stopped")

    def test_cancelling_allocation_stops_both_steps(self):
        process = self.start_pair()
        deadline = time.monotonic() + 5
        while len(self.calls()) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(len(self.calls()), 2)
        process.terminate()
        process.communicate(timeout=10)
        self.assertEqual(process.returncode, 143)
        for name in ("gemma3-27b", "medgemma-27b-text"):
            self.assertTrue((self.root / (name + ".stopped")).exists())


if __name__ == "__main__":
    unittest.main()
