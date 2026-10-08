from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.checkout = root / "renamed checkout"
        (self.checkout / "scripts").mkdir(parents=True)
        self.script = self.checkout / "scripts" / "start_gpt_oss.sh"
        shutil.copy2(ROOT / "scripts" / "start_gpt_oss.sh", self.script)
        venv_bin = self.checkout / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        python = venv_bin / "python"
        python.write_text('#!/bin/bash\n[[ "$1" == "-c" ]] || exit 1\n'
                          '[[ "$2" != *cuda* && "$2" != *vllm* ]] || exit 1\n'
                          'exit "${STUB_PYTHON_EXIT:-0}"\n')
        python.chmod(0o755)
        # Stale console/activation paths must never be followed on the CPU node.
        (venv_bin / "vllm").write_text("#!/missing-old-interpreter\n")
        (venv_bin / "activate").write_text("exit 98\n")
        module_home = root / "modules"
        (module_home / "init").mkdir(parents=True)
        (module_home / "init" / "bash").write_text(
            'module() { printf "%s\\n" "$*" >> "$MODULE_LOG"; '
            'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:loaded-module-libs"; }\n')
        binaries = root / "bin"
        binaries.mkdir()
        sbatch = binaries / "sbatch"
        submission_impl = root / "submit.py"
        sbatch.write_text('#!/bin/bash\nexec "$STUB_INTERPRETER" "$STUB_SUBMIT_IMPL" "$@"\n')
        submission_impl.write_text(
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'keys = ["SERVER_ROOT", "VENV_DIR", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "PATH"]\n'
            'keys += ["PYTHONPATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH"]\n'
            'Path(os.environ["SUBMISSION"]).write_text(json.dumps({'
            '"args": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}, "cwd": os.getcwd()}))\n')
        sbatch.chmod(0o755)
        self.submission = root / "submission.json"
        self.module_log = root / "module.log"
        self.environment = dict(os.environ)
        for key in list(self.environment):
            if key.startswith("BASH_FUNC_") or key in {"PARTITION", "NODE", "WALLTIME", "ADVERTISE_HOST", "SERVER_VENV_DIR", "SERVER_GCC_MODULE", "SERVER_PYTHON_MODULE"}:
                self.environment.pop(key)
        self.environment.update(PATH=str(binaries) + os.pathsep + os.defpath,
            MODULESHOME=str(module_home), MODULE_LOG=str(self.module_log), SUBMISSION=str(self.submission),
            STUB_INTERPRETER=sys.executable, STUB_SUBMIT_IMPL=str(submission_impl),
            SERVER_ROOT="wrong-checkout", VENV_DIR="wrong-env", VIRTUAL_ENV="wrong-env",
            PYTHONPATH="wrong-packages", GCC_EXEC_PREFIX="wrong-gcc", COMPILER_PATH="wrong-compiler", LIBRARY_PATH="wrong-libs")

    def run_helper(self, *args):
        return subprocess.run(["bash", str(self.script), *args],
            env=self.environment, cwd=self.checkout.parent, text=True, capture_output=True)

    def test_loads_modules_and_submits_from_selected_checkout(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(self.submission.read_text())
        self.assertEqual(self.module_log.read_text().splitlines(), ["load gcc/15.2.0", "load python/cpu/3.10.6"])
        self.assertEqual(payload["args"][:3], ["--partition=superpod", "--nodelist=sp-0003", "--time=5-00:00:00"])
        self.assertEqual(Path(payload["cwd"]).resolve(), self.checkout.resolve())
        self.assertEqual(payload["env"]["VENV_DIR"], str(self.checkout / ".venv"))
        self.assertEqual(payload["env"]["VIRTUAL_ENV"], str(self.checkout / ".venv"))
        for key in ("PYTHONPATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH"):
            self.assertIsNone(payload["env"][key])
        self.assertIn("loaded-module-libs", payload["env"]["LD_LIBRARY_PATH"])
        self.assertEqual(payload["args"][-1], str(self.checkout / "slurm" / "serve.sbatch"))
        self.assertTrue((self.checkout / "logs").is_dir())

    def test_overrides_and_additional_slurm_options(self):
        self.environment.update(NODE="sp-0004", WALLTIME="2-00:00:00", PARTITION="other-partition")
        result = self.run_helper("--account=research-account")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.submission.read_text())["args"]
        self.assertEqual(args[:3], ["--partition=other-partition", "--nodelist=sp-0004", "--time=2-00:00:00"])
        self.assertIn("ADVERTISE_HOST=sp-0004", args[3])
        self.assertIn("--account=research-account", args)

    def test_interpreter_failure_prevents_submission(self):
        self.environment["STUB_PYTHON_EXIT"] = "127"
        result = self.run_helper()
        self.assertEqual(result.returncode, 127)
        self.assertFalse(self.submission.exists())


if __name__ == "__main__":
    unittest.main()
