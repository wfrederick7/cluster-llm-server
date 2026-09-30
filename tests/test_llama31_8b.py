from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replica_proxy import serve


class Backend(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        data = json.dumps({"choices": [{"message": {"content": body["model"]}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


class Llama31ServerTests(unittest.TestCase):
    def test_launcher_and_profile_are_valid(self) -> None:
        launcher = ROOT / "slurm" / "serve_llama31_8b.sbatch"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text()
        self.assertIn("#SBATCH --gres=gpu:8", text)
        self.assertIn("server.env", text)
        self.assertIn("replica_proxy.py", text)
        profile = (ROOT / "profiles" / "llama31-8b.env").read_text()
        self.assertIn("Llama-3.1-8B-Instruct", profile)
        self.assertIn('REPLICA_COUNT="8"', profile)

    def test_proxy_requires_key_and_forwards_chat(self) -> None:
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        threading.Thread(
            target=serve,
            args=("127.0.0.1", port, [f"http://127.0.0.1:{backend.server_port}"], "secret", 2),
            daemon=True,
        ).start()
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except urllib.error.URLError:
                time.sleep(0.02)
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({"model": "llama31", "messages": []}).encode(),
                headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "llama31")
            bad = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}"
            )
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad, timeout=3)
            self.assertEqual(error.exception.code, 401)
        finally:
            backend.shutdown()
            backend.server_close()


if __name__ == "__main__":
    unittest.main()
