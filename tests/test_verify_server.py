from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "verify_server.py"
SPEC = importlib.util.spec_from_file_location("verify_server", MODULE_PATH)
assert SPEC and SPEC.loader
verify_server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_server)


class VerifyServerTests(unittest.TestCase):
    def test_normalize_base_url(self) -> None:
        self.assertEqual(
            verify_server.normalize_base_url("http://node:8000"),
            "http://node:8000/v1",
        )
        self.assertEqual(
            verify_server.normalize_base_url(
                "http://node:8000/v1/chat/completions"
            ),
            "http://node:8000/v1",
        )

    def test_extracts_first_json_object(self) -> None:
        self.assertEqual(
            verify_server.first_json_object('Reasoning omitted. {"status":"ok"}'),
            {"status": "ok"},
        )

    def test_chat_payload_uses_openai_compatible_transport(self) -> None:
        payload = verify_server.chat_payload("model", "prompt")
        self.assertEqual(payload["model"], "model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
