#!/usr/bin/env python3
"""Authenticated non-streaming Chat Completions proxy for independent replicas."""

from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(host: str, port: int, backends: list[str], api_key: str, limit: int) -> None:
    active = [0] * len(backends)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {api_key}"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200, b'{"status":"ok"}')
                return
            if not self._authorized():
                self._reply(401, b'{"error":"unauthorized"}')
                return
            if self.path != "/v1/models":
                self._reply(404, b'{"error":"not found"}')
                return
            try:
                request = urllib.request.Request(backends[0] + "/v1/models", headers={
                    "Authorization": f"Bearer {api_key}"
                })
                with urllib.request.urlopen(request, timeout=10) as response:
                    self._reply(response.status, response.read())
            except (OSError, urllib.error.URLError):
                self._reply(502, b'{"error":"backend unavailable"}')

        def do_POST(self) -> None:
            if not self._authorized():
                self._reply(401, b'{"error":"unauthorized"}')
                return
            if self.path != "/v1/chat/completions":
                self._reply(404, b'{"error":"not found"}')
                return
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 32 * 1024 * 1024:
                self._reply(413, b'{"error":"invalid request size"}')
                return
            body = self.rfile.read(size)
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict) or payload.get("stream"):
                    raise ValueError("streaming is not supported by this proxy")
            except (ValueError, json.JSONDecodeError):
                self._reply(400, b'{"error":"expected non-streaming JSON request"}')
                return
            with lock:
                choices = [idx for idx, count in enumerate(active) if count < limit]
                if not choices:
                    self._reply(503, b'{"error":"all backends busy"}')
                    return
                index = min(choices, key=lambda idx: (active[idx], idx))
                active[index] += 1
            try:
                request = urllib.request.Request(
                    backends[index] + self.path, data=body, method="POST",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=6000) as response:
                    self._reply(response.status, response.read(), response.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as error:
                self._reply(error.code, error.read())
            except (OSError, urllib.error.URLError):
                self._reply(502, b'{"error":"backend unavailable"}')
            finally:
                with lock:
                    active[index] -= 1

        def log_message(self, format: str, *args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--max-in-flight", type=int, default=16)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("VLLM_API_KEY is required")
    serve(args.host, args.port, args.backend, args.api_key, args.max_in_flight)
