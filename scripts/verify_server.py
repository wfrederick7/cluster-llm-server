#!/usr/bin/env python3
"""Verify the authenticated GPT-OSS vLLM API without logging response text."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any


def normalize_base_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    if not url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return url


def first_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    if start < 0:
        raise ValueError("assistant response contained no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(raw[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("assistant JSON was not an object")
                return value
    raise ValueError("assistant response contained incomplete JSON")


def request_json(
    url: str,
    *,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return int(response.status), body
    except urllib.error.HTTPError as error:
        body_raw = error.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body_raw)
        except json.JSONDecodeError:
            body = {"error": "non-JSON error response"}
        return int(error.code), body


def chat_payload(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 2048,
    reasoning_effort: str = "high",
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 123,
        "stream": False,
        "frequency_penalty": 0,
        "reasoning_effort": reasoning_effort,
    }


def assistant_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response contains no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("content") is None:
        raise ValueError("chat response contains no assistant content")
    return str(message["content"])


def build_long_prompt(model: str, revision: str, target_tokens: int) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    unit = " context token"
    low, high = 1, max(2, target_tokens * 2)
    while low < high:
        middle = (low + high + 1) // 2
        count = len(tokenizer.encode(unit * middle, add_special_tokens=False))
        if count <= target_tokens:
            low = middle
        else:
            high = middle - 1
    prompt = unit * low
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual < int(target_tokens * 0.98):
        raise RuntimeError(f"could not construct the requested prompt length: {actual}")
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=os.getenv("MODEL_REVISION", ""))
    parser.add_argument("--long-context-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = normalize_base_url(args.base_url)
    api_key = str(os.getenv("VLLM_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("VLLM_API_KEY is required in the environment")

    status, _ = request_json(f"{base_url}/models", api_key=None, timeout=args.timeout)
    if status != 401:
        raise RuntimeError(f"unauthenticated request returned HTTP {status}, expected 401")
    status, _ = request_json(
        f"{base_url}/models", api_key=f"{api_key}-incorrect", timeout=args.timeout
    )
    if status != 401:
        raise RuntimeError(f"incorrect API key returned HTTP {status}, expected 401")

    status, models = request_json(
        f"{base_url}/models", api_key=api_key, timeout=args.timeout
    )
    if status != 200 or not isinstance(models.get("data"), list):
        raise RuntimeError("authenticated model listing failed")

    status, chat = request_json(
        f"{base_url}/chat/completions",
        api_key=api_key,
        payload=chat_payload(
            args.model,
            'Return only this JSON object: {"status":"ok"}',
        ),
        timeout=args.timeout,
    )
    if status != 200:
        raise RuntimeError(f"Chat Completions returned HTTP {status}")
    parsed = first_json_object(assistant_content(chat))
    if parsed.get("status") != "ok":
        raise RuntimeError("assistant JSON smoke test returned an unexpected value")

    status, responses = request_json(
        f"{base_url}/responses",
        api_key=api_key,
        payload={
            "model": args.model,
            "input": "Return exactly: ok",
            "max_output_tokens": 512,
        },
        timeout=args.timeout,
    )
    if status != 200 or not responses.get("output"):
        raise RuntimeError("Responses API smoke test failed")

    if args.long_context_tokens:
        if not args.revision:
            raise RuntimeError("--revision is required for the long-context test")
        prompt = build_long_prompt(
            args.model, args.revision, int(args.long_context_tokens)
        )
        payload = chat_payload(
            args.model,
            prompt,
            max_tokens=512,
            reasoning_effort="low",
        )
        status, chat = request_json(
            f"{base_url}/chat/completions",
            api_key=api_key,
            payload=payload,
            timeout=max(args.timeout, 3600),
        )
        if status != 200 or not assistant_content(chat).strip():
            raise RuntimeError("long-context Chat Completions test failed")

    print("Server verification passed")


if __name__ == "__main__":
    main()
