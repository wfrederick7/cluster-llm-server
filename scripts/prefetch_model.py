#!/usr/bin/env python3
"""Download one pinned Hugging Face model snapshot into the shared cache."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_ID"))
    parser.add_argument("--revision", default=os.getenv("MODEL_REVISION"))
    parser.add_argument(
        "--min-free-gb", type=int, default=int(os.getenv("MIN_FREE_GB", "150"))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = str(args.model or "").strip()
    revision = str(args.revision or "").strip()
    hf_home = str(os.getenv("HF_HOME") or "").strip()
    if not model or not revision:
        raise SystemExit("MODEL_ID and MODEL_REVISION are required")
    if not hf_home:
        raise SystemExit("HF_HOME must point to a cache outside the repository")

    cache_root = Path(hf_home).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    token = str(os.getenv("HF_TOKEN") or "").strip() or None

    try:
        snapshot = snapshot_download(
            repo_id=model,
            revision=revision,
            token=token,
            local_files_only=True,
        )
        print(f"Model snapshot already cached: {snapshot}")
        return
    except LocalEntryNotFoundError:
        pass

    free_gb = shutil.disk_usage(cache_root).free / (1024**3)
    if free_gb < int(args.min_free_gb):
        raise SystemExit(
            f"HF_HOME has {free_gb:.1f} GB free; {args.min_free_gb} GB is required"
        )

    snapshot = snapshot_download(
        repo_id=model,
        revision=revision,
        token=token,
    )
    print(f"Downloaded {model}@{revision} to {snapshot}")


if __name__ == "__main__":
    main()
