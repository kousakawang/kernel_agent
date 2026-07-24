#!/usr/bin/env python3
"""Stable Python entry point for task-pack benchmarks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--target",
        choices=["reference", "candidate", "both"],
        default=os.environ.get("TARGET", "both"),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "20")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "100")))
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    python = os.environ.get("PYTHON") or sys.executable
    command = [
        python,
        "-B",
        "benchmark.py",
        "--device",
        args.device,
        "--target",
        args.target,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    if args.group_id:
        command.extend(["--group-id", args.group_id])
    if args.sample_id:
        command.extend(["--sample-id", args.sample_id])
    if args.all_priorities:
        command.append("--all-priorities")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(command, cwd=TASK_DIR, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
