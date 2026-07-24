#!/usr/bin/env python3
"""Stable Python entry point for task-pack correctness."""

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
        "--mode",
        choices=["reference-replay", "snapshot-golden"],
        default=os.environ.get("CORRECTNESS_MODE", "snapshot-golden"),
    )
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--all-priorities", action="store_true")
    parser.add_argument(
        "--output-format",
        choices=["human", "json"],
        default=os.environ.get("OUTPUT_FORMAT", "human"),
    )
    args = parser.parse_args()

    python = os.environ.get("PYTHON") or sys.executable
    command = [
        python,
        "-B",
        "correctness_test.py",
        "--device",
        args.device,
        "--mode",
        args.mode,
        "--output-format",
        args.output_format,
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
