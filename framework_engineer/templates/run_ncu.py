#!/usr/bin/env python3
"""Profile one selected snapshot group with Nsight Compute."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("group_id")
    parser.add_argument("sample_id", nargs="?", default=None)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--target",
        choices=["reference", "candidate", "both"],
        default=os.environ.get("TARGET", "candidate"),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "5")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "20")))
    args = parser.parse_args()

    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("ncu is not available on PATH")
    python = os.environ.get("PYTHON") or sys.executable
    benchmark = [
        python,
        "-B",
        "benchmark.py",
        "--group-id",
        args.group_id,
        "--device",
        args.device,
        "--target",
        args.target,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    if args.sample_id:
        benchmark.extend(["--sample-id", args.sample_id])
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [ncu, "--set", "full", "--target-processes", "all", *benchmark],
        cwd=TASK_DIR,
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
