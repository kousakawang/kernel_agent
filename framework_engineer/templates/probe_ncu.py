#!/usr/bin/env python3
"""Probe Nsight Compute and optionally run a command under it."""

from __future__ import annotations

import argparse
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("ncu is not available on PATH")

    version = subprocess.run([ncu, "--version"], check=False)
    if version.returncode != 0:
        return version.returncode
    if not args.command:
        return 0
    return subprocess.run(
        [ncu, "--set", "full", "--target-processes", "all", *args.command],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
