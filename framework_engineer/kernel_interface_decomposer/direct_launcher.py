"""Run a direct KID workload once for warmup and once during collection.

This process is launched as the application owned by an interactive Nsight
Systems session.  Keeping it alive around both child commands lets the Runtime
Capture runner start and stop collection without attaching a second,
unrelated process to the session.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_file(path: Path, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {description}: {path}")


def _run_command(command: str, *, workdir: Path, log_path: Path, timeout: float) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=workdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nKID command timed out after {timeout:.3f}s\n")
            return 124
    return int(result.returncode)


def run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    warmup_ready = Path(args.warmup_ready_file)
    gate = Path(args.recording_gate_file)
    done = Path(args.test_done_file)
    shutdown = Path(args.shutdown_file)
    for marker in (warmup_ready, gate, done, shutdown):
        marker.unlink(missing_ok=True)

    warmup_returncode = _run_command(
        args.command,
        workdir=workdir,
        log_path=Path(args.warmup_log),
        timeout=args.timeout_sec,
    )
    if warmup_returncode != 0:
        _write_json(
            done,
            {"phase": "warmup", "returncode": warmup_returncode},
        )
        return warmup_returncode

    warmup_ready.touch()
    try:
        _wait_for_file(gate, args.timeout_sec, "recording gate")
    except TimeoutError:
        _write_json(done, {"phase": "gate_wait", "returncode": 124})
        return 124

    test_returncode = _run_command(
        args.command,
        workdir=workdir,
        log_path=Path(args.test_log),
        timeout=args.timeout_sec,
    )
    _write_json(done, {"phase": "test", "returncode": test_returncode})

    # Keep the Nsight-owned process alive until the runner has stopped and
    # exported the interactive session.
    try:
        _wait_for_file(shutdown, args.timeout_sec, "runner shutdown")
    except TimeoutError:
        return 124
    return test_returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--warmup-log", required=True)
    parser.add_argument("--test-log", required=True)
    parser.add_argument("--warmup-ready-file", required=True)
    parser.add_argument("--recording-gate-file", required=True)
    parser.add_argument("--test-done-file", required=True)
    parser.add_argument("--shutdown-file", required=True)
    parser.add_argument("--timeout-sec", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
