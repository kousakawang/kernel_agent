"""Validation helpers for snapshot task packs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..task_pack_layout import task_dir


REQUIRED_TASK_PACK_FILES = [
    "README.md",
    "validate_task_pack.py",
    "task/task.yaml",
    "task/shape_list.json",
    "task/env_manifest.yaml",
    "task/snapshot_runtime.py",
    "task/snapshots/manifest.json",
    "task/original_impl.py",
    "task/reference_impl.py",
    "task/candidate_impl.py",
    "task/correctness_test.py",
    "task/benchmark.py",
    "task/scripts/run_correctness.py",
    "task/scripts/run_benchmark.py",
    "task/scripts/run_ncu.py",
    "task/kernel_translate/README.md",
    "task/kernel_engineer_ws/README.md",
]


def validate_files(task_pack: Path) -> list[str]:
    return validate_structure(task_pack)["errors"]


def validate_structure(task_pack: Path) -> dict[str, Any]:
    file_errors = []
    snapshot_errors = []
    missing = []
    present = []
    errors = []
    for rel in REQUIRED_TASK_PACK_FILES:
        if not (task_pack / rel).exists():
            message = f"missing required file: {rel}"
            file_errors.append(message)
            missing.append(rel)
        else:
            present.append(rel)
    payload = task_dir(task_pack)
    manifest = payload / "snapshots" / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not data.get("case_groups"):
            snapshot_errors.append("snapshots/manifest.json has no selected case_groups")
        for group in data.get("case_groups", []):
            group_dir = payload / "snapshots" / "selected" / group["group_id"]
            if not (group_dir / "group_meta.json").exists():
                snapshot_errors.append(f"missing group_meta.json for {group['group_id']}")
            for sample in group.get("samples", []):
                sample_dir = group_dir / "samples" / sample["sample_id"]
                for rel in ("meta.json", "pre_inputs.pt", "post_inputs.pt", "outputs.pt"):
                    if not (sample_dir / rel).exists():
                        snapshot_errors.append(f"missing snapshot file for {group['group_id']}/{sample['sample_id']}: {rel}")
    errors = file_errors + snapshot_errors
    return {
        "errors": errors,
        "file_check": {
            "status": "passed" if not file_errors else "failed",
            "missing": missing,
            "present_count": len(present),
            "required_count": len(REQUIRED_TASK_PACK_FILES),
            "errors": file_errors,
        },
        "snapshot_check": {
            "status": "passed" if not snapshot_errors else "failed",
            "errors": snapshot_errors,
        },
    }


def run_smoke(task_pack: Path, *, correctness: bool, benchmark: bool, timeout: int) -> list[dict[str, Any]]:
    results = []
    commands = []
    if correctness:
        commands.append([sys.executable, "-B", "task/scripts/run_correctness.py"])
    if benchmark:
        commands.append([sys.executable, "-B", "task/scripts/run_benchmark.py"])
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=task_pack,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        results.append(
            {
                "command": " ".join(cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        )
    return results
