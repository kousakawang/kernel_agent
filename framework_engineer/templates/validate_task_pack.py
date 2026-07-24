#!/usr/bin/env python3
"""Self-contained delivery validator for a Framework Engineer task pack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TASK = ROOT / "task"
DOCS = TASK / "docs"
EXPECTED_ROOT_ENTRIES = {"README.md", "validate_task_pack.py", "task"}
FORBIDDEN_DIRECTORY_NAMES = {"original_source", "kernel_sources", "__pycache__"}
REQUIRED_FILES = [
    "task.yaml",
    "shape_list.json",
    "env_manifest.yaml",
    "snapshot_runtime.py",
    "snapshots/manifest.json",
    "original_impl.py",
    "reference_impl.py",
    "candidate_impl.py",
    "correctness_test.py",
    "benchmark.py",
    "scripts/run_correctness.py",
    "scripts/run_benchmark.py",
    "scripts/run_ncu.py",
    "env_probe/probe_triton.py",
    "env_probe/probe_cutedsl.py",
    "env_probe/probe_cuda_extension.py",
    "env_probe/probe_ncu.py",
    "kernel_translate/README.md",
    "kernel_engineer_ws/README.md",
]
REQUIRED_DIRECTORIES = [
    "docs",
    "env_probe",
    "scripts",
    "snapshots",
    "snapshots/raw",
    "snapshots/selected",
    "kernel_translate",
    "kernel_engineer_ws",
]
REQUIRED_POLICY_SNIPPETS = [
    "kernel_translate_scope:",
    "task/kernel_translate/",
    "kernel_engineer_scope:",
    "task/candidate_impl.py",
    "task/kernel_engineer_ws/",
    "task/docs/",
]


def _status(errors: list[str]) -> str:
    return "failed" if errors else "passed"


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": proc.returncode,
            "elapsed_sec": time.time() - started,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except FileNotFoundError as exc:
        return {
            "command": " ".join(command),
            "returncode": 127,
            "elapsed_sec": time.time() - started,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": 124,
            "elapsed_sec": time.time() - started,
            "stdout": _text(exc.stdout)[-8000:],
            "stderr": (_text(exc.stderr) or f"timed out after {timeout}s")[-8000:],
        }


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _structure_check() -> dict[str, Any]:
    errors: list[str] = []
    actual_root_entries = {path.name for path in ROOT.iterdir()}
    missing_root = sorted(EXPECTED_ROOT_ENTRIES - actual_root_entries)
    unexpected_root = sorted(actual_root_entries - EXPECTED_ROOT_ENTRIES)
    errors.extend(f"missing required root entry: {name}" for name in missing_root)
    errors.extend(f"unexpected root entry: {name}" for name in unexpected_root)

    missing_files = [rel for rel in REQUIRED_FILES if not (TASK / rel).is_file()]
    missing_dirs = [rel for rel in REQUIRED_DIRECTORIES if not (TASK / rel).is_dir()]
    errors.extend(f"missing required file: task/{rel}" for rel in missing_files)
    errors.extend(f"missing required directory: task/{rel}" for rel in missing_dirs)

    forbidden_paths: list[str] = []
    shell_files: list[str] = []
    if TASK.exists():
        for path in TASK.rglob("*"):
            relative = str(path.relative_to(ROOT))
            if path.is_dir() and path.name in FORBIDDEN_DIRECTORY_NAMES:
                forbidden_paths.append(relative)
            if path.is_file() and path.suffix == ".sh":
                shell_files.append(relative)
    errors.extend(f"forbidden directory: {path}" for path in sorted(forbidden_paths))
    errors.extend(f"shell script is not allowed: {path}" for path in sorted(shell_files))
    return {
        "status": _status(errors),
        "errors": errors,
        "missing_files": missing_files,
        "missing_directories": missing_dirs,
        "unexpected_root_entries": unexpected_root,
        "forbidden_paths": sorted(forbidden_paths),
        "shell_files": sorted(shell_files),
    }


def _workspace_policy_check() -> dict[str, Any]:
    errors: list[str] = []
    task_yaml = TASK / "task.yaml"
    text = task_yaml.read_text(encoding="utf-8") if task_yaml.is_file() else ""
    missing = [snippet for snippet in REQUIRED_POLICY_SNIPPETS if snippet not in text]
    errors.extend(f"task.yaml missing workspace policy declaration: {snippet}" for snippet in missing)
    return {"status": _status(errors), "errors": errors, "missing_declarations": missing}


def _syntax_check() -> dict[str, Any]:
    errors: list[str] = []
    checked: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        checked.append(relative)
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            errors.append(f"invalid Python syntax: {relative}: {exc}")
    return {"status": _status(errors), "errors": errors, "checked_files": checked}


def _snapshot_check() -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = TASK / "snapshots" / "manifest.json"
    group_count = 0
    sample_count = 0
    if not manifest_path.is_file():
        errors.append("missing task/snapshots/manifest.json")
        return {
            "status": "failed",
            "errors": errors,
            "selected_group_count": 0,
            "selected_sample_count": 0,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid task/snapshots/manifest.json: {exc!r}")
        return {
            "status": "failed",
            "errors": errors,
            "selected_group_count": 0,
            "selected_sample_count": 0,
        }
    groups = manifest.get("case_groups", [])
    if not isinstance(groups, list) or not groups:
        errors.append("task/snapshots/manifest.json has no selected case_groups")
        groups = []
    for group in groups:
        group_id = group.get("group_id")
        if not group_id:
            errors.append("snapshot group is missing group_id")
            continue
        group_count += 1
        group_dir = TASK / "snapshots" / "selected" / str(group_id)
        if not (group_dir / "group_meta.json").is_file():
            errors.append(f"missing group_meta.json for {group_id}")
        samples = group.get("samples", [])
        if not isinstance(samples, list) or not samples:
            errors.append(f"snapshot group has no samples: {group_id}")
            continue
        for sample in samples:
            sample_id = sample.get("sample_id")
            if not sample_id:
                errors.append(f"snapshot sample is missing sample_id in {group_id}")
                continue
            sample_count += 1
            sample_dir = group_dir / "samples" / str(sample_id)
            for filename in ("meta.json", "pre_inputs.pt", "post_inputs.pt", "outputs.pt"):
                if not (sample_dir / filename).is_file():
                    errors.append(f"missing snapshot file for {group_id}/{sample_id}: {filename}")
    return {
        "status": _status(errors),
        "errors": errors,
        "selected_group_count": group_count,
        "selected_sample_count": sample_count,
    }


def _kernel_source_package_check() -> dict[str, Any]:
    errors: list[str] = []
    task_yaml_path = TASK / "task.yaml"
    task_yaml = task_yaml_path.read_text(encoding="utf-8") if task_yaml_path.is_file() else ""
    configured = 'kernel_source_package: "task/kernel_source_package"' in task_yaml
    package = TASK / "kernel_source_package"
    if not configured:
        return {"status": "skipped", "configured": False, "errors": []}
    if not package.is_dir():
        errors.append("task.yaml declares kernel_source_package but task/kernel_source_package is missing")
        return {"status": "failed", "configured": True, "errors": errors}
    json_files = sorted(path.name for path in package.glob("*.json") if path.is_file())
    source_dirs = sorted(path.name for path in package.iterdir() if path.is_dir())
    if not json_files:
        errors.append("task/kernel_source_package has no top-level JSON manifest")
    if not source_dirs:
        errors.append("task/kernel_source_package has no matched low-level source directory")
    return {
        "status": _status(errors),
        "configured": True,
        "errors": errors,
        "json_files": json_files,
        "source_directories": source_dirs,
    }


def _python_probe(code: str, *, timeout: int, env: dict[str, str]) -> dict[str, Any]:
    python = os.environ.get("PYTHON") or sys.executable
    result = _run([python, "-B", "-c", code], cwd=TASK, timeout=timeout, env=env)
    return {"available": result["returncode"] == 0, **result}


def _probe_environment(*, timeout: int) -> dict[str, Any]:
    python = os.environ.get("PYTHON") or sys.executable
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return {
        "task_id": ROOT.name,
        "generated_by": "validate_task_pack.py",
        "python": {"executable": python, "version": sys.version.split()[0]},
        "pytorch": _python_probe(
            "import torch; print(torch.__version__); print(torch.version.cuda); "
            "print(torch.cuda.is_available())",
            timeout=timeout,
            env=env,
        ),
        "gpu": _python_probe(
            "import torch\n"
            "print(torch.cuda.device_count())\n"
            "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None\n"
            "print('' if p is None else p.name)\n"
            "print('' if p is None else f'{p.major}.{p.minor}')\n"
            "print('' if p is None else p.total_memory)",
            timeout=timeout,
            env=env,
        ),
        "triton": _available_result(
            _run([python, "-B", "env_probe/probe_triton.py"], cwd=TASK, timeout=timeout, env=env)
        ),
        "cutedsl": _available_result(
            _run([python, "-B", "env_probe/probe_cutedsl.py"], cwd=TASK, timeout=timeout, env=env)
        ),
        "cuda_extension": _available_result(
            _run(
                [python, "-B", "env_probe/probe_cuda_extension.py"],
                cwd=TASK,
                timeout=timeout,
                env=env,
            )
        ),
        "ncu": _available_result(
            _run([python, "-B", "env_probe/probe_ncu.py"], cwd=TASK, timeout=timeout, env=env)
        ),
    }


def _available_result(result: dict[str, Any]) -> dict[str, Any]:
    return {"available": result["returncode"] == 0, **result}


def _availability_map(data: Any, prefix: str = "") -> dict[str, bool]:
    out: dict[str, bool] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "available" and isinstance(value, bool):
                out[prefix] = value
            else:
                out.update(_availability_map(value, path))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            out.update(_availability_map(item, f"{prefix}.{index}"))
    return out


def _environment_check(*, timeout: int) -> dict[str, Any]:
    expected_path = DOCS / "env_probe_result.json"
    if not expected_path.is_file():
        return {
            "status": "failed",
            "errors": ["missing task/docs/env_probe_result.json"],
            "mismatches": [],
        }
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "failed",
            "errors": [f"invalid task/docs/env_probe_result.json: {exc!r}"],
            "mismatches": [],
        }
    current = _probe_environment(timeout=timeout)
    expected_map = _availability_map(expected)
    current_map = _availability_map(current)
    mismatches = [
        f"{key}: expected {value}, got {current_map.get(key)}"
        for key, value in sorted(expected_map.items())
        if current_map.get(key) != value
    ]
    return {
        "status": "failed" if mismatches else "passed",
        "errors": [f"environment availability mismatch: {item}" for item in mismatches],
        "mismatches": mismatches,
        "current": current,
    }


def _smoke(
    *,
    script: str,
    device: str,
    timeout: int,
    benchmark: bool = False,
) -> dict[str, Any]:
    python = os.environ.get("PYTHON") or sys.executable
    command = [python, "-B", script, "--device", device]
    if benchmark:
        command.extend(
            [
                "--target",
                os.environ.get("TARGET", "both"),
                "--warmup",
                os.environ.get("WARMUP", "1"),
                "--repeat",
                os.environ.get("REPEAT", "2"),
            ]
        )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return _run(command, cwd=ROOT, timeout=timeout, env=env)


def _original_capture_benchmark_check() -> dict[str, Any]:
    path = DOCS / "original_capture_benchmark_summary.json"
    if not path.is_file():
        return {"status": "missing", "required": False, "speedup_baseline": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid", "required": False, "error": repr(exc)}
    return {
        "status": data.get("status", "unknown"),
        "required": False,
        "path": str(path),
        "speedup_baseline": bool(data.get("speedup_baseline", False)),
        "overall": data.get("overall", {}),
    }


def _write_report(report: dict[str, Any]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "task_pack_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _print_human(report: dict[str, Any]) -> None:
    print(f"Task pack validation: {'PASS' if report['valid'] else 'FAIL'}")
    for key in (
        "structure_check",
        "workspace_policy_check",
        "syntax_check",
        "snapshot_check",
        "kernel_source_package_check",
        "env_check",
        "correctness_smoke",
        "benchmark_smoke",
    ):
        print(f"- {key}: {report[key]['status']}")
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-format", choices=["human", "json"], default="human")
    args = parser.parse_args()

    structure = _structure_check()
    workspace = _workspace_policy_check()
    syntax = _syntax_check()
    snapshots = _snapshot_check()
    kernel_sources = _kernel_source_package_check()
    errors = [
        *structure["errors"],
        *workspace["errors"],
        *syntax["errors"],
        *snapshots["errors"],
        *kernel_sources["errors"],
    ]

    if args.skip_env_check:
        env_check = {"status": "skipped", "errors": [], "mismatches": []}
    elif errors:
        env_check = {
            "status": "skipped",
            "errors": [],
            "reason": "static delivery checks failed",
            "mismatches": [],
        }
    else:
        env_check = _environment_check(timeout=args.timeout)
        errors.extend(env_check["errors"])

    smoke: list[dict[str, Any]] = []
    if args.skip_correctness:
        correctness = {"status": "skipped"}
    elif errors:
        correctness = {"status": "skipped", "reason": "earlier delivery checks failed"}
    else:
        result = _smoke(
            script="task/scripts/run_correctness.py",
            device=args.device,
            timeout=args.timeout,
        )
        smoke.append(result)
        correctness = {"status": "passed" if result["returncode"] == 0 else "failed"}
        if result["returncode"] != 0:
            errors.append(f"command failed: {result['command']}")

    if not args.run_benchmark:
        benchmark = {"status": "skipped"}
    elif errors:
        benchmark = {"status": "skipped", "reason": "earlier delivery checks failed"}
    else:
        result = _smoke(
            script="task/scripts/run_benchmark.py",
            device=args.device,
            timeout=args.timeout,
            benchmark=True,
        )
        smoke.append(result)
        benchmark = {"status": "passed" if result["returncode"] == 0 else "failed"}
        if result["returncode"] != 0:
            errors.append(f"command failed: {result['command']}")

    report = {
        "valid": not errors,
        "errors": errors,
        "task_pack": str(ROOT),
        "structure_check": structure,
        "file_check": structure,
        "workspace_policy_check": workspace,
        "syntax_check": syntax,
        "snapshot_check": snapshots,
        "kernel_source_package_check": kernel_sources,
        "original_capture_benchmark_check": _original_capture_benchmark_check(),
        "env_check": env_check,
        "correctness_smoke": correctness,
        "benchmark_smoke": benchmark,
        "smoke": smoke,
    }
    _write_report(report)
    if args.output_format == "json":
        print(json.dumps(report, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
