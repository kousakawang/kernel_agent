"""Framework Engineer Phase 1 CLI.

Run as:

    python -m framework_engineer.cli <subcommand>
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .snapshot.harness_builder import SnapshotHarnessBuilder, copy_probe_templates
from .snapshot.selector import SnapshotSelector, write_shape_list_summary
from .snapshot.store import SnapshotStore
from .snapshot.validation import run_smoke, validate_files, validate_structure


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
TEMPLATE_DIR = PACKAGE_ROOT / "templates"
OUTPUT_FORMATS = ("auto", "human", "json")


@dataclass(frozen=True)
class SourceInterface:
    file: Path
    function_name: str
    qualified_name: str
    line: int
    end_line: int | None
    class_path: list[str]
    module_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.file),
            "function_name": self.function_name,
            "qualified_name": self.qualified_name,
            "line": self.line,
            "end_line": self.end_line,
            "class_path": self.class_path,
            "module_name": self.module_name,
        }


@dataclass(frozen=True)
class TargetConfig:
    task_id: str
    task_pack: Path
    target_file: Path
    target_line: int
    drop_first_arg: bool
    signature: str
    mode: str
    backend: str
    layer_id: str
    candidate_function: str


@dataclass(frozen=True)
class Phase1Config:
    config_path: Path
    task_group_id: str
    output_root: Path
    service_cmd: str
    workload_cmd: str
    forward_boundary_file: Path
    forward_boundary_line: int
    non_cudagraph_service_cmd: str | None
    health_url: str | None
    startup_timeout: int
    workload_timeout: int
    extra_env: dict[str, str]
    run_baseline: bool
    run_probe_env: bool
    skip_env_check: bool
    run_benchmark_smoke: bool
    validate_device: str
    validate_warmup: int
    validate_repeat: int
    force: bool
    max_capture_groups: int
    max_samples_per_group: int
    max_samples_per_forward_per_group: int
    max_selected_groups: int
    max_selected_samples_per_group: int
    target_model: str
    target_framework: str
    target_hardware: str
    objective: str
    targets: list[TargetConfig]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="framework-engineer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = _add_command_parser(sub, "validate-config")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=cmd_validate_config)

    p = _add_command_parser(sub, "run-phase1")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=cmd_run_phase1)

    p = _add_command_parser(sub, "scaffold-task-pack")
    p.add_argument("--task-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_scaffold_task_pack)

    p = _add_command_parser(sub, "run-baseline")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--service-cmd", required=True)
    p.add_argument("--workload-cmd", required=True)
    p.add_argument("--health-url", default=None)
    p.add_argument("--startup-timeout", type=int, default=120)
    p.add_argument("--workload-timeout", type=int, default=600)
    p.set_defaults(func=cmd_run_baseline)

    p = _add_command_parser(sub, "resolve-interface")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--line", type=int, required=True)
    p.set_defaults(func=cmd_resolve_interface)

    p = _add_command_parser(sub, "probe-target-calls")
    _add_run_and_instrument_args(p)
    p.set_defaults(func=cmd_probe_target_calls)

    p = _add_command_parser(sub, "capture-snapshots")
    _add_run_and_instrument_args(p)
    p.add_argument("--signature", default="candidate(*args, **kwargs)")
    p.add_argument("--mutable-arg-path", action="append", default=[], help=argparse.SUPPRESS)
    p.add_argument("--mode", default="")
    p.add_argument("--backend", default="")
    p.add_argument("--layer-id", default="")
    p.add_argument("--calls-per-forward", type=int, default=None)
    p.add_argument("--max-capture-groups", type=int, default=64)
    p.add_argument("--max-samples-per-group", type=int, default=8)
    p.add_argument("--max-samples-per-forward-per-group", type=int, default=3)
    p.add_argument("--max-raw-cases", type=int, default=None, help="Deprecated alias for --max-capture-groups.")
    p.set_defaults(func=cmd_capture_snapshots)

    p = _add_command_parser(sub, "select-snapshots")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--max-selected-samples-per-group", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=None, help="Deprecated alias for --max-groups.")
    p.set_defaults(func=cmd_select_snapshots)

    p = _add_command_parser(sub, "generate-harness")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--candidate-function", default="candidate")
    p.set_defaults(func=cmd_generate_harness)

    p = _add_command_parser(sub, "probe-env")
    p.add_argument("--task-pack", type=Path, required=True)
    p.set_defaults(func=cmd_probe_env)

    p = _add_command_parser(sub, "validate-task-pack")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--run-correctness", action="store_true")
    p.add_argument("--run-benchmark", action="store_true")
    p.add_argument("--skip-env-check", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.set_defaults(func=cmd_validate_task_pack)

    args = parser.parse_args(argv)
    if _use_human_output(args):
        _terminal_log(f"command={args.command}")
    try:
        return int(args.func(args))
    except SystemExit as exc:
        returncode = int(exc.code) if isinstance(exc.code, int) else 1
        _emit_result(
            args,
            {"status": "failed", "command": args.command, "error": str(exc)},
            title=f"{args.command} failed",
        )
        return returncode
    except Exception as exc:
        _emit_result(
            args,
            {
                "status": "failed",
                "command": args.command,
                "error": repr(exc),
                "traceback_tail": traceback.format_exc()[-8000:],
            },
            title=f"{args.command} failed",
        )
        return 1


def _add_command_parser(subparsers: Any, name: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default="auto",
        help="Terminal output mode: auto uses human output on a TTY and one-line JSON otherwise.",
    )
    return parser


def _add_run_and_instrument_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-pack", type=Path, required=True)
    parser.add_argument("--service-cmd", required=True)
    parser.add_argument("--non-cudagraph-service-cmd", default=None)
    parser.add_argument("--workload-cmd", required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--target-line", type=int, default=None)
    parser.add_argument("--function-name", default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--drop-first-arg", action="store_true")
    parser.add_argument("--forward-boundary-file", type=Path, default=None)
    parser.add_argument("--forward-boundary-line", type=int, default=None)
    parser.add_argument("--forward-boundary-function", default=None)
    parser.add_argument("--forward-boundary-name", default=None)
    parser.add_argument("--health-url", default=None)
    parser.add_argument("--startup-timeout", type=int, default=120)
    parser.add_argument("--workload-timeout", type=int, default=600)


def _use_human_output(args: argparse.Namespace) -> bool:
    output_format = getattr(args, "output_format", "auto")
    if output_format == "human":
        return True
    if output_format == "json":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _emit_result(
    args: argparse.Namespace,
    payload: dict[str, Any],
    *,
    title: str,
    formatter: Any | None = None,
) -> None:
    if not _use_human_output(args):
        print(json.dumps(payload, sort_keys=True))
        return
    render = formatter or _format_human_payload
    print(render(payload, title=title))


def _format_human_payload(payload: dict[str, Any], *, title: str) -> str:
    lines = [title]
    failed = _payload_failed(payload)
    for key, value in payload.items():
        if not failed and (_is_log_field(key) or key == "smoke"):
            continue
        if value in (None, "", [], {}):
            continue
        _append_human_value(lines, key, value, indent=0)
    return "\n".join(lines)


def _format_phase1_report(payload: dict[str, Any], *, title: str) -> str:
    targets = payload.get("targets", [])
    succeeded = bool(targets) and all(item.get("status") == "ok" for item in targets)
    lines = [title, f"status: {'ok' if succeeded else 'failed'}"]
    for key in ("config", "task_group_id", "output_root"):
        if payload.get(key):
            lines.append(f"{key}: {payload[key]}")

    baseline = payload.get("baseline") or {}
    if baseline.get("status") == "skipped":
        lines.append("baseline: skipped")
    elif baseline:
        lines.append(f"baseline: {'ok' if baseline.get('returncode') == 0 else 'failed'}")

    lines.append("targets:")
    for item in targets:
        target = item.get("target", {})
        task_id = target.get("task_id", "unknown")
        status = item.get("status", "unknown")
        lines.append(f"  - {task_id}: {status}")
        if target.get("task_pack"):
            lines.append(f"    task_pack: {target['task_pack']}")
        failed_step = next(
            (step for step in item.get("steps", []) if step.get("returncode") != 0),
            None,
        )
        if failed_step:
            lines.append(f"    failed_step: {failed_step.get('step', 'unknown')}")
            if failed_step.get("error"):
                _append_human_value(lines, "error", failed_step["error"], indent=4)

    output_root = payload.get("output_root")
    if output_root:
        lines.extend(
            [
                "reports:",
                f"  json: {Path(output_root) / 'multi_target_report.json'}",
                f"  markdown: {Path(output_root) / 'multi_target_report.md'}",
            ]
        )
    return "\n".join(lines)


def _append_human_value(lines: list[str], key: str, value: Any, *, indent: int) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        for child_key, child_value in value.items():
            if child_value in (None, "", [], {}):
                continue
            _append_human_value(lines, str(child_key), child_value, indent=indent + 2)
        return
    if isinstance(value, list):
        lines.append(f"{prefix}{key}:")
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{' ' * (indent + 2)}-")
                for child_key, child_value in item.items():
                    if child_value in (None, "", [], {}):
                        continue
                    _append_human_value(lines, str(child_key), child_value, indent=indent + 4)
            else:
                item_text = _human_scalar(item)
                if "\n" in item_text:
                    lines.append(f"{' ' * (indent + 2)}-")
                    lines.extend(f"{' ' * (indent + 4)}| {line}" for line in item_text.splitlines())
                else:
                    lines.append(f"{' ' * (indent + 2)}- {item_text}")
        return

    text = _human_scalar(value)
    if "\n" in text:
        lines.append(f"{prefix}{key}:")
        lines.extend(f"{' ' * (indent + 2)}| {line}" for line in text.splitlines())
    else:
        lines.append(f"{prefix}{key}: {text}")


def _human_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _payload_failed(payload: dict[str, Any]) -> bool:
    if payload.get("valid") is False:
        return True
    if payload.get("status") in ("failed", "blocked"):
        return True
    if payload.get("errors"):
        return True
    workload_returncode = payload.get("workload_returncode")
    return workload_returncode is not None and workload_returncode != 0


def _is_log_field(key: str) -> bool:
    return key.endswith(("_stdout_tail", "_stderr_tail")) or key in (
        "stdout_tail",
        "stderr_tail",
        "traceback_tail",
    )


def _terminal_log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[framework-engineer] {message}", file=sys.stderr, flush=True)


def _terminal_detail(label: str, value: Any, *, enabled: bool) -> None:
    if not enabled or value in (None, "", [], {}):
        return
    if isinstance(value, (dict, list)):
        text = json.dumps(value, indent=2, sort_keys=True)
    else:
        text = str(value)
    _terminal_log(f"  {label}:", enabled=True)
    for line in text.rstrip().splitlines():
        print(f"[framework-engineer]    | {line}", file=sys.stderr, flush=True)


def _print_step_failure_details(step: dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    summary = step.get("summary") if isinstance(step.get("summary"), dict) else {}
    _terminal_detail("error", step.get("error"), enabled=True)
    _terminal_detail("errors", summary.get("errors"), enabled=True)
    if summary.get("workload_returncode") is not None:
        _terminal_detail("workload_returncode", summary["workload_returncode"], enabled=True)
    if summary.get("health") and not summary["health"].get("ready", True):
        _terminal_detail("health", summary["health"], enabled=True)
    for key in (
        "service_stderr_tail",
        "service_stdout_tail",
        "workload_stderr_tail",
        "workload_stdout_tail",
    ):
        _terminal_detail(key, summary.get(key), enabled=True)
    for smoke in summary.get("smoke", []):
        if smoke.get("returncode") == 0:
            continue
        _terminal_detail("failed_command", smoke.get("command"), enabled=True)
        _terminal_detail("command_stderr", smoke.get("stderr"), enabled=True)
        _terminal_detail("command_stdout", smoke.get("stdout"), enabled=True)
    _terminal_detail("captured_stderr", step.get("stderr_tail"), enabled=True)
    _terminal_detail("captured_stdout", step.get("stdout_tail"), enabled=True)
    _terminal_detail("traceback", step.get("traceback_tail"), enabled=True)


def _execution_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    workload = result.get("workload", {})
    service = result.get("service", {})
    return {
        "health": result.get("health"),
        "startup_elapsed_sec": result.get("startup_elapsed_sec"),
        "service_returncode": service.get("returncode_before_termination"),
        "service_stdout_tail": service.get("stdout", "")[-2000:],
        "service_stderr_tail": service.get("stderr", "")[-2000:],
        "workload_returncode": workload.get("returncode"),
        "workload_elapsed_sec": workload.get("elapsed_sec"),
        "workload_timed_out": workload.get("timed_out", False),
        "workload_stdout_tail": workload.get("stdout", "")[-2000:],
        "workload_stderr_tail": workload.get("stderr", "")[-2000:],
    }


def cmd_validate_config(args: argparse.Namespace) -> int:
    cfg, errors = _load_phase1_config(args.config)
    report = {
        "valid": not errors,
        "config": str(args.config),
        "errors": errors,
        "task_group_id": cfg.task_group_id if cfg else None,
        "output_root": str(cfg.output_root) if cfg else None,
        "targets": [_target_report(target) for target in cfg.targets] if cfg else [],
    }
    _emit_result(args, report, title="Configuration validation")
    return 0 if not errors else 1


def cmd_run_phase1(args: argparse.Namespace) -> int:
    cfg, errors = _load_phase1_config(args.config)
    if cfg is None or errors:
        _emit_result(
            args,
            {"valid": False, "config": str(args.config), "errors": errors},
            title="Phase 1.2 configuration error",
        )
        return 1

    progress = _use_human_output(args)
    _terminal_log(
        f"config={cfg.config_path} targets={len(cfg.targets)} output_root={cfg.output_root}",
        enabled=progress,
    )

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "task_group_id": cfg.task_group_id,
        "config": str(cfg.config_path),
        "output_root": str(cfg.output_root),
        "targets": [],
        "baseline": {"status": "skipped"} if not cfg.run_baseline else None,
    }

    with _temporary_env(cfg.extra_env):
        for target in cfg.targets:
            if cfg.force and target.task_pack.exists():
                shutil.rmtree(target.task_pack)
            step = _run_step(
                "scaffold-task-pack",
                cmd_scaffold_task_pack,
                argparse.Namespace(task_id=target.task_id, out=target.task_pack, force=cfg.force),
                context=target.task_id,
                progress=progress,
            )
            target_report = {"target": _target_report(target), "steps": [step], "status": "running"}
            report["targets"].append(target_report)
            if step["returncode"] != 0:
                target_report["status"] = "failed"
            else:
                _write_task_contract(target.task_pack, cfg, target)

        if cfg.run_baseline and any(item["status"] != "failed" for item in report["targets"]):
            first_pack = next(target.task_pack for target, item in zip(cfg.targets, report["targets"]) if item["status"] != "failed")
            baseline_step = _run_step(
                "run-baseline",
                cmd_run_baseline,
                argparse.Namespace(
                    task_pack=first_pack,
                    service_cmd=cfg.service_cmd,
                    workload_cmd=cfg.workload_cmd,
                    health_url=cfg.health_url,
                    startup_timeout=cfg.startup_timeout,
                    workload_timeout=cfg.workload_timeout,
                ),
                context="group",
                progress=progress,
            )
            report["baseline"] = baseline_step
            _copy_baseline_to_targets(first_pack, [target.task_pack for target in cfg.targets])
            if baseline_step["returncode"] != 0:
                for item in report["targets"]:
                    if item["status"] != "failed":
                        item["status"] = "blocked"
                _write_phase1_reports(cfg.output_root, report)
                _emit_result(args, report, title="Phase 1.2 summary", formatter=_format_phase1_report)
                return 1

        for target, target_report in zip(cfg.targets, report["targets"]):
            if target_report["status"] in ("failed", "blocked"):
                continue
            target_report["status"] = _run_phase1_target(
                cfg,
                target,
                target_report["steps"],
                progress=progress,
            )

    _write_phase1_reports(cfg.output_root, report)
    _emit_result(args, report, title="Phase 1.2 summary", formatter=_format_phase1_report)
    return 0 if all(item["status"] == "ok" for item in report["targets"]) else 1


def cmd_scaffold_task_pack(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} already exists and is not empty; pass --force to overwrite scaffold files.")
    out.mkdir(parents=True, exist_ok=True)
    for rel in ("docs", "scripts", "snapshots/raw", "snapshots/selected", "env_probe", "kernel_sources", "original_source"):
        (out / rel).mkdir(parents=True, exist_ok=True)

    _copy_template("task_pack_README.md", out / "README.md")
    _copy_template("task_pack_manifest.yaml", out / "task.yaml")
    _copy_template("env_manifest.yaml", out / "env_manifest.yaml")
    (out / "snapshots" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1.snapshot.v1",
                "selection_policy": "not_selected_yet",
                "raw_group_count": 0,
                "raw_sample_count": 0,
                "selected_group_count": 0,
                "selected_sample_count": 0,
                "case_groups": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out / "shape_list.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1.shape_summary.v1",
                "source": "snapshots/manifest.json",
                "note": "Selected snapshot samples are the replay source; this file is only a group index/summary.",
                "shape_groups": [],
                "shape_cases": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out / "snapshots" / "raw_index.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1.snapshot.v1",
                "index_type": "raw_group_index",
                "raw_group_count": 0,
                "raw_sample_count": 0,
                "total_hit_count": 0,
                "groups": {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    copy_probe_templates(TEMPLATE_DIR, out)
    _write_json(out / "docs" / "scaffold_result.json", {"task_id": args.task_id, "task_pack": str(out)})
    _emit_result(args, {"status": "ok", "task_pack": str(out)}, title="Task pack scaffold")
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    result = _run_service_and_workload(
        service_cmd=args.service_cmd,
        workload_cmd=args.workload_cmd,
        health_url=args.health_url,
        startup_timeout=args.startup_timeout,
        workload_timeout=args.workload_timeout,
    )
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "baseline_result.json", result)
    _write_baseline_report(docs / "baseline_run_report.md", args, result)
    succeeded = result["workload"]["returncode"] == 0
    payload = {
        "status": "ok" if succeeded else "failed",
        "report": str(docs / "baseline_run_report.md"),
        **_execution_result_summary(result),
    }
    _emit_result(args, payload, title="Baseline result")
    return 0 if succeeded else 1


def cmd_resolve_interface(args: argparse.Namespace) -> int:
    interface = _resolve_source_interface(
        file=args.file,
        line=args.line,
        function_name=None,
        qualified_name=None,
        role="interface",
    )
    payload = {
        **interface.to_dict(),
        "target_file": str(interface.file),
        "function_name": interface.function_name,
        "target_name": interface.qualified_name,
    }
    _emit_result(args, payload, title="Resolved interface")
    return 0


def cmd_probe_target_calls(args: argparse.Namespace) -> int:
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    log_path = docs / "target_call_probe.jsonl"
    target = _resolve_target_interface(args)
    boundary = _resolve_forward_boundary_interface(args)
    decorator_expr = (
        "__import__('framework_engineer.snapshot.recorder', "
        "fromlist=['make_probe_decorator']).make_probe_decorator("
        f"{str(log_path)!r}, {target.qualified_name!r}, drop_first_arg={bool(args.drop_first_arg)!r})"
    )
    service_cmd = _resolve_non_cudagraph_service_cmd(args.service_cmd, args.non_cudagraph_service_cmd)
    with _instrumentation_context(target, boundary, decorator_expr):
        result = _run_service_and_workload(
            service_cmd=service_cmd,
            workload_cmd=args.workload_cmd,
            health_url=args.health_url,
            startup_timeout=args.startup_timeout,
            workload_timeout=args.workload_timeout,
        )
    calls = _read_jsonl(log_path)
    succeeded = result["workload"]["returncode"] == 0 and bool(calls)
    runtime_qualified_name = next(
        (str(call["qualified_name"]) for call in calls if call.get("qualified_name")),
        target.qualified_name,
    )
    runtime_target = _interface_with_runtime_qualified_name(target, runtime_qualified_name)
    report = {
        "status": "ok" if succeeded else "failed",
        "target_name": runtime_qualified_name,
        "target_interface": runtime_target,
        "forward_boundary_interface": boundary.to_dict() if boundary else None,
        "call_count": len(calls),
        "workload_returncode": result["workload"]["returncode"],
        "health": result["health"],
        "service_returncode": result["service"].get("returncode_before_termination"),
        "service_stdout_tail": result["service"].get("stdout", "")[-2000:],
        "service_stderr_tail": result["service"].get("stderr", "")[-2000:],
        "workload_timed_out": result["workload"].get("timed_out", False),
        "workload_stdout_tail": result["workload"].get("stdout", "")[-2000:],
        "workload_stderr_tail": result["workload"].get("stderr", "")[-2000:],
        "log_path": str(log_path),
        "service_cmd": service_cmd,
        "workload_cmd": args.workload_cmd,
    }
    _write_json(docs / "target_call_probe_report.json", report)
    (docs / "target_call_probe_report.md").write_text(
        f"# Target Call Probe Report\n\n- target: `{runtime_qualified_name}`\n- call_count: {len(calls)}\n- workload_returncode: {result['workload']['returncode']}\n- log: `{log_path}`\n",
        encoding="utf-8",
    )
    _emit_result(args, report, title="Target call probe")
    return 0 if succeeded else 1


def cmd_capture_snapshots(args: argparse.Namespace) -> int:
    snapshot_root = args.task_pack / "snapshots"
    mutable_paths = ",".join(args.mutable_arg_path)
    max_capture_groups = args.max_raw_cases if args.max_raw_cases is not None else args.max_capture_groups
    target = _resolve_target_interface(args)
    boundary = _resolve_forward_boundary_interface(args)
    decorator_expr = (
        "__import__('framework_engineer.snapshot.recorder', "
        "fromlist=['make_snapshot_decorator']).make_snapshot_decorator("
        f"{str(snapshot_root)!r}, {args.task_pack.name!r}, {target.qualified_name!r}, {args.signature!r}, "
        f"mutable_arg_paths={mutable_paths!r}, mode={args.mode!r}, backend={args.backend!r}, "
        f"layer_id={args.layer_id!r}, drop_first_arg={bool(args.drop_first_arg)!r}, "
        f"source_info={target.to_dict()!r}, "
        f"calls_per_forward={args.calls_per_forward!r}, max_capture_groups={max_capture_groups!r}, "
        f"max_samples_per_group={args.max_samples_per_group!r}, "
        f"max_samples_per_forward_per_group={args.max_samples_per_forward_per_group!r})"
    )
    service_cmd = _resolve_non_cudagraph_service_cmd(args.service_cmd, args.non_cudagraph_service_cmd)
    with _instrumentation_context(target, boundary, decorator_expr):
        result = _run_service_and_workload(
            service_cmd=service_cmd,
            workload_cmd=args.workload_cmd,
            health_url=args.health_url,
            startup_timeout=args.startup_timeout,
            workload_timeout=args.workload_timeout,
        )
    raw_index = SnapshotStore(snapshot_root).read_raw_index()
    raw_timing_summary = _original_capture_timing_summary_from_raw_index(raw_index)
    succeeded = result["workload"]["returncode"] == 0 and bool(raw_index.get("raw_sample_count", 0))
    runtime_target = _runtime_target_interface_from_raw_index(raw_index) or target.to_dict()
    report = {
        "status": "ok" if succeeded else "failed",
        "target_name": runtime_target.get("qualified_name", target.qualified_name),
        "target_interface": runtime_target,
        "forward_boundary_interface": boundary.to_dict() if boundary else None,
        "windowing_mode": _windowing_mode(args),
        "raw_group_count": raw_index.get("raw_group_count", 0),
        "raw_sample_count": raw_index.get("raw_sample_count", 0),
        "raw_snapshot_count": raw_index.get("raw_sample_count", 0),
        "total_hit_count": raw_index.get("total_hit_count", 0),
        "dropped_hit_count": raw_index.get("dropped_hit_count", 0),
        "original_capture_timing": raw_timing_summary.get("overall", {}),
        "mutation_warning_count": _mutation_warning_count(raw_index),
        "workload_returncode": result["workload"]["returncode"],
        "health": result["health"],
        "service_returncode": result["service"].get("returncode_before_termination"),
        "service_stdout_tail": result["service"].get("stdout", "")[-2000:],
        "service_stderr_tail": result["service"].get("stderr", "")[-2000:],
        "workload_timed_out": result["workload"].get("timed_out", False),
        "service_cmd": service_cmd,
        "workload_cmd": args.workload_cmd,
        "max_raw_cases_deprecated_alias_used": args.max_raw_cases is not None,
        "workload_stdout_tail": result["workload"].get("stdout", "")[-2000:],
        "workload_stderr_tail": result["workload"].get("stderr", "")[-2000:],
    }
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "snapshot_capture_report.json", report)
    _write_json(docs / "original_capture_timing_raw_summary.json", raw_timing_summary)
    _emit_result(args, report, title="Snapshot capture")
    return 0 if succeeded else 1


def cmd_select_snapshots(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.task_pack / "snapshots")
    max_groups = args.max_groups if args.max_groups is not None else args.max_cases
    manifest = SnapshotSelector(store).select(
        max_groups=max_groups,
        max_samples_per_group=args.max_selected_samples_per_group,
    )
    write_shape_list_summary(args.task_pack, manifest)
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    capture_benchmark_summary = _original_capture_benchmark_summary_from_manifest(manifest)
    _write_json(docs / "snapshot_selection_report.json", manifest)
    _write_json(docs / "original_capture_benchmark_summary.json", capture_benchmark_summary)
    _write_original_capture_benchmark_report(docs / "original_capture_benchmark_report.md", capture_benchmark_summary)
    (docs / "snapshot_selection_report.md").write_text(
        "# Snapshot Selection Report\n\n"
        f"- raw_group_count: {manifest['raw_group_count']}\n"
        f"- raw_sample_count: {manifest['raw_sample_count']}\n"
        f"- selected_group_count: {manifest['selected_group_count']}\n"
        f"- selected_sample_count: {manifest['selected_sample_count']}\n"
        f"- policy: `{manifest['selection_policy']}`\n",
        encoding="utf-8",
    )
    _emit_result(
        args,
        {
            "selected_group_count": manifest["selected_group_count"],
            "selected_sample_count": manifest["selected_sample_count"],
            "selected_case_count": manifest["selected_group_count"],
            "manifest": str(store.manifest_path),
            "original_capture_benchmark_summary": str(docs / "original_capture_benchmark_summary.json"),
        },
        title="Snapshot selection",
    )
    return 0


def cmd_generate_harness(args: argparse.Namespace) -> int:
    SnapshotHarnessBuilder(args.task_pack).generate(candidate_function=args.candidate_function)
    _emit_result(args, {"status": "ok", "task_pack": str(args.task_pack)}, title="Harness generation")
    return 0


def cmd_probe_env(args: argparse.Namespace) -> int:
    result = probe_environment(args.task_pack)
    (args.task_pack / "env_manifest.yaml").write_text(_env_to_yaml(result), encoding="utf-8")
    _write_json(args.task_pack / "docs" / "env_probe_result.json", result)
    _emit_result(
        args,
        {"status": "ok", "env_manifest": str(args.task_pack / "env_manifest.yaml")},
        title="Environment probe",
    )
    return 0


def cmd_validate_task_pack(args: argparse.Namespace) -> int:
    structure = validate_structure(args.task_pack)
    errors = list(structure["errors"])
    env_check: dict[str, Any] | None = None
    env_status = "skipped" if args.skip_env_check else "passed"
    if not args.skip_env_check:
        expected_path = args.task_pack / "docs" / "env_probe_result.json"
        if not expected_path.exists():
            errors.append("missing docs/env_probe_result.json; run probe-env before validate-task-pack or pass --skip-env-check")
            env_status = "failed"
        else:
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            current = probe_environment(args.task_pack)
            mismatches = _compare_availability(expected, current)
            env_check = {"status": "failed" if mismatches else "passed", "mismatches": mismatches}
            env_status = env_check["status"]
            errors.extend(f"env availability mismatch: {item}" for item in mismatches)
    smoke = run_smoke(
        args.task_pack,
        correctness=args.run_correctness,
        benchmark=args.run_benchmark,
        timeout=args.timeout,
    )
    correctness_status = "skipped"
    benchmark_status = "skipped"
    for item in smoke:
        if "run_correctness" in item["command"]:
            correctness_status = "passed" if item["returncode"] == 0 else "failed"
        if "run_benchmark" in item["command"]:
            benchmark_status = "passed" if item["returncode"] == 0 else "failed"
        if item["returncode"] != 0:
            errors.append(f"command failed: {item['command']}")
    report = {
        "valid": not errors,
        "errors": errors,
        "file_check": structure["file_check"],
        "snapshot_check": structure["snapshot_check"],
        "original_capture_benchmark_check": _original_capture_benchmark_check(args.task_pack),
        "env_check": env_check or {"status": env_status},
        "correctness_smoke": {"status": correctness_status},
        "benchmark_smoke": {"status": benchmark_status},
        "smoke": smoke,
    }
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "task_pack_validation_report.json", report)
    _emit_result(args, report, title="Task pack validation")
    return 0 if not errors else 1


def _run_phase1_target(
    cfg: Phase1Config,
    target: TargetConfig,
    steps: list[dict[str, Any]],
    *,
    progress: bool = False,
) -> str:
    common = {
        "task_pack": target.task_pack,
        "service_cmd": cfg.service_cmd,
        "non_cudagraph_service_cmd": cfg.non_cudagraph_service_cmd,
        "workload_cmd": cfg.workload_cmd,
        "target_file": target.target_file,
        "target_line": target.target_line,
        "function_name": None,
        "target_name": None,
        "drop_first_arg": target.drop_first_arg,
        "forward_boundary_file": cfg.forward_boundary_file,
        "forward_boundary_line": cfg.forward_boundary_line,
        "forward_boundary_function": None,
        "forward_boundary_name": None,
        "health_url": cfg.health_url,
        "startup_timeout": cfg.startup_timeout,
        "workload_timeout": cfg.workload_timeout,
    }
    step_specs = [
        ("resolve-target", cmd_resolve_interface, argparse.Namespace(file=target.target_file, line=target.target_line)),
        ("resolve-forward-boundary", cmd_resolve_interface, argparse.Namespace(file=cfg.forward_boundary_file, line=cfg.forward_boundary_line)),
        ("probe-target-calls", cmd_probe_target_calls, argparse.Namespace(**common)),
        (
            "capture-snapshots",
            cmd_capture_snapshots,
            argparse.Namespace(
                **common,
                signature=target.signature,
                mutable_arg_path=[],
                mode=target.mode,
                backend=target.backend,
                layer_id=target.layer_id,
                calls_per_forward=None,
                max_capture_groups=cfg.max_capture_groups,
                max_samples_per_group=cfg.max_samples_per_group,
                max_samples_per_forward_per_group=cfg.max_samples_per_forward_per_group,
                max_raw_cases=None,
            ),
        ),
        (
            "select-snapshots",
            cmd_select_snapshots,
            argparse.Namespace(
                task_pack=target.task_pack,
                max_groups=cfg.max_selected_groups,
                max_selected_samples_per_group=cfg.max_selected_samples_per_group,
                max_cases=None,
            ),
        ),
        (
            "generate-harness",
            cmd_generate_harness,
            argparse.Namespace(task_pack=target.task_pack, candidate_function=target.candidate_function),
        ),
    ]
    if cfg.run_probe_env:
        step_specs.append(("probe-env", cmd_probe_env, argparse.Namespace(task_pack=target.task_pack)))

    validation_env = {
        "DEVICE": cfg.validate_device,
        "WARMUP": str(cfg.validate_warmup),
        "REPEAT": str(cfg.validate_repeat),
        "PYTHON": sys.executable,
    }
    for name, func, namespace in step_specs:
        step = _run_step(name, func, namespace, context=target.task_id, progress=progress)
        steps.append(step)
        if step["returncode"] != 0:
            return "failed"
    with _temporary_env(validation_env):
        validate_step = _run_step(
            "validate-task-pack",
            cmd_validate_task_pack,
            argparse.Namespace(
                task_pack=target.task_pack,
                run_correctness=True,
                run_benchmark=cfg.run_benchmark_smoke,
                skip_env_check=cfg.skip_env_check,
                timeout=300,
            ),
            context=target.task_id,
            progress=progress,
        )
    steps.append(validate_step)
    return "ok" if validate_step["returncode"] == 0 else "failed"


def _run_step(
    name: str,
    func: Any,
    namespace: argparse.Namespace,
    *,
    context: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    started = time.time()
    label = f" target={context}" if context else ""
    _terminal_log(f"START {name}{label}", enabled=progress)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = int(func(namespace))
        error = None
        traceback_text = None
    except SystemExit as exc:
        returncode = int(exc.code) if isinstance(exc.code, int) else 1
        error = str(exc)
        traceback_text = None
    except Exception as exc:
        returncode = 1
        error = repr(exc)
        traceback_text = traceback.format_exc()
    text = stdout.getvalue()
    stderr_text = stderr.getvalue()
    payload = _last_json_line(text)
    elapsed = time.time() - started
    step = {
        "step": name,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": _non_result_stdout(text)[-4000:],
        "stderr_tail": stderr_text[-4000:],
        "summary": payload,
        "error": error,
        "traceback_tail": traceback_text[-8000:] if traceback_text else None,
    }
    if returncode == 0:
        _terminal_log(f"OK    {name}{label} ({elapsed:.2f}s)", enabled=progress)
    else:
        _terminal_log(f"FAIL  {name}{label} ({elapsed:.2f}s, rc={returncode})", enabled=progress)
        _print_step_failure_details(step, enabled=progress)
    return step


def _last_json_line(text: str) -> Any:
    for line in reversed([item for item in text.splitlines() if item.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _non_result_stdout(text: str) -> str:
    """Remove the final structured JSON line from captured command output."""
    lines = text.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if not lines[idx].strip():
            continue
        try:
            json.loads(lines[idx])
        except json.JSONDecodeError:
            break
        del lines[idx]
        break
    return "\n".join(lines).strip()


def _load_phase1_config(path: Path) -> tuple[Phase1Config | None, list[str]]:
    errors: list[str] = []
    path = path.expanduser().resolve()
    if not path.exists():
        return None, [f"config file does not exist: {path}"]
    try:
        module = _load_config_module(path)
    except Exception as exc:
        return None, [f"failed to load config: {exc!r}"]

    def get(name: str, default: Any = None) -> Any:
        return getattr(module, name, default)

    service_cmd = str(get("service_cmd", "") or "")
    workload_cmd = str(get("workload_cmd", "") or "")
    forward_boundary_file_raw = get("forward_boundary_file")
    forward_boundary_line = get("forward_boundary_line")
    if not service_cmd:
        errors.append("missing required config: service_cmd")
    if not workload_cmd:
        errors.append("missing required config: workload_cmd")
    if not forward_boundary_file_raw:
        errors.append("missing required config: forward_boundary_file")
    if forward_boundary_line is None:
        errors.append("missing required config: forward_boundary_line")

    raw_targets = get("targets", None)
    single_target = raw_targets is None
    targets: list[TargetConfig] = []
    output_root_raw = get("output_root", None)
    force = bool(get("force", False))

    if raw_targets is None:
        raw_targets = [
            {
                "task_id": get("task_id", ""),
                "task_pack": get("task_pack", ""),
                "target_file": get("target_file", ""),
                "target_line": get("target_line", None),
                "drop_first_arg": get("drop_first_arg", False),
            }
        ]
    if not isinstance(raw_targets, list) or not raw_targets:
        errors.append("targets must be a non-empty list, or provide single target_file/target_line/task_id/task_pack")
        raw_targets = []

    if output_root_raw:
        output_root = Path(str(output_root_raw)).expanduser().resolve()
    elif single_target and raw_targets and isinstance(raw_targets[0], dict) and raw_targets[0].get("task_pack"):
        output_root = Path(str(raw_targets[0]["task_pack"])).expanduser().resolve().parent
    else:
        output_root = Path.cwd() / "phase1_task_packs"
        if not single_target:
            errors.append("missing required config for multi-target mode: output_root")

    shared_signature = str(get("signature", "candidate(*args, **kwargs)"))
    shared_candidate = str(get("candidate_function", "candidate"))
    for idx, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            errors.append(f"targets[{idx}] must be a dict")
            continue
        task_id = str(raw.get("task_id") or f"target_{idx + 1}")
        target_file = Path(str(raw.get("target_file", ""))).expanduser().resolve()
        target_line = raw.get("target_line")
        if not raw.get("task_id"):
            errors.append(f"targets[{idx}] missing task_id")
        if not raw.get("target_file"):
            errors.append(f"targets[{idx}] missing target_file")
        if target_line is None:
            errors.append(f"targets[{idx}] missing target_line")
            target_line = 0
        task_pack_raw = raw.get("task_pack")
        if task_pack_raw:
            task_pack = Path(str(task_pack_raw)).expanduser().resolve()
        else:
            task_pack = output_root / task_id
        targets.append(
            TargetConfig(
                task_id=task_id,
                task_pack=task_pack,
                target_file=target_file,
                target_line=int(target_line),
                drop_first_arg=bool(raw.get("drop_first_arg", get("drop_first_arg", False))),
                signature=str(raw.get("signature", shared_signature)),
                mode=str(raw.get("mode", get("target_mode", "")) or ""),
                backend=str(raw.get("backend", get("target_backend", "")) or ""),
                layer_id=str(raw.get("layer_id", get("target_layer_id", "")) or ""),
                candidate_function=str(raw.get("candidate_function", shared_candidate)),
            )
        )

    task_group_id = str(get("task_group_id", "") or (targets[0].task_id if len(targets) == 1 else "phase1_targets"))
    cfg = Phase1Config(
        config_path=path,
        task_group_id=task_group_id,
        output_root=output_root,
        service_cmd=service_cmd,
        workload_cmd=workload_cmd,
        forward_boundary_file=Path(str(forward_boundary_file_raw or "")).expanduser().resolve(),
        forward_boundary_line=int(forward_boundary_line or 0),
        non_cudagraph_service_cmd=get("non_cudagraph_service_cmd", None),
        health_url=get("health_url", None),
        startup_timeout=int(get("startup_timeout", 120)),
        workload_timeout=int(get("workload_timeout", 600)),
        extra_env={str(k): str(v) for k, v in dict(get("extra_env", {}) or {}).items()},
        run_baseline=bool(get("run_baseline", True)),
        run_probe_env=bool(get("run_probe_env", False)),
        skip_env_check=bool(get("skip_env_check", True)),
        run_benchmark_smoke=bool(get("run_benchmark_smoke", False)),
        validate_device=str(get("validate_device", "cuda")),
        validate_warmup=int(get("validate_warmup", 3)),
        validate_repeat=int(get("validate_repeat", 5)),
        force=force,
        max_capture_groups=int(get("max_capture_groups", 64)),
        max_samples_per_group=int(get("max_samples_per_group", 8)),
        max_samples_per_forward_per_group=int(get("max_samples_per_forward_per_group", 3)),
        max_selected_groups=int(get("max_selected_groups", 8)),
        max_selected_samples_per_group=int(get("max_selected_samples_per_group", 8)),
        target_model=str(get("target_model", "") or "unknown"),
        target_framework=str(get("target_framework", "") or "unknown"),
        target_hardware=str(get("target_hardware", "") or "unknown"),
        objective=str(get("objective", "") or "Optimize captured framework target interfaces."),
        targets=targets,
    )
    errors.extend(_validate_phase1_config(cfg))
    return cfg, errors


def _load_config_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_framework_engineer_phase1_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_phase1_config(cfg: Phase1Config) -> list[str]:
    errors: list[str] = []
    if not cfg.forward_boundary_file.exists():
        errors.append(f"forward_boundary_file does not exist: {cfg.forward_boundary_file}")
    else:
        errors.extend(_validate_interface_ref(cfg.forward_boundary_file, cfg.forward_boundary_line, "forward boundary"))
    if cfg.output_root.exists() and not cfg.output_root.is_dir():
        errors.append(f"output_root exists but is not a directory: {cfg.output_root}")
    for target in cfg.targets:
        if not target.target_file.exists():
            errors.append(f"{target.task_id}: target_file does not exist: {target.target_file}")
        else:
            errors.extend(_validate_interface_ref(target.target_file, target.target_line, f"{target.task_id} target"))
        if target.task_pack.exists() and any(target.task_pack.iterdir()) and not cfg.force:
            errors.append(f"{target.task_id}: task_pack exists and is not empty; set force=True or choose a new path: {target.task_pack}")
    return errors


def _validate_interface_ref(file: Path, line: int, role: str) -> list[str]:
    try:
        _resolve_source_interface(file=file, line=line, function_name=None, qualified_name=None, role=role)
        return []
    except SystemExit as exc:
        return [str(exc)]


def _target_report(target: TargetConfig) -> dict[str, Any]:
    return {
        "task_id": target.task_id,
        "task_pack": str(target.task_pack),
        "target_file": str(target.target_file),
        "target_line": target.target_line,
        "drop_first_arg": target.drop_first_arg,
    }


def _interface_with_runtime_qualified_name(target: SourceInterface, qualified_name: str) -> dict[str, Any]:
    payload = target.to_dict()
    suffix = ".".join([*target.class_path, target.function_name])
    suffix_with_dot = f".{suffix}"
    if suffix and qualified_name.endswith(suffix_with_dot):
        module_name = qualified_name[: -len(suffix_with_dot)]
    else:
        module_name = qualified_name.rsplit(".", 1)[0] if "." in qualified_name else target.module_name
    payload.update(
        {
            "qualified_name": qualified_name,
            "module_name": module_name,
            "runtime_qualname": qualified_name[len(module_name) + 1 :] if module_name else qualified_name,
            "identity_source": "runtime_probe_callable",
        }
    )
    return payload


def _runtime_target_interface_from_raw_index(raw_index: dict[str, Any]) -> dict[str, Any] | None:
    fallback: dict[str, Any] | None = None
    for group in raw_index.get("groups", {}).values():
        target = group.get("target", {})
        source = target.get("source") if isinstance(target, dict) else None
        if not isinstance(source, dict) or not source.get("qualified_name"):
            continue
        candidate = dict(source)
        if candidate.get("identity_source") == "runtime_decorated_callable":
            return candidate
        fallback = fallback or candidate
    return fallback


def _copy_baseline_to_targets(source_pack: Path, task_packs: list[Path]) -> None:
    source_docs = source_pack / "docs"
    for task_pack in task_packs:
        docs = task_pack / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        for name in ("baseline_result.json", "baseline_run_report.md"):
            src = source_docs / name
            if src.exists() and src != docs / name:
                shutil.copy2(src, docs / name)


def _write_phase1_reports(output_root: Path, report: dict[str, Any]) -> None:
    _write_json(output_root / "multi_target_report.json", report)
    lines = [
        "# Phase 1 Multi-target Report",
        "",
        f"- task_group_id: `{report['task_group_id']}`",
        f"- output_root: `{report['output_root']}`",
        f"- baseline: `{(report.get('baseline') or {}).get('returncode', 'skipped')}`",
        "",
        "| target | status | task_pack | failed_step |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("targets", []):
        failed = next((step["step"] for step in item.get("steps", []) if step.get("returncode") != 0), "")
        target = item["target"]
        lines.append(f"| `{target['task_id']}` | `{item['status']}` | `{target['task_pack']}` | `{failed}` |")
    (output_root / "multi_target_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_task_contract(task_pack: Path, cfg: Phase1Config, target: TargetConfig) -> None:
    task_pack.mkdir(parents=True, exist_ok=True)
    task_yaml = f'''task_id: "{target.task_id}"
task_group_id: "{cfg.task_group_id}"
task_type: "core_kernel_optimization"
owner: "framework_engineer"
target_model: "{cfg.target_model}"
target_framework: "{cfg.target_framework}"
target_hardware: "{cfg.target_hardware}"

objective: |
  {cfg.objective.replace(chr(10), chr(10) + "  ")}

source_entrypoints:
  target_file: "{target.target_file}"
  target_line: {target.target_line}
  forward_boundary_file: "{cfg.forward_boundary_file}"
  forward_boundary_line: {cfg.forward_boundary_line}

kernel_abi:
  candidate_function: "{target.candidate_function}"
  signature: |
    {target.signature}
  required_semantics: |
    Match captured snapshot-golden behavior on all required selected samples in snapshots/manifest.json.
  forbidden_changes:
    - "Do not change snapshots/"
    - "Do not change snapshot_runtime.py"
    - "Do not change shape_list.json except via Framework Engineer snapshot selection"
    - "Do not change original_source/"
    - "Do not change original_impl.py"
    - "Do not change reference_impl.py"
    - "Do not relax tolerance in correctness_test.py"
    - "Do not change benchmark timing/reset rules"

inputs_are_framework_owned: true
input_construction_policy: |
  Framework Engineer owns input construction. Kernel Engineer may inspect selected snapshots,
  but must not replace them with unrelated random inputs. selected snapshots are the replay source.

commands:
  correctness: "bash scripts/run_correctness.sh"
  benchmark: "bash scripts/run_benchmark.sh"
  profile: "bash scripts/run_ncu.sh"
'''
    (task_pack / "task.yaml").write_text(task_yaml, encoding="utf-8")
    env_manifest = f'''task_id: "{target.task_id}"
generated_by: "framework_engineer"
target_framework: "{cfg.target_framework}"
target_hardware: "{cfg.target_hardware}"

python:
  executable: ""
  version: ""

dependency_policy:
  kernel_engineer_may_install_packages: false
  allowed_paths_only: true
  notes: "Run probe-env to populate concrete toolchain availability."
'''
    (task_pack / "env_manifest.yaml").write_text(env_manifest, encoding="utf-8")


@contextlib.contextmanager
def _temporary_env(values: dict[str, str]):
    old: dict[str, str | None] = {}
    for key, value in values.items():
        old[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def probe_environment(task_pack: Path) -> dict[str, Any]:
    env_probe = task_pack / "env_probe"
    env_probe.mkdir(parents=True, exist_ok=True)
    copy_probe_templates(TEMPLATE_DIR, task_pack)
    return {
        "task_id": task_pack.name,
        "generated_by": "framework_engineer.cli probe-env",
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "pytorch": _run_python_probe("import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"),
        "gpu": _run_python_probe(
            "import torch\n"
            "print(torch.cuda.device_count())\n"
            "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None\n"
            "print('' if p is None else p.name)\n"
            "print('' if p is None else f'{p.major}.{p.minor}')\n"
            "print('' if p is None else p.total_memory)"
        ),
        "triton": _run_command([sys.executable, "env_probe/probe_triton.py"], cwd=task_pack),
        "cutedsl": _run_command([sys.executable, "env_probe/probe_cutedsl.py"], cwd=task_pack),
        "cuda_extension": _run_command([sys.executable, "env_probe/probe_cuda_extension.py"], cwd=task_pack),
        "ncu": _run_command(["bash", "env_probe/probe_ncu.sh"], cwd=task_pack),
        "dependency_policy": {
            "kernel_engineer_may_install_packages": False,
            "allowed_paths_only": True,
        },
    }


def _copy_template(name: str, dst: Path) -> None:
    src = TEMPLATE_DIR / name
    if src.exists():
        shutil.copy2(src, dst)


def _run_python_probe(code: str) -> dict[str, Any]:
    return _run_command([sys.executable, "-c", code], cwd=Path.cwd())


def _run_command(cmd: list[str], *, cwd: Path, timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {
            "available": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": " ".join(cmd),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except FileNotFoundError as exc:
        return {"available": False, "returncode": 127, "command": " ".join(cmd), "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "returncode": 124, "command": " ".join(cmd), "stdout": exc.stdout or "", "stderr": exc.stderr or "timeout"}


def _run_service_and_workload(
    *,
    service_cmd: str,
    workload_cmd: str,
    health_url: str | None,
    startup_timeout: int,
    workload_timeout: int,
) -> dict[str, Any]:
    started = time.time()
    env = _subprocess_env()
    result: dict[str, Any] | None = None
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as service_stdout, tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as service_stderr:
        service = subprocess.Popen(
            service_cmd,
            shell=True,
            stdout=service_stdout,
            stderr=service_stderr,
            text=True,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        service_returncode_before_termination: int | None = None
        try:
            health = _wait_for_service(service, health_url=health_url, timeout=startup_timeout)
            startup_elapsed = time.time() - started
            workload_start = time.time()
            try:
                workload = subprocess.run(
                    workload_cmd,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=workload_timeout,
                    check=False,
                )
                workload_returncode = workload.returncode
                workload_stdout = workload.stdout
                workload_stderr = workload.stderr
                workload_timed_out = False
            except subprocess.TimeoutExpired as exc:
                workload_returncode = 124
                workload_stdout = _subprocess_text(exc.stdout)
                workload_stderr = _subprocess_text(exc.stderr)
                timeout_message = f"workload timed out after {workload_timeout}s"
                workload_stderr = f"{workload_stderr.rstrip()}\n{timeout_message}".lstrip()
                workload_timed_out = True
            workload_elapsed = time.time() - workload_start
            result = {
                "service_cmd": service_cmd,
                "workload_cmd": workload_cmd,
                "health": health,
                "startup_elapsed_sec": startup_elapsed,
                "workload": {
                    "returncode": workload_returncode,
                    "elapsed_sec": workload_elapsed,
                    "timed_out": workload_timed_out,
                    "stdout": workload_stdout[-8000:],
                    "stderr": workload_stderr[-8000:],
                },
            }
        finally:
            service_returncode_before_termination = service.poll()
            _terminate_process(service)
            service_stdout.flush()
            service_stderr.flush()
            service_stdout.seek(0)
            service_stderr.seek(0)
            if result is not None:
                result["service"] = {
                    "returncode_before_termination": service_returncode_before_termination,
                    "returncode_after_termination": service.returncode,
                    "stdout": service_stdout.read()[-8000:],
                    "stderr": service_stderr.read()[-8000:],
                }
    if result is None:  # pragma: no cover - unexpected exceptions propagate through the finally block.
        raise RuntimeError("service/workload execution did not produce a result")
    return result


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _wait_for_service(proc: subprocess.Popen, *, health_url: str | None, timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    if health_url is None:
        time.sleep(min(10, timeout))
        return {"mode": "sleep", "ready": proc.poll() is None}
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            return {"mode": "http", "ready": False, "error": f"service exited with {proc.returncode}"}
        try:
            with urlopen(health_url, timeout=5) as response:
                return {"mode": "http", "ready": 200 <= response.status < 500, "status": response.status}
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2)
    return {"mode": "http", "ready": False, "error": last_error or "timeout"}


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    project_root = str(PROJECT_ROOT)
    current = env.get("PYTHONPATH", "")
    parts = [project_root]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _resolve_target_interface(args: argparse.Namespace) -> SourceInterface:
    return _resolve_source_interface(
        file=args.target_file,
        line=args.target_line,
        function_name=args.function_name,
        qualified_name=args.target_name,
        role="target",
    )


def _resolve_forward_boundary_interface(args: argparse.Namespace) -> SourceInterface | None:
    if args.forward_boundary_file is None:
        return None
    return _resolve_source_interface(
        file=args.forward_boundary_file,
        line=args.forward_boundary_line,
        function_name=args.forward_boundary_function,
        qualified_name=args.forward_boundary_name,
        role="forward boundary",
    )


def _resolve_source_interface(
    *,
    file: Path,
    line: int | None,
    function_name: str | None,
    qualified_name: str | None,
    role: str,
) -> SourceInterface:
    file = file.expanduser().resolve()
    if not file.exists():
        raise SystemExit(f"{role}: file does not exist: {file}")
    source = file.read_text(encoding="utf-8")
    module_name = _infer_module_name(file)
    if line is not None:
        resolved = _resolve_interface_by_line(file, source, line, module_name)
        if function_name is not None and function_name != resolved.function_name:
            raise SystemExit(
                f"{role}: --function-name {function_name!r} does not match function at line {line}: "
                f"{resolved.function_name!r}"
            )
        if qualified_name is not None:
            return SourceInterface(
                file=resolved.file,
                function_name=resolved.function_name,
                qualified_name=qualified_name,
                line=resolved.line,
                end_line=resolved.end_line,
                class_path=resolved.class_path,
                module_name=resolved.module_name,
            )
        return resolved
    if function_name is None:
        raise SystemExit(f"{role}: provide either --{_role_prefix(role)}-line or --function-name")
    resolved = _resolve_interface_by_function_name(file, source, function_name, module_name)
    if qualified_name is not None:
        return SourceInterface(
            file=resolved.file,
            function_name=resolved.function_name,
            qualified_name=qualified_name,
            line=resolved.line,
            end_line=resolved.end_line,
            class_path=resolved.class_path,
            module_name=resolved.module_name,
        )
    return resolved


def _role_prefix(role: str) -> str:
    return "forward-boundary" if role == "forward boundary" else "target"


def _resolve_interface_by_line(file: Path, source: str, line: int, module_name: str) -> SourceInterface:
    candidates = _function_candidates(file, source, module_name)
    matching = [
        candidate
        for candidate in candidates
        if candidate.line <= line <= (candidate.end_line or candidate.line)
    ]
    if not matching:
        raise SystemExit(f"No function definition in {file} contains line {line}")
    return min(matching, key=lambda item: ((item.end_line or item.line) - item.line, -len(item.class_path)))


def _resolve_interface_by_function_name(file: Path, source: str, function_name: str, module_name: str) -> SourceInterface:
    matches = [candidate for candidate in _function_candidates(file, source, module_name) if candidate.function_name == function_name]
    if not matches:
        raise SystemExit(f"Could not find function definition {function_name!r} in {file}")
    return sorted(matches, key=lambda item: item.line)[0]


def _function_candidates(file: Path, source: str, module_name: str) -> list[SourceInterface]:
    tree = ast.parse(source, filename=str(file))
    out: list[SourceInterface] = []

    def visit(node: ast.AST, class_path: list[str]) -> None:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit(child, [*class_path, node.name])
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts = [module_name, *class_path, node.name]
            out.append(
                SourceInterface(
                    file=file,
                    function_name=node.name,
                    qualified_name=".".join(part for part in parts if part),
                    line=int(node.lineno),
                    end_line=getattr(node, "end_lineno", None),
                    class_path=list(class_path),
                    module_name=module_name,
                )
            )
            for child in node.body:
                visit(child, class_path)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, class_path)

    visit(tree, [])
    return out


def _infer_module_name(file: Path) -> str:
    package_parts = [] if file.stem == "__init__" else [file.stem]
    parent = file.parent
    while (parent / "__init__.py").exists():
        package_parts.insert(0, parent.name)
        parent = parent.parent
    if len(package_parts) > (0 if file.stem == "__init__" else 1):
        return ".".join(package_parts)

    parts = list(file.with_suffix("").parts)
    if "python" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("python")
        module_parts = parts[idx + 1 :]
        if module_parts:
            return ".".join(module_parts)
    return file.stem


def _instrumentation_context(
    target: SourceInterface,
    boundary: SourceInterface | None,
    target_decorator_expr: str,
):
    stack = contextlib.ExitStack()
    entries: list[tuple[SourceInterface, str]] = []
    if boundary is not None:
        boundary_name = boundary.qualified_name
        boundary_expr = (
            "__import__('framework_engineer.snapshot.recorder', "
            "fromlist=['make_forward_boundary_decorator']).make_forward_boundary_decorator("
            f"{boundary_name!r})"
        )
        entries.append((boundary, boundary_expr))
    entries.append((target, target_decorator_expr))

    by_file: dict[Path, list[tuple[SourceInterface, str]]] = {}
    for interface, expr in entries:
        by_file.setdefault(interface.file, []).append((interface, expr))
    for file, file_entries in by_file.items():
        if len(file_entries) == 1:
            interface, expr = file_entries[0]
            stack.enter_context(_temporary_decorator(file, interface.function_name, expr, line=interface.line))
        else:
            stack.enter_context(_temporary_decorators(file, file_entries))
    return stack


def _resolve_non_cudagraph_service_cmd(service_cmd: str, explicit_cmd: str | None) -> str:
    if explicit_cmd:
        return _dedupe_disable_cuda_graph(explicit_cmd)
    cmd = _dedupe_disable_cuda_graph(service_cmd)
    if "--disable-cuda-graph" not in cmd.split():
        cmd = cmd.rstrip() + " --disable-cuda-graph"
    return cmd


def _dedupe_disable_cuda_graph(cmd: str) -> str:
    parts = cmd.split()
    seen = False
    changed = False
    out = []
    for part in parts:
        if part == "--disable-cuda-graph":
            if seen:
                changed = True
                continue
            seen = True
        out.append(part)
    return " ".join(out) if changed else cmd


def _windowing_mode(args: argparse.Namespace) -> str:
    if args.forward_boundary_file and (args.forward_boundary_function or args.forward_boundary_line):
        return "forward_boundary"
    if getattr(args, "calls_per_forward", None):
        return "calls_per_forward"
    return "unknown_forward"


def _mutation_warning_count(raw_index: dict[str, Any]) -> int:
    count = 0
    for group in raw_index.get("groups", {}).values():
        for sample in group.get("samples", []):
            count += int(sample.get("capture", {}).get("mutation_warning_count", 0))
    return count


def _original_capture_timing_summary_from_raw_index(raw_index: dict[str, Any]) -> dict[str, Any]:
    groups = []
    timings = []
    for group in raw_index.get("groups", {}).values():
        summary = dict(group.get("original_call_timing_summary", {}) or {})
        sample_timings = [
            sample.get("capture", {}).get("original_call_timing", {})
            for sample in group.get("samples", [])
            if sample.get("capture", {}).get("original_call_timing")
        ]
        timings.extend(sample_timings)
        groups.append(
            {
                "group_id": group.get("group_id"),
                "total_hit_count": int(group.get("total_hit_count", 0)),
                "sample_count": int(group.get("sample_count", 0)),
                "original_call_timing_summary": summary,
                "saved_sample_timing_count": len(sample_timings),
            }
        )
    payload = _original_capture_benchmark_payload(groups=groups, timings=timings, source="raw_index_saved_samples")
    payload["overall"] = _timing_stats_from_group_summaries(groups)
    payload["status"] = "present" if payload["overall"].get("count") else "missing"
    payload["source"] = "raw_index_all_tracked_hits"
    return payload


def _original_capture_benchmark_summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    groups = []
    timings = []
    for group in manifest.get("case_groups", []):
        sample_timings = [
            sample.get("capture", {}).get("original_call_timing", {})
            for sample in group.get("samples", [])
            if sample.get("capture", {}).get("original_call_timing")
        ]
        timings.extend(sample_timings)
        groups.append(
            {
                "group_id": group.get("group_id"),
                "selection": group.get("selection", {}),
                "selected_sample_count": len(group.get("samples", [])),
                "original_call_timing_summary": group.get("original_call_timing_summary", {}),
                "selected_sample_timing_count": len(sample_timings),
            }
        )
    return _original_capture_benchmark_payload(groups=groups, timings=timings, source="selected_manifest")


def _original_capture_benchmark_payload(
    *,
    groups: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    elapsed_values = [float(item["elapsed_us"]) for item in timings if item.get("elapsed_us") is not None]
    python_values = [float(item.get("python_call_us", 0.0)) for item in timings if item.get("elapsed_us") is not None]
    post_sync_values = [float(item.get("post_call_sync_us", 0.0)) for item in timings if item.get("elapsed_us") is not None]
    overall = _timing_stats(elapsed_values)
    if overall["count"]:
        overall["python_call_us"] = _timing_stats(python_values)
        overall["post_call_sync_us"] = _timing_stats(post_sync_values)
    status = "present" if overall["count"] else "missing"
    return {
        "schema_version": "phase1.original_capture_benchmark.v1",
        "status": status,
        "source": source,
        "baseline_kind": "captured_original_call_timing_reference",
        "timing_source": "snapshot_capture_decorator",
        "speedup_baseline": False,
        "dump_time_excluded": True,
        "reference_priority": "advisory_only",
        "note": (
            "This is capture-time original framework call timing. It excludes snapshot dump time. "
            "When linked original_impl is executable, benchmark.py reference timing should be used for speedup."
        ),
        "overall": overall,
        "group_count": len(groups),
        "groups": groups,
    }


def _timing_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": len(values),
        "mean_us": sum(values) / len(values),
        "median_us": median,
        "min_us": min(values),
        "max_us": max(values),
    }


def _timing_stats_from_group_summaries(groups: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    total = 0.0
    mins = []
    maxes = []
    for group in groups:
        summary = group.get("original_call_timing_summary", {}) or {}
        group_count = int(summary.get("count", 0) or 0)
        if not group_count:
            continue
        count += group_count
        total += float(summary.get("total_elapsed_us", 0.0) or 0.0)
        if summary.get("min_elapsed_us") is not None:
            mins.append(float(summary["min_elapsed_us"]))
        if summary.get("max_elapsed_us") is not None:
            maxes.append(float(summary["max_elapsed_us"]))
    if not count:
        return {"count": 0}
    return {
        "count": count,
        "mean_us": total / count,
        "median_us": None,
        "min_us": min(mins) if mins else None,
        "max_us": max(maxes) if maxes else None,
    }


def _write_original_capture_benchmark_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary.get("overall", {})
    lines = [
        "# Original Capture Benchmark Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- baseline_kind: `{summary.get('baseline_kind')}`",
        f"- source: `{summary.get('source')}`",
        f"- speedup_baseline: `{summary.get('speedup_baseline')}`",
        f"- dump_time_excluded: `{summary.get('dump_time_excluded')}`",
        "",
        "This timing is advisory. Prefer `benchmark.py --target reference` when linked `original_impl.py` is executable.",
        "",
        "## Overall",
        "",
        f"- count: `{overall.get('count', 0)}`",
    ]
    if overall.get("count"):
        lines.extend(
            [
                f"- mean_us: `{overall.get('mean_us', 0.0):.3f}`",
                f"- median_us: `{overall.get('median_us', 0.0):.3f}`",
                f"- min_us: `{overall.get('min_us', 0.0):.3f}`",
                f"- max_us: `{overall.get('max_us', 0.0):.3f}`",
            ]
        )
    lines.extend(["", "## Groups", "", "| group | samples | mean_us | median_us |", "| --- | ---: | ---: | ---: |"])
    for group in summary.get("groups", []):
        group_summary = group.get("original_call_timing_summary", {}) or {}
        lines.append(
            "| `{}` | `{}` | `{:.3f}` | `{}` |".format(
                group.get("group_id"),
                group.get("selected_sample_timing_count", group.get("saved_sample_timing_count", 0)),
                float(group_summary.get("mean_elapsed_us", 0.0) or 0.0),
                "n/a",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _original_capture_benchmark_check(task_pack: Path) -> dict[str, Any]:
    path = task_pack / "docs" / "original_capture_benchmark_summary.json"
    if not path.exists():
        return {
            "status": "missing",
            "required": False,
            "speedup_baseline": False,
            "note": "Capture-time timing reference is missing; task pack may predate this feature.",
        }
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


class _temporary_decorator:
    def __init__(self, target_file: Path, function_name: str, decorator_expr: str, *, line: int | None = None):
        self.target_file = target_file
        self.function_name = function_name
        self.decorator_expr = decorator_expr
        self.line = line
        self.original = ""

    def __enter__(self):
        self.original = self.target_file.read_text(encoding="utf-8")
        self.target_file.write_text(
            _insert_decorator(self.original, self.function_name, self.decorator_expr, line=self.line),
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.target_file.write_text(self.original, encoding="utf-8")
        return False


class _temporary_decorators:
    def __init__(self, target_file: Path, entries: list[tuple[SourceInterface, str]]):
        self.target_file = target_file
        self.entries = entries
        self.original = ""

    def __enter__(self):
        self.original = self.target_file.read_text(encoding="utf-8")
        source = self.original
        for interface, expr in sorted(self.entries, key=lambda item: item[0].line, reverse=True):
            source = _insert_decorator(source, interface.function_name, expr, line=interface.line)
        self.target_file.write_text(source, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.target_file.write_text(self.original, encoding="utf-8")
        return False


def _insert_decorator(source: str, function_name: str, decorator_expr: str, *, line: int | None = None) -> str:
    lines = source.splitlines()
    needle = f"def {function_name}("
    async_needle = f"async def {function_name}("
    target_line_idx = line - 1 if line is not None else None
    for idx, source_line in enumerate(lines):
        stripped = source_line.lstrip()
        if stripped.startswith(needle) or stripped.startswith(async_needle):
            if target_line_idx is not None and idx != target_line_idx:
                continue
            indent = source_line[: len(source_line) - len(stripped)]
            lines.insert(idx, f"{indent}@{decorator_expr}")
            return "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if target_line_idx is not None:
        raise ValueError(f"Could not find function definition {function_name!r} at line {line}")
    raise ValueError(f"Could not find function definition {function_name!r}")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_baseline_report(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    service = result.get("service", {})
    path.write_text(
        textwrap.dedent(
            f"""\
            # Baseline Run Report

            - service_cmd: `{args.service_cmd}`
            - workload_cmd: `{args.workload_cmd}`
            - health_url: `{args.health_url}`
            - health: `{json.dumps(result.get('health'), sort_keys=True)}`
            - service_returncode_before_termination: `{service.get('returncode_before_termination')}`
            - workload_returncode: `{result['workload']['returncode']}`
            - workload_timed_out: `{result['workload'].get('timed_out', False)}`
            - workload_elapsed_sec: `{result['workload']['elapsed_sec']:.3f}`

            ## Service Stdout Tail

            ```text
            {service.get('stdout', '')}
            ```

            ## Service Stderr Tail

            ```text
            {service.get('stderr', '')}
            ```

            ## Workload Stdout Tail

            ```text
            {result['workload']['stdout']}
            ```

            ## Workload Stderr Tail

            ```text
            {result['workload']['stderr']}
            ```
            """
        ),
        encoding="utf-8",
    )


def _env_to_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(value: Any, indent: int, key: str | None = None) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                indent += 2
                prefix = " " * indent
            for k, v in value.items():
                emit(v, indent, str(k))
        elif isinstance(value, list):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                indent += 2
                prefix = " " * indent
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    emit(item, indent + 2)
                else:
                    lines.append(f"{prefix}- {json.dumps(item)}")
        else:
            scalar = json.dumps(value)
            if key is None:
                lines.append(f"{prefix}{scalar}")
            else:
                lines.append(f"{prefix}{key}: {scalar}")

    emit(data, 0)
    return "\n".join(lines) + "\n"


def _compare_availability(expected: dict[str, Any], current: dict[str, Any]) -> list[str]:
    expected_map = _availability_map(expected)
    current_map = _availability_map(current)
    mismatches = []
    for key, expected_value in sorted(expected_map.items()):
        current_value = current_map.get(key)
        if current_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {current_value}")
    return mismatches


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
        for idx, item in enumerate(data):
            out.update(_availability_map(item, f"{prefix}.{idx}"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
