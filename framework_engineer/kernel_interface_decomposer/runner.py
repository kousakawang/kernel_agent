from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .config import DecomposerConfig
from .source_resolver import SourceResolver
from .trace_parser import TraceParser


def _write_runtime_files(config: DecomposerConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "events").mkdir(exist_ok=True)
    runtime_config = config.to_runtime_dict()
    config.runtime_config_path().write_text(json.dumps(runtime_config, indent=2))

    inject_dir = config.output_dir / "_inject"
    inject_dir.mkdir(exist_ok=True)
    (inject_dir / "sitecustomize.py").write_text(
        "from framework_engineer.kernel_interface_decomposer.runtime_instrumentation import install_from_env\n"
        "install_from_env()\n"
    )
    return inject_dir


def _env_with_injection(config: DecomposerConfig, inject_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(inject_dir), str(config.workdir)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["KID_RUNTIME_CONFIG"] = str(config.runtime_config_path())
    env["KID_ENABLE"] = "1"
    return env


def _nsys_command(config: DecomposerConfig, service_cmd: str) -> list[str]:
    profiling = config.profiling
    nsys = str(profiling.get("nsys_bin", "nsys"))
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,cublas,cudnn,osrt",
        "--target-processes=all",
        "--trace-fork-before-exec=true",
        f"--output={config.nsys_output_base()}",
    ]
    if profiling.get("include_python_backtrace", True):
        cmd.extend(["--python-backtrace=cuda", "--cudabacktrace=all"])
    if profiling.get("trace_cuda_graph_nodes", True):
        cmd.append("--cuda-graph-trace=node")
    cmd.append("--pytorch=functions-trace-shapes,autograd-nvtx")
    cmd.extend(["bash", "-lc", service_cmd])
    return cmd


def _wait_ready(config: DecomposerConfig, proc: subprocess.Popen[Any]) -> None:
    ready = config.ready or {}
    timeout = float(ready.get("timeout_sec", 300))
    deadline = time.time() + timeout
    ready_type = ready.get("type", "none")
    if ready_type == "none" or not ready:
        time.sleep(float(ready.get("sleep_sec", 5)))
        return
    if ready_type == "http":
        url = ready.get("url")
        if not url:
            raise RuntimeError("commands.ready.url is required for http readiness")
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"service exited before readiness with code {proc.returncode}")
            try:
                with urllib.request.urlopen(str(url), timeout=3) as resp:
                    if 200 <= resp.status < 500:
                        return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"service did not become ready before {timeout}s: {url}")
    if ready_type == "sleep":
        time.sleep(float(ready.get("seconds", ready.get("sleep_sec", 5))))
        return
    raise RuntimeError(f"unsupported ready.type: {ready_type}")


def _terminate_process_group(proc: subprocess.Popen[Any], config: DecomposerConfig) -> None:
    stop = config.stop or {}
    sig_name = str(stop.get("signal", "SIGINT"))
    grace = float(stop.get("grace_sec", 30))
    sig = getattr(signal, sig_name, signal.SIGINT)
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def export_sqlite(config: DecomposerConfig, nsys_rep: Path, sqlite_path: Path | None = None) -> Path:
    sqlite_path = sqlite_path or config.sqlite_path()
    if sqlite_path.exists():
        sqlite_path.unlink()
    nsys = str(config.profiling.get("nsys_bin", "nsys"))
    cmd = [
        nsys,
        "export",
        "--force-overwrite=true",
        "--type=sqlite",
        f"--output={sqlite_path}",
        str(nsys_rep),
    ]
    result = subprocess.run(cmd, cwd=config.workdir, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "nsys export failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not sqlite_path.exists():
        alt = sqlite_path.with_suffix(sqlite_path.suffix + ".sqlite")
        if alt.exists():
            return alt
        raise RuntimeError(f"nsys export did not create sqlite file: {sqlite_path}")
    return sqlite_path


def _build_schema(config: DecomposerConfig, nsys_rep: Path, sqlite_path: Path) -> dict[str, Any]:
    resolver = SourceResolver(config)
    parser = TraceParser(config, resolver)
    invocations = parser.parse(sqlite_path)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": uuid.uuid4().hex,
            "workdir": str(config.workdir),
            "nsys_rep": str(nsys_rep),
            "nsys_sqlite": str(sqlite_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "target": {
            "file": str(config.target_file),
            "line": config.target_line,
            "qualified_name": resolver.target_qualified_name,
        },
        "selection": {
            "per_invocation": bool(config.selection.get("per_invocation", True)),
            "top_k": int(config.selection.get("top_k", 20)),
            "min_duration_us": float(config.selection.get("min_duration_us", 0)),
            "min_share_in_invocation": float(config.selection.get("min_share_in_invocation", 0.0)),
        },
        "invocations": invocations,
    }
    config.schema_path().write_text(json.dumps(schema, indent=2, sort_keys=False))
    return schema


def analyze_existing_trace(
    config: DecomposerConfig,
    *,
    nsys_rep: Path,
    sqlite_path: Path | None = None,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if sqlite_path is None:
        sqlite_path = export_sqlite(config, nsys_rep)
    return _build_schema(config, nsys_rep, sqlite_path)


def run_workflow(config: DecomposerConfig) -> dict[str, Any]:
    if not config.service_cmd:
        raise RuntimeError("commands.service is required for run")
    if not config.test_cmd:
        raise RuntimeError("commands.test is required for run")
    if not shutil.which(str(config.profiling.get("nsys_bin", "nsys"))):
        raise RuntimeError("nsys binary not found; set profiling.nsys_bin")

    inject_dir = _write_runtime_files(config)
    env = _env_with_injection(config, inject_dir)
    service_log = (config.output_dir / "service.log").open("w")
    test_log = (config.output_dir / "test.log").open("w")
    service_proc: subprocess.Popen[Any] | None = None
    try:
        service_proc = subprocess.Popen(
            _nsys_command(config, config.service_cmd),
            cwd=config.workdir,
            env=env,
            stdout=service_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        _wait_ready(config, service_proc)
        test_result = subprocess.run(
            ["bash", "-lc", config.test_cmd],
            cwd=config.workdir,
            env=env,
            stdout=test_log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(config.profiling.get("max_runtime_sec", 1800)),
        )
        if test_result.returncode != 0:
            raise RuntimeError(f"test command failed with exit code {test_result.returncode}")
    finally:
        if service_proc is not None:
            _terminate_process_group(service_proc, config)
            service_proc.wait(timeout=60)
        service_log.close()
        test_log.close()

    nsys_rep = config.nsys_rep_path()
    if not nsys_rep.exists():
        raise RuntimeError(f"nsys report not found: {nsys_rep}")
    sqlite_path = export_sqlite(config, nsys_rep)
    return _build_schema(config, nsys_rep, sqlite_path)
