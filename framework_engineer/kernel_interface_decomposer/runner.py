from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .config import RuntimeCaptureConfig
from .trace_parser import RuntimeTraceParser


ENVIRONMENT_VERSION = "kid-runtime-environment/v1"


def _prepare_layout(root: Path) -> None:
    (root / "capture_events").mkdir(parents=True, exist_ok=True)
    (root / "trace").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "_inject").mkdir(parents=True, exist_ok=True)


def _write_injection(config: RuntimeCaptureConfig, root: Path) -> Path:
    inject = root / "_inject"
    runtime_config = config.runtime_config(events_dir=root / "capture_events")
    runtime_config["output_dir"] = str(root)
    runtime_config["recording_gate_file"] = str(
        root / "_inject" / "recording.enabled"
    )
    runtime_config["active_ranges_dir"] = str(
        root / "_inject" / "active_ranges"
    )
    runtime_path = inject / "runtime_config.json"
    runtime_path.write_text(
        json.dumps(runtime_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (inject / "sitecustomize.py").write_text(
        "from framework_engineer.kernel_interface_decomposer.runtime_instrumentation "
        "import install_from_env\ninstall_from_env()\n",
        encoding="utf-8",
    )
    return runtime_path


def _base_env(config: RuntimeCaptureConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(config.env)
    return env


def _injected_env(
    config: RuntimeCaptureConfig, root: Path, runtime_config_path: Path
) -> dict[str, str]:
    env = _base_env(config)
    package_root = Path(__file__).resolve().parents[2]
    paths = [str(root / "_inject"), str(package_root), str(config.workdir)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["KID_ENABLE"] = "1"
    env["KID_RUNTIME_CONFIG"] = str(runtime_config_path)
    return env


def _package_probe(module: str, distribution: str | None = None) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    version = None
    try:
        version = importlib.metadata.version(distribution or module)
    except importlib.metadata.PackageNotFoundError:
        pass
    return {
        "present": spec is not None,
        "version": version,
        "origin": getattr(spec, "origin", None) if spec else None,
    }


def probe_environment(config: RuntimeCaptureConfig, root: Path) -> dict[str, Any]:
    nsys_requested = str(config.profiling["nsys_bin"])
    nsys_resolved = shutil.which(nsys_requested)
    if not nsys_resolved:
        raise RuntimeError(f"nsys binary not found: {nsys_requested}")
    version_result = subprocess.run(
        [nsys_resolved, "--version"], text=True, capture_output=True, check=False
    )
    nsys_version = (version_result.stdout or version_result.stderr).strip()
    gpu: dict[str, Any] = {"cuda_available": False}
    torch_error = None
    try:
        import torch

        available = bool(torch.cuda.is_available())
        gpu = {
            "cuda_available": available,
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability(0)) if available else None
            ),
        }
    except Exception as exc:
        torch_error = f"{type(exc).__name__}: {exc}"
    dependencies = {
        "torch": _package_probe("torch"),
        "triton": _package_probe("triton"),
        "cutlass": _package_probe("cutlass", "nvidia-cutlass-dsl"),
        "tilelang": _package_probe("tilelang"),
        "tvm_ffi": _package_probe("tvm_ffi", "apache-tvm-ffi"),
        "deep_gemm": _package_probe("deep_gemm", "sgl-deep-gemm"),
    }
    probe = {
        "schema_version": ENVIRONMENT_VERSION,
        "created_at_unix": time.time(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "gpu": gpu,
        "nsight_systems": {
            "requested": nsys_requested,
            "resolved": nsys_resolved,
            "version": nsys_version,
        },
        "capture_adapter_dependencies": dependencies,
        "capture_adapters": {
            "pytorch_dispatch": dependencies["torch"]["present"],
            "triton_launch": dependencies["triton"]["present"],
            "cute_dsl_launch": dependencies["cutlass"]["present"],
            "tilelang_launch": dependencies["tilelang"]["present"],
            "tvm_ffi_call": dependencies["tvm_ffi"]["present"],
            "inductor_launch": dependencies["torch"]["present"],
            "python_binding": dependencies["deep_gemm"]["present"],
        },
        "torch_probe_error": torch_error,
    }
    probe_path = root / "environment_probe.json"
    probe_path.write_text(
        json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "logs" / "probe.log").write_text(
        json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not gpu.get("cuda_available"):
        raise RuntimeError("PyTorch CUDA is unavailable; inspect logs/probe.log")
    return probe


def _eager_command(command: str) -> str:
    if "sglang.launch_server" in command and "--disable-cuda-graph" not in command.split():
        return command.rstrip() + " --disable-cuda-graph"
    return command


def _nsys_launch_command(
    config: RuntimeCaptureConfig,
    session_name: str,
    application_command: str,
) -> list[str]:
    """Launch a service in a paused interactive Nsight session."""

    return [
        str(config.profiling["nsys_bin"]),
        "launch",
        f"--session-new={session_name}",
        "--trace=cuda,nvtx",
        "--trace-fork-before-exec=true",
        "--show-output=true",
        "--wait=all",
        "bash",
        "-lc",
        application_command,
    ]


def _nsys_start_command(
    config: RuntimeCaptureConfig, root: Path, session_name: str
) -> list[str]:
    return [
        str(config.profiling["nsys_bin"]),
        "start",
        f"--session={session_name}",
        "--force-overwrite=true",
        f"--output={root / '_profile'}",
    ]


def _nsys_stop_command(
    config: RuntimeCaptureConfig, session_name: str
) -> list[str]:
    return [
        str(config.profiling["nsys_bin"]),
        "stop",
        f"--session={session_name}",
    ]


def _wait_ready(config: RuntimeCaptureConfig, process: subprocess.Popen[Any]) -> None:
    ready = config.ready or {}
    if not ready:
        return
    ready_type = str(ready.get("type", "none"))
    timeout = float(ready.get("timeout_sec", 300))
    if ready_type == "sleep":
        time.sleep(float(ready.get("seconds", ready.get("sleep_sec", 5))))
        return
    if ready_type in {"none", "null"}:
        return
    if ready_type != "http":
        raise RuntimeError(f"unsupported ready.type: {ready_type}")
    url = ready.get("url")
    if not url:
        raise RuntimeError("ready.url is required for HTTP readiness")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"profiled service exited before readiness with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(str(url), timeout=3) as response:
                if 200 <= response.status < 500:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"service did not become ready in {timeout}s: {url}")


def _stop_process_group(
    process: subprocess.Popen[Any], config: RuntimeCaptureConfig
) -> None:
    if process.poll() is not None:
        return
    stop = config.stop or {}
    signal_name = str(stop.get("signal", "SIGINT"))
    stop_signal = getattr(signal, signal_name, signal.SIGINT)
    grace = float(stop.get("grace_sec", 30))
    try:
        os.killpg(process.pid, stop_signal)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline and process.poll() is None:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_recording_drain(
    root: Path, process: subprocess.Popen[Any], timeout: float
) -> None:
    """Wait until high-level calls admitted before gate close have returned."""

    active_dir = root / "_inject" / "active_ranges"
    deadline = time.monotonic() + min(timeout, 60.0)
    empty_since: float | None = None
    while time.monotonic() < deadline:
        if not list(active_dir.glob("*.active")):
            now = time.monotonic()
            if empty_since is None:
                empty_since = now
            elif now - empty_since >= 0.1:
                return
        else:
            empty_since = None
        if process.poll() is not None:
            raise RuntimeError(
                "profiled service exited while KID high-level ranges were active"
            )
        time.sleep(0.05)
    active = sorted(path.name for path in active_dir.glob("*.active"))
    raise TimeoutError(f"KID high-level ranges did not drain: {active}")


def _wait_for_marker(
    marker: Path,
    process: subprocess.Popen[Any],
    timeout: float,
    description: str,
    *,
    failure_marker: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return
        if failure_marker is not None and failure_marker.exists():
            status = json.loads(failure_marker.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"direct workload failed during {status.get('phase')} with exit code "
                f"{status.get('returncode')}"
            )
        if process.poll() is not None:
            raise RuntimeError(
                f"Nsight-owned process exited before {description} with code "
                f"{process.returncode}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {description}: {marker}")


def _direct_launcher_command(
    config: RuntimeCaptureConfig, root: Path, timeout: float
) -> str:
    inject = root / "_inject"
    return shlex.join(
        [
            sys.executable,
            "-m",
            "framework_engineer.kernel_interface_decomposer.direct_launcher",
            "--command",
            config.test_command,
            "--workdir",
            str(config.workdir),
            "--warmup-log",
            str(root / "logs" / "warmup.log"),
            "--test-log",
            str(root / "logs" / "test.log"),
            "--warmup-ready-file",
            str(inject / "warmup.ready"),
            "--recording-gate-file",
            str(inject / "recording.enabled"),
            "--test-done-file",
            str(inject / "test.done.json"),
            "--shutdown-file",
            str(inject / "shutdown.requested"),
            "--timeout-sec",
            str(timeout),
        ]
    )


def _run_nsys_control(
    command: list[str],
    *,
    config: RuntimeCaptureConfig,
    log: Any,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    log.write("command: " + shlex.join(command) + "\n")
    log.flush()
    return subprocess.run(
        command,
        cwd=config.workdir,
        env=_base_env(config),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _run_profile(config: RuntimeCaptureConfig, root: Path, env: dict[str, str]) -> Path:
    nsys_log_path = root / "logs" / "nsys.log"
    warmup_log_path = root / "logs" / "warmup.log"
    test_log_path = root / "logs" / "test.log"
    timeout = float(config.profiling["max_runtime_sec"])
    session_name = "KID" + uuid.uuid4().hex
    gate_path = root / "_inject" / "recording.enabled"
    active_dir = root / "_inject" / "active_ranges"
    active_dir.mkdir(parents=True, exist_ok=True)
    direct_mode = config.command is None
    application_command = (
        _direct_launcher_command(config, root, timeout)
        if direct_mode
        else _eager_command(config.command or "")
    )
    command = _nsys_launch_command(config, session_name, application_command)
    profiled_process: subprocess.Popen[Any] | None = None
    collection_started = False
    failure: BaseException | None = None
    shutdown_path = root / "_inject" / "shutdown.requested"
    with nsys_log_path.open("w", encoding="utf-8") as nsys_log:
        nsys_log.write("command: " + shlex.join(command) + "\n")
        nsys_log.flush()
        try:
            profiled_process = subprocess.Popen(
                command,
                cwd=config.workdir,
                env=env,
                stdout=nsys_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            if direct_mode:
                _wait_for_marker(
                    root / "_inject" / "warmup.ready",
                    profiled_process,
                    timeout,
                    "direct workload warmup",
                    failure_marker=root / "_inject" / "test.done.json",
                )
            else:
                _wait_ready(config, profiled_process)
                warmup_log_path.write_text(
                    "service startup/readiness completed outside Nsight collection\n",
                    encoding="utf-8",
                )

            start_result = _run_nsys_control(
                _nsys_start_command(config, root, session_name),
                config=config,
                log=nsys_log,
                timeout=timeout,
            )
            if start_result.returncode != 0:
                raise RuntimeError(
                    f"nsys start failed with exit code {start_result.returncode}"
                )
            collection_started = True
            gate_path.touch()

            if direct_mode:
                done_path = root / "_inject" / "test.done.json"
                _wait_for_marker(
                    done_path,
                    profiled_process,
                    timeout,
                    "profiled direct workload",
                )
                status = json.loads(done_path.read_text(encoding="utf-8"))
                if status.get("phase") != "test" or status.get("returncode") != 0:
                    raise RuntimeError(
                        f"direct workload failed during {status.get('phase')} with "
                        f"exit code {status.get('returncode')}"
                    )
            else:
                with test_log_path.open("w", encoding="utf-8") as test_log:
                    test_result = subprocess.run(
                        ["bash", "-lc", config.test_command],
                        cwd=config.workdir,
                        env=_base_env(config),
                        stdout=test_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout,
                    )
                if test_result.returncode != 0:
                    raise RuntimeError(
                        f"test command failed with exit code {test_result.returncode}"
                    )
        except BaseException as exc:
            failure = exc
        finally:
            gate_path.unlink(missing_ok=True)
            if collection_started and profiled_process is not None:
                try:
                    _wait_recording_drain(root, profiled_process, timeout)
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                try:
                    stop_result = _run_nsys_control(
                        _nsys_stop_command(config, session_name),
                        config=config,
                        log=nsys_log,
                        timeout=timeout,
                    )
                    if stop_result.returncode != 0 and failure is None:
                        failure = RuntimeError(
                            "nsys stop failed with exit code "
                            f"{stop_result.returncode}"
                        )
                except BaseException as exc:
                    if failure is None:
                        failure = exc
            shutdown_path.touch()
            if profiled_process is not None:
                if not direct_mode:
                    _stop_process_group(profiled_process, config)
                try:
                    profiled_process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    _stop_process_group(profiled_process, config)
                    try:
                        profiled_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        profiled_process.kill()
        if failure is not None:
            raise failure
    report = root / "_profile.nsys-rep"
    if not report.exists():
        alternatives = sorted(root.glob("_profile*.nsys-rep"))
        if alternatives:
            report = alternatives[0]
    if not report.exists():
        raise RuntimeError(f"nsys did not create a report under {root}")
    if not test_log_path.exists() or test_log_path.stat().st_size == 0:
        test_log_path.write_text("test command completed successfully\n", encoding="utf-8")
    return report


def export_sqlite(
    config: RuntimeCaptureConfig, report: Path, sqlite_path: Path, log_path: Path
) -> Path:
    command = [
        str(config.profiling["nsys_bin"]),
        "export",
        "--force-overwrite=true",
        "--type=sqlite",
        f"--output={sqlite_path}",
        str(report),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write("export: " + shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=config.workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError(f"nsys export failed with exit code {result.returncode}")
    return sqlite_path


def _add_artifact_metadata(
    result: dict[str, Any],
    *,
    created_at: float | None = None,
    sqlite_retained: bool = True,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "environment_probe": "environment_probe.json",
        "capture_events": "capture_events/events_*.jsonl",
        "logs": {
            "probe": "logs/probe.log",
            "nsys": "logs/nsys.log",
            "test": "logs/test.log",
            "summary": "logs/summary.log",
        },
    }
    if sqlite_retained:
        artifacts["sqlite"] = "trace/profile.sqlite"
    result["artifacts"] = artifacts
    result["run"] = {
        "run_id": uuid.uuid4().hex,
        "created_at_unix": created_at if created_at is not None else time.time(),
    }
    return result


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        "KID Runtime Capture",
        "===================",
        f"Backend: {result.get('backend_name')}",
        "GPU metric: sum of correlated Nsight kernel activity durations.",
        "",
    ]
    for index, invocation in enumerate(result.get("invocations", []), start=1):
        high = invocation["high_level"]
        lines.append(
            f"Invocation {index}: {high['interface']} call_id={high['call_id']} "
            f"stage={high.get('stage', 'unknown')}"
        )
        lines.append(
            f"  CPU NVTX={high['nvtx_cpu_duration_us']:.3f} us | "
            f"GPU sum={high['gpu_kernel_sum_us']:.3f} us | "
            f"coverage={invocation['coverage']:.2%}"
        )
        for capture in sorted(
            invocation.get("execution_captures", []),
            key=lambda item: int(item.get("hotspot_rank", 10**9)),
        ):
            metrics = capture["metrics"]
            lines.append(
                f"  #{capture['hotspot_rank']} {capture['archetype']} "
                f"{capture['execution_interface']}: "
                f"direct={metrics['direct_gpu_kernel_sum_us']:.3f} us "
                f"inclusive={metrics['inclusive_gpu_kernel_sum_us']:.3f} us"
            )
        if invocation.get("unattributed_kernel_ids"):
            lines.append(
                "  unattributed: " + ", ".join(invocation["unattributed_kernel_ids"])
            )
        lines.append("")
    diagnostics = result.get("diagnostics", {})
    lines.append(
        "Observed/eligible/selected invocations: "
        f"{diagnostics.get('observed_invocation_count', 0)}/"
        f"{diagnostics.get('eligible_invocation_count', 0)}/"
        f"{diagnostics.get('selected_invocation_count', 0)}"
    )
    if "unique_decomposition_count" in diagnostics:
        lines.append(
            "Unique decomposition groups: "
            f"{diagnostics.get('unique_decomposition_count', 0)}"
        )
    return "\n".join(lines) + "\n"


def _publish(staging: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)


def analyze_existing_trace(
    config: RuntimeCaptureConfig,
    *,
    sqlite_path: Path,
    events_dir: Path,
    write_output: bool = True,
) -> dict[str, Any]:
    retain_sqlite = config.profiling["trace_retention"] == "always"
    result = _add_artifact_metadata(
        RuntimeTraceParser(config).parse(sqlite_path.resolve(), events_dir.resolve()),
        sqlite_retained=retain_sqlite,
    )
    if write_output:
        destination = config.cli_dir()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.kid-{uuid.uuid4().hex}"
        _prepare_layout(staging)
        try:
            preserve_explicit_sqlite = (
                sqlite_path.resolve()
                == (destination / "trace" / "profile.sqlite").resolve()
            )
            if retain_sqlite or preserve_explicit_sqlite:
                shutil.copy2(sqlite_path, staging / "trace" / "profile.sqlite")
            else:
                shutil.rmtree(staging / "trace")
            for source_event in sorted(events_dir.glob("events_*.jsonl")):
                shutil.copy2(source_event, staging / "capture_events" / source_event.name)
            (staging / "environment_probe.json").write_text(
                json.dumps(
                    {
                        "schema_version": ENVIRONMENT_VERSION,
                        "created_at_unix": time.time(),
                        "mode": "offline_analyze",
                        "python": {
                            "version": sys.version,
                            "executable": sys.executable,
                            "platform": platform.platform(),
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            for log_name in ("probe.log", "nsys.log", "warmup.log", "test.log"):
                (staging / "logs" / log_name).write_text(
                    "offline analyze: no command log\n", encoding="utf-8"
                )
            (staging / "runtime_capture.schema.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (staging / "logs" / "summary.log").write_text(
                format_summary(result), encoding="utf-8"
            )
            shutil.rmtree(staging / "_inject", ignore_errors=True)
            _publish(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return result


def capture_runtime(config: RuntimeCaptureConfig) -> dict[str, Any]:
    if not config.workdir.is_dir():
        raise RuntimeError(f"workdir does not exist: {config.workdir}")
    if not config.target_file.is_file():
        raise RuntimeError(f"target.file does not exist: {config.target_file}")
    destination = config.cli_dir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.kid-{uuid.uuid4().hex}"
    _prepare_layout(staging)
    created_at = time.time()
    retention = str(config.profiling["trace_retention"])
    retain_on_success = retention == "always"
    try:
        probe_environment(config, staging)
        runtime_config_path = _write_injection(config, staging)
        env = _injected_env(config, staging, runtime_config_path)
        report = _run_profile(config, staging, env)
        try:
            sqlite_path = export_sqlite(
                config,
                report,
                staging / "trace" / "profile.sqlite",
                staging / "logs" / "nsys.log",
            )
        finally:
            report.unlink(missing_ok=True)
        result = _add_artifact_metadata(
            RuntimeTraceParser(config).parse(sqlite_path, staging / "capture_events"),
            created_at=created_at,
            sqlite_retained=retain_on_success,
        )
        (staging / "runtime_capture.schema.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (staging / "logs" / "summary.log").write_text(
            format_summary(result), encoding="utf-8"
        )
        from .artifact_validator import RuntimeArtifactValidator

        validator = RuntimeArtifactValidator(staging)
        if not validator.validate():
            raise RuntimeError(
                "Runtime artifact validation failed: " + "; ".join(validator.errors)
            )
        if not retain_on_success:
            sqlite_path.unlink(missing_ok=True)
            try:
                (staging / "trace").rmdir()
            except OSError:
                pass
        shutil.rmtree(staging / "_inject", ignore_errors=True)
    except Exception:
        shutil.rmtree(staging / "_inject", ignore_errors=True)
        for stale_report in staging.rglob("*.nsys-rep"):
            stale_report.unlink(missing_ok=True)
        if retention == "never":
            shutil.rmtree(staging / "trace", ignore_errors=True)
        _publish(staging, destination)
        raise
    _publish(staging, destination)
    return result
