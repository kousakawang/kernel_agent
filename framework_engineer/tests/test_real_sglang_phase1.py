from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLI_STEPS = (
    "scaffold-task-pack",
    "run-baseline",
    "probe-target-calls",
    "capture-snapshots",
    "select-snapshots",
    "generate-harness",
    "probe-env",
    "validate-task-pack",
)


@dataclass
class RealSGLangConfig:
    service_cmd: str
    workload_cmd: str
    target_file: str
    function_name: str | None = None
    target_name: str | None = None
    target_line: int | None = None
    forward_boundary_file: str | None = None
    forward_boundary_line: int | None = None
    forward_boundary_function: str | None = None
    forward_boundary_name: str | None = None

    task_id: str = "real_sglang_phase1"
    task_pack: str | None = None
    keep_task_pack: bool = True
    skip_baseline: bool = False

    non_cudagraph_service_cmd: str | None = None
    health_url: str | None = None
    startup_timeout: int = 240
    workload_timeout: int = 1200
    test_timeout: int = 3600

    signature: str = "candidate(*args, **kwargs)"
    target_mode: str = ""
    target_backend: str = ""
    target_layer_id: str = ""
    drop_first_arg: bool = False
    calls_per_forward: int | None = None
    max_capture_groups: int = 64
    max_samples_per_group: int = 8
    max_samples_per_forward_per_group: int = 3
    max_raw_cases: int | None = None
    max_selected_groups: int | None = 8
    max_selected_samples_per_group: int = 8
    max_selected_cases: int | None = None
    candidate_function: str = "candidate"

    run_probe_env: bool = False
    skip_env_check: bool = True
    run_benchmark: bool = False
    validate_device: str = "cuda"
    validate_warmup: int = 3
    validate_repeat: int = 5

    cli_tests: dict[str, bool] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)

    def should_run_cli(self, name: str) -> bool:
        if name not in CLI_STEPS:
            raise KeyError(f"Unknown CLI step: {name}")
        return bool(self.cli_tests.get(name, True))


def _load_config() -> RealSGLangConfig:
    config_path = os.environ.get("KA_REAL_SGLANG_CONFIG")
    if not config_path:
        raise unittest.SkipTest("Set KA_REAL_SGLANG_CONFIG=/path/to/real_sglang_phase1_config.py")
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise unittest.SkipTest(f"KA_REAL_SGLANG_CONFIG does not exist: {path}")

    spec = importlib.util.spec_from_file_location("ka_real_sglang_config", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"Cannot import config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "CONFIG"):
        raw = getattr(module, "CONFIG")
        if not isinstance(raw, dict):
            raise TypeError("CONFIG must be a dict when provided")
    else:
        raw = {
            key: getattr(module, key)
            for key in dir(module)
            if key.islower() and not key.startswith("_")
        }

    required = ["service_cmd", "workload_cmd", "target_file"]
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise unittest.SkipTest(f"Missing required config fields: {', '.join(missing)}")
    if not raw.get("target_line") and not raw.get("function_name"):
        raise unittest.SkipTest("Missing target locator: provide target_line or function_name")
    return RealSGLangConfig(**raw)


class RealSGLangPhase1CliTests(unittest.TestCase):
    """Run the Framework Engineer CLI against a real SGLang/workload target."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = _load_config()

    def test_real_sglang_snapshot_task_pack_flow(self) -> None:
        cfg: RealSGLangConfig = self.cfg

        tmpdir: str | None = None
        if cfg.task_pack:
            task_pack = Path(cfg.task_pack).expanduser().resolve()
        else:
            tmpdir = tempfile.mkdtemp(prefix="ka_real_sglang_")
            task_pack = Path(tmpdir) / "task_pack"

        print(f"[ka-real] task_pack={task_pack}")
        try:
            if cfg.should_run_cli("scaffold-task-pack"):
                self._run_cli(
                    "scaffold-task-pack",
                    "--task-id",
                    cfg.task_id,
                    "--out",
                    str(task_pack),
                    "--force",
                )

            if cfg.should_run_cli("run-baseline") and not cfg.skip_baseline:
                self._run_cli(
                    "run-baseline",
                    "--task-pack",
                    str(task_pack),
                    "--service-cmd",
                    cfg.service_cmd,
                    "--workload-cmd",
                    cfg.workload_cmd,
                    *self._service_args(),
                )

            if cfg.should_run_cli("probe-target-calls"):
                probe = self._run_cli(
                    "probe-target-calls",
                    "--task-pack",
                    str(task_pack),
                    "--service-cmd",
                    self._non_cudagraph_service_cmd(cfg.service_cmd),
                    "--workload-cmd",
                    cfg.workload_cmd,
                    "--target-file",
                    cfg.target_file,
                    *self._target_locator_args(),
                    *self._drop_first_arg(),
                    *self._forward_boundary_args(),
                    *self._service_args(),
                )
                self.assertGreater(probe["call_count"], 0, probe)

            if cfg.should_run_cli("capture-snapshots"):
                capture = self._run_cli(
                    "capture-snapshots",
                    "--task-pack",
                    str(task_pack),
                    "--service-cmd",
                    self._non_cudagraph_service_cmd(cfg.service_cmd),
                    "--workload-cmd",
                    cfg.workload_cmd,
                    "--target-file",
                    cfg.target_file,
                    *self._target_locator_args(),
                    "--signature",
                    cfg.signature,
                    "--mode",
                    cfg.target_mode,
                    "--backend",
                    cfg.target_backend,
                    "--layer-id",
                    cfg.target_layer_id,
                    "--max-capture-groups",
                    str(cfg.max_capture_groups),
                    "--max-samples-per-group",
                    str(cfg.max_samples_per_group),
                    "--max-samples-per-forward-per-group",
                    str(cfg.max_samples_per_forward_per_group),
                    *self._calls_per_forward_args(),
                    *self._drop_first_arg(),
                    *self._forward_boundary_args(),
                    *self._service_args(),
                )
                self.assertGreater(capture["raw_sample_count"], 0, capture)

            if cfg.should_run_cli("select-snapshots"):
                selected = self._run_cli(
                    "select-snapshots",
                    "--task-pack",
                    str(task_pack),
                    *self._max_groups_args(),
                    "--max-selected-samples-per-group",
                    str(cfg.max_selected_samples_per_group),
                )
                self.assertGreater(selected["selected_group_count"], 0, selected)

            if cfg.should_run_cli("generate-harness"):
                self._run_cli(
                    "generate-harness",
                    "--task-pack",
                    str(task_pack),
                    "--candidate-function",
                    cfg.candidate_function,
                )

            ran_probe_env = False
            if cfg.should_run_cli("probe-env") and cfg.run_probe_env:
                self._run_cli("probe-env", "--task-pack", str(task_pack))
                ran_probe_env = True

            if cfg.should_run_cli("validate-task-pack"):
                validate_args = [
                    "validate-task-pack",
                    "--task-pack",
                    str(task_pack),
                    "--run-correctness",
                ]
                if cfg.run_benchmark:
                    validate_args.append("--run-benchmark")
                if not ran_probe_env or cfg.skip_env_check:
                    validate_args.append("--skip-env-check")

                validate = self._run_cli(
                    *validate_args,
                    extra_env={
                        "DEVICE": cfg.validate_device,
                        "WARMUP": str(cfg.validate_warmup),
                        "REPEAT": str(cfg.validate_repeat),
                        "PYTHON": sys.executable,
                    },
                )
                self.assertTrue(validate["valid"], validate)

            manifest_path = task_pack / "task" / "snapshots" / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if cfg.should_run_cli("select-snapshots"):
                    self.assertGreater(manifest["selected_group_count"], 0)
        finally:
            if tmpdir and not cfg.keep_task_pack:
                shutil.rmtree(tmpdir, ignore_errors=True)
            elif cfg.keep_task_pack:
                print(f"[ka-real] kept task_pack={task_pack}")

    def _run_cli(self, *args: str, extra_env: dict[str, str] | None = None) -> dict:
        cfg: RealSGLangConfig = self.cfg
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env.update(cfg.extra_env)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            [sys.executable, "-m", "framework_engineer.cli", *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=cfg.test_timeout,
        )
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        self.assertEqual(proc.returncode, 0, f"args={args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, f"no stdout for args={args}")
        return json.loads(lines[-1])

    def _service_args(self) -> list[str]:
        cfg: RealSGLangConfig = self.cfg
        args: list[str] = [
            "--startup-timeout",
            str(cfg.startup_timeout),
            "--workload-timeout",
            str(cfg.workload_timeout),
        ]
        if cfg.health_url:
            args.extend(["--health-url", cfg.health_url])
        return args

    def _drop_first_arg(self) -> list[str]:
        return ["--drop-first-arg"] if self.cfg.drop_first_arg else []

    def _target_locator_args(self) -> list[str]:
        cfg = self.cfg
        if cfg.target_line is not None:
            return ["--target-line", str(cfg.target_line)]
        args = ["--function-name", str(cfg.function_name)]
        if cfg.target_name:
            args.extend(["--target-name", cfg.target_name])
        return args

    def _max_groups_args(self) -> list[str]:
        if self.cfg.max_selected_groups is not None:
            return ["--max-groups", str(self.cfg.max_selected_groups)]
        if self.cfg.max_selected_cases is not None:
            return ["--max-cases", str(self.cfg.max_selected_cases)]
        return []

    def _calls_per_forward_args(self) -> list[str]:
        if self.cfg.calls_per_forward is not None:
            return ["--calls-per-forward", str(self.cfg.calls_per_forward)]
        return []

    def _forward_boundary_args(self) -> list[str]:
        cfg = self.cfg
        if not cfg.forward_boundary_file:
            return []
        args = ["--forward-boundary-file", cfg.forward_boundary_file]
        if cfg.forward_boundary_line is not None:
            args.extend(["--forward-boundary-line", str(cfg.forward_boundary_line)])
        elif cfg.forward_boundary_function:
            args.extend(["--forward-boundary-function", cfg.forward_boundary_function])
        else:
            raise ValueError("forward_boundary_file requires forward_boundary_line or forward_boundary_function")
        if cfg.forward_boundary_name:
            args.extend(["--forward-boundary-name", cfg.forward_boundary_name])
        return args

    def _non_cudagraph_service_cmd(self, service_cmd: str) -> str:
        if self.cfg.non_cudagraph_service_cmd:
            return self.cfg.non_cudagraph_service_cmd
        if "--disable-cuda-graph" in service_cmd:
            return service_cmd
        return service_cmd + " --disable-cuda-graph"


if __name__ == "__main__":
    unittest.main()
