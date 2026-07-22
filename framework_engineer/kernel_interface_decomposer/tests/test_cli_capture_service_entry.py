from __future__ import annotations

import ast
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework_engineer.kernel_interface_decomposer.artifact_validator import (  # noqa: E402
    RuntimeArtifactValidator,
)


MODULE = "framework_engineer.kernel_interface_decomposer"
FORCE_GPU_E2E = "--gpu-e2e" in sys.argv
while "--gpu-e2e" in sys.argv:
    sys.argv.remove("--gpu-e2e")
RUN_GPU_E2E = os.environ.get("KID_RUN_GPU_E2E") == "1" or FORCE_GPU_E2E


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _function_line(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _high_call_line(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "do_GET"
    )
    return next(
        node.lineno
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "high_level"
    )


def _failure_logs(output: Path) -> str:
    sections: list[str] = []
    for relative in ("logs/nsys.log", "logs/test.log", "logs/probe.log"):
        path = output / relative
        if path.is_file():
            sections.append(
                f"===== {relative} =====\n{path.read_text(encoding='utf-8')}"
            )
    return "\n".join(sections)


@unittest.skipUnless(
    RUN_GPU_E2E,
    "set KID_RUN_GPU_E2E=1 (or pass --gpu-e2e) for the Nsight/GPU regression",
)
class TestServiceHighEntryCapture(unittest.TestCase):
    maxDiff = None

    def test_service_capture_records_direct_high_caller(self) -> None:
        self.assertIsNotNone(shutil.which("nsys"), "nsys is not installed")
        try:
            import torch
        except ImportError as exc:
            self.fail(f"PyTorch is not installed: {exc}")
        self.assertTrue(torch.cuda.is_available(), "PyTorch CUDA is unavailable")

        fixture = Path(__file__).resolve().parent / "fixtures/capture_window_workload.py"
        python = shutil.which("python3") or sys.executable
        port = _free_port()
        with tempfile.TemporaryDirectory(prefix="kid-service-entry-") as temporary:
            root = Path(temporary)
            output_root = root / "output"
            backend = "service_entry"
            cli_dir = output_root / backend / "cli_log"
            config: dict[str, Any] = {
                "schema_version": "kid-runtime-config/v3",
                "backend_name": backend,
                "workdir": str(REPO_ROOT.parent),
                "output_dir": str(output_root),
                "target": {
                    "file": str(fixture),
                    "line": _function_line(fixture, "high_level"),
                    "qualified_name": "high_level",
                },
                "cmd": f"{python} {fixture} server --port {port}",
                "test_cmd": f"{python} {fixture} client --port {port}",
                "ready": {
                    "type": "http",
                    "url": f"http://127.0.0.1:{port}/health",
                    "timeout_sec": 120,
                },
                "stop": {"signal": "SIGINT", "grace_sec": 10},
                "env": {},
                "selection": {
                    "skip_invocations": 0,
                    "sample_count_per_stage": 1,
                    "sampling": "single",
                    "aggregation": "single",
                },
                "profiling": {
                    "nsys_bin": "nsys",
                    "max_runtime_sec": 300,
                    "disable_cuda_graph": True,
                    "min_capture_coverage": 1.0,
                    "trace_retention": "always",
                },
            }
            config_path = root / "runtime_capture_config.json"
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(REPO_ROOT), env.get("PYTHONPATH")) if value
            )
            completed = subprocess.run(
                [sys.executable, "-m", MODULE, "capture", str(config_path)],
                cwd=REPO_ROOT.parent,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=360,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=(
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n"
                    f"{_failure_logs(cli_dir)}"
                ),
            )

            runtime = json.loads(
                (cli_dir / "runtime_capture.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(runtime["diagnostics"]["observed_invocation_count"], 1)
            self.assertEqual(runtime["diagnostics"]["high_invocation_event_count"], 1)
            high = runtime["invocations"][0]["high_level"]
            self.assertTrue(high["entry_python_stack"])
            direct_caller = high["entry_python_stack"][-1]
            self.assertEqual(Path(direct_caller["file"]).resolve(), fixture.resolve())
            self.assertEqual(
                direct_caller["call_site_to_next"]["line"], _high_call_line(fixture)
            )
            validator = RuntimeArtifactValidator(cli_dir)
            self.assertTrue(validator.validate(), msg="; ".join(validator.errors))


if __name__ == "__main__":
    unittest.main()
