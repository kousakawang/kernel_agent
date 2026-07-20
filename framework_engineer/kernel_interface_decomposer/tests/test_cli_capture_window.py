from __future__ import annotations

import ast
import json
import os
import shutil
import socket
import sqlite3
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
from framework_engineer.kernel_interface_decomposer.sampling import (  # noqa: E402
    decomposition_signature,
)
MODULE = "framework_engineer.kernel_interface_decomposer"
FORCE_GPU_E2E = "--gpu-e2e" in sys.argv
while "--gpu-e2e" in sys.argv:
    sys.argv.remove("--gpu-e2e")
RUN_GPU_E2E = os.environ.get("KID_RUN_GPU_E2E") == "1" or FORCE_GPU_E2E


def _failure_logs(output: Path) -> str:
    sections: list[str] = []
    for relative in ("logs/nsys.log", "logs/test.log", "logs/probe.log"):
        path = output / relative
        if path.is_file():
            sections.append(f"===== {relative} =====\n{path.read_text(encoding='utf-8')}")
    return "\n".join(sections)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _target_line(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "high_level"
    )


def _kid_nvtx_count(sqlite_path: Path, kind: str) -> int:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text LIKE ?",
                (f"KID:type={kind}%",),
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _stable_result(payload: dict[str, Any]) -> dict[str, Any]:
    invocation = payload["invocations"][0]
    signature = decomposition_signature(invocation)
    return {
        "signature": signature,
        "raw_capture_event_count": invocation["raw_capture_event_count"],
        "materialized_capture_count": len(invocation["execution_captures"]),
        "kernel_count": len(payload["kernels"]),
        "coverage": invocation["coverage"],
    }


@unittest.skipUnless(
    RUN_GPU_E2E,
    "set KID_RUN_GPU_E2E=1 (or pass --gpu-e2e) for the Nsight/GPU regression",
)
class TestCaptureWindow(unittest.TestCase):
    maxDiff = None

    def _capture(
        self,
        root: Path,
        *,
        backend: str,
        fixture: Path,
        command: str | None,
        test_command: str,
        ready: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], Path]:
        output = root / backend
        config = {
            "schema_version": "kid-runtime-config/v2",
            "backend_name": backend,
            "workdir": str(REPO_ROOT.parent),
            "output_dir": str(output),
            "target": {
                "file": str(fixture),
                "line": _target_line(fixture),
                "qualified_name": "high_level",
            },
            "cmd": command,
            "test_cmd": test_command,
            "ready": ready,
            "stop": {"signal": "SIGINT", "grace_sec": 10} if command else None,
            "env": {},
            "selection": {
                "skip_invocations": 0,
                "stages": ["unknown"],
                "sample_count_per_stage": 1,
                "sampling": "single",
                "aggregation": "single",
            },
            "profiling": {
                "nsys_bin": "nsys",
                "max_runtime_sec": 300,
                "disable_cuda_graph": True,
                "min_capture_coverage": 1.0,
            },
        }
        config_path = root / f"{backend}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
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
        if completed.returncode != 0:
            self.fail(
                f"capture {backend} failed with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n"
                f"{_failure_logs(output)}"
            )
        payload = json.loads(
            (output / "runtime_capture.schema.json").read_text(encoding="utf-8")
        )
        validator = RuntimeArtifactValidator(output)
        self.assertTrue(validator.validate(), msg="; ".join(validator.errors))
        return payload, output

    def test_server_startup_target_calls_are_excluded(self) -> None:
        self.assertIsNotNone(shutil.which("nsys"), "nsys is not installed")
        try:
            import torch
        except ImportError as exc:
            self.fail(f"PyTorch is not installed: {exc}")
        self.assertTrue(torch.cuda.is_available(), "PyTorch CUDA is unavailable")

        fixture = Path(__file__).resolve().parent / "fixtures/capture_window_workload.py"
        python = shutil.which("python3") or sys.executable
        port = _free_port()
        with tempfile.TemporaryDirectory(prefix="kid-capture-window-") as temporary:
            root = Path(temporary)
            direct, direct_output = self._capture(
                root,
                backend="capture_window_direct",
                fixture=fixture,
                command=None,
                test_command=f"{python} {fixture} direct",
                ready=None,
            )
            service, service_output = self._capture(
                root,
                backend="capture_window_service",
                fixture=fixture,
                command=f"{python} {fixture} server --port {port}",
                test_command=f"{python} {fixture} client --port {port}",
                ready={
                    "type": "http",
                    "url": f"http://127.0.0.1:{port}/health",
                    "timeout_sec": 120,
                },
            )

            self.assertEqual(_stable_result(service), _stable_result(direct))
            for output in (direct_output, service_output):
                sqlite_path = output / "trace/profile.sqlite"
                self.assertEqual(_kid_nvtx_count(sqlite_path, "high"), 1)
                self.assertEqual(_kid_nvtx_count(sqlite_path, "execution"), 1)
            self.assertEqual(service["diagnostics"]["observed_invocation_count"], 1)
            self.assertEqual(service["diagnostics"]["capture_event_count"], 1)


if __name__ == "__main__":
    unittest.main()
