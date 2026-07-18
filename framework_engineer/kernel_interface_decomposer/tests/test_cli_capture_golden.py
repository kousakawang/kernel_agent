from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
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
RUN_GPU_E2E = os.environ.get("KID_RUN_GPU_E2E") == "1"


def _failure_logs(output_dir: Path) -> str:
    sections: list[str] = []
    for relative in ("logs/test.log", "logs/nsys.log", "logs/probe.log"):
        path = output_dir / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Keep unittest output readable while retaining the final traceback and
        # Nsight error, which normally occur at the end of each log.
        if len(text) > 20_000:
            text = "...<last 20000 characters>...\n" + text[-20_000:]
        sections.append(f"===== {relative} =====\n{text}")
    return "\n".join(sections) or "<no Runtime Capture logs were published>"


def _target_line(path: Path, function_name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {function_name!r} definition in {path}; got {matches}"
        )
    return matches[0]


def _capture_depth(capture_id: str, captures: dict[str, dict[str, Any]]) -> int:
    depth = 0
    current = captures[capture_id]
    seen = {capture_id}
    while current.get("parent_capture_id") is not None:
        parent_id = str(current["parent_capture_id"])
        if parent_id in seen or parent_id not in captures:
            raise AssertionError(f"invalid capture ancestry at {capture_id}")
        seen.add(parent_id)
        current = captures[parent_id]
        depth += 1
    return depth


def _stable_capture_signature(payload: dict[str, Any]) -> dict[str, Any]:
    """Represent golden facts that should survive a fresh Nsight execution."""

    invocations = payload.get("invocations", [])
    if len(invocations) != 1:
        raise AssertionError(f"expected one selected invocation; got {len(invocations)}")
    invocation = invocations[0]
    captures = {
        str(capture["capture_id"]): capture
        for capture in invocation.get("execution_captures", [])
    }
    capture_rows = sorted(
        (
            capture["archetype"],
            capture["common_interface"],
            capture["execution_interface"],
            capture.get("provider_hint"),
            _capture_depth(capture_id, captures),
            len(capture.get("child_capture_ids", [])),
            len(capture.get("kernel_ids", [])),
            len(capture.get("inclusive_kernel_ids", [])),
        )
        for capture_id, capture in captures.items()
    )
    kernel_rows = sorted(
        (
            captures[str(kernel["owner_capture_id"])]["archetype"],
            captures[str(kernel["owner_capture_id"])]["execution_interface"],
            kernel["name"],
        )
        for kernel in payload.get("kernels", [])
    )
    diagnostics = payload.get("diagnostics", {})
    return {
        "schema_version": payload.get("schema_version"),
        "backend_name": payload.get("backend_name"),
        "target_interface": payload.get("target", {}).get("interface"),
        "raw_capture_event_count": invocation.get("raw_capture_event_count"),
        "capture_without_kernel_count": invocation.get(
            "capture_without_kernel_count"
        ),
        "materialized_capture_count": len(captures),
        "kernel_count": len(payload.get("kernels", [])),
        "selected_invocation_count": diagnostics.get(
            "selected_invocation_count"
        ),
        "capture_rows": capture_rows,
        "kernel_rows": kernel_rows,
    }


@unittest.skipUnless(
    RUN_GPU_E2E,
    "set KID_RUN_GPU_E2E=1 to run the Nsight/GPU capture regression",
)
class TestCaptureCliGolden(unittest.TestCase):
    def test_capture_cli_matches_runtime_golden_structure(self) -> None:
        self.assertIsNotNone(shutil.which("nsys"), "nsys is not installed")
        try:
            import torch
        except ImportError as exc:
            self.fail(f"PyTorch is not installed: {exc}")
        self.assertTrue(torch.cuda.is_available(), "PyTorch CUDA is unavailable")

        golden_root = REPO_ROOT / "example_kernels/nsys_poc_kid_golden"
        golden_cli = golden_root / "cli_log/nsys_poc"
        golden = json.loads(
            (golden_cli / "runtime_capture.schema.json").read_text(encoding="utf-8")
        )
        base_config = json.loads(
            (
                golden_root / "config/nsys_poc/runtime_capture_config.json"
            ).read_text(encoding="utf-8")
        )
        poc_path = (
            REPO_ROOT / "framework_engineer/kernel_interface_decomposer/nsys_poc.py"
        )
        cases = base_config["test_cmd"].split("--cases", 1)[1].strip()

        with tempfile.TemporaryDirectory(prefix="kid-capture-golden-") as tempdir:
            temp_root = Path(tempdir)
            output_dir = temp_root / "nsys_poc"
            config_path = temp_root / "runtime_capture_config.json"
            base_config["workdir"] = str(REPO_ROOT.parent)
            base_config["output_dir"] = str(output_dir)
            base_config["target"] = {
                "file": str(poc_path),
                "line": _target_line(poc_path, "high_level"),
                "qualified_name": "high_level",
            }
            base_config["test_cmd"] = (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(poc_path))} "
                f"--worker --cases {shlex.quote(cases)}"
            )
            config_path.write_text(
                json.dumps(base_config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(REPO_ROOT), existing_pythonpath) if value
            )
            completed = subprocess.run(
                [sys.executable, "-m", MODULE, "capture", str(config_path)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                self.fail(
                    f"capture CLI exited with {completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}\n"
                    f"{_failure_logs(output_dir)}"
                )

            actual = json.loads(
                (output_dir / "runtime_capture.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            validator = RuntimeArtifactValidator(output_dir)
            self.assertTrue(validator.validate(), msg="; ".join(validator.errors))
            self.assertEqual(
                _stable_capture_signature(actual),
                _stable_capture_signature(golden),
            )
            self.assertEqual(actual["invocations"][0]["coverage"], 1.0)
            self.assertTrue(
                all(float(kernel["duration_us"]) > 0 for kernel in actual["kernels"])
            )


if __name__ == "__main__":
    unittest.main()
