from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE = "framework_engineer.kernel_interface_decomposer"
RECORD_CANDIDATE = "--record-candidate" in sys.argv
FORCE_GPU_E2E = "--gpu-e2e" in sys.argv
DISTINCT_ONLY = "--distinct-only" in sys.argv
for custom_argument in ("--record-candidate", "--gpu-e2e", "--distinct-only"):
    while custom_argument in sys.argv:
        sys.argv.remove(custom_argument)
RUN_GPU_E2E = (
    os.environ.get("KID_RUN_GPU_E2E") == "1" or FORCE_GPU_E2E or RECORD_CANDIDATE
)

from framework_engineer.kernel_interface_decomposer.artifact_validator import (  # noqa: E402
    RuntimeArtifactValidator,
)
from framework_engineer.kernel_interface_decomposer.config import (  # noqa: E402
    RuntimeCaptureConfig,
)
from framework_engineer.kernel_interface_decomposer.sampling import (  # noqa: E402
    decomposition_signature,
)
from framework_engineer.kernel_interface_decomposer.tests.test_cli_capture_golden import (  # noqa: E402
    _failure_logs,
    _stable_capture_signature,
    _target_line,
)

def _compact_signature(invocation: dict[str, Any]) -> dict[str, Any]:
    """Golden-facing subset; the sampler still hashes the complete call path."""

    signature = decomposition_signature(invocation)
    return {
        "kernel_owner_captures": [
            {
                key: capture[key]
                for key in (
                    "archetype",
                    "common_interface",
                    "execution_interface",
                    "capture_depth",
                )
            }
            for capture in signature["kernel_owner_captures"]
        ],
        "unattributed_kernel_count": signature["unattributed_kernel_count"],
    }


def _raw_high_count(sqlite_path: Path) -> int:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM NVTX_EVENTS "
                "WHERE text LIKE 'KID:type=high%'"
            ).fetchone()[0]
        )
    finally:
        connection.close()


class TestConvergenceFixtures(unittest.TestCase):
    def test_configs_and_compact_golden_contract(self) -> None:
        test_root = Path(__file__).resolve().parent
        poc_path = REPO_ROOT / "framework_engineer/kernel_interface_decomposer/nsys_poc.py"
        target_line = _target_line(poc_path, "high_level")
        for backend, variants in (
            ("nsys_poc_repeat", "old,old"),
            ("nsys_poc_distinct", "old,softmax"),
        ):
            config = RuntimeCaptureConfig.load(test_root / f"configs/{backend}.json")
            self.assertEqual(config.backend_name, backend)
            self.assertEqual(config.target_line, target_line)
            self.assertEqual(config.selection["sampling"], "unique_decomposition")
            self.assertIn(f"--invocation-variants {variants}", config.test_command)

        golden = json.loads(
            (test_root / "golden/nsys_poc_convergence.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            golden["scenarios"],
            {"repeat": ["old"], "distinct": ["old", "softmax"]},
        )
        old = golden["variants"]["old"]
        softmax = golden["variants"]["softmax"]
        old_rows = old["kernel_owner_captures"]
        softmax_rows = softmax["kernel_owner_captures"]
        self.assertEqual(len(old_rows), len(softmax_rows))
        differences = [
            (before["execution_interface"], after["execution_interface"])
            for before, after in zip(old_rows, softmax_rows)
            if before != after
        ]
        self.assertEqual(differences, [("aten.mm.default", "aten._softmax.default")])


@unittest.skipUnless(
    RUN_GPU_E2E,
    "set KID_RUN_GPU_E2E=1 (or pass --gpu-e2e) for the Nsight/GPU regression",
)
class TestCaptureCliConvergence(unittest.TestCase):
    maxDiff = None

    def _capture(self, config_path: Path) -> tuple[dict[str, Any], Path]:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        poc_path = REPO_ROOT / "framework_engineer/kernel_interface_decomposer/nsys_poc.py"
        self.assertEqual(config["target"]["line"], _target_line(poc_path, "high_level"))
        output_dir = Path(config["output_dir"])
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(REPO_ROOT), env.get("PYTHONPATH")) if value
        )
        process = subprocess.Popen(
            [sys.executable, "-m", MODULE, "capture", str(config_path)],
            cwd=REPO_ROOT.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started = time.monotonic()
        while True:
            try:
                return_code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - started)
                print(
                    f"[GPU E2E heartbeat] backend={config['backend_name']} "
                    f"elapsed={elapsed}s",
                    flush=True,
                )
        stdout, stderr = process.communicate()
        if return_code != 0:
            self.fail(
                f"capture CLI exited with {return_code}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}\n"
                f"{_failure_logs(output_dir)}"
            )
        payload = json.loads(
            (output_dir / "runtime_capture.schema.json").read_text(encoding="utf-8")
        )
        validator = RuntimeArtifactValidator(output_dir)
        self.assertTrue(validator.validate(), msg="; ".join(validator.errors))
        self.assertEqual(_raw_high_count(output_dir / "trace/profile.sqlite"), 2)
        self.assertEqual(payload["diagnostics"]["observed_invocation_count"], 2)
        return payload, output_dir

    def test_repeated_and_distinct_invocations_converge(self) -> None:
        config_root = Path(__file__).resolve().parent / "configs"
        repeat = None
        if not DISTINCT_ONLY:
            repeat, _ = self._capture(config_root / "nsys_poc_repeat.json")
        distinct, _ = self._capture(config_root / "nsys_poc_distinct.json")

        if repeat is not None:
            repeat_diagnostics = repeat["diagnostics"]
            self.assertEqual(repeat_diagnostics["unique_decomposition_count"], 1)
            self.assertEqual(repeat_diagnostics["selected_invocation_count"], 1)
            self.assertEqual(
                len(
                    repeat_diagnostics["decomposition_groups"][0][
                        "member_call_ids"
                    ]
                ),
                2,
            )
        distinct_diagnostics = distinct["diagnostics"]
        self.assertEqual(distinct_diagnostics["unique_decomposition_count"], 2)
        self.assertEqual(distinct_diagnostics["selected_invocation_count"], 2)

        existing_golden = json.loads(
            (
                REPO_ROOT
                / "example_kernels/nsys_poc_kid_golden/cli_log/nsys_poc/"
                "runtime_capture.schema.json"
            ).read_text(encoding="utf-8")
        )
        if repeat is not None:
            repeat_stable = _stable_capture_signature(repeat)
            existing_stable = _stable_capture_signature(existing_golden)
            repeat_stable["backend_name"] = existing_stable["backend_name"]
            self.assertEqual(repeat_stable, existing_stable)

        actual_signatures = {
            "old": _compact_signature(distinct["invocations"][0]),
            "softmax": _compact_signature(distinct["invocations"][1]),
        }
        candidate = {
            "schema_version": "kid-runtime-convergence-golden/v1",
            "variants": actual_signatures,
            "scenarios": {
                "repeat": ["old"],
                "distinct": ["old", "softmax"],
            },
        }
        if RECORD_CANDIDATE:
            candidate_path = (
                Path("/mnt/infra_agent/kid_cli_convergence_output")
                / "convergence_golden_candidate.json"
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(
                json.dumps(candidate, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"recorded convergence golden candidate: {candidate_path}")
            return

        golden_path = Path(__file__).resolve().parent / "golden/nsys_poc_convergence.json"
        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        self.assertEqual(candidate, expected)


if __name__ == "__main__":
    unittest.main()
