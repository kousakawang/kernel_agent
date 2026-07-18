from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.kernel_interface_decomposer.config import (
    ConfigError,
    RuntimeCaptureConfig,
)
from framework_engineer.kernel_interface_decomposer.sampling import select_invocations


def _invocation(call_id: str, stage: str, start: int) -> dict[str, object]:
    return {
        "high_level": {"call_id": call_id, "stage": stage},
        "_nvtx_start_ns": start,
    }


class TestConfigAndSampling(unittest.TestCase):
    def test_runtime_v2_config_accepts_direct_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("def high():\n    pass\n", encoding="utf-8")
            path = root / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "kid-runtime-config/v2",
                        "backend_name": "demo",
                        "workdir": str(root),
                        "output_dir": str(root / "out" / "demo"),
                        "target": {
                            "file": str(target),
                            "line": 1,
                            "qualified_name": "high",
                        },
                        "cmd": None,
                        "test_cmd": "python target.py",
                        "ready": None,
                        "stop": None,
                        "env": {},
                        "selection": {
                            "sampling": "last_n",
                            "sample_count_per_stage": 2,
                        },
                        "profiling": {"disable_cuda_graph": True},
                    }
                ),
                encoding="utf-8",
            )
            config = RuntimeCaptureConfig.load(path)
            self.assertIsNone(config.command)
            self.assertEqual(config.selection["sampling"], "last_n")
            self.assertEqual(config.profiling["min_capture_coverage"], 1.0)

    def test_invalid_legacy_or_cuda_graph_config_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                RuntimeCaptureConfig.load(path)

    def test_last_n_is_per_stage_and_restores_chronological_order(self) -> None:
        invocations = [
            _invocation("p0", "prefill", 10),
            _invocation("d0", "decode", 20),
            _invocation("d1", "decode", 30),
            _invocation("p1", "prefill", 40),
            _invocation("d2", "decode", 50),
        ]
        selected, diagnostics = select_invocations(
            invocations,
            {
                "skip_invocations": 0,
                "stages": ["prefill", "decode"],
                "sampling": "last_n",
                "sample_count_per_stage": 1,
            },
        )
        self.assertEqual(
            [item["high_level"]["call_id"] for item in selected], ["p1", "d2"]
        )
        self.assertEqual(diagnostics["discarded_call_ids"], ["p0", "d0", "d1"])

    def test_all_honors_global_skip_and_stage_filter(self) -> None:
        invocations = [
            _invocation("warm", "unknown", 10),
            _invocation("p", "prefill", 20),
            _invocation("d", "decode", 30),
        ]
        selected, diagnostics = select_invocations(
            invocations,
            {
                "skip_invocations": 1,
                "stages": ["decode"],
                "sampling": "all",
                "sample_count_per_stage": 1,
            },
        )
        self.assertEqual([item["high_level"]["call_id"] for item in selected], ["d"])
        self.assertEqual(diagnostics["skipped_invocation_count"], 1)

    def test_single_is_last_invocation(self) -> None:
        selected, _ = select_invocations(
            [_invocation("1", "unknown", 10), _invocation("2", "unknown", 20)],
            {
                "sampling": "single",
                "sample_count_per_stage": 1,
                "skip_invocations": 0,
                "stages": ["unknown"],
            },
        )
        self.assertEqual(selected[0]["high_level"]["call_id"], "2")


if __name__ == "__main__":
    unittest.main()
