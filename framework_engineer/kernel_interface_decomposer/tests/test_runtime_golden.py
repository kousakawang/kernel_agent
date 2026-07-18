from __future__ import annotations

import json
import unittest
from pathlib import Path

from framework_engineer.kernel_interface_decomposer.config import RuntimeCaptureConfig
from framework_engineer.kernel_interface_decomposer.trace_parser import RuntimeTraceParser


class TestRuntimeGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = (
            Path(__file__).resolve().parents[3]
            / "example_kernels"
            / "nsys_poc_kid_golden"
        )
        cls.config = RuntimeCaptureConfig.load(
            cls.root / "config/nsys_poc/runtime_capture_config.json"
        )
        cls.actual = RuntimeTraceParser(cls.config).parse(
            cls.root / "cli_log/nsys_poc/trace/profile.sqlite",
            cls.root / "cli_log/nsys_poc/capture_events",
        )
        cls.expected = json.loads(
            (cls.root / "cli_log/nsys_poc/runtime_capture.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_core_trace_facts_match_golden(self) -> None:
        invocation = self.actual["invocations"][0]
        self.assertEqual(self.actual["schema_version"], "kid-runtime-capture/v1")
        self.assertEqual(invocation["raw_capture_event_count"], 36)
        self.assertEqual(len(invocation["execution_captures"]), 13)
        self.assertEqual(invocation["capture_without_kernel_count"], 23)
        self.assertEqual(len(self.actual["kernels"]), 12)
        self.assertAlmostEqual(invocation["high_level"]["gpu_kernel_sum_us"], 48.096)
        self.assertEqual(invocation["coverage"], 1.0)

    def test_all_capture_archetypes_are_observed(self) -> None:
        observed = {
            item["archetype"]
            for item in self.actual["invocations"][0]["execution_captures"]
        }
        self.assertEqual(
            observed,
            {
                "pytorch_dispatch",
                "triton_launch",
                "cute_dsl_launch",
                "tilelang_launch",
                "tvm_ffi_call",
                "inductor_launch",
                "python_binding",
            },
        )

    def test_kernel_and_capture_metrics_match_golden(self) -> None:
        expected_kernels = {
            item["kernel_id"]: item for item in self.expected["kernels"]
        }
        for kernel in self.actual["kernels"]:
            expected = expected_kernels[kernel["kernel_id"]]
            self.assertEqual(kernel["name"], expected["name"])
            self.assertEqual(kernel["owner_capture_id"], expected["owner_capture_id"])
            self.assertAlmostEqual(kernel["duration_us"], expected["duration_us"])

    def test_runtime_output_contains_no_semantic_or_source_location(self) -> None:
        serialized = json.dumps(self.actual)
        for forbidden in (
            "semantic_target",
            "source_files",
            "binding_provider",
            "archetype_code",
            "workload_case",
        ):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
