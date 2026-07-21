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


def _decomposed_invocation(
    call_id: str,
    stage: str,
    start: int,
    execution_interface: str,
    *,
    repeat: int = 1,
    provider_hint: str = "provider-a",
) -> dict[str, object]:
    captures = []
    for index in range(repeat):
        captures.append(
            {
                "capture_id": f"{call_id}-capture-{index}",
                "parent_capture_id": None,
                "archetype": "pytorch_dispatch",
                "common_interface": "TorchDispatchMode.__torch_dispatch__",
                "execution_interface": execution_interface,
                "provider_hint": provider_hint,
                "kernel_ids": [f"{call_id}-kernel-{index}"],
                "metrics": {"direct_gpu_kernel_sum_us": 10.0 + index},
                "python_stack": [
                    {
                        "file": "/repo/workload.py",
                        "definition_line": 10,
                        "function": "high",
                        "qualname": "high",
                        "call_site_to_next": {
                            "file": "/repo/workload.py",
                            "line": 11,
                        },
                    }
                ],
            }
        )
    return {
        "high_level": {"call_id": call_id, "stage": stage},
        "execution_captures": captures,
        "unattributed_kernel_ids": [],
        "_nvtx_start_ns": start,
    }


class TestConfigAndSampling(unittest.TestCase):
    def test_runtime_v3_config_derives_backend_first_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("def high():\n    pass\n", encoding="utf-8")
            path = root / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "kid-runtime-config/v3",
                        "backend_name": "demo",
                        "workdir": str(root),
                        "output_dir": str(root / "out"),
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
                            "sample_count_per_stage": 2,
                        },
                        "profiling": {"disable_cuda_graph": True},
                    }
                ),
                encoding="utf-8",
            )
            config = RuntimeCaptureConfig.load(path)
            self.assertIsNone(config.command)
            self.assertEqual(config.selection["sampling"], "unique_decomposition")
            self.assertEqual(config.profiling["min_capture_coverage"], 1.0)
            self.assertEqual(config.profiling["trace_retention"], "on_failure")
            self.assertEqual(config.cli_dir(), (root / "out/demo/cli_log").resolve())
            self.assertEqual(config.ref_dir(), (root / "out/demo/ref").resolve())
            self.assertEqual(
                config.final_output_dir(), (root / "out/demo/output").resolve()
            )

    def test_invalid_legacy_or_cuda_graph_config_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                RuntimeCaptureConfig.load(path)

    def test_v2_and_removed_stages_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("def high():\n    pass\n", encoding="utf-8")
            raw = {
                "schema_version": "kid-runtime-config/v2",
                "backend_name": "demo",
                "workdir": str(root),
                "output_dir": str(root / "out"),
                "target": {"file": str(target), "line": 1},
                "cmd": None,
                "test_cmd": "python target.py",
            }
            path = root / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "kid-runtime-config/v3"):
                RuntimeCaptureConfig.load(path)
            raw["schema_version"] = "kid-runtime-config/v3"
            raw["selection"] = {"stages": ["decode"]}
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "selection.stages was removed"):
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
                "sampling": "last_n",
                "sample_count_per_stage": 1,
            },
        )
        self.assertEqual(
            [item["high_level"]["call_id"] for item in selected], ["p1", "d2"]
        )
        self.assertEqual(diagnostics["discarded_call_ids"], ["p0", "d0", "d1"])

    def test_all_honors_global_skip_without_stage_filter(self) -> None:
        invocations = [
            _invocation("warm", "unknown", 10),
            _invocation("p", "prefill", 20),
            _invocation("d", "decode", 30),
        ]
        selected, diagnostics = select_invocations(
            invocations,
            {
                "skip_invocations": 1,
                "sampling": "all",
                "sample_count_per_stage": 1,
            },
        )
        self.assertEqual(
            [item["high_level"]["call_id"] for item in selected], ["p", "d"]
        )
        self.assertEqual(diagnostics["skipped_invocation_count"], 1)

    def test_single_is_last_invocation(self) -> None:
        selected, _ = select_invocations(
            [_invocation("1", "unknown", 10), _invocation("2", "unknown", 20)],
            {
                "sampling": "single",
                "sample_count_per_stage": 1,
                "skip_invocations": 0,
            },
        )
        self.assertEqual(selected[0]["high_level"]["call_id"], "2")

    def test_unique_decomposition_chooses_last_across_stages(self) -> None:
        first = _decomposed_invocation("p0", "prefill", 10, "aten.mm.default")
        last = _decomposed_invocation(
            "d0",
            "decode",
            20,
            "aten.mm.default",
            repeat=2,
            provider_hint="provider-b",
        )
        selected, diagnostics = select_invocations(
            [first, last],
            {
                "sampling": "unique_decomposition",
                "sample_count_per_stage": 1,
                "skip_invocations": 0,
            },
        )
        self.assertEqual(
            [item["high_level"]["call_id"] for item in selected], ["d0"]
        )
        self.assertEqual(diagnostics["unique_decomposition_count"], 1)
        self.assertEqual(
            diagnostics["decomposition_groups"],
            [
                {
                    "signature_hash": diagnostics["decomposition_groups"][0][
                        "signature_hash"
                    ],
                    "member_call_ids": ["p0", "d0"],
                    "observed_stages": ["prefill", "decode"],
                    "representative_call_id": "d0",
                }
            ],
        )
        self.assertEqual(diagnostics["discarded_call_ids"], ["p0"])

    def test_unique_decomposition_keeps_distinct_execution_interfaces(self) -> None:
        invocations = [
            _decomposed_invocation("old", "unknown", 10, "aten.mm.default"),
            _decomposed_invocation(
                "softmax", "unknown", 20, "aten._softmax.default"
            ),
        ]
        selected, diagnostics = select_invocations(
            invocations,
            {
                "sampling": "unique_decomposition",
                "sample_count_per_stage": 1,
                "skip_invocations": 0,
            },
        )
        self.assertEqual(
            [item["high_level"]["call_id"] for item in selected],
            ["old", "softmax"],
        )
        self.assertEqual(diagnostics["unique_decomposition_count"], 2)

    def test_unique_decomposition_distinguishes_unattributed_kernel_count(self) -> None:
        clean = _decomposed_invocation("clean", "unknown", 10, "aten.mm.default")
        unattributed = _decomposed_invocation(
            "unattributed", "unknown", 20, "aten.mm.default"
        )
        unattributed["unattributed_kernel_ids"] = ["kernel-u"]
        selected, _ = select_invocations(
            [clean, unattributed],
            {
                "sampling": "unique_decomposition",
                "sample_count_per_stage": 1,
                "skip_invocations": 0,
            },
        )
        self.assertEqual(len(selected), 2)

    def test_unique_decomposition_distinguishes_python_call_sites(self) -> None:
        first = _decomposed_invocation("first", "unknown", 10, "aten.mm.default")
        second = _decomposed_invocation("second", "unknown", 20, "aten.mm.default")
        second["execution_captures"][0]["python_stack"][0][
            "call_site_to_next"
        ]["line"] = 99
        selected, diagnostics = select_invocations(
            [first, second],
            {
                "sampling": "unique_decomposition",
                "sample_count_per_stage": 1,
                "skip_invocations": 0,
            },
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(diagnostics["unique_decomposition_count"], 2)


if __name__ == "__main__":
    unittest.main()
