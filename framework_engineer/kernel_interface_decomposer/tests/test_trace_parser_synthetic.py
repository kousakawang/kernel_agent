from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from framework_engineer.kernel_interface_decomposer.config import RuntimeCaptureConfig
from framework_engineer.kernel_interface_decomposer.trace_parser import RuntimeTraceParser


class TestSyntheticTraceParser(unittest.TestCase):
    def test_driver_correlation_delayed_multi_kernel_nested_and_unattributed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "workload.py"
            target.write_text("def high():\n    pass\n", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "kid-runtime-config/v2",
                        "backend_name": "synthetic",
                        "workdir": str(root),
                        "output_dir": str(root / "synthetic"),
                        "target": {
                            "file": str(target),
                            "line": 1,
                            "qualified_name": "high",
                        },
                        "cmd": None,
                        "test_cmd": "python workload.py",
                        "ready": None,
                        "stop": None,
                        "env": {},
                        "selection": {
                            "skip_invocations": 0,
                            "stages": ["decode"],
                            "sample_count_per_stage": 1,
                            "sampling": "unique_decomposition",
                            "aggregation": "single",
                        },
                        "profiling": {
                            "disable_cuda_graph": True,
                            "min_capture_coverage": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            sqlite_path = root / "profile.sqlite"
            connection = sqlite3.connect(sqlite_path)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE NVTX_EVENTS (
                    start INTEGER, end INTEGER, text TEXT,
                    globalTid INTEGER, globalPid INTEGER
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
                    start INTEGER, end INTEGER, correlationId INTEGER,
                    nameId INTEGER, globalTid INTEGER, globalPid INTEGER
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, correlationId INTEGER,
                    shortName INTEGER, globalPid INTEGER,
                    deviceId INTEGER, streamId INTEGER
                );
                """
            )
            connection.executemany(
                "INSERT INTO StringIds VALUES (?, ?)",
                [(1, "cuLaunchKernel"), (11, "kernel_a"), (12, "kernel_b"), (13, "kernel_u")],
            )
            connection.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
                [
                    (0, 70_000, "KID:type=high|call_id=1|interface=high|stage=decode", 77, 77),
                    (5_000, 40_000, "KID:type=execution|capture_id=o1|parent_call_id=1|archetype=pytorch_dispatch|interface=custom.default", 77, 77),
                    (10_000, 30_000, "KID:type=execution|capture_id=i1|parent_capture_id=o1|parent_call_id=1|archetype=triton_launch|interface=kernel", 77, 77),
                    (100_000, 170_000, "KID:type=high|call_id=2|interface=high|stage=decode", 77, 77),
                    (110_000, 140_000, "KID:type=execution|capture_id=o2|parent_call_id=2|archetype=pytorch_dispatch|interface=custom.default", 77, 77),
                    (115_000, 130_000, "KID:type=execution|capture_id=i2|parent_capture_id=o2|parent_call_id=2|archetype=triton_launch|interface=kernel", 77, 77),
                ],
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (15_000, 15_100, 11, 1, 77, 77),
                    (20_000, 20_100, 12, 1, 77, 77),
                    (50_000, 50_100, 13, 1, 77, 77),
                    (120_000, 120_100, 21, 1, 77, 77),
                    (125_000, 125_100, 22, 1, 77, 77),
                    (150_000, 150_100, 23, 1, 77, 77),
                ],
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (80_000, 110_000, 11, 11, 77, 0, 1),
                    (110_000, 150_000, 12, 12, 77, 0, 1),
                    (150_000, 170_000, 13, 13, 77, 0, 1),
                    (220_000, 250_000, 21, 11, 77, 0, 1),
                    (250_000, 290_000, 22, 12, 77, 0, 1),
                    (290_000, 310_000, 23, 13, 77, 0, 1),
                ],
            )
            connection.commit()
            connection.close()

            events = root / "events"
            events.mkdir()
            stack = [
                {
                    "file": str(target),
                    "definition_line": 1,
                    "function": "high",
                    "qualname": "high",
                    "call_site_to_next": {"file": str(target), "line": 2},
                }
            ]
            (events / "events_77.jsonl").write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        {
                            "event": "execution_capture",
                            "capture_id": "o1",
                            "parent_capture_id": None,
                            "parent_call_id": "1",
                            "archetype": "pytorch_dispatch",
                            "common_interface": "dispatch",
                            "execution_interface": "custom.default",
                            "python_stack": stack,
                            "pid": 77,
                        },
                        {
                            "event": "execution_capture",
                            "capture_id": "i1",
                            "parent_capture_id": "o1",
                            "parent_call_id": "1",
                            "archetype": "triton_launch",
                            "common_interface": "launcher",
                            "execution_interface": "kernel",
                            "python_stack": stack,
                            "pid": 77,
                        },
                        {
                            "event": "execution_capture",
                            "capture_id": "o2",
                            "parent_capture_id": None,
                            "parent_call_id": "2",
                            "archetype": "pytorch_dispatch",
                            "common_interface": "dispatch",
                            "execution_interface": "custom.default",
                            "python_stack": stack,
                            "pid": 77,
                        },
                        {
                            "event": "execution_capture",
                            "capture_id": "i2",
                            "parent_capture_id": "o2",
                            "parent_call_id": "2",
                            "archetype": "triton_launch",
                            "common_interface": "launcher",
                            "execution_interface": "kernel",
                            "python_stack": stack,
                            "pid": 77,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = RuntimeTraceParser(RuntimeCaptureConfig.load(config_path)).parse(
                sqlite_path, events
            )
            self.assertEqual(len(result["invocations"]), 1)
            invocation = result["invocations"][0]
            self.assertEqual(invocation["high_level"]["call_id"], "2")
            self.assertEqual(result["diagnostics"]["observed_invocation_count"], 2)
            self.assertEqual(result["diagnostics"]["unique_decomposition_count"], 1)
            self.assertEqual(
                result["diagnostics"]["decomposition_groups"][0]["member_call_ids"],
                ["1", "2"],
            )
            self.assertEqual(len(result["kernels"]), 3)
            captures = {item["capture_id"]: item for item in invocation["execution_captures"]}
            self.assertEqual(len(captures["i2"]["kernel_ids"]), 2)
            self.assertEqual(captures["o2"]["kernel_ids"], [])
            self.assertEqual(len(captures["o2"]["inclusive_kernel_ids"]), 2)
            self.assertEqual(len(invocation["unattributed_kernel_ids"]), 1)
            self.assertAlmostEqual(invocation["coverage"], 70.0 / 90.0)


if __name__ == "__main__":
    unittest.main()
