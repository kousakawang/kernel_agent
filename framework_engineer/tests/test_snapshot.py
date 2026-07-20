from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from framework_engineer.snapshot.harness_builder import SnapshotHarnessBuilder
from framework_engineer.snapshot.hashing import value_hash
from framework_engineer.snapshot.recorder import SnapshotRecorder, make_forward_boundary_decorator
from framework_engineer.snapshot.selector import SnapshotSelector, write_shape_list_summary
from framework_engineer.snapshot.store import SnapshotStore
from framework_engineer.snapshot.tree import tree_meta


class SnapshotTests(unittest.TestCase):
    def _capture_primitive_calls(self, store: SnapshotStore, *, task_id: str = "task_pack") -> None:
        recorder = SnapshotRecorder(
            store,
            task_id=task_id,
            target={
                "qualified_name": "framework_engineer_missing_original_for_test.extend",
                "logical_name": "extend",
                "mode": "extend",
                "backend": "test",
            },
            signature="candidate(*args, **kwargs)",
            max_capture_groups=8,
            max_samples_per_group=4,
            max_samples_per_forward_per_group=2,
        )

        @recorder.decorate
        def target(*, values, state):
            state["total"] += sum(values)
            return {"out": [v + 1 for v in values]}

        @make_forward_boundary_decorator("toy.forward")
        def forward(values):
            target(values=values, state={"total": 0})
            target(values=values, state={"total": 0})
            target(values=values, state={"total": 0})

        forward([1, 2, 3])
        forward([1, 2, 3])

    @unittest.skipIf(torch is None, "torch is required for tensor metadata test")
    def test_tensor_meta_records_layout(self) -> None:
        tensor = torch.zeros(2, 3).t()
        meta = tree_meta({"x": tensor})["items"]["x"]["meta"]
        self.assertEqual(meta["shape"], [3, 2])
        self.assertEqual(meta["stride"], [1, 3])
        self.assertFalse(meta["is_contiguous"])
        self.assertEqual(meta["storage_offset"], 0)

    @unittest.skipIf(torch is None, "torch is required for bfloat16 hash test")
    def test_value_hash_supports_bfloat16(self) -> None:
        tensor = torch.arange(8, dtype=torch.float32).to(torch.bfloat16)
        digest = value_hash({"x": tensor})
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 24)

    def test_group_hit_count_and_sample_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "snapshots")
            self._capture_primitive_calls(store)
            index = store.read_raw_index()
            self.assertEqual(index["raw_group_count"], 1)
            self.assertEqual(index["total_hit_count"], 6)
            group = next(iter(index["groups"].values()))
            self.assertEqual(group["total_hit_count"], 6)
            self.assertEqual(group["forward_hit_count"], 2)
            self.assertEqual(group["sample_count"], 4)
            timing_summary = group["original_call_timing_summary"]
            self.assertEqual(timing_summary["baseline_kind"], "captured_original_call_timing_reference")
            self.assertEqual(timing_summary["count"], 6)
            self.assertTrue(timing_summary["dump_time_excluded"])
            self.assertGreaterEqual(timing_summary["mean_elapsed_us"], 0)
            for count in group["samples_per_forward"].values():
                self.assertLessEqual(count, 2)

    def test_select_and_generated_harness_passes_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_pack = Path(tmp) / "task_pack"
            (task_pack / "snapshots" / "raw").mkdir(parents=True)
            (task_pack / "snapshots" / "selected").mkdir(parents=True)
            store = SnapshotStore(task_pack / "snapshots")
            self._capture_primitive_calls(store, task_id=task_pack.name)
            manifest = SnapshotSelector(store).select(max_groups=1, max_samples_per_group=4)
            write_shape_list_summary(task_pack, manifest)
            SnapshotHarnessBuilder(task_pack).generate()
            original_manifest = json.loads((task_pack / "original_source" / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(original_manifest["source_available"])
            self.assertFalse(original_manifest["executable"])
            proc = subprocess.run(
                [sys.executable, "correctness_test.py", "--device", "cpu"],
                cwd=task_pack,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('"status": "PASS"', proc.stdout)
            bench = subprocess.run(
                [
                    sys.executable,
                    "benchmark.py",
                    "--device",
                    "cpu",
                    "--target",
                    "candidate",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                ],
                cwd=task_pack,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(bench.returncode, 0, bench.stderr)
            self.assertIn('"candidate"', bench.stdout)
            both = subprocess.run(
                [
                    sys.executable,
                    "benchmark.py",
                    "--device",
                    "cpu",
                    "--target",
                    "both",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                ],
                cwd=task_pack,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(both.returncode, 0, both.stderr)
            self.assertIn('"reference": {"available": false', both.stdout)

    def test_generated_candidate_falls_back_when_source_relative_import_cannot_be_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_pack = tmp_path / "task_pack"
            (task_pack / "snapshots" / "raw").mkdir(parents=True)
            (task_pack / "snapshots" / "selected").mkdir(parents=True)
            store = SnapshotStore(task_pack / "snapshots")
            self._capture_primitive_calls(store, task_id=task_pack.name)
            manifest = SnapshotSelector(store).select(max_groups=1, max_samples_per_group=4)
            write_shape_list_summary(task_pack, manifest)

            package_dir = tmp_path / "third_party" / "external_ns"
            package_dir.mkdir(parents=True)
            (package_dir / "constants.py").write_text("OFFSET = 7\n", encoding="utf-8")
            source_file = package_dir / "ops.py"
            source_file.write_text(
                "from .constants import OFFSET\n\n"
                "def external_target(value):\n"
                "    return value + OFFSET\n",
                encoding="utf-8",
            )
            docs = task_pack / "docs"
            docs.mkdir(parents=True)
            (docs / "snapshot_capture_report.json").write_text(
                json.dumps(
                    {
                        "target_interface": {
                            "file": str(source_file),
                            "function_name": "external_target",
                            "qualified_name": "ops.external_target",
                            "line": 3,
                            "end_line": 4,
                            "class_path": [],
                            "module_name": "ops",
                        }
                    }
                ),
                encoding="utf-8",
            )

            SnapshotHarnessBuilder(task_pack).generate()
            original_manifest = json.loads(
                (task_pack / "original_source" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(original_manifest["executable"], original_manifest)

            proc = subprocess.run(
                [sys.executable, "correctness_test.py", "--device", "cpu"],
                cwd=task_pack,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('"status": "PASS"', proc.stdout)


if __name__ == "__main__":
    unittest.main()
