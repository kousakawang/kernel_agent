"""Unit tests for Layer 3 extraction (source_location.extractor). CPU-only."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.source_location import cli as sl_cli
from framework_engineer.source_location.extractor import extract_workspace


def _schema_with(kernels: list[dict]) -> dict:
    return {"schema_version": "test/v1", "kernels": kernels}


def _kernel(kid: str, archetype: str, layers: dict) -> dict:
    return {
        "low_level_id": kid,
        "interface": f"iface::{kid}",
        "archetype": archetype,
        "source_locations": {"archetype": archetype, "needs_agent": False, "layers": layers},
    }


def _resolved(file: str, start: int, end: int) -> dict:
    return {"status": "resolved", "hits": [{"file": file, "line_start": start, "line_end": end}], "repo_hint": None}


class TestExtract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="l3_ut_"))
        # A real small source file to slice.
        self.src = self.tmp / "src_kernel.py"
        self.src.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")

    def _write_schema(self, schema: dict) -> Path:
        p = self.tmp / "decomposition_test.schema.json"
        p.write_text(json.dumps(schema, indent=2))
        return p

    def test_triton_like_resolved_ab_and_na_cd(self) -> None:
        # sglang_triton: a/b resolved, c/d not_applicable.
        schema = _schema_with([
            _kernel(
                "k0",
                "sglang_triton",
                {
                    "interface_definition": _resolved(str(self.src), 5, 8),
                    "kernel_impl": _resolved(str(self.src), 1, 3),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertFalse(report.stopped)

        kdir = self.tmp / "kernel_sources" / "k0"
        # a/b written with real content
        self.assertIn("line5", (kdir / "interface_definition.py").read_text())
        self.assertIn("line1", (kdir / "kernel_impl.py").read_text())
        # c/d empty + comment
        cc = (kdir / "py_cpp_binding.cc").read_text()
        self.assertIn("不适用", cc)
        self.assertTrue(cc.lstrip().startswith("//"))
        # read_hints has all four layers with right wording
        hints = (kdir / "read_hints.txt").read_text()
        self.assertIn("interface_definition.py: read lines 5-8", hints)
        self.assertIn("N/A", hints)
        # kernel_sources_dir backfilled
        written = json.loads(schema_path.read_text())
        self.assertEqual(
            Path(written["kernels"][0]["kernel_sources_dir"]).resolve(), kdir.resolve()
        )

    def test_missed_required_layer_hard_stops(self) -> None:
        schema = _schema_with([
            _kernel(
                "k1",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.src), 1, 2),
                    "kernel_impl": {"status": "missed", "hits": [], "repo_hint": "/some/repo"},
                    "py_cpp_binding": {"status": "missed", "hits": []},
                    "kernel_header": {"status": "missed", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        self.assertTrue(any(m["layer"] == "kernel_impl" for m in report.missing))
        # No kernel_sources produced on hard stop.
        self.assertFalse((self.tmp / "kernel_sources").exists())

    def test_allow_empty_emits_placeholder_for_missed(self) -> None:
        schema = _schema_with([
            _kernel(
                "k2",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.src), 1, 2),
                    "kernel_impl": {"status": "missed", "hits": []},
                    "py_cpp_binding": {"status": "missed", "hits": []},
                    "kernel_header": {"status": "missed", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp, allow_empty=True)
        self.assertFalse(report.stopped)
        kdir = self.tmp / "kernel_sources" / "k2"
        impl = (kdir / "kernel_impl.py").read_text()
        self.assertIn("未定位", impl)
        self.assertIn("MISSING", (kdir / "read_hints.txt").read_text())

    def test_fill_placeholder_counts_as_missing(self) -> None:
        # A layer marked resolved but still carrying <FILL> must be treated as missing.
        schema = _schema_with([
            _kernel(
                "k3",
                "sgl_kernel_builtin",
                {
                    "interface_definition": {
                        "status": "resolved",
                        "hits": [{"file": "<FILL: path>", "line_start": "<FILL>", "line_end": "<FILL>"}],
                    },
                    "kernel_impl": _resolved(str(self.src), 1, 2),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        self.assertTrue(any(m["layer"] == "interface_definition" for m in report.missing))

    def test_cli_extract_returns_2_on_hard_stop(self) -> None:
        schema = _schema_with([
            _kernel(
                "k4",
                "sgl_kernel_builtin",
                {
                    "interface_definition": {"status": "missed", "hits": []},
                    "kernel_impl": {"status": "missed", "hits": []},
                    "py_cpp_binding": {"status": "missed", "hits": []},
                    "kernel_header": {"status": "missed", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = sl_cli.main(["extract", "--schema", str(schema_path), "--workspace-out", str(self.tmp)])
        self.assertEqual(rc, 2)

    def test_kernel_impl_extension_follows_source(self) -> None:
        cu = self.tmp / "impl.cu"
        cu.write_text("\n".join(f"cu{i}" for i in range(1, 11)) + "\n")
        schema = _schema_with([
            _kernel(
                "k5",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.src), 1, 2),
                    "kernel_impl": _resolved(str(cu), 1, 5),
                    "py_cpp_binding": _resolved(str(self.src), 1, 1),
                    "kernel_header": _resolved(str(self.src), 1, 1),
                },
            )
        ])
        schema_path = self._write_schema(schema)
        extract_workspace(schema_path, self.tmp)
        kdir = self.tmp / "kernel_sources" / "k5"
        self.assertTrue((kdir / "kernel_impl.cu").exists())
        self.assertIn("cu1", (kdir / "kernel_impl.cu").read_text())

    def test_nonexistent_path_in_required_layer_hard_stops(self) -> None:
        # User filled a bogus path -> must be treated exactly like "not filled".
        schema = _schema_with([
            _kernel(
                "k6",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved("/no/such/file.py", 1, 2),
                    "kernel_impl": _resolved(str(self.src), 1, 2),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        bad = [m for m in report.missing if m["layer"] == "interface_definition"]
        self.assertTrue(bad)
        self.assertIn("file not found", bad[0]["reason"])
        self.assertFalse((self.tmp / "kernel_sources").exists())

    def test_nonexistent_path_in_optional_layer_becomes_placeholder(self) -> None:
        # Required layers valid; a non-required layer points nowhere -> placeholder
        # (with reason), NOT a silent success, NOT a hard stop.
        schema = _schema_with([
            _kernel(
                "k7",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.src), 1, 2),
                    "kernel_impl": _resolved(str(self.src), 1, 2),
                    "py_cpp_binding": _resolved("/no/such/binding.cc", 1, 5),
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertFalse(report.stopped)
        kdir = self.tmp / "kernel_sources" / "k7"
        # binding written as placeholder carrying the reason
        binding = (kdir / "py_cpp_binding.cc").read_text()
        self.assertIn("未定位", binding)
        hints = (kdir / "read_hints.txt").read_text()
        self.assertIn("MISSING", hints)
        self.assertIn("file not found", hints)

    def test_invalid_line_range_treated_as_missing(self) -> None:
        schema = _schema_with([
            _kernel(
                "k8",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.src), 10, 3),  # end < start
                    "kernel_impl": _resolved(str(self.src), 1, 2),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        self.assertTrue(any("invalid line range" in m.get("reason", "") for m in report.missing))


if __name__ == "__main__":
    unittest.main(verbosity=2)
