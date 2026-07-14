"""Unit tests for Layer 3 extraction (source_location.extractor). CPU-only.

New schema shape (locate standard §1/§2): each hit is ``{file, def_line}`` (no
line_start/line_end — the end is computed here by range-completion).
``interface_definition``/``py_cpp_binding`` are single-file; ``kernel_impl``/
``kernel_header`` are directory layers whose hits go into a ``<layer>/`` subdir.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.source_location import cli as sl_cli
from framework_engineer.source_location.extractor import (
    _end_line_c_family,
    _end_line_python,
    extract_workspace,
)


def _schema_with(kernels: list[dict]) -> dict:
    return {"schema_version": "test/v1", "kernels": kernels}


def _kernel(kid: str, archetype: str, layers: dict) -> dict:
    return {
        "low_level_id": kid,
        "interface": f"iface::{kid}",
        "archetype": archetype,
        "source_locations": {"archetype": archetype, "needs_agent": False, "layers": layers},
    }


def _resolved(file: str, def_line: int) -> dict:
    """A resolved single-hit layer in the new {file, def_line} shape."""
    return {"status": "resolved", "hits": [{"file": file, "def_line": def_line}], "repo_hint": None}


def _resolved_multi(hits: list[tuple[str, int]]) -> dict:
    """A resolved directory layer with multiple {file, def_line} hits."""
    return {
        "status": "resolved",
        "hits": [{"file": f, "def_line": d} for f, d in hits],
        "repo_hint": None,
    }


class TestExtract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="l3_ut_"))
        # A real python file with a couple of functions to slice via AST.
        self.py = self.tmp / "src_kernel.py"
        self.py.write_text(
            "import torch\n"                       # 1
            "\n"                                    # 2
            "def alpha(x):\n"                       # 3
            "    y = x + 1\n"                        # 4
            "    return y\n"                         # 5
            "\n"                                     # 6
            "def beta(x):\n"                         # 7
            "    return x * 2\n"                      # 8
        )
        # A real C-family file with a function + a header-style prototype.
        self.cu = self.tmp / "impl.cu"
        self.cu.write_text(
            "#include <foo>\n"                       # 1
            "void launch(int d) {\n"                 # 2
            "  int a = d;\n"                          # 3
            "  kern<<<1,1>>>(a);\n"                   # 4
            "}\n"                                     # 5
            "void proto(int x);\n"                    # 6
        )

    def _write_schema(self, schema: dict) -> Path:
        p = self.tmp / "decomposition_test.schema.json"
        p.write_text(json.dumps(schema, indent=2))
        return p

    # --- range-completion helpers (py AST / c-family braces) -----------------

    def test_range_completion_python_ast(self) -> None:
        lines = self.py.read_text().splitlines(keepends=True)
        self.assertEqual(_end_line_python(lines, 3), 5)  # def alpha -> lines 3-5
        self.assertEqual(_end_line_python(lines, 7), 8)  # def beta  -> lines 7-8

    def test_range_completion_c_family_braces_and_proto(self) -> None:
        lines = self.cu.read_text().splitlines(keepends=True)
        self.assertEqual(_end_line_c_family(lines, 2), 5)  # void launch { .. }
        self.assertEqual(_end_line_c_family(lines, 6), 6)  # bare prototype ends at ;

    # --- single-file layers --------------------------------------------------

    def test_triton_like_resolved_ab_and_na_cd(self) -> None:
        # sglang_triton: a/b resolved, c/d not_applicable.
        schema = _schema_with([
            _kernel(
                "k0",
                "sglang_triton",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    "kernel_impl": _resolved(str(self.py), 7),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertFalse(report.stopped)

        kdir = self.tmp / "kernel_sources" / "k0"
        # interface_definition single-file: WHOLE file copied verbatim (no
        # truncation — Layer 3 copies, the focus range lives in read_hints).
        iface = (kdir / "interface_definition.py").read_text()
        self.assertEqual(iface, self.py.read_text())
        self.assertIn("def alpha", iface)
        self.assertIn("def beta", iface)  # whole file, not just the alpha span
        # kernel_impl is a DIRECTORY layer -> kernel_impl/<n>_<basename>
        impl = (kdir / "kernel_impl" / "1_src_kernel.py").read_text()
        self.assertEqual(impl, self.py.read_text())  # whole file
        # c/d not_applicable -> placeholder in their subdir
        cc = (kdir / "py_cpp_binding.cc").read_text()
        self.assertIn("不适用", cc)
        self.assertTrue(cc.lstrip().startswith("//"))
        self.assertIn("不适用", (kdir / "kernel_header" / "kernel_header.h").read_text())
        # read_hints: focus range still points at the definition span (3-5 / 7-8)
        hints = (kdir / "read_hints.txt").read_text()
        self.assertIn("interface_definition.py: read lines 3-5", hints)
        self.assertIn("kernel_impl/1_src_kernel.py: read lines 7-8", hints)
        self.assertIn("N/A", hints)
        # kernel_sources_dir backfilled
        written = json.loads(schema_path.read_text())
        self.assertEqual(
            Path(written["kernels"][0]["kernel_sources_dir"]).resolve(), kdir.resolve()
        )

    # --- directory layers: multiple ordered hits -----------------------------

    def test_kernel_impl_directory_multi_hit_ordered(self) -> None:
        schema = _schema_with([
            _kernel(
                "k1",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    # call chain: launcher (.cu) -> real kernel (.py beta)
                    "kernel_impl": _resolved_multi([(str(self.cu), 2), (str(self.py), 7)]),
                    "py_cpp_binding": _resolved(str(self.cu), 2),
                    "kernel_header": _resolved_multi([(str(self.cu), 6)]),
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertFalse(report.stopped)
        kdir = self.tmp / "kernel_sources" / "k1"
        # kernel_impl subdir, numbered by call order
        self.assertTrue((kdir / "kernel_impl" / "1_impl.cu").exists())
        self.assertTrue((kdir / "kernel_impl" / "2_src_kernel.py").exists())
        self.assertIn("void launch", (kdir / "kernel_impl" / "1_impl.cu").read_text())
        # kernel_header subdir, not numbered
        self.assertTrue((kdir / "kernel_header" / "impl.cu").exists())
        # binding single-file at top; keeps its source (.cu) suffix
        self.assertTrue((kdir / "py_cpp_binding.cu").exists())
        hints = (kdir / "read_hints.txt").read_text()
        self.assertIn("kernel_impl/1_impl.cu: read lines 2-5", hints)
        self.assertIn("kernel_impl/2_src_kernel.py: read lines 7-8", hints)

    def test_copies_whole_file_not_truncated_span(self) -> None:
        # def is a small span mid-file; the extracted file must be the WHOLE file
        # (no truncation), while read_hints records only the definition span.
        big = self.tmp / "big.py"
        head = "".join(f"pre_{i} = {i}\n" for i in range(1, 11))   # lines 1-10
        fn = "def target(x):\n    return x + 1\n"                   # lines 11-12
        tail = "".join(f"post_{i} = {i}\n" for i in range(1, 11))  # lines 13-22
        big.write_text(head + fn + tail)
        schema = _schema_with([
            _kernel("k13", "sglang_triton", {
                "interface_definition": _resolved(str(big), 11),
                "kernel_impl": _resolved(str(big), 11),
                "py_cpp_binding": {"status": "not_applicable", "hits": []},
                "kernel_header": {"status": "not_applicable", "hits": []},
            })
        ])
        schema_path = self._write_schema(schema)
        extract_workspace(schema_path, self.tmp)
        kdir = self.tmp / "kernel_sources" / "k13"
        iface = (kdir / "interface_definition.py").read_text()
        self.assertEqual(iface, big.read_text())     # whole file, byte-for-byte
        self.assertIn("pre_1 = 1", iface)            # content before the def kept
        self.assertIn("post_10 = 10", iface)         # content after the def kept
        # but the focus hint narrows to the def span (11-12)
        self.assertIn("interface_definition.py: read lines 11-12", (kdir / "read_hints.txt").read_text())

    def test_single_file_layer_multi_hit_is_ambiguous(self) -> None:
        # interface_definition is single-file: >1 hit must be treated as missing.
        schema = _schema_with([
            _kernel(
                "k2",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved_multi([(str(self.py), 3), (str(self.py), 7)]),
                    "kernel_impl": _resolved(str(self.cu), 2),
                    "py_cpp_binding": _resolved(str(self.cu), 2),
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        bad = [m for m in report.missing if m["layer"] == "interface_definition"]
        self.assertTrue(bad)
        self.assertIn("ambiguous", bad[0]["reason"])

    # --- hard-stop gate ------------------------------------------------------

    def test_missed_required_layer_hard_stops(self) -> None:
        schema = _schema_with([
            _kernel(
                "k3",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
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
        self.assertFalse((self.tmp / "kernel_sources").exists())

    def test_allow_empty_emits_placeholder_for_missed(self) -> None:
        schema = _schema_with([
            _kernel(
                "k4",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    "kernel_impl": {"status": "missed", "hits": []},
                    "py_cpp_binding": {"status": "missed", "hits": []},
                    "kernel_header": {"status": "missed", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp, allow_empty=True)
        self.assertFalse(report.stopped)
        kdir = self.tmp / "kernel_sources" / "k4"
        # kernel_impl directory placeholder
        impl = (kdir / "kernel_impl" / "kernel_impl.py").read_text()
        self.assertIn("未定位", impl)
        self.assertIn("MISSING", (kdir / "read_hints.txt").read_text())

    def test_fill_placeholder_counts_as_missing(self) -> None:
        schema = _schema_with([
            _kernel(
                "k5",
                "sgl_kernel_builtin",
                {
                    "interface_definition": {
                        "status": "resolved",
                        "hits": [{"file": "<FILL: path>", "def_line": "<FILL>"}],
                    },
                    "kernel_impl": _resolved(str(self.cu), 2),
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
                "k6",
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

    # --- filesystem validity: wrong paths / bad def_line ---------------------

    def test_nonexistent_path_in_required_layer_hard_stops(self) -> None:
        schema = _schema_with([
            _kernel(
                "k7",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved("/no/such/file.py", 1),
                    "kernel_impl": _resolved(str(self.cu), 2),
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
        schema = _schema_with([
            _kernel(
                "k8",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    "kernel_impl": _resolved(str(self.cu), 2),
                    "py_cpp_binding": _resolved("/no/such/binding.cc", 1),
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertFalse(report.stopped)
        kdir = self.tmp / "kernel_sources" / "k8"
        binding = (kdir / "py_cpp_binding.cc").read_text()
        self.assertIn("未定位", binding)
        hints = (kdir / "read_hints.txt").read_text()
        self.assertIn("MISSING", hints)
        self.assertIn("file not found", hints)

    def test_out_of_range_def_line_treated_as_missing(self) -> None:
        # def_line past EOF -> treated exactly like "not filled" (required -> stop).
        schema = _schema_with([
            _kernel(
                "k9",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 999),
                    "kernel_impl": _resolved(str(self.cu), 2),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        self.assertTrue(any("out of range" in m.get("reason", "") for m in report.missing))

    def test_directory_layer_bad_hit_reports_index(self) -> None:
        # 2nd hit of kernel_impl points nowhere -> reason names hit[1].
        schema = _schema_with([
            _kernel(
                "k10",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    "kernel_impl": _resolved_multi([(str(self.cu), 2), ("/no/such/k.cu", 1)]),
                    "py_cpp_binding": {"status": "not_applicable", "hits": []},
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema)
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        bad = [m for m in report.missing if m["layer"] == "kernel_impl"]
        self.assertTrue(bad)
        self.assertIn("hit[1]", bad[0]["reason"])

    # --- re-run cleanliness --------------------------------------------------

    def test_rerun_wipes_stale_files(self) -> None:
        # First run: kernel_impl resolves to the .py (directory layer file).
        schema_v1 = _schema_with([
            _kernel(
                "k11",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    "kernel_impl": _resolved(str(self.py), 7),
                    "py_cpp_binding": _resolved(str(self.cu), 2),
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path = self._write_schema(schema_v1)
        extract_workspace(schema_path, self.tmp)
        kdir = self.tmp / "kernel_sources" / "k11"
        self.assertTrue((kdir / "kernel_impl" / "1_src_kernel.py").exists())
        stray = kdir / "kernel_impl" / "OLD.txt"
        stray.write_text("stale")

        # Second run: kernel_impl now the .cu.
        schema_v2 = _schema_with([
            _kernel(
                "k11",
                "sgl_kernel_builtin",
                {
                    "interface_definition": _resolved(str(self.py), 3),
                    "kernel_impl": _resolved(str(self.cu), 2),
                    "py_cpp_binding": _resolved(str(self.cu), 2),
                    "kernel_header": {"status": "not_applicable", "hits": []},
                },
            )
        ])
        schema_path.write_text(json.dumps(schema_v2, indent=2))
        extract_workspace(schema_path, self.tmp)
        self.assertTrue((kdir / "kernel_impl" / "1_impl.cu").exists())
        self.assertFalse((kdir / "kernel_impl" / "1_src_kernel.py").exists())
        self.assertFalse(stray.exists())

    def test_rerun_removes_orphaned_kernel_dir(self) -> None:
        two = _schema_with([
            _kernel("keep", "sglang_triton", {
                "interface_definition": _resolved(str(self.py), 3),
                "kernel_impl": _resolved(str(self.py), 7),
                "py_cpp_binding": {"status": "not_applicable", "hits": []},
                "kernel_header": {"status": "not_applicable", "hits": []},
            }),
            _kernel("drop_me", "sglang_triton", {
                "interface_definition": _resolved(str(self.py), 3),
                "kernel_impl": _resolved(str(self.py), 7),
                "py_cpp_binding": {"status": "not_applicable", "hits": []},
                "kernel_header": {"status": "not_applicable", "hits": []},
            }),
        ])
        schema_path = self._write_schema(two)
        extract_workspace(schema_path, self.tmp)
        self.assertTrue((self.tmp / "kernel_sources" / "drop_me").exists())

        one = _schema_with([two["kernels"][0]])
        schema_path.write_text(json.dumps(one, indent=2))
        extract_workspace(schema_path, self.tmp)
        self.assertTrue((self.tmp / "kernel_sources" / "keep").exists())
        self.assertFalse((self.tmp / "kernel_sources" / "drop_me").exists())

    def test_hard_stop_rerun_preserves_previous_output(self) -> None:
        good = _schema_with([
            _kernel("k12", "sglang_triton", {
                "interface_definition": _resolved(str(self.py), 3),
                "kernel_impl": _resolved(str(self.py), 7),
                "py_cpp_binding": {"status": "not_applicable", "hits": []},
                "kernel_header": {"status": "not_applicable", "hits": []},
            }),
        ])
        schema_path = self._write_schema(good)
        extract_workspace(schema_path, self.tmp)
        kdir = self.tmp / "kernel_sources" / "k12"
        self.assertTrue((kdir / "kernel_impl" / "1_src_kernel.py").exists())

        bad = _schema_with([
            _kernel("k12", "sgl_kernel_builtin", {
                "interface_definition": _resolved(str(self.py), 3),
                "kernel_impl": {"status": "missed", "hits": []},
                "py_cpp_binding": {"status": "missed", "hits": []},
                "kernel_header": {"status": "missed", "hits": []},
            }),
        ])
        schema_path.write_text(json.dumps(bad, indent=2))
        report = extract_workspace(schema_path, self.tmp)
        self.assertTrue(report.stopped)
        # Previous good output survives (wipe happens only past the gate).
        self.assertTrue((kdir / "kernel_impl" / "1_src_kernel.py").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
