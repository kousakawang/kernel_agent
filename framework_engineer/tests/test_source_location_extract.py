"""CPU-only tests for Agent-result extraction and range completion."""

from __future__ import annotations

import contextlib
import copy
import filecmp
import io
import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.source_location import cli as sl_cli
from framework_engineer.source_location.contracts import (
    ContractError,
    kid_projection,
    validate_agent_schema,
)
from framework_engineer.source_location.extractor import (
    ExtractError,
    _end_line_c_family,
    _end_line_python,
    extract_workspace,
)


def _layer(
    status: str,
    hits: list[tuple[str, int]] | None = None,
    repo_hint: str | None = None,
) -> dict:
    return {
        "status": status,
        "hits": [
            {"file": file, "def_line": def_line}
            for file, def_line in (hits or [])
        ],
        "repo_hint": repo_hint,
    }


def _kernel(kid: str, layers: dict, *, rank: int = 1) -> dict:
    return {
        "rank": rank,
        "low_level_id": kid,
        "kernel": {"raw_name": f"raw::{kid}", "normalized_name": kid},
        "interface": f"test.api.{kid}",
        "archetype": "python_binding",
        "provider": "test",
        "metrics": {"duration_us": None, "share_in_invocation": None},
        "measurement": {
            "metric": "gpu_kernel_duration_us",
            "aggregation": "sum",
            "sample_count": 0,
        },
        "runtime_event": {
            "call_site": {"file": "/tmp/caller.py", "line": 1},
            "attribution": {"method": "test", "confidence": "high"},
        },
        "source_locations": {"layers": layers},
    }


def _schema(kernels: list[dict]) -> dict:
    return {
        "schema_version": "kernel-interface-decomposition/v2",
        "backend_name": "test",
        "target": {"interface": "high", "file": "/tmp/high.py", "line": 1},
        "coverage_report": {
            "per_invocation": [],
            "min_coverage": None,
            "semantic_target_count": len(kernels),
            "gpu_kernel_count": None,
            "uncaptured_hint": "unit test",
        },
        "kernels": kernels,
    }


class ExtractFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="source_extract_v2_")
        self.tmp = Path(self.tempdir.name)
        self.py = self.tmp / "src_kernel.py"
        self.py.write_text(
            "import torch\n"  # 1
            "\n"  # 2
            "def alpha(x):\n"  # 3
            "    y = x + 1\n"  # 4
            "    return y\n"  # 5
            "\n"  # 6
            "def beta(x):\n"  # 7
            "    return x * 2\n"  # 8
        )
        self.cu = self.tmp / "impl.cu"
        self.cu.write_text(
            "#include <foo>\n"  # 1
            "void launch(int d) {\n"  # 2
            "  int a = d;\n"  # 3
            "  kern<<<1,1>>>(a);\n"  # 4
            "}\n"  # 5
            "void proto(int x);\n"  # 6
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _standard_layers(self) -> dict:
        return {
            "interface_definition": _layer("resolved", [(str(self.py), 3)]),
            "kernel_impl": _layer("resolved", [(str(self.cu), 2)]),
            "py_cpp_binding": _layer("not_applicable"),
            "kernel_header": _layer("not_applicable"),
        }

    def _write_schema(self, payload: dict, *, name: str = "schema.json") -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload, indent=2))
        return path


class TestRangeCompletion(ExtractFixture):
    def test_python_ast_ranges(self) -> None:
        lines = self.py.read_text().splitlines(keepends=True)
        self.assertEqual(_end_line_python(lines, 3), 5)
        self.assertEqual(_end_line_python(lines, 7), 8)

    def test_c_family_braces_prototype_comments_and_strings(self) -> None:
        lines = self.cu.read_text().splitlines(keepends=True)
        self.assertEqual(_end_line_c_family(lines, 2), 5)
        self.assertEqual(_end_line_c_family(lines, 6), 6)

        tricky = self.tmp / "tricky.cu"
        tricky.write_text(
            "void f(int d) {\n"
            "  // if (d) {\n"
            "  const char* s = \"}{ \";\n"
            "  char c = '}';\n"
            "  /* } } nested\n"
            "     still } comment */\n"
            "  const char* r = R\"raw()}{)raw\";\n"
            "  int x = 1'000;\n"
            "}\n"
            "void after();\n"
        )
        tricky_lines = tricky.read_text().splitlines(keepends=True)
        self.assertEqual(_end_line_c_family(tricky_lines, 1), 9)
        self.assertEqual(_end_line_c_family(tricky_lines, 10), 10)


class TestExtraction(ExtractFixture):
    def test_resolved_multi_hit_copies_whole_files_and_writes_ranges(self) -> None:
        layers = self._standard_layers()
        layers["kernel_impl"] = _layer(
            "resolved", [(str(self.cu), 2), (str(self.py), 7)]
        )
        layers["py_cpp_binding"] = _layer(
            "resolved", [(str(self.py), 7), (str(self.cu), 2)]
        )
        layers["kernel_header"] = _layer("resolved", [(str(self.cu), 6)])
        schema_path = self._write_schema(_schema([_kernel("k1", layers)]))

        report = extract_workspace(schema_path, self.tmp)
        self.assertEqual(report.summary()["written"], 4)
        output = self.tmp / "kernel_sources" / "k1"
        self.assertEqual(
            (output / "interface_definition.py").read_bytes(), self.py.read_bytes()
        )
        self.assertEqual(
            (output / "kernel_impl" / "1_impl.cu").read_bytes(),
            self.cu.read_bytes(),
        )
        self.assertTrue((output / "kernel_impl" / "2_src_kernel.py").is_file())
        self.assertTrue((output / "py_cpp_binding" / "1_src_kernel.py").is_file())
        self.assertTrue((output / "py_cpp_binding" / "2_impl.cu").is_file())
        self.assertTrue((output / "kernel_header" / "impl.cu").is_file())
        hints = (output / "read_hints.txt").read_text()
        self.assertIn("interface_definition.py: read lines 3-5", hints)
        self.assertIn("kernel_impl/1_impl.cu: read lines 2-5", hints)
        self.assertIn("kernel_impl/2_src_kernel.py: read lines 7-8", hints)
        written = json.loads(schema_path.read_text())
        self.assertEqual(
            written["kernels"][0]["kernel_sources_dir"], str(output.resolve())
        )
        self.assertNotIn("end_line", schema_path.read_text())

    def test_best_effort_hits_are_copied_and_tagged(self) -> None:
        layers = self._standard_layers()
        layers["kernel_impl"] = _layer("best_effort", [(str(self.cu), 2)])
        schema_path = self._write_schema(_schema([_kernel("best", layers)]))
        extract_workspace(schema_path, self.tmp)
        output = self.tmp / "kernel_sources" / "best"
        self.assertTrue((output / "kernel_impl" / "1_impl.cu").is_file())
        self.assertIn("[best_effort]", (output / "read_hints.txt").read_text())

    def test_missed_and_not_applicable_are_normal_placeholders(self) -> None:
        layers = {
            "interface_definition": _layer("missed"),
            "kernel_impl": _layer("missed"),
            "py_cpp_binding": _layer("not_applicable"),
            "kernel_header": _layer("not_applicable"),
        }
        schema_path = self._write_schema(_schema([_kernel("missing", layers)]))
        report = extract_workspace(schema_path, self.tmp)
        self.assertEqual(report.summary()["placeholders"], 4)
        output = self.tmp / "kernel_sources" / "missing"
        self.assertIn("未定位", (output / "interface_definition.py").read_text())
        self.assertIn(
            "未定位", (output / "kernel_impl" / "kernel_impl.py").read_text()
        )
        self.assertIn(
            "不适用",
            (output / "py_cpp_binding" / "py_cpp_binding.cc").read_text(),
        )
        hints = (output / "read_hints.txt").read_text()
        self.assertIn("MISSING (status=missed)", hints)
        self.assertIn("N/A (not applicable)", hints)
        self.assertNotIn("archetype", hints)

    def test_invalid_hit_preserves_previous_output(self) -> None:
        good = _schema([_kernel("stable", self._standard_layers())])
        schema_path = self._write_schema(good)
        extract_workspace(schema_path, self.tmp)
        output_file = (
            self.tmp / "kernel_sources" / "stable" / "kernel_impl" / "1_impl.cu"
        )
        self.assertTrue(output_file.is_file())

        bad = copy.deepcopy(good)
        bad["kernels"][0]["source_locations"]["layers"]["kernel_impl"] = _layer(
            "resolved", [("/no/such/kernel.cu", 1)]
        )
        schema_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ExtractError, "file not found"):
            extract_workspace(schema_path, self.tmp)
        self.assertTrue(output_file.is_file())

    def test_out_of_range_line_is_rejected(self) -> None:
        layers = self._standard_layers()
        layers["interface_definition"] = _layer("resolved", [(str(self.py), 999)])
        schema_path = self._write_schema(_schema([_kernel("bad_line", layers)]))
        with self.assertRaisesRegex(ExtractError, "out of range"):
            extract_workspace(schema_path, self.tmp)
        self.assertFalse((self.tmp / "kernel_sources").exists())

    def test_rerun_wipes_stale_and_orphaned_files(self) -> None:
        first = _schema(
            [
                _kernel("keep", self._standard_layers(), rank=1),
                _kernel("drop", self._standard_layers(), rank=2),
            ]
        )
        schema_path = self._write_schema(first)
        extract_workspace(schema_path, self.tmp)
        stray = self.tmp / "kernel_sources" / "keep" / "OLD.txt"
        stray.write_text("stale")
        self.assertTrue((self.tmp / "kernel_sources" / "drop").is_dir())

        second = _schema([_kernel("keep", self._standard_layers())])
        schema_path.write_text(json.dumps(second))
        extract_workspace(schema_path, self.tmp)
        self.assertFalse(stray.exists())
        self.assertFalse((self.tmp / "kernel_sources" / "drop").exists())


class TestContractAndCli(ExtractFixture):
    def test_final_contract_rejects_old_fields_statuses_and_candidates(self) -> None:
        valid = _schema([_kernel("strict", self._standard_layers())])
        cases: list[dict] = []

        old_source = copy.deepcopy(valid)
        old_source["kernels"][0]["source_locations"]["source"] = "manual"
        cases.append(old_source)

        old_status = copy.deepcopy(valid)
        old_status["kernels"][0]["source_locations"]["layers"][
            "kernel_impl"
        ] = _layer("not_found")
        cases.append(old_status)

        with_candidates = copy.deepcopy(valid)
        with_candidates["kernels"][0]["locate_candidates"] = {}
        cases.append(with_candidates)

        missing_layer = copy.deepcopy(valid)
        del missing_layer["kernels"][0]["source_locations"]["layers"][
            "kernel_header"
        ]
        cases.append(missing_layer)

        impossible_na = copy.deepcopy(valid)
        impossible_na["kernels"][0]["source_locations"]["layers"][
            "kernel_impl"
        ] = _layer("not_applicable")
        cases.append(impossible_na)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ContractError):
                    validate_agent_schema(payload)

    def test_cli_success_and_invalid_path_return_codes(self) -> None:
        valid_path = self._write_schema(
            _schema([_kernel("ok", self._standard_layers())]), name="valid.json"
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(
                sl_cli.main(
                    [
                        "extract",
                        "--schema",
                        str(valid_path),
                        "--workspace-out",
                        str(self.tmp / "valid_workspace"),
                    ]
                ),
                0,
            )

        bad_layers = self._standard_layers()
        bad_layers["kernel_impl"] = _layer(
            "resolved", [("/no/such/file.cu", 1)]
        )
        bad_path = self._write_schema(
            _schema([_kernel("bad", bad_layers)]), name="bad.json"
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(
                sl_cli.main(
                    [
                        "extract",
                        "--schema",
                        str(bad_path),
                        "--workspace-out",
                        str(self.tmp / "bad_workspace"),
                    ]
                ),
                2,
            )


class TestRealGolden(unittest.TestCase):
    def test_agent_and_extract_goldens(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        workspace = (
            repo
            / "example_kernels"
            / "source_locate_golden"
            / "workspaces"
            / "all_backends"
        )
        kid_path = (
            repo
            / "example_kernels"
            / "source_locate_golden"
            / "input"
            / "all_backends"
            / "decomposition.kid.schema.json"
        )
        agent_path = workspace / "agent" / "located.schema.json"
        extract_path = workspace / "extract" / "decomposition.extracted.schema.json"
        golden_tree = workspace / "extract" / "kernel_sources"
        kid = json.loads(kid_path.read_text())
        agent = json.loads(agent_path.read_text())
        validate_agent_schema(agent)
        self.assertEqual(kid_projection(agent), kid)

        with tempfile.TemporaryDirectory(prefix="source_extract_golden_") as tmp:
            temp_workspace = Path(tmp)
            schema_path = temp_workspace / "located.json"
            schema_path.write_text(json.dumps(agent, indent=2))
            extract_workspace(schema_path, temp_workspace)

            actual_schema = json.loads(schema_path.read_text())
            expected_schema = json.loads(extract_path.read_text())
            expected_dirs = {
                entry["low_level_id"]: entry["kernel_sources_dir"]
                for entry in expected_schema["kernels"]
            }
            for entry in actual_schema["kernels"]:
                entry["kernel_sources_dir"] = expected_dirs[entry["low_level_id"]]
            self.assertEqual(actual_schema, expected_schema)

            actual_tree = temp_workspace / "kernel_sources"
            expected_files = sorted(
                path.relative_to(golden_tree)
                for path in golden_tree.rglob("*")
                if path.is_file()
            )
            actual_files = sorted(
                path.relative_to(actual_tree)
                for path in actual_tree.rglob("*")
                if path.is_file()
            )
            self.assertEqual(actual_files, expected_files)
            for relative in expected_files:
                self.assertTrue(
                    filecmp.cmp(
                        actual_tree / relative,
                        golden_tree / relative,
                        shallow=False,
                    ),
                    str(relative),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
