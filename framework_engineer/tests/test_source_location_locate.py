"""CPU-only tests for the Layer-1 deterministic source locator."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

try:  # Package-style invocation from the workspace parent.
    from ..source_location import cli as sl_cli
    from ..source_location.locator import LocateError, locate_schema
except ImportError:  # unittest discovery from the kernel_agent repo root.
    from framework_engineer.source_location import cli as sl_cli
    from framework_engineer.source_location.locator import LocateError, locate_schema


class LocateFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="locate_layer1_")
        self.tmp = Path(self.tempdir.name)
        self.sglang = self.tmp / "sglang"
        self.third = self.tmp / "third"
        self.sglang.mkdir()
        self.third.mkdir()
        self.manifest = self.tmp / "third_party_manifest.json"
        self._write_manifest()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_manifest(self, *, missing: Path | None = None) -> None:
        repos = [
            {
                "name": "third",
                "status": "ok",
                "local_path": str(self.third),
            }
        ]
        if missing is not None:
            repos.append(
                {"name": "missing", "status": "ok", "local_path": str(missing)}
            )
        self.manifest.write_text(json.dumps({"repos": repos}))

    def _entry(
        self,
        interface: str,
        *,
        kid: str = "k1",
        code: str = "F4",
        provider: str | None = "provider",
        call_file: Path | None = None,
        call_line: int | None = None,
        implementation: dict | None = None,
    ) -> dict:
        runtime_event: dict = {
            "call_site": {
                "file": str(call_file or self.tmp / "missing_call.py"),
                "line": call_line or 1,
            },
            "implementation": implementation or {"source_files": []},
        }
        return {
            "low_level_id": kid,
            "interface": interface,
            "archetype": "test_archetype" if code != "F0" else "pytorch_native",
            "archetype_code": code,
            "binding_provider": provider,
            "kernel": {"normalized_name": interface.rsplit(".", 1)[-1]},
            "runtime_event": runtime_event,
            "untouched": {"value": [1, 2, 3]},
        }

    def _locate(self, payload: dict, *, name: str = "schema.json") -> tuple[dict, object]:
        schema = self.tmp / name
        output = self.tmp / f"out_{name}"
        schema.write_text(json.dumps(payload, indent=2))
        result = locate_schema(
            schema,
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            output_path=output,
        )
        return json.loads(output.read_text()), result


class TestRuntimeAndImports(LocateFixture):
    def test_runtime_fact_uses_first_decorator_line(self) -> None:
        source = self.third / "runtime_impl.py"
        source.write_text(
            "@first\n"
            "@second(value=1)\n"
            "def target(x):\n"
            "    return x\n"
        )
        entry = self._entry(
            "target",
            code="F6",
            provider=None,
            implementation={
                "source_files": [str(source)],
                "definition_line": 4,
            },
        )
        enriched, result = self._locate({"kernels": [entry]})

        layer = enriched["kernels"][0]["source_locations"]["layers"][
            "interface_definition"
        ]
        self.assertEqual(layer["status"], "resolved")
        self.assertEqual(
            layer["hits"], [{"file": str(source), "def_line": 1}]
        )
        self.assertEqual(result.kernels[0].evidence, "runtime_event.implementation")

    def test_function_import_alias_relative_import_and_reexport(self) -> None:
        package = self.third / "pkg"
        package.mkdir()
        implementation = package / "implementation.py"
        implementation.write_text("def api(value):\n    return value\n")
        (package / "__init__.py").write_text(
            "from .implementation import api as exported_api\n"
        )
        caller = self.sglang / "caller.py"
        caller.write_text(
            "def invoke(value):\n"
            "    from pkg import exported_api as run\n"
            "    return run(value)\n"
        )
        entry = self._entry(
            "pkg.exported_api", call_file=caller, call_line=3
        )

        enriched, result = self._locate({"kernels": [entry]})
        layer = enriched["kernels"][0]["source_locations"]["layers"][
            "interface_definition"
        ]
        self.assertEqual(layer["status"], "resolved")
        self.assertEqual(
            layer["hits"],
            [{"file": str(implementation), "def_line": 1}],
        )
        self.assertEqual(result.kernels[0].evidence, "call_site_import")

    def test_module_alias_resolves_module_function(self) -> None:
        package = self.third / "modpkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        implementation = package / "ops.py"
        implementation.write_text("def launch():\n    return 1\n")
        caller = self.sglang / "caller_alias.py"
        caller.write_text("import modpkg.ops as ops\nresult = ops.launch()\n")
        entry = self._entry("modpkg.ops.launch", call_file=caller, call_line=2)

        enriched, _ = self._locate({"kernels": [entry]})
        hit = enriched["kernels"][0]["source_locations"]["layers"][
            "interface_definition"
        ]["hits"]
        self.assertEqual(
            hit, [{"file": str(implementation), "def_line": 1}]
        )

    def test_wrapper_run_resolves_class_method_and_collapses_overloads(self) -> None:
        implementation = self.third / "wrapper.py"
        implementation.write_text(
            "class Wrapper:\n"
            "    @overload\n"
            "    def run(self, x: int): ...\n"
            "    @overload\n"
            "    def run(self, x: str): ...\n"
            "    def run(self, x):\n"
            "        return x\n"
        )
        caller = self.sglang / "wrapper_caller.py"
        caller.write_text("output = wrapper.run(value)\n")
        entry = self._entry("Wrapper.run", call_file=caller, call_line=1)

        enriched, _ = self._locate({"kernels": [entry]})
        layer = enriched["kernels"][0]["source_locations"]["layers"][
            "interface_definition"
        ]
        self.assertEqual(layer["status"], "resolved")
        self.assertEqual(
            layer["hits"],
            [{"file": str(implementation), "def_line": 3}],
        )

    def test_binary_reexport_line_is_interface_definition(self) -> None:
        package = self.third / "binary_pkg"
        package.mkdir()
        init = package / "__init__.py"
        init.write_text("from ._C import (\n    other_api,\n    binary_api,\n)\n")
        entry = self._entry("binary_api")

        enriched, _ = self._locate({"kernels": [entry]})
        layer = enriched["kernels"][0]["source_locations"]["layers"][
            "interface_definition"
        ]
        self.assertEqual(layer["status"], "resolved")
        self.assertEqual(
            layer["hits"], [{"file": str(init), "def_line": 3}]
        )


class TestCandidatesAndTemplates(LocateFixture):
    def test_exact_name_unique_ambiguous_and_not_found(self) -> None:
        unique = self.third / "unique.py"
        unique.write_text("def only_here():\n    pass\n")
        first = self.sglang / "first.py"
        second = self.third / "second.py"
        first.write_text("def duplicate():\n    pass\n")
        second.write_text("def duplicate():\n    pass\n")
        entries = [
            self._entry("only_here", kid="unique"),
            self._entry("duplicate", kid="duplicate"),
            self._entry("nowhere", kid="missing"),
        ]

        enriched, result = self._locate({"kernels": entries})
        layers = [
            entry["source_locations"]["layers"]["interface_definition"]
            for entry in enriched["kernels"]
        ]
        self.assertEqual(
            [layer["status"] for layer in layers],
            ["resolved", "ambiguous", "not_found"],
        )
        self.assertEqual(
            layers[1]["hits"],
            sorted(
                [
                    {"file": str(first), "def_line": 1},
                    {"file": str(second), "def_line": 1},
                ],
                key=lambda hit: (hit["file"], hit["def_line"]),
            ),
        )
        self.assertIsNone(layers[1]["repo_hint"])
        self.assertTrue(any("call-site file not found" in message for message in result.kernels[2].diagnostics))

    def test_flat_schema_templates_and_original_fields(self) -> None:
        definition = self.third / "target.py"
        definition.write_text("def target():\n    pass\n")
        f0 = self._entry("native", kid="f0", code="F0", provider=None)
        f1 = self._entry("target", kid="f1", code="F1", provider=None)
        f2 = self._entry("target", kid="f2", code="F2", provider="registered_later")
        original = copy.deepcopy([f0, f1, f2])

        enriched, _ = self._locate({"meta": "keep", "kernels": [f0, f1, f2]})
        self.assertEqual(enriched["meta"], "keep")
        for before, after in zip(original, enriched["kernels"]):
            without_locations = {k: v for k, v in after.items() if k != "source_locations"}
            self.assertEqual(without_locations, before)

        locations = [entry["source_locations"] for entry in enriched["kernels"]]
        self.assertFalse(locations[0]["needs_agent"])
        self.assertEqual(
            {layer["status"] for layer in locations[0]["layers"].values()},
            {"not_applicable"},
        )
        self.assertEqual(locations[1]["layers"]["py_cpp_binding"]["status"], "not_applicable")
        self.assertEqual(locations[1]["layers"]["kernel_header"]["status"], "not_applicable")
        self.assertEqual(locations[2]["layers"]["py_cpp_binding"], {
            "status": "not_found", "hits": [], "repo_hint": None, "source": "locate_layer1"
        })
        self.assertEqual(locations[2]["layers"]["kernel_header"]["status"], "missed")
        self.assertNotIn("kernel_sources_dir", enriched["kernels"][2])

    def test_nested_invocations_shape(self) -> None:
        definition = self.third / "nested.py"
        definition.write_text("def nested_api():\n    pass\n")
        payload = {
            "invocations": [
                {"selected_kernels": [self._entry("nested_api", kid="nested")]}
            ]
        }
        enriched, result = self._locate(payload)
        kernel = enriched["invocations"][0]["selected_kernels"][0]
        self.assertEqual(
            kernel["source_locations"]["layers"]["interface_definition"]["status"],
            "resolved",
        )
        self.assertEqual(result.summary()["total"], 1)

    def test_layer1_rerun_allowed_layer2_and_manual_protected(self) -> None:
        definition = self.third / "rerun.py"
        definition.write_text("def rerun():\n    pass\n")
        schema = self.tmp / "rerun.json"
        schema.write_text(json.dumps({"kernels": [self._entry("rerun")]}))
        locate_schema(
            schema,
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
        )
        locate_schema(
            schema,
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
        )

        for protected in ("locate_layer2_agent", "manual"):
            payload = json.loads(schema.read_text())
            payload["kernels"][0]["source_locations"]["layers"][
                "kernel_impl"
            ]["source"] = protected
            blocked = self.tmp / f"blocked_{protected}.json"
            blocked.write_text(json.dumps(payload))
            with self.assertRaisesRegex(LocateError, "refusing to overwrite"):
                locate_schema(
                    blocked,
                    manifest_path=self.manifest,
                    sglang_repo_root=self.sglang,
                )


class TestLocateCli(LocateFixture):
    def test_cli_success_summary_report_and_missing_manifest_repo(self) -> None:
        definition = self.third / "cli_api.py"
        definition.write_text("def cli_api():\n    pass\n")
        schema = self.tmp / "cli_schema.json"
        output = self.tmp / "result" / "enriched.json"
        schema.write_text(json.dumps({"kernels": [self._entry("cli_api")]}))
        self._write_manifest(missing=self.tmp / "absent_repo")
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = sl_cli.main(
                [
                    "locate",
                    "--schema",
                    str(schema),
                    "--manifest",
                    str(self.manifest),
                    "--sglang-repo-root",
                    str(self.sglang),
                    "--out",
                    str(output),
                ]
            )

        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["interface_resolved"], 1)
        self.assertIn("resolved=1", stderr.getvalue())
        self.assertIn("skipped manifest repo missing", stderr.getvalue())
        report_path = output.parent / "ref" / "locate_report.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["interface_resolved"], 1)
        self.assertEqual(len(report["needs_agent"]), 1)
        self.assertEqual(report["needs_agent"][0]["status"], "resolved")
        self.assertEqual(
            report["needs_agent"][0]["unresolved_layers"],
            ["kernel_impl", "py_cpp_binding", "kernel_header"],
        )
        self.assertEqual(len(report["search_roots_skipped"]), 1)

    def test_cli_not_found_is_nonfatal(self) -> None:
        schema = self.tmp / "not_found_schema.json"
        schema.write_text(json.dumps({"kernels": [self._entry("absent_api")]}))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = sl_cli.main(
                [
                    "locate",
                    "--schema",
                    str(schema),
                    "--manifest",
                    str(self.manifest),
                    "--sglang-repo-root",
                    str(self.sglang),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["interface_not_found"], 1)

    def test_cli_invalid_global_inputs_return_two(self) -> None:
        invalid_schema = self.tmp / "invalid_schema.json"
        invalid_schema.write_text("{not JSON")
        invalid_manifest = self.tmp / "invalid_manifest.json"
        invalid_manifest.write_text("[not JSON")
        cases = [
            [
                "locate",
                "--schema",
                str(self.tmp / "missing.json"),
                "--manifest",
                str(self.manifest),
                "--sglang-repo-root",
                str(self.sglang),
            ],
            [
                "locate",
                "--schema",
                str(self._valid_schema()),
                "--manifest",
                str(self.tmp / "missing_manifest.json"),
                "--sglang-repo-root",
                str(self.sglang),
            ],
            [
                "locate",
                "--schema",
                str(self._valid_schema()),
                "--manifest",
                str(self.manifest),
                "--sglang-repo-root",
                str(self.tmp / "missing_sglang"),
            ],
            [
                "locate",
                "--schema",
                str(invalid_schema),
                "--manifest",
                str(self.manifest),
                "--sglang-repo-root",
                str(self.sglang),
            ],
            [
                "locate",
                "--schema",
                str(self._valid_schema()),
                "--manifest",
                str(invalid_manifest),
                "--sglang-repo-root",
                str(self.sglang),
            ],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code = sl_cli.main(arguments)
                self.assertEqual(code, 2)
                self.assertIn("error:", stderr.getvalue())

    def _valid_schema(self) -> Path:
        schema = self.tmp / "valid_schema.json"
        if not schema.exists():
            schema.write_text(json.dumps({"kernels": [self._entry("not_found")]}))
        return schema


if __name__ == "__main__":
    unittest.main()
