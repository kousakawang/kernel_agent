"""CPU-only tests for the KID v3 Python interface candidate locator."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.source_location import cli as sl_cli
from framework_engineer.source_location.contracts import kid_projection
from framework_engineer.source_location.locator import LocateError, locate_schema


class LocateFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="source_locate_v3_")
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
            {"name": "third", "status": "ok", "local_path": str(self.third)}
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
        call_file: Path | None = None,
        call_line: int = 1,
        provider: str | None = "third",
        archetype: str = "python_binding",
    ) -> dict:
        return {
            "rank": 1,
            "low_level_id": kid,
            "kernel": {
                "raw_name": f"raw::{kid}",
                "normalized_name": kid,
            },
            "interface": interface,
            "archetype": archetype,
            "provider": provider,
            "metrics": {"duration_us": None, "share_in_invocation": None},
            "measurement": {
                "metric": "gpu_kernel_duration_us",
                "aggregation": "sum",
                "sample_count": 0,
            },
            "runtime_event": {
                "call_site": {
                    "file": str(call_file or self.tmp / "missing_call.py"),
                    "line": call_line,
                },
                "attribution": {"method": "test", "confidence": "high"},
            },
            "untouched": {"value": [1, 2, 3]},
        }

    def _schema(self, entries: list[dict]) -> dict:
        return {
            "schema_version": "kernel-interface-decomposition/v3",
            "backend_name": "test",
            "target": {"interface": "high", "file": "/tmp/high.py", "line": 1},
            "coverage_report": {
                "per_invocation": [],
                "min_coverage": None,
                "semantic_target_count": len(entries),
                "gpu_kernel_count": None,
                "uncaptured_hint": "unit test",
            },
            "kernels": entries,
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


class TestResolution(LocateFixture):
    def test_callsite_alias_relative_import_and_reexport(self) -> None:
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
        payload = self._schema(
            [self._entry("pkg.exported_api", call_file=caller, call_line=3)]
        )
        original = copy.deepcopy(payload)

        located, result = self._locate(payload)
        candidate = located["kernels"][0]["locate_candidates"][
            "interface_definition"
        ]
        self.assertEqual(candidate["status"], "resolved")
        self.assertEqual(candidate["evidence"], "call_site_import")
        self.assertEqual(
            candidate["candidates"],
            [{"file": str(implementation), "def_line": 1}],
        )
        self.assertEqual(kid_projection(located), original)
        self.assertEqual(result.summary()["interface_resolved"], 1)

    def test_qualified_reexported_class_method_prefers_concrete_over_overloads(self) -> None:
        package = self.third / "wrapper_pkg"
        package.mkdir()
        implementation = package / "wrapper.py"
        implementation.write_text(
            "class Wrapper:\n"
            "    @overload\n"
            "    def run(self, x: int): ...\n"
            "    @overload\n"
            "    def run(self, x: str): ...\n"
            "    def run(self, x):\n"
            "        return x\n"
        )
        (package / "__init__.py").write_text(
            "from .wrapper import Wrapper as Wrapper\n"
        )
        caller = self.sglang / "wrapper_caller.py"
        caller.write_text("output = instance.run(value)\n")

        located, _ = self._locate(
            self._schema(
                [
                    self._entry(
                        "wrapper_pkg.Wrapper.run",
                        call_file=caller,
                        call_line=1,
                    )
                ]
            )
        )
        candidate = located["kernels"][0]["locate_candidates"][
            "interface_definition"
        ]
        self.assertEqual(candidate["status"], "resolved")
        self.assertEqual(candidate["evidence"], "interface_qualified_name")
        self.assertEqual(
            candidate["candidates"],
            [{"file": str(implementation), "def_line": 6}],
        )

    def test_binary_reexport_is_interface_definition(self) -> None:
        package = self.third / "binary_pkg"
        package.mkdir()
        init = package / "__init__.py"
        init.write_text("from ._C import (\n    other_api,\n    binary_api,\n)\n")
        caller = self.sglang / "binary_caller.py"
        caller.write_text("from binary_pkg import binary_api\nbinary_api()\n")

        located, _ = self._locate(
            self._schema(
                [
                    self._entry(
                        "binary_pkg.binary_api", call_file=caller, call_line=2
                    )
                ]
            )
        )
        candidate = located["kernels"][0]["locate_candidates"][
            "interface_definition"
        ]
        self.assertEqual(
            candidate["candidates"], [{"file": str(init), "def_line": 3}]
        )

    def test_qualified_duplicate_modules_are_ambiguous(self) -> None:
        for root in (self.sglang, self.third):
            package = root / "duplicate_pkg"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "api.py").write_text("def target():\n    return 1\n")
        located, _ = self._locate(
            self._schema([self._entry("duplicate_pkg.api.target")])
        )
        candidate = located["kernels"][0]["locate_candidates"][
            "interface_definition"
        ]
        self.assertEqual(candidate["status"], "ambiguous")
        self.assertEqual(len(candidate["candidates"]), 2)
        self.assertIsNone(candidate["repo_hint"])

    def test_global_leaf_name_fallback_is_disabled(self) -> None:
        (self.third / "false_positive.py").write_text(
            "def matmul():\n    return 'wrong'\n"
        )
        caller = self.sglang / "torch_caller.py"
        caller.write_text("import torch\ntorch.matmul(a, b)\n")
        located, _ = self._locate(
            self._schema(
                [
                    self._entry(
                        "torch.matmul",
                        call_file=caller,
                        call_line=2,
                        provider="pytorch",
                        archetype="pytorch_dispatch",
                    )
                ]
            )
        )
        candidate = located["kernels"][0]["locate_candidates"][
            "interface_definition"
        ]
        self.assertEqual(candidate["status"], "not_found")
        self.assertEqual(candidate["candidates"], [])
        self.assertTrue(any("fallback" in item for item in candidate["diagnostics"]))

    def test_archetype_and_provider_do_not_dispatch_resolution(self) -> None:
        package = self.third / "neutral"
        package.mkdir()
        (package / "__init__.py").write_text("")
        definition = package / "api.py"
        definition.write_text("def execute():\n    return 1\n")
        entries = [
            self._entry(
                "neutral.api.execute",
                kid="a",
                provider="unrelated",
                archetype="triton_launch",
            ),
            self._entry(
                "neutral.api.execute",
                kid="b",
                provider=None,
                archetype="python_binding",
            ),
        ]
        entries[1]["rank"] = 2
        located, _ = self._locate(self._schema(entries))
        candidates = [
            entry["locate_candidates"]["interface_definition"]["candidates"]
            for entry in located["kernels"]
        ]
        self.assertEqual(candidates[0], candidates[1])
        self.assertEqual(candidates[0][0]["file"], str(definition))


class TestContractsAndCli(LocateFixture):
    def test_public_cli_exposes_only_locate_and_extract(self) -> None:
        parser = sl_cli.build_parser()
        subcommands = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None)
        )
        self.assertEqual(set(subcommands), {"locate", "extract"})

    def test_rerun_replaces_candidates_but_rejects_agent_result(self) -> None:
        package = self.third / "rerun_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "api.py").write_text("def rerun():\n    pass\n")
        first, _ = self._locate(self._schema([self._entry("rerun_pkg.api.rerun")]))
        second, _ = self._locate(first, name="second.json")
        self.assertEqual(
            first["kernels"][0]["locate_candidates"],
            second["kernels"][0]["locate_candidates"],
        )

        first["kernels"][0]["source_locations"] = {"layers": {}}
        with self.assertRaisesRegex(LocateError, "source_locations"):
            self._locate(first, name="agent.json")

    def test_cli_success_reports_skipped_repo_and_not_found_is_nonfatal(self) -> None:
        schema = self.tmp / "input.json"
        output = self.tmp / "output.json"
        schema.write_text(json.dumps(self._schema([self._entry("absent.api")])) )
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
        self.assertEqual(json.loads(stdout.getvalue())["interface_not_found"], 1)
        self.assertIn("skipped manifest repo missing", stderr.getvalue())

    def test_invalid_inputs_and_in_place_output_return_two(self) -> None:
        valid = self.tmp / "valid.json"
        valid.write_text(json.dumps(self._schema([self._entry("absent.api")])))
        legacy = self._schema([self._entry("absent.api")])
        legacy["kernels"][0]["binding_provider"] = "old"
        legacy_path = self.tmp / "legacy.json"
        legacy_path.write_text(json.dumps(legacy))
        old_version = self._schema([self._entry("absent.api")])
        old_version["schema_version"] = "kernel-interface-decomposition/v2"
        old_version_path = self.tmp / "old_version.json"
        old_version_path.write_text(json.dumps(old_version))
        cases = [
            (self.tmp / "missing.json", self.tmp / "out1.json"),
            (legacy_path, self.tmp / "out2.json"),
            (old_version_path, self.tmp / "out3.json"),
            (valid, valid),
        ]
        for schema, output in cases:
            with self.subTest(schema=schema, output=output):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
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
                self.assertEqual(code, 2)


class TestRealGolden(unittest.TestCase):
    def test_all_backends_locate_candidates_golden(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        case = repo / "example_kernels" / "source_locate_golden"
        schema = case / "input" / "all_backends" / "decomposition.kid.schema.json"
        expected = (
            case
            / "workspaces"
            / "all_backends"
            / "locate"
            / "locate_candidates.schema.json"
        )
        manifest = case / "config" / "all_backends" / "third_party_manifest.json"
        with tempfile.TemporaryDirectory(prefix="source_locate_golden_") as tmp:
            output = Path(tmp) / "located.json"
            result = locate_schema(
                schema,
                manifest_path=manifest,
                sglang_repo_root=repo.parent / "sglang",
                output_path=output,
            )
            self.assertEqual(json.loads(output.read_text()), json.loads(expected.read_text()))
            self.assertEqual(result.summary()["interface_resolved"], 9)
            self.assertEqual(result.summary()["interface_not_found"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
