"""CPU-only tests for the private source-locate Agent helpers."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.source_location import agent_helper
from framework_engineer.source_location.agent_contracts import (
    DECISIONS_SCHEMA_VERSION,
    validate_decisions,
)
from framework_engineer.source_location.agent_helper import (
    AgentHelperError,
    evaluate_agent_result,
    finalize_agent_result,
    inspect_target,
    search_sources,
)
from framework_engineer.source_location.contracts import (
    ContractError,
    kid_projection,
)


def _candidate(file: Path, line: int = 1) -> dict:
    return {
        "interface_definition": {
            "status": "resolved",
            "candidates": [{"file": str(file), "def_line": line}],
            "repo_hint": str(file.parent),
            "evidence": "call_site_import",
            "diagnostics": [],
        }
    }


def _decision_layer(
    status: str,
    rationale: str,
    hits: list[tuple[Path, int, str, str]] | None = None,
) -> dict:
    return {
        "status": status,
        "rationale": rationale,
        "hits": [
            {
                "file": str(file),
                "def_line": line,
                "symbol": symbol,
                "reason": reason,
            }
            for file, line, symbol, reason in (hits or [])
        ],
    }


class AgentHelperFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="source_agent_helper_")
        self.tmp = Path(self.tempdir.name)
        self.sglang = self.tmp / "sglang"
        self.third = self.tmp / "third"
        self.sglang.mkdir()
        self.third.mkdir()

        self.caller = self.sglang / "caller.py"
        self.caller.write_text(
            "from pkg.api import launch\n"
            "result = launch(value)\n",
            encoding="utf-8",
        )
        package = self.third / "pkg"
        package.mkdir()
        self.interface = package / "api.py"
        self.interface.write_text(
            "def launch(value):\n"
            "    return torch.ops.demo.op(value)\n",
            encoding="utf-8",
        )
        self.binding = self.third / "binding.cc"
        self.binding.write_text(
            "TORCH_LIBRARY(demo, m) {\n"
            "  m.def(\"op(Tensor value) -> Tensor\");\n"
            "}\n",
            encoding="utf-8",
        )
        self.host = self.third / "host.cu"
        self.host.write_text(
            "void launch_op(int value) {\n"
            "  core_kernel<<<1, 1>>>(value);\n"
            "}\n",
            encoding="utf-8",
        )
        self.device = self.sglang / "device.cuh"
        self.device.write_text(
            "__global__ void core_kernel(int value) {\n"
            "  int output = value + 1;\n"
            "}\n",
            encoding="utf-8",
        )
        self.loader = self.third / "loader.py"
        self.loader.write_text(
            "def get_module():\n"
            "    return gen_jit_spec(\"demo\").build_and_load()\n",
            encoding="utf-8",
        )
        self.cmake = self.third / "CMakeLists.txt"
        self.cmake.write_text(
            "FetchContent_Declare(demo GIT_REPOSITORY example.invalid/demo)\n",
            encoding="utf-8",
        )

        self.missing = self.tmp / "missing_repo"
        self.manifest = self.tmp / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "repos": [
                        {
                            "name": "third",
                            "status": "ok",
                            "local_path": str(self.third),
                        },
                        {
                            "name": "missing",
                            "status": "ok",
                            "local_path": str(self.missing),
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.schema_path = self.tmp / "candidates.json"
        self.schema = self._schema()
        self.schema_path.write_text(
            json.dumps(self.schema, indent=2), encoding="utf-8"
        )
        self.decisions_path = self.tmp / "decisions.json"
        self.decisions = self._decisions()
        self._write_decisions(self.decisions)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _schema(self, *, provider: str | None = None) -> dict:
        return {
            "schema_version": "kernel-interface-decomposition/v2",
            "backend_name": "test",
            "opaque_kid_field": {"must": ["stay", "unchanged"]},
            "kernels": [
                {
                    "rank": 1,
                    "low_level_id": "demo_op",
                    "kernel": {
                        "raw_name": "void demo::core_kernel()",
                        "normalized_name": "demo_core",
                    },
                    "interface": "pkg.api.launch",
                    "archetype": "python_binding",
                    "provider": provider,
                    "metrics": {},
                    "coverage": None,
                    "runtime_event": {
                        "call_site": {"file": str(self.caller), "line": 2},
                        "attribution": {"method": "test", "confidence": "high"},
                    },
                    "locate_candidates": _candidate(self.interface),
                }
            ],
        }

    def _decisions(self) -> dict:
        return {
            "schema_version": DECISIONS_SCHEMA_VERSION,
            "kernels": [
                {
                    "low_level_id": "demo_op",
                    "summary": "launch crosses torch registration and reaches core_kernel",
                    "layers": {
                        "interface_definition": _decision_layer(
                            "resolved",
                            "The call-site import resolves to this Python API.",
                            [
                                (
                                    self.interface,
                                    1,
                                    "pkg.api.launch",
                                    "Matches the KID interface and calls torch.ops.demo.op.",
                                )
                            ],
                        ),
                        "kernel_impl": _decision_layer(
                            "resolved",
                            "The host launcher calls the device kernel.",
                            [
                                (
                                    self.host,
                                    1,
                                    "launch_op",
                                    "Host launch entry for the registered op.",
                                ),
                                (
                                    self.device,
                                    1,
                                    "core_kernel",
                                    "Core device implementation launched by launch_op.",
                                ),
                            ],
                        ),
                        "py_cpp_binding": _decision_layer(
                            "resolved",
                            "TORCH_LIBRARY publishes the Python/native boundary.",
                            [
                                (
                                    self.binding,
                                    1,
                                    "TORCH_LIBRARY(demo)",
                                    "Registers the demo namespace and op schema.",
                                )
                            ],
                        ),
                        "kernel_header": _decision_layer(
                            "not_applicable",
                            "The device implementation is header-implementation combined.",
                        ),
                    },
                    "gaps": [],
                    "manual_followup": None,
                }
            ],
        }

    def _write_decisions(self, value: dict) -> None:
        self.decisions_path.write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )

    def _finalize(self, *, output_name: str = "located.json") -> tuple[Path, Path]:
        output = self.tmp / output_name
        notes = self.tmp / f"{output_name}.md"
        finalize_agent_result(
            self.schema_path,
            decisions_path=self.decisions_path,
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            output_path=output,
            notes_path=notes,
        )
        return output, notes


class TestInspectAndSearch(AgentHelperFixture):
    def test_inspect_target_returns_candidate_context_roots_and_missing_repo(self) -> None:
        report = inspect_target(
            self.schema_path,
            kernel_id="demo_op",
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            max_lines=20,
        )
        self.assertEqual(report["interface"], "pkg.api.launch")
        self.assertEqual(report["candidate_contexts"][0]["end_line"], 2)
        self.assertEqual(
            report["candidate_contexts"][0]["allowed_source_root"]["path"],
            str(self.third),
        )
        self.assertEqual(report["call_site_context"]["line"], 2)
        self.assertIn(str(self.third), {root["path"] for root in report["search_roots"]})
        self.assertEqual(report["skipped_roots"][0]["name"], "missing")

    def test_inspect_does_not_require_provider(self) -> None:
        self.schema_path.write_text(json.dumps(self._schema(provider=None)))
        report = inspect_target(
            self.schema_path,
            kernel_id="demo_op",
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
        )
        self.assertIsNone(report["provider_hint"])

    def test_search_modes_are_raw_scoped_candidates(self) -> None:
        registration = search_sources(
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            mode="registration",
            queries=["demo.op", "op"],
        )
        self.assertTrue(
            any(match["file"] == str(self.binding) for match in registration["matches"])
        )
        self.assertTrue(
            all(match["category"] == "registration" for match in registration["matches"])
        )

        loader = search_sources(
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            mode="loader",
            queries=["gen_jit_spec"],
        )
        self.assertEqual(loader["matches"][0]["file"], str(self.loader))

        build = search_sources(
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            mode="build",
            queries=["demo"],
        )
        self.assertTrue(any(match["file"] == str(self.cmake) for match in build["matches"]))

    def test_search_limit_reports_truncation(self) -> None:
        report = search_sources(
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
            mode="literal",
            queries=["demo"],
            limit=1,
        )
        self.assertGreater(report["total_matches"], 1)
        self.assertTrue(report["truncated"])
        self.assertEqual(len(report["matches"]), 1)


class TestDecisionContract(AgentHelperFixture):
    def test_interface_and_impl_cannot_be_not_applicable(self) -> None:
        decisions = copy.deepcopy(self.decisions)
        layer = decisions["kernels"][0]["layers"]["kernel_impl"]
        layer.update(
            {
                "status": "not_applicable",
                "rationale": "invalid",
                "hits": [],
            }
        )
        with self.assertRaisesRegex(ContractError, "cannot be not_applicable"):
            validate_decisions(decisions, expected_kernel_ids=["demo_op"])

    def test_best_effort_requires_gap_and_missed_requires_followup(self) -> None:
        decisions = copy.deepcopy(self.decisions)
        decisions["kernels"][0]["layers"]["kernel_impl"]["status"] = "best_effort"
        with self.assertRaisesRegex(ContractError, "gaps must explain"):
            validate_decisions(decisions, expected_kernel_ids=["demo_op"])

        decisions = copy.deepcopy(self.decisions)
        decisions["kernels"][0]["layers"]["kernel_impl"] = _decision_layer(
            "missed", "The implementation repository is unavailable."
        )
        decisions["kernels"][0]["gaps"] = ["Implementation source is missing."]
        with self.assertRaisesRegex(ContractError, "manual_followup is required"):
            validate_decisions(decisions, expected_kernel_ids=["demo_op"])


class TestFinalizeAndEvaluate(AgentHelperFixture):
    def test_finalize_preserves_kid_strips_reasoning_and_computes_repo_hints(self) -> None:
        output, notes = self._finalize()
        located = json.loads(output.read_text())
        self.assertEqual(kid_projection(located), kid_projection(self.schema))
        entry = located["kernels"][0]
        self.assertNotIn("locate_candidates", entry)
        self.assertNotIn("kernel_sources_dir", entry)
        layers = entry["source_locations"]["layers"]
        self.assertEqual(layers["interface_definition"]["repo_hint"], str(self.third))
        self.assertIsNone(layers["kernel_impl"]["repo_hint"])
        self.assertEqual(
            set(layers["kernel_impl"]["hits"][0]), {"file", "def_line"}
        )
        note_text = notes.read_text()
        self.assertIn("core_kernel", note_text)
        self.assertIn("Host launch entry", note_text)

    def test_invalid_source_preserves_existing_outputs(self) -> None:
        outside = self.tmp / "outside.py"
        outside.write_text("def unexpected():\n    pass\n")
        decisions = copy.deepcopy(self.decisions)
        hit = decisions["kernels"][0]["layers"]["interface_definition"]["hits"][0]
        hit["file"] = str(outside)
        self._write_decisions(decisions)
        output = self.tmp / "existing.json"
        notes = self.tmp / "existing.md"
        output.write_text("old schema")
        notes.write_text("old notes")
        with self.assertRaisesRegex(AgentHelperError, "outside the allowed"):
            finalize_agent_result(
                self.schema_path,
                decisions_path=self.decisions_path,
                manifest_path=self.manifest,
                sglang_repo_root=self.sglang,
                output_path=output,
                notes_path=notes,
            )
        self.assertEqual(output.read_text(), "old schema")
        self.assertEqual(notes.read_text(), "old notes")

    def test_finalize_refuses_to_write_into_source_roots(self) -> None:
        with self.assertRaisesRegex(AgentHelperError, "must not write inside"):
            finalize_agent_result(
                self.schema_path,
                decisions_path=self.decisions_path,
                manifest_path=self.manifest,
                sglang_repo_root=self.sglang,
                output_path=self.sglang / "located.json",
                notes_path=self.tmp / "notes.md",
            )

    def test_out_of_range_line_is_rejected(self) -> None:
        decisions = copy.deepcopy(self.decisions)
        decisions["kernels"][0]["layers"]["kernel_impl"]["hits"][0][
            "def_line"
        ] = 999
        self._write_decisions(decisions)
        with self.assertRaisesRegex(AgentHelperError, "out of range"):
            self._finalize()

    def test_evaluate_requires_golden_core_order_but_allows_explained_extra(self) -> None:
        golden, _ = self._finalize(output_name="golden.json")

        decisions = copy.deepcopy(self.decisions)
        extra = {
            "file": str(self.host),
            "def_line": 2,
            "symbol": "launch statement",
            "reason": "Explicit launch edge between the host and device nodes.",
        }
        decisions["kernels"][0]["layers"]["kernel_impl"]["hits"].insert(1, extra)
        self._write_decisions(decisions)
        actual, _ = self._finalize(output_name="actual.json")
        report = evaluate_agent_result(
            actual,
            decisions_path=self.decisions_path,
            golden_path=golden,
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
        )
        self.assertTrue(report["ok"], report["errors"])

        reordered = copy.deepcopy(decisions)
        hits = reordered["kernels"][0]["layers"]["kernel_impl"]["hits"]
        reordered["kernels"][0]["layers"]["kernel_impl"]["hits"] = [
            hits[2],
            hits[1],
            hits[0],
        ]
        self._write_decisions(reordered)
        reordered_actual, _ = self._finalize(output_name="reordered.json")
        report = evaluate_agent_result(
            reordered_actual,
            decisions_path=self.decisions_path,
            golden_path=golden,
            manifest_path=self.manifest,
            sglang_repo_root=self.sglang,
        )
        self.assertFalse(report["ok"])
        self.assertIn("ordered subsequence", "\n".join(report["errors"]))

    def test_private_cli_returns_two_for_invalid_contract(self) -> None:
        stderr: list[str] = []

        class Sink:
            def write(self, value: str) -> int:
                stderr.append(value)
                return len(value)

            def flush(self) -> None:
                return None

        original = agent_helper.sys.stderr
        try:
            agent_helper.sys.stderr = Sink()  # type: ignore[assignment]
            code = agent_helper.main(
                [
                    "inspect-target",
                    "--schema",
                    str(self.schema_path),
                    "--kernel-id",
                    "unknown",
                    "--manifest",
                    str(self.manifest),
                    "--sglang-repo-root",
                    str(self.sglang),
                ]
            )
        finally:
            agent_helper.sys.stderr = original
        self.assertEqual(code, 2)
        self.assertIn("unknown low_level_id", "".join(stderr))


class TestRealAgentGolden(unittest.TestCase):
    def test_finalize_and_evaluate_current_ten_target_golden(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        candidates_path = repo / "example_kernels" / "to_fill_locate_candidates.json"
        golden_path = repo / "example_kernels" / "to_fill_locate.json"
        manifest = (
            repo
            / "framework_engineer"
            / "source_location"
            / "example"
            / "third_party_manifest.json"
        )
        golden = json.loads(golden_path.read_text())
        decisions = {
            "schema_version": DECISIONS_SCHEMA_VERSION,
            "kernels": [],
        }
        for entry in golden["kernels"]:
            has_missed = False
            decision_layers: dict[str, dict] = {}
            for layer_name, layer in entry["source_locations"]["layers"].items():
                if layer["status"] == "missed":
                    has_missed = True
                decision_layers[layer_name] = {
                    "status": layer["status"],
                    "rationale": (
                        "The current source-locate golden records this layer "
                        f"as {layer['status']}."
                    ),
                    "hits": [
                        {
                            "file": hit["file"],
                            "def_line": hit["def_line"],
                            "symbol": f"{Path(hit['file']).name}:{hit['def_line']}",
                            "reason": "Required core-chain evidence from the Agent golden.",
                        }
                        for hit in layer["hits"]
                    ],
                }
            decisions["kernels"].append(
                {
                    "low_level_id": entry["low_level_id"],
                    "summary": "Current ten-target source-locate golden call chain.",
                    "layers": decision_layers,
                    "gaps": (
                        ["The required source repository is not present in the manifest."]
                        if has_missed
                        else []
                    ),
                    "manual_followup": (
                        "Add the missing source repository to the manifest and rerun locate."
                        if has_missed
                        else None
                    ),
                }
            )

        with tempfile.TemporaryDirectory(prefix="source_agent_real_golden_") as tmp:
            temp = Path(tmp)
            decisions_path = temp / "decisions.json"
            decisions_path.write_text(json.dumps(decisions, indent=2))
            output = temp / "located.json"
            notes = temp / "ref" / "locate_agent_notes.md"
            report = finalize_agent_result(
                candidates_path,
                decisions_path=decisions_path,
                manifest_path=manifest,
                sglang_repo_root=repo.parent / "sglang",
                output_path=output,
                notes_path=notes,
            )
            self.assertEqual(report["kernels"], 10)
            self.assertEqual(json.loads(output.read_text()), golden)
            self.assertTrue(notes.is_file())

            evaluation = evaluate_agent_result(
                output,
                decisions_path=decisions_path,
                golden_path=golden_path,
                manifest_path=manifest,
                sglang_repo_root=repo.parent / "sglang",
            )
            self.assertTrue(evaluation["ok"], evaluation["errors"])


if __name__ == "__main__":
    unittest.main()
