from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from framework_engineer.kernel_interface_decomposer.artifact_validator import (
    ArtifactValidator,
)
from framework_engineer.kernel_interface_decomposer.semantic_resolver import (
    SemanticResolver,
    SemanticResolverConfig,
    SemanticResolverError,
)
from framework_engineer.kernel_interface_decomposer.config import ConfigError


MODULE = (
    "framework_engineer.kernel_interface_decomposer.semantic_resolver_tools"
)


class SemanticResolverFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.golden = cls.repo_root / "example_kernels/nsys_poc_kid_golden"
        cls.golden_config = json.loads(
            (
                cls.golden
                / "config/nsys_poc/semantic_resolver_config.json"
            ).read_text(encoding="utf-8")
        )
        cls.golden_decisions = json.loads(
            (
                cls.golden
                / "nsys_poc/ref/semantic_resolver_decisions.json"
            ).read_text(encoding="utf-8")
        )
        cls.golden_final = json.loads(
            (
                cls.golden
                / "nsys_poc/output/decomposition.schema.json"
            ).read_text(encoding="utf-8")
        )

    def make_workspace(
        self,
        root: Path,
        *,
        runtime: dict[str, Any] | None = None,
        decisions: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> tuple[Path, SemanticResolverConfig]:
        config_dir = root / "config/nsys_poc"
        backend_cli = root / "nsys_poc/cli_log"
        backend_ref = root / "nsys_poc/ref"
        backend_output = root / "nsys_poc/output"
        config_dir.mkdir(parents=True)
        backend_cli.mkdir(parents=True)
        backend_ref.mkdir(parents=True)
        backend_output.mkdir(parents=True)
        config = copy.deepcopy(self.golden_config)
        runtime_value = runtime or json.loads(
            (self.golden / "nsys_poc/cli_log/runtime_capture.schema.json").read_text(
                encoding="utf-8"
            )
        )
        (backend_cli / "runtime_capture.schema.json").write_text(
            json.dumps(runtime_value, indent=2) + "\n", encoding="utf-8"
        )
        runtime_config = json.loads(
            (
                self.golden / "config/nsys_poc/runtime_capture_config.json"
            ).read_text(encoding="utf-8")
        )
        runtime_config["workdir"] = str(self.repo_root.parent)
        runtime_config["output_dir"] = str(root)
        runtime_config["target"]["file"] = str(
            self.repo_root
            / "framework_engineer/kernel_interface_decomposer/nsys_poc.py"
        )
        (config_dir / "runtime_capture_config.json").write_text(
            json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8"
        )
        decisions_value = decisions or copy.deepcopy(self.golden_decisions)
        (backend_ref / "semantic_resolver_decisions.json").write_text(
            json.dumps(decisions_value, indent=2) + "\n", encoding="utf-8"
        )
        (backend_ref / "kid_semantic_resolver_notes.md").write_text(
            notes
            if notes is not None
            else (
                self.golden / "nsys_poc/ref/kid_semantic_resolver_notes.md"
            ).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        config_path = config_dir / "semantic_resolver_config.json"
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        return config_path, SemanticResolverConfig.load(config_path)


class TestSemanticResolverGolden(SemanticResolverFixture):
    def test_complete_artifact_validator_accepts_semantic_golden(self) -> None:
        validator = ArtifactValidator(self.golden)
        self.assertTrue(validator.validate(), msg="; ".join(validator.errors))

    def test_prepare_context_has_owner_evidence_and_current_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-semantic-") as temporary:
            _, config = self.make_workspace(Path(temporary))
            context = SemanticResolver(config).prepare()
            owners = [
                capture
                for invocation in context["invocations"]
                for capture in invocation["owner_captures"]
            ]
            self.assertEqual(len(owners), 11)
            self.assertEqual(
                {capture["member_ref"]["capture_id"] for capture in owners},
                {"1", "3", "4", "6", "7", "9", "21", "23", "24", "35", "36"},
            )
            matmul = next(
                item for item in owners if item["member_ref"]["capture_id"] == "1"
            )
            edge = next(
                item
                for item in matmul["stack_edges"]
                if item["call_site"]["line"] == 818
            )
            self.assertEqual(edge["call_expression"], "torch.matmul(a, b)")
            self.assertTrue(edge["analysis_file"].endswith("nsys_poc.py"))
            self.assertNotIn("source_snapshot", edge["analysis_file"])
            serialized = json.dumps(context)
            self.assertNotIn("low_level_id", serialized)
            self.assertNotIn("normalized_kernel_name", serialized)
            self.assertEqual(
                {hint["name"] for hint in context["repository_hints"]},
                {"sglang", "flashinfer", "flash_attn", "deep_gemm"},
            )
            self.assertTrue(
                all(
                    hint["source"] == "third_party_manifest"
                    for hint in context["repository_hints"]
                )
            )

    def test_empty_mapping_keeps_paths_and_outputs_are_derived(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-semantic-empty-map-") as temporary:
            root = Path(temporary)
            config_path, _ = self.make_workspace(root)
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw["source_context"]["runtime_to_local_path_mappings"] = []
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            config = SemanticResolverConfig.load(config_path)
            self.assertEqual(config.map_runtime_path("/runtime/source.py"), "/runtime/source.py")
            self.assertEqual(
                config.runtime_capture,
                (root / "nsys_poc/cli_log/runtime_capture.schema.json").resolve(),
            )
            self.assertEqual(
                config.context_output,
                (root / "nsys_poc/ref/semantic_resolver_context.json").resolve(),
            )
            self.assertEqual(
                config.output,
                (root / "nsys_poc/output/decomposition.schema.json").resolve(),
            )

    def test_shared_output_root_keeps_backend_paths_isolated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-semantic-backends-") as temporary:
            root = Path(temporary)
            first_path, first = self.make_workspace(root)
            second_dir = root / "config/other"
            second_dir.mkdir(parents=True)
            runtime_raw = json.loads(
                (first_path.parent / "runtime_capture_config.json").read_text(
                    encoding="utf-8"
                )
            )
            runtime_raw["backend_name"] = "other"
            (second_dir / "runtime_capture_config.json").write_text(
                json.dumps(runtime_raw), encoding="utf-8"
            )
            semantic_raw = json.loads(first_path.read_text(encoding="utf-8"))
            semantic_raw["backend_name"] = "other"
            second_path = second_dir / "semantic_resolver_config.json"
            second_path.write_text(json.dumps(semantic_raw), encoding="utf-8")
            second = SemanticResolverConfig.load(second_path)
            self.assertEqual(first.output_root, second.output_root)
            self.assertNotEqual(first.runtime_capture, second.runtime_capture)
            self.assertNotEqual(first.context_output, second.context_output)
            self.assertNotEqual(first.output, second.output)

    def test_longest_path_mapping_wins_and_removed_fields_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-semantic-ast-") as temporary:
            root = Path(temporary)
            config_path, config = self.make_workspace(root)
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw["source_context"]["runtime_to_local_path_mappings"] = [
                {
                    "runtime_prefix": "/runtime",
                    "local_prefix": "/local/general",
                },
                {
                    "runtime_prefix": "/runtime/pkg",
                    "local_prefix": "/local/specific",
                },
            ]
            config_path.write_text(
                json.dumps(raw, indent=2) + "\n", encoding="utf-8"
            )
            mapped = SemanticResolverConfig.load(config_path)
            self.assertEqual(
                mapped.map_runtime_path("/runtime/pkg/source.py"),
                "/local/specific/source.py",
            )
            for removed in (
                "runtime_capture",
                "analysis_source_overrides",
                "context_output",
                "decisions_output",
                "notes_output",
                "output",
            ):
                candidate = copy.deepcopy(raw)
                candidate[removed] = [] if removed == "analysis_source_overrides" else "unused"
                config_path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(removed=removed), self.assertRaisesRegex(
                    ConfigError, f"unsupported.*{removed}"
                ):
                    SemanticResolverConfig.load(config_path)
            candidate = copy.deepcopy(raw)
            candidate["source_context"]["source_roots"] = []
            config_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unsupported.*source_roots"):
                SemanticResolverConfig.load(config_path)
            candidate = copy.deepcopy(raw)
            candidate["schema_version"] = "kid-semantic-resolver-config/v2"
            config_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "kid-semantic-resolver-config/v3"):
                SemanticResolverConfig.load(config_path)

    def test_helper_cli_reproduces_final_golden(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-semantic-cli-") as temporary:
            config_path, config = self.make_workspace(Path(temporary))
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(self.repo_root), env.get("PYTHONPATH")) if value
            )
            for command in ("prepare", "finalize", "validate"):
                completed = subprocess.run(
                    [sys.executable, "-m", MODULE, command, str(config_path)],
                    cwd=self.repo_root,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"{command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )
            actual = json.loads(config.output.read_text(encoding="utf-8"))
            self.assertEqual(actual, self.golden_final)

    def test_decision_contract_rejects_missing_owner_invalid_edge_and_low_confidence(self) -> None:
        mutations = []
        missing = copy.deepcopy(self.golden_decisions)
        missing["targets"].pop()
        mutations.append((missing, "unassigned"))
        invalid_edge = copy.deepcopy(self.golden_decisions)
        invalid_edge["targets"][0]["members"][0]["semantic_call_site"]["line"] = 999999
        mutations.append((invalid_edge, "lacks Runtime stack evidence"))
        low_confidence = copy.deepcopy(self.golden_decisions)
        low_confidence["targets"][0]["confidence"] = "low"
        mutations.append((low_confidence, "not publishable"))
        duplicate = copy.deepcopy(self.golden_decisions)
        duplicate_target = copy.deepcopy(duplicate["targets"][0])
        duplicate_target["low_level_id"] = "duplicate_matmul"
        duplicate["targets"].append(duplicate_target)
        mutations.append((duplicate, "duplicate semantic interface"))

        for index, (decisions, expected) in enumerate(mutations):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory(
                prefix=f"kid-semantic-invalid-{index}-"
            ) as temporary:
                _, config = self.make_workspace(
                    Path(temporary), decisions=decisions
                )
                resolver = SemanticResolver(config)
                resolver.prepare()
                with self.assertRaisesRegex(SemanticResolverError, expected):
                    resolver.build_final()

    def test_mixed_archetype_requires_notes_disclosure(self) -> None:
        runtime = json.loads(
            (
                self.golden
                / "nsys_poc/cli_log/runtime_capture.schema.json"
            ).read_text(encoding="utf-8")
        )
        capture = next(
            item
            for item in runtime["invocations"][0]["execution_captures"]
            if item["capture_id"] == "3"
        )
        capture["archetype"] = "triton_launch"
        decisions = copy.deepcopy(self.golden_decisions)
        first = decisions["targets"][0]
        second = decisions["targets"].pop(1)
        first["members"].extend(second["members"])
        with tempfile.TemporaryDirectory(prefix="kid-semantic-mixed-") as temporary:
            _, config = self.make_workspace(
                Path(temporary), runtime=runtime, decisions=decisions, notes="# notes\n"
            )
            resolver = SemanticResolver(config)
            resolver.prepare()
            with self.assertRaisesRegex(SemanticResolverError, "mixed-archetype"):
                resolver.build_final()


class TestSemanticResolverMultiInvocation(SemanticResolverFixture):
    def _two_invocation_runtime(self) -> dict[str, Any]:
        runtime = json.loads(
            (
                self.golden
                / "nsys_poc/cli_log/runtime_capture.schema.json"
            ).read_text(encoding="utf-8")
        )
        original = runtime["invocations"][0]
        second = copy.deepcopy(original)
        second["high_level"]["call_id"] = "2"
        kernel_mapping: dict[str, str] = {}
        duplicate_kernels: list[dict[str, Any]] = []
        for kernel in runtime["kernels"]:
            duplicate = copy.deepcopy(kernel)
            old_id = str(kernel["kernel_id"])
            new_id = old_id + "-inv2"
            kernel_mapping[old_id] = new_id
            duplicate["kernel_id"] = new_id
            duplicate_kernels.append(duplicate)
        second["high_level"]["kernel_ids"] = [
            kernel_mapping[item] for item in second["high_level"]["kernel_ids"]
        ]
        for capture in second["execution_captures"]:
            capture["parent_call_id"] = "2"
            capture["kernel_ids"] = [
                kernel_mapping[item] for item in capture["kernel_ids"]
            ]
            capture["inclusive_kernel_ids"] = [
                kernel_mapping[item] for item in capture["inclusive_kernel_ids"]
            ]
        softmax_capture = next(
            item for item in second["execution_captures"] if item["capture_id"] == "1"
        )
        softmax_capture["execution_interface"] = "aten._softmax.default"
        softmax_kernel = next(
            item
            for item in duplicate_kernels
            if item["kernel_id"] == kernel_mapping[original["execution_captures"][0]["kernel_ids"][0]]
        )
        softmax_kernel["name"] = "softmax_warp_forward"
        runtime["invocations"].append(second)
        runtime["kernels"].extend(duplicate_kernels)
        return runtime

    def _union_decisions(self) -> dict[str, Any]:
        decisions = copy.deepcopy(self.golden_decisions)
        for target in decisions["targets"]:
            if target["interface"] != "torch.matmul":
                member = copy.deepcopy(target["members"][0])
                member["call_id"] = "2"
                target["members"].append(member)
        softmax = copy.deepcopy(decisions["targets"][0])
        softmax["low_level_id"] = "poc_pytorch_softmax"
        softmax["interface"] = "torch.softmax"
        softmax["normalized_kernel_name"] = "torch_softmax"
        softmax["members"][0]["call_id"] = "2"
        decisions["targets"].append(softmax)
        return decisions

    def test_interface_union_and_sample_count(self) -> None:
        runtime = self._two_invocation_runtime()
        decisions = self._union_decisions()
        with tempfile.TemporaryDirectory(prefix="kid-semantic-union-") as temporary:
            _, config = self.make_workspace(
                Path(temporary), runtime=runtime, decisions=decisions
            )
            resolver = SemanticResolver(config)
            resolver.prepare()
            final = resolver.build_final()
            by_interface = {item["interface"]: item for item in final["kernels"]}
            self.assertEqual(len(by_interface), 12)
            self.assertEqual(by_interface["torch.matmul"]["measurement"]["sample_count"], 1)
            self.assertEqual(by_interface["torch.softmax"]["measurement"]["sample_count"], 1)
            self.assertEqual(
                by_interface["sgl_kernel.silu_and_mul"]["measurement"]["sample_count"],
                2,
            )
            semantic_total = sum(
                item["metrics"]["duration_us"] for item in final["kernels"]
            )
            runtime_total = sum(
                item["total_gpu_us"]
                for item in final["coverage_report"]["per_invocation"]
            )
            self.assertAlmostEqual(semantic_total, runtime_total)


class TestKidAgentPromptContract(SemanticResolverFixture):
    def test_prompt_owns_runtime_and_semantic_workflow(self) -> None:
        core = (self.repo_root / "framework_engineer/prompts/kid.md").read_text(
            encoding="utf-8"
        )
        starter = (
            self.repo_root / "framework_engineer/prompts/start_kid.md"
        ).read_text(encoding="utf-8")
        for command in ("capture", "prepare", "finalize", "validate"):
            self.assertIn(command, core)
            self.assertIn(command, starter)
        self.assertIn("runtime_capture_config.json", core)
        self.assertIn("semantic_resolver_config.json", core)
        self.assertIn("配置目录", starter)
        self.assertIn("不负责定位接口定义", core)
        self.assertIn("source_locate", starter)
        self.assertIn("confidence", core)

        user_readme = (self.repo_root / "KID_README.md").read_text(encoding="utf-8")
        self.assertIn("framework_engineer/prompts/start_kid.md", user_readme)
        self.assertIn("runtime_capture_config.json", user_readme)
        self.assertIn("semantic_resolver_config.json", user_readme)
        self.assertIn("decomposition.schema.json", user_readme)


if __name__ == "__main__":
    unittest.main()
