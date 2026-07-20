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
                / "ref/nsys_poc/semantic_resolver_decisions.json"
            ).read_text(encoding="utf-8")
        )
        cls.golden_final = json.loads(
            (
                cls.golden
                / "output/nsys_poc/decomposition.schema.json"
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
        backend_ref = root / "ref/nsys_poc"
        backend_output = root / "output/nsys_poc"
        backend_ref.mkdir(parents=True)
        backend_output.mkdir(parents=True)
        config = copy.deepcopy(self.golden_config)
        if runtime is not None:
            runtime_path = root / "runtime_capture.schema.json"
            runtime_path.write_text(
                json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
            )
            config["runtime_capture"] = str(runtime_path)
        config["context_output"] = str(backend_ref / "semantic_resolver_context.json")
        config["decisions_output"] = str(backend_ref / "semantic_resolver_decisions.json")
        config["notes_output"] = str(backend_ref / "kid_semantic_resolver_notes.md")
        config["output"] = str(backend_output / "decomposition.schema.json")
        decisions_value = decisions or copy.deepcopy(self.golden_decisions)
        Path(config["decisions_output"]).write_text(
            json.dumps(decisions_value, indent=2) + "\n", encoding="utf-8"
        )
        Path(config["notes_output"]).write_text(
            notes
            if notes is not None
            else (
                self.golden / "ref/nsys_poc/kid_semantic_resolver_notes.md"
            ).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        config_path = root / "semantic_resolver_config.json"
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        return config_path, SemanticResolverConfig.load(config_path)


class TestSemanticResolverGolden(SemanticResolverFixture):
    def test_complete_artifact_validator_accepts_semantic_golden(self) -> None:
        validator = ArtifactValidator(self.golden)
        self.assertTrue(validator.validate(), msg="; ".join(validator.errors))

    def test_prepare_context_has_owner_evidence_and_frozen_source(self) -> None:
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
                if item["call_site"]["line"] == 810
            )
            self.assertEqual(edge["call_expression"], "torch.matmul(a, b)")
            self.assertTrue(edge["analysis_file"].endswith("nsys_poc.trace_snapshot.json"))
            serialized = json.dumps(context)
            self.assertNotIn("low_level_id", serialized)
            self.assertNotIn("normalized_kernel_name", serialized)

    def test_python_source_override_uses_ast_call_expression(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-semantic-ast-") as temporary:
            root = Path(temporary)
            analysis_source = root / "trace_source.py"
            analysis_source.write_text(
                "\n" * 809 + "value = torch.matmul(a, b)\n", encoding="utf-8"
            )
            config_path, _ = self.make_workspace(root)
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw["analysis_source_overrides"][0]["analysis_path"] = str(
                analysis_source
            )
            config_path.write_text(
                json.dumps(raw, indent=2) + "\n", encoding="utf-8"
            )
            context = SemanticResolver(
                SemanticResolverConfig.load(config_path)
            ).prepare_context()
            matmul = next(
                capture
                for capture in context["invocations"][0]["owner_captures"]
                if capture["member_ref"]["capture_id"] == "1"
            )
            edge = next(
                item for item in matmul["stack_edges"] if item["call_site"]["line"] == 810
            )
            self.assertEqual(edge["call_expression"], "torch.matmul(a, b)")

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
                / "cli_log/nsys_poc/runtime_capture.schema.json"
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
                / "cli_log/nsys_poc/runtime_capture.schema.json"
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
