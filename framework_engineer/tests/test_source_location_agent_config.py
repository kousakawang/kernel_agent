"""Tests for the config-driven source-locate Agent entry workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.source_location.agent_config import (
    AGENT_CONFIG_SCHEMA_VERSION,
    load_agent_config,
)
from framework_engineer.source_location.agent_helper import (
    AgentHelperError,
    prepare_agent_run,
    validate_agent_run,
)
from framework_engineer.source_location.contracts import ContractError


class AgentConfigFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="source_agent_config_")
        self.tmp = Path(self.tempdir.name)
        self.config_dir = self.tmp / "config" / "case_a"
        self.input_dir = self.tmp / "input" / "case_a"
        self.config_dir.mkdir(parents=True)
        self.input_dir.mkdir(parents=True)
        self.sglang = self.tmp / "sglang"
        self.third = self.tmp / "third"
        self.sglang.mkdir()
        self.third.mkdir()
        self.kid = self.input_dir / "decomposition.kid.schema.json"
        self.kid.write_text(
            json.dumps(
                {
                    "schema_version": "kernel-interface-decomposition/v2",
                    "kernels": [
                        {
                            "low_level_id": "demo",
                            "interface": "pkg.api",
                            "archetype": "python_binding",
                            "provider": "third",
                            "kernel": {
                                "raw_name": "raw::demo",
                                "normalized_name": "demo",
                            },
                            "runtime_event": {
                                "call_site": {
                                    "file": str(self.sglang / "caller.py"),
                                    "line": 1,
                                }
                            },
                        }
                    ],
                },
                indent=2,
            )
        )
        self.manifest = self.config_dir / "third_party_manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "repos": [
                        {
                            "name": "third",
                            "status": "ok",
                            "local_path": str(self.third),
                        }
                    ]
                }
            )
        )
        self.config = self.config_dir / "source_locate_config.json"
        self.payload = {
            "schema_version": AGENT_CONFIG_SCHEMA_VERSION,
            "testcase_id": "case_a",
            "kid_schema": "../../input/case_a/decomposition.kid.schema.json",
            "third_party_manifest": "third_party_manifest.json",
            "sglang_repo_root": str(self.sglang),
            "workspace": "../../workspaces/case_a",
        }
        self._write_config()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_config(self) -> None:
        self.config.write_text(json.dumps(self.payload, indent=2))


class TestAgentConfig(AgentConfigFixture):
    def test_relative_paths_and_derived_workspace(self) -> None:
        loaded = load_agent_config(self.config)
        self.assertEqual(loaded.testcase_id, "case_a")
        self.assertEqual(loaded.kid_schema, self.kid)
        self.assertEqual(
            loaded.run.candidate_schema,
            self.tmp
            / "workspaces"
            / "case_a"
            / "locate"
            / "locate_candidates.schema.json",
        )
        self.assertEqual(
            loaded.run.extracted_schema,
            self.tmp
            / "workspaces"
            / "case_a"
            / "extract"
            / "decomposition.extracted.schema.json",
        )

    def test_prepare_run_validates_inputs_and_creates_stage_dirs(self) -> None:
        report = prepare_agent_run(self.config)
        loaded = load_agent_config(self.config)
        self.assertEqual(report["kernels"], ["demo"])
        self.assertEqual(report["search_roots_skipped"], [])
        self.assertTrue(loaded.run.locate_dir.is_dir())
        self.assertTrue(loaded.run.notes.parent.is_dir())
        self.assertTrue(loaded.run.extract_dir.is_dir())

    def test_config_rejects_unknown_fields(self) -> None:
        self.payload["candidate_schema"] = "manual-output.json"
        self._write_config()
        with self.assertRaisesRegex(ContractError, "extra"):
            load_agent_config(self.config)

    def test_prepare_rejects_workspace_inside_source_root(self) -> None:
        self.payload["workspace"] = str(self.sglang / "source_locate_output")
        self._write_config()
        with self.assertRaisesRegex(AgentHelperError, "inside source root"):
            prepare_agent_run(self.config)


class TestRealConfigDrivenGolden(unittest.TestCase):
    def test_all_backends_workspace_validates_from_one_config(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        config = (
            repo
            / "example_kernels"
            / "source_locate_golden"
            / "config"
            / "all_backends"
            / "source_locate_config.json"
        )
        report = validate_agent_run(config)
        self.assertTrue(report["ok"])
        self.assertEqual(report["testcase_id"], "all_backends")
        self.assertEqual(report["kernels"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
