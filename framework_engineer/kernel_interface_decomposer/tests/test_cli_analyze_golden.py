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
    RuntimeArtifactValidator,
)


MODULE = "framework_engineer.kernel_interface_decomposer"


def _portable_path(value: str) -> str:
    """Keep a stable repository/package suffix instead of a machine-local prefix."""

    normalized = value.replace("\\", "/")
    for marker in ("/kernel_agent/", "/site-packages/", "/dist-packages/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    return Path(normalized).name


def _normalize_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only run-local identity and filesystem-prefix differences."""

    normalized = copy.deepcopy(payload)
    normalized.pop("run", None)

    target = normalized.get("target", {})
    if target.get("file"):
        target["file"] = _portable_path(str(target["file"]))

    diagnostics = normalized.get("diagnostics", {})
    if "capture_event_files" in diagnostics:
        diagnostics["capture_event_files"] = sorted(
            Path(str(path)).name for path in diagnostics["capture_event_files"]
        )

    return normalized


class TestAnalyzeCliGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.golden_root = cls.repo_root / "example_kernels/nsys_poc_kid_golden"
        cls.golden_cli = cls.golden_root / "cli_log/nsys_poc"
        cls.golden_config_path = (
            cls.golden_root / "config/nsys_poc/runtime_capture_config.json"
        )
        cls.expected = json.loads(
            (cls.golden_cli / "runtime_capture.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_analyze_cli_reproduces_runtime_golden(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kid-analyze-golden-") as tempdir:
            temp_root = Path(tempdir)
            output_dir = temp_root / "nsys_poc"
            config_path = temp_root / "runtime_capture_config.json"

            config = json.loads(self.golden_config_path.read_text(encoding="utf-8"))
            config["workdir"] = str(self.repo_root.parent)
            config["output_dir"] = str(output_dir)
            config["target"]["file"] = str(
                self.repo_root
                / "framework_engineer/kernel_interface_decomposer/nsys_poc.py"
            )
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (str(self.repo_root), existing_pythonpath)
                if value
            )
            command = [
                sys.executable,
                "-m",
                MODULE,
                "analyze",
                str(config_path),
                "--sqlite",
                str(self.golden_cli / "trace/profile.sqlite"),
                "--events-dir",
                str(self.golden_cli / "capture_events"),
            ]
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

            cli_report = json.loads(completed.stdout)
            self.assertEqual(cli_report["backend_name"], "nsys_poc")
            self.assertEqual(cli_report["selected_invocations"], 1)
            self.assertEqual(cli_report["kernels"], 12)
            self.assertEqual(cli_report["raw_capture_events"], 36)

            actual_path = output_dir / "runtime_capture.schema.json"
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
            self.assertRegex(actual["run"]["run_id"], r"^[0-9a-f]{32}$")
            self.assertGreater(actual["run"]["created_at_unix"], 0)
            self.assertEqual(
                _normalize_runtime(actual),
                _normalize_runtime(self.expected),
            )

            validator = RuntimeArtifactValidator(output_dir)
            self.assertTrue(validator.validate(), msg="; ".join(validator.errors))
            self.assertEqual(
                (output_dir / "trace/profile.sqlite").stat().st_size,
                (self.golden_cli / "trace/profile.sqlite").stat().st_size,
            )
            self.assertEqual(
                sorted(
                    path.name
                    for path in (output_dir / "capture_events").glob("*.jsonl")
                ),
                sorted(
                    path.name
                    for path in (self.golden_cli / "capture_events").glob("*.jsonl")
                ),
            )

    def test_analyze_default_does_not_copy_or_delete_external_sqlite(self) -> None:
        source_sqlite = self.golden_cli / "trace/profile.sqlite"
        with tempfile.TemporaryDirectory(prefix="kid-analyze-retention-") as tempdir:
            temp_root = Path(tempdir)
            output_dir = temp_root / "nsys_poc"
            config_path = temp_root / "runtime_capture_config.json"
            config = json.loads(self.golden_config_path.read_text(encoding="utf-8"))
            config["workdir"] = str(self.repo_root.parent)
            config["output_dir"] = str(output_dir)
            config["target"]["file"] = str(
                self.repo_root
                / "framework_engineer/kernel_interface_decomposer/nsys_poc.py"
            )
            config["profiling"].pop("trace_retention", None)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "analyze",
                    str(config_path),
                    "--sqlite",
                    str(source_sqlite),
                    "--events-dir",
                    str(self.golden_cli / "capture_events"),
                ],
                cwd=self.repo_root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        value
                        for value in (str(self.repo_root), os.environ.get("PYTHONPATH"))
                        if value
                    ),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            runtime = json.loads(
                (output_dir / "runtime_capture.schema.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("sqlite", runtime["artifacts"])
            self.assertFalse((output_dir / "trace/profile.sqlite").exists())
            self.assertTrue(source_sqlite.is_file())
            validator = RuntimeArtifactValidator(output_dir)
            self.assertTrue(validator.validate(), msg="; ".join(validator.errors))


if __name__ == "__main__":
    unittest.main()
