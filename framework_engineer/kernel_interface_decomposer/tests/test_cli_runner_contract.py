from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

from framework_engineer.kernel_interface_decomposer.cli import _build_parser
from framework_engineer.kernel_interface_decomposer.runner import (
    _eager_command,
    _nsys_profile_command,
)


class TestCliRunnerContract(unittest.TestCase):
    def test_public_commands_are_capture_and_analyze_only(self) -> None:
        parser = _build_parser()
        capture = parser.parse_args(["capture", "config.json"])
        analyze = parser.parse_args(
            [
                "analyze",
                "config.json",
                "--sqlite",
                "profile.sqlite",
                "--events-dir",
                "events",
            ]
        )
        self.assertEqual(capture.command, "capture")
        self.assertEqual(analyze.command, "analyze")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "config.json"])

    def test_sglang_cuda_graph_flag_is_appended_once(self) -> None:
        command = "python -m sglang.launch_server --model-path /models/demo"
        eager = _eager_command(command)
        self.assertEqual(eager.count("--disable-cuda-graph"), 1)
        self.assertEqual(_eager_command(eager).count("--disable-cuda-graph"), 1)
        unrelated = "python workload.py"
        self.assertEqual(_eager_command(unrelated), unrelated)

    def test_nsys_systems_command_does_not_use_ncu_target_processes(self) -> None:
        config = SimpleNamespace(profiling={"nsys_bin": "nsys"})
        command = _nsys_profile_command(
            config,  # type: ignore[arg-type]
            Path("/tmp/kid-output"),
            "python workload.py",
        )
        self.assertNotIn("--target-processes=all", command)
        self.assertIn("--trace-fork-before-exec=true", command)
        self.assertEqual(command[-3:], ["bash", "-lc", "python workload.py"])


if __name__ == "__main__":
    unittest.main()
