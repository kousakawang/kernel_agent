from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from framework_engineer.kernel_interface_decomposer.cli import _build_parser
from framework_engineer.kernel_interface_decomposer.runner import (
    _direct_launcher_command,
    _eager_command,
    emit_progress,
    _nsys_launch_command,
    _nsys_start_command,
    _nsys_stop_command,
)


class TestCliRunnerContract(unittest.TestCase):
    def test_progress_uses_stderr_and_keeps_stdout_machine_readable(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            emit_progress("testing progress")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("[KID ", stderr.getvalue())
        self.assertIn("testing progress", stderr.getvalue())

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

    def test_direct_mode_uses_a_two_phase_launcher(self) -> None:
        config = SimpleNamespace(
            workdir=Path("/tmp/work"), test_command="python workload.py"
        )
        command = _direct_launcher_command(
            config,  # type: ignore[arg-type]
            Path("/tmp/kid-output"),
            30,
        )
        self.assertIn("direct_launcher", command)
        self.assertIn("python workload.py", command)
        self.assertIn("warmup.ready", command)
        self.assertIn("recording.enabled", command)

    def test_service_uses_paused_interactive_nsys_session(self) -> None:
        config = SimpleNamespace(profiling={"nsys_bin": "nsys"})
        launch = _nsys_launch_command(
            config,  # type: ignore[arg-type]
            "KIDsession",
            "python server.py",
        )
        start = _nsys_start_command(
            config,  # type: ignore[arg-type]
            Path("/tmp/kid-output"),
            "KIDsession",
        )
        stop = _nsys_stop_command(
            config,  # type: ignore[arg-type]
            "KIDsession",
        )
        self.assertEqual(launch[1], "launch")
        self.assertIn("--session-new=KIDsession", launch)
        self.assertIn("--trace=cuda,nvtx", launch)
        self.assertNotIn("osrt", " ".join(launch))
        self.assertNotIn("--target-processes=all", launch)
        self.assertNotIn("profile", launch)
        self.assertEqual(start[1], "start")
        self.assertIn("--session=KIDsession", start)
        self.assertIn("--output=/tmp/kid-output/_profile", start)
        self.assertEqual(stop, ["nsys", "stop", "--session=KIDsession"])

    def test_direct_launcher_runs_warmup_then_profiled_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "workload.py"
            counter = root / "counter.txt"
            script.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "path = Path(sys.argv[1])\n"
                "value = int(path.read_text() or '0') if path.exists() else 0\n"
                "path.write_text(str(value + 1))\n"
                "print(f'run={value + 1}')\n",
                encoding="utf-8",
            )
            warmup_ready = root / "warmup.ready"
            gate = root / "recording.enabled"
            done = root / "test.done.json"
            shutdown = root / "shutdown.requested"
            command = shlex.join([sys.executable, str(script), str(counter)])
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "framework_engineer.kernel_interface_decomposer.direct_launcher",
                    "--command",
                    command,
                    "--workdir",
                    str(root),
                    "--warmup-log",
                    str(root / "warmup.log"),
                    "--test-log",
                    str(root / "test.log"),
                    "--warmup-ready-file",
                    str(warmup_ready),
                    "--recording-gate-file",
                    str(gate),
                    "--test-done-file",
                    str(done),
                    "--shutdown-file",
                    str(shutdown),
                    "--timeout-sec",
                    "10",
                ],
                env=os.environ.copy(),
            )

            deadline = time.monotonic() + 10
            while not warmup_ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(warmup_ready.exists())
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

            gate.touch()
            deadline = time.monotonic() + 10
            while not done.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(done.exists())
            self.assertEqual(
                json.loads(done.read_text(encoding="utf-8")),
                {"phase": "test", "returncode": 0},
            )
            self.assertEqual(counter.read_text(encoding="utf-8"), "2")
            shutdown.touch()
            self.assertEqual(process.wait(timeout=10), 0)
            self.assertIn("run=1", (root / "warmup.log").read_text())
            self.assertIn("run=2", (root / "test.log").read_text())


if __name__ == "__main__":
    unittest.main()
