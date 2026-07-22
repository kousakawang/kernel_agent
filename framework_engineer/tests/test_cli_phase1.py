from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Phase1CliEndToEndTests(unittest.TestCase):
    """Exercise the Framework Engineer CLI on a tiny pure-Python target."""

    def test_cli_probe_capture_select_generate_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_pack = tmp_path / "task_pack"
            target_file = tmp_path / "toy_kernel.py"
            workload_file = tmp_path / "workload.py"
            service_file = tmp_path / "service.py"

            self._write_target(target_file)
            self._write_workload(workload_file, tmp_path)
            service_file.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

            service_cmd = f"{sys.executable} {service_file}"
            workload_cmd = f"{sys.executable} {workload_file}"
            target_line = self._line_for(target_file, "def extend(*, values, state):")
            boundary_line = self._line_for(target_file, "def forward_window(self, values):")

            self._run_cli("scaffold-task-pack", "--task-id", "toy_extend", "--out", str(task_pack))

            baseline = self._run_cli(
                "run-baseline",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd,
                "--workload-cmd",
                workload_cmd,
                "--startup-timeout",
                "1",
            )
            self.assertEqual(baseline["status"], "ok")

            probe = self._run_cli(
                "probe-target-calls",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd,
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                str(target_file),
                "--target-line",
                str(target_line),
                "--forward-boundary-file",
                str(target_file),
                "--forward-boundary-line",
                str(boundary_line),
                "--startup-timeout",
                "1",
            )
            self.assertEqual(probe["call_count"], 6)
            self.assertEqual(probe["target_interface"]["function_name"], "extend")
            self.assertEqual(probe["target_interface"]["qualified_name"], "toy_kernel.extend")
            self.assertEqual(probe["forward_boundary_interface"]["qualified_name"], "toy_kernel.Worker.forward_window")
            probe_rows = [
                json.loads(line)
                for line in (task_pack / "docs" / "target_call_probe.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(probe_rows[0]["positional_arg_count"], 0)
            self.assertEqual(probe_rows[0]["captured_positional_arg_count"], 0)
            self.assertEqual(probe_rows[0]["kwarg_count"], 2)
            self.assertIsNotNone(probe_rows[0]["forward_id"])
            self.assertIn("--disable-cuda-graph", probe["service_cmd"])

            capture = self._run_cli(
                "capture-snapshots",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd + " --disable-cuda-graph --disable-cuda-graph",
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                str(target_file),
                "--target-line",
                str(target_line),
                "--signature",
                "candidate(*args, **kwargs)",
                "--forward-boundary-file",
                str(target_file),
                "--forward-boundary-line",
                str(boundary_line),
                "--max-capture-groups",
                "8",
                "--max-samples-per-group",
                "4",
                "--max-samples-per-forward-per-group",
                "2",
                "--startup-timeout",
                "1",
            )
            self.assertEqual(capture["raw_group_count"], 1)
            self.assertEqual(capture["raw_sample_count"], 4)
            self.assertEqual(capture["total_hit_count"], 6)
            self.assertEqual(capture["mutation_warning_count"], 0)
            self.assertEqual(capture["service_cmd"].count("--disable-cuda-graph"), 1)
            self.assertNotIn("@__import__", target_file.read_text(encoding="utf-8"))

            selected = self._run_cli(
                "select-snapshots",
                "--task-pack",
                str(task_pack),
                "--max-groups",
                "1",
                "--max-selected-samples-per-group",
                "4",
            )
            self.assertEqual(selected["selected_group_count"], 1)
            self.assertEqual(selected["selected_sample_count"], 4)

            self._run_cli("generate-harness", "--task-pack", str(task_pack))

            original_source_manifest = json.loads(
                (task_pack / "original_source" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(original_source_manifest["source_available"])
            self.assertTrue(original_source_manifest["executable"])
            self.assertEqual(original_source_manifest["target_info"]["qualified_name"], "toy_kernel.extend")
            copied_source = task_pack / original_source_manifest["copied_file"]
            self.assertTrue(copied_source.exists())
            self.assertIn("def extend", copied_source.read_text(encoding="utf-8"))

            validate = self._run_cli(
                "validate-task-pack",
                "--task-pack",
                str(task_pack),
                "--skip-env-check",
                "--run-correctness",
                "--run-benchmark",
                extra_env={
                    "DEVICE": "cpu",
                    "WARMUP": "1",
                    "REPEAT": "2",
                    "PYTHON": sys.executable,
                },
            )
            self.assertTrue(validate["valid"], validate)
            self.assertFalse(validate["errors"], validate)
            self.assertEqual(validate["original_capture_benchmark_check"]["status"], "present")
            self.assertFalse(validate["original_capture_benchmark_check"]["speedup_baseline"])

            manifest = json.loads((task_pack / "snapshots" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_group_count"], 1)
            self.assertEqual(manifest["selected_sample_count"], 4)
            self.assertIn("original_call_timing_summary", manifest["case_groups"][0])
            first_sample = manifest["case_groups"][0]["samples"][0]
            sample_meta_path = (
                task_pack
                / "snapshots"
                / "selected"
                / manifest["case_groups"][0]["group_id"]
                / "samples"
                / first_sample["sample_id"]
                / "meta.json"
            )
            sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(sample_meta["mutation"]["mutable_arg_paths"], ["kwargs.state.total"])
            self.assertEqual(sample_meta["mutation"]["detection_mode"], "auto_pre_post_diff")
            self.assertEqual(sample_meta["mutation"]["ignored_mutable_arg_paths"], [])
            timing = sample_meta["capture"]["original_call_timing"]
            self.assertEqual(timing["baseline_kind"], "captured_original_call_timing_reference")
            self.assertTrue(timing["dump_time_excluded"])
            self.assertGreaterEqual(timing["elapsed_us"], 0)

            capture_benchmark = json.loads(
                (task_pack / "docs" / "original_capture_benchmark_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_benchmark["status"], "present")
            self.assertFalse(capture_benchmark["speedup_baseline"])
            self.assertEqual(capture_benchmark["overall"]["count"], 4)

            shape_list = json.loads((task_pack / "shape_list.json").read_text(encoding="utf-8"))
            self.assertEqual(shape_list["source"], "snapshots/manifest.json")
            self.assertEqual(len(shape_list["shape_groups"]), 1)
            self.assertIn("original_call_timing_summary", shape_list["shape_groups"][0])

            resolved = self._run_cli(
                "resolve-interface",
                "--file",
                str(target_file),
                "--line",
                str(target_line),
            )
            self.assertEqual(resolved["function_name"], "extend")
            self.assertEqual(resolved["target_name"], "toy_kernel.extend")

    def test_run_phase1_multi_target_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "packs"
            target_file = tmp_path / "toy_kernel.py"
            workload_file = tmp_path / "workload.py"
            service_file = tmp_path / "service.py"
            config_file = tmp_path / "phase1_config.py"

            self._write_target(target_file)
            self._write_workload(workload_file, tmp_path)
            service_file.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

            extend_line = self._line_for(target_file, "def extend(*, values, state):")
            scale_line = self._line_for(target_file, "def scale(*, values):")
            boundary_line = self._line_for(target_file, "def forward_window(self, values):")
            config_file.write_text(
                textwrap.dedent(
                    f"""
                    task_group_id = "toy_group"
                    output_root = {str(output_root)!r}
                    service_cmd = {f"{sys.executable} {service_file}"!r}
                    workload_cmd = {f"{sys.executable} {workload_file}"!r}
                    forward_boundary_file = {str(target_file)!r}
                    forward_boundary_line = {boundary_line}
                    startup_timeout = 1
                    workload_timeout = 120
                    force = True
                    run_baseline = True
                    run_probe_env = False
                    skip_env_check = True
                    run_benchmark_smoke = False
                    validate_device = "cpu"
                    targets = [
                        {{"task_id": "extend", "target_file": {str(target_file)!r}, "target_line": {extend_line}}},
                        {{"task_id": "scale", "target_file": {str(target_file)!r}, "target_line": {scale_line}}},
                    ]
                    """
                ),
                encoding="utf-8",
            )

            validated = self._run_cli("validate-config", "--config", str(config_file))
            self.assertTrue(validated["valid"], validated)
            self.assertEqual(len(validated["targets"]), 2)

            run = self._run_cli("run-phase1", "--config", str(config_file), timeout=240)
            self.assertEqual(run["task_group_id"], "toy_group")
            self.assertTrue((output_root / "multi_target_report.json").exists())
            self.assertEqual([item["status"] for item in run["targets"]], ["ok", "ok"])
            for task_id in ("extend", "scale"):
                self.assertTrue((output_root / task_id / "correctness_test.py").exists())
                self.assertTrue((output_root / task_id / "snapshots" / "manifest.json").exists())

    def test_human_failure_output_renders_multiline_service_and_workload_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_pack = tmp_path / "task_pack"
            service_file = tmp_path / "service.py"
            workload_file = tmp_path / "workload.py"
            service_file.write_text(
                "import sys, time\n"
                "print('service stdout line 1', flush=True)\n"
                "print('service stderr line 2', file=sys.stderr, flush=True)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            workload_file.write_text(
                "import sys\n"
                "print('workload stdout line 1')\n"
                "print('workload stderr line 2', file=sys.stderr)\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "framework_engineer.cli",
                    "run-baseline",
                    "--output-format",
                    "human",
                    "--task-pack",
                    str(task_pack),
                    "--service-cmd",
                    f"{sys.executable} {service_file}",
                    "--workload-cmd",
                    f"{sys.executable} {workload_file}",
                    "--startup-timeout",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )

            self.assertEqual(proc.returncode, 1, proc)
            self.assertIn("Baseline result\nstatus: failed", proc.stdout)
            self.assertIn("| service stdout line 1", proc.stdout)
            self.assertIn("| service stderr line 2", proc.stdout)
            self.assertIn("| workload stdout line 1", proc.stdout)
            self.assertIn("| workload stderr line 2", proc.stdout)
            self.assertNotIn(r"\n", proc.stdout)

            report = json.loads((task_pack / "docs" / "baseline_result.json").read_text(encoding="utf-8"))
            self.assertIn("service stdout line 1", report["service"]["stdout"])
            self.assertIn("service stderr line 2", report["service"]["stderr"])
            self.assertEqual(report["workload"]["returncode"], 7)

            target_file = tmp_path / "target.py"
            target_file.write_text(
                "def target(value):\n"
                "    return value\n\n"
                "def forward(value):\n"
                "    return target(value)\n",
                encoding="utf-8",
            )
            output_root = tmp_path / "phase1_output"
            config_file = tmp_path / "phase1_config.py"
            config_file.write_text(
                textwrap.dedent(
                    f"""
                    output_root = {str(output_root)!r}
                    service_cmd = {f"{sys.executable} {service_file}"!r}
                    workload_cmd = {f"{sys.executable} {workload_file}"!r}
                    forward_boundary_file = {str(target_file)!r}
                    forward_boundary_line = 4
                    startup_timeout = 1
                    force = True
                    targets = [
                        {{"task_id": "target", "target_file": {str(target_file)!r}, "target_line": 1}},
                    ]
                    """
                ),
                encoding="utf-8",
            )
            phase1 = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "framework_engineer.cli",
                    "run-phase1",
                    "--output-format",
                    "human",
                    "--config",
                    str(config_file),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
            self.assertEqual(phase1.returncode, 1, phase1)
            self.assertIn("Phase 1.2 summary\nstatus: failed", phase1.stdout)
            self.assertIn("START scaffold-task-pack target=target", phase1.stderr)
            self.assertIn("FAIL  run-baseline target=group", phase1.stderr)
            self.assertIn("| workload stderr line 2", phase1.stderr)
            self.assertNotIn(r"\n", phase1.stderr)

    def test_run_phase1_uses_runtime_identity_for_external_namespace_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_root = tmp_path / "third_party"
            package_dir = package_root / "external_ns"
            package_dir.mkdir(parents=True)
            (package_dir / "constants.py").write_text("OFFSET = 7\n", encoding="utf-8")
            target_file = package_dir / "ops.py"
            target_file.write_text(
                "from .constants import OFFSET\n\n"
                "def external_target(value):\n"
                "    return value + OFFSET\n\n"
                "def forward(value):\n"
                "    return external_target(value)\n",
                encoding="utf-8",
            )
            workload_file = tmp_path / "workload.py"
            workload_file.write_text(
                "from external_ns.ops import forward\n"
                "assert forward(5) == 12\n",
                encoding="utf-8",
            )
            service_file = tmp_path / "service.py"
            service_file.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

            output_root = tmp_path / "output"
            config_file = tmp_path / "phase1_config.py"
            config_file.write_text(
                textwrap.dedent(
                    f"""
                    output_root = {str(output_root)!r}
                    service_cmd = {f"{sys.executable} {service_file}"!r}
                    workload_cmd = {f"{sys.executable} {workload_file}"!r}
                    forward_boundary_file = {str(target_file)!r}
                    forward_boundary_line = 6
                    startup_timeout = 1
                    force = True
                    validate_device = "cpu"
                    extra_env = {{"PYTHONPATH": {str(package_root)!r}}}
                    targets = [
                        {{"task_id": "external", "target_file": {str(target_file)!r}, "target_line": 3}},
                    ]
                    """
                ),
                encoding="utf-8",
            )

            run = self._run_cli("run-phase1", "--config", str(config_file), timeout=240)
            self.assertEqual(run["targets"][0]["status"], "ok", run)

            task_pack = output_root / "external"
            capture_report = json.loads(
                (task_pack / "docs" / "snapshot_capture_report.json").read_text(encoding="utf-8")
            )
            target_info = capture_report["target_interface"]
            self.assertEqual(target_info["module_name"], "external_ns.ops")
            self.assertEqual(target_info["qualified_name"], "external_ns.ops.external_target")
            self.assertEqual(target_info["identity_source"], "runtime_decorated_callable")

            original_manifest = json.loads(
                (task_pack / "original_source" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(original_manifest["executable"], original_manifest)
            self.assertEqual(original_manifest["target_info"]["module_name"], "external_ns.ops")

    def test_run_phase1_maps_local_package_definition_to_installed_runtime_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_name = "_phase1_installed_mapping_pkg"
            local_package = tmp_path / "local_checkout" / package_name
            installed_root = tmp_path / "runtime_site_packages"
            installed_package = installed_root / package_name
            local_package.mkdir(parents=True)
            installed_package.mkdir(parents=True)
            (local_package / "__init__.py").write_text("", encoding="utf-8")
            (installed_package / "__init__.py").write_text("", encoding="utf-8")

            local_target = local_package / "ops.py"
            local_target.write_text(
                '"""Local checkout with different leading lines."""\n\n'
                "LOCAL_ONLY = True\n\n\n"
                "def external_target(value):\n"
                "    return value + 7\n",
                encoding="utf-8",
            )
            configured_line = self._line_for(local_target, "return value + 7")

            runtime_target = installed_package / "ops.py"
            runtime_target.write_text(
                "def external_target(value):\n"
                "    return value + 7\n\n"
                "def forward(value):\n"
                "    return external_target(value)\n",
                encoding="utf-8",
            )
            runtime_original = runtime_target.read_text(encoding="utf-8")
            workload_file = tmp_path / "workload.py"
            workload_file.write_text(
                f"from {package_name}.ops import forward\n"
                "assert forward(5) == 12\n",
                encoding="utf-8",
            )
            service_file = tmp_path / "service.py"
            service_file.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

            output_root = tmp_path / "output"
            config_file = tmp_path / "phase1_config.py"
            config_file.write_text(
                textwrap.dedent(
                    f"""
                    output_root = {str(output_root)!r}
                    service_cmd = {f"{sys.executable} {service_file}"!r}
                    workload_cmd = {f"{sys.executable} {workload_file}"!r}
                    forward_boundary_file = {str(runtime_target)!r}
                    forward_boundary_line = 4
                    startup_timeout = 1
                    force = True
                    validate_device = "cpu"
                    extra_env = {{"PYTHONPATH": {str(installed_root)!r}}}
                    targets = [
                        {{"task_id": "external", "target_file": {str(local_target)!r}, "target_line": {configured_line}}},
                    ]
                    """
                ),
                encoding="utf-8",
            )

            validated = self._run_cli("validate-config", "--config", str(config_file))
            self.assertEqual(validated["targets"][0]["target_file"], str(runtime_target.resolve()))
            self.assertEqual(validated["targets"][0]["target_line"], 1)
            self.assertEqual(validated["targets"][0]["target_resolution"]["status"], "mapped")

            run = self._run_cli("run-phase1", "--config", str(config_file), timeout=240)
            target_report = run["targets"][0]["target"]
            self.assertEqual(target_report["configured_target_file"], str(local_target.resolve()))
            self.assertEqual(target_report["configured_target_line"], configured_line)
            self.assertEqual(target_report["target_file"], str(runtime_target.resolve()))
            self.assertEqual(target_report["target_line"], 1)
            self.assertEqual(target_report["target_resolution"]["status"], "mapped")
            self.assertTrue(target_report["target_resolution"]["mapping_applied"])

            resolution = json.loads(
                (output_root / "external" / "docs" / "target_definition_resolution.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(resolution["identity"]["module_name"], f"{package_name}.ops")
            self.assertEqual(resolution["runtime"]["file"], str(runtime_target.resolve()))
            self.assertEqual(resolution["runtime"]["line"], 1)
            self.assertEqual(runtime_target.read_text(encoding="utf-8"), runtime_original)

    def _run_cli(self, *args: str, extra_env: dict[str, str] | None = None, timeout: int = 120) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            [sys.executable, "-m", "framework_engineer.cli", *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        self.assertEqual(proc.returncode, 0, f"args={args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, "CLI produced no stdout")
        return json.loads(lines[-1])

    @staticmethod
    def _write_target(path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """
                def extend(*, values, state):
                    state["total"] += sum(values)
                    return {"out": [v + 1 for v in values], "total": state["total"]}


                def scale(*, values):
                    return {"scaled": [v * 2 for v in values]}


                class Other:
                    def forward_window(self):
                        return "not the boundary"


                class Worker:
                    def forward_window(self, values):
                        extend(values=values, state={"total": 0})
                        extend(values=values, state={"total": 0})
                        extend(values=values, state={"total": 0})
                        scale(values=values)


                _WORKER = Worker()


                def run(values):
                    _WORKER.forward_window(values)
                """
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_workload(path: Path, module_dir: Path) -> None:
        path.write_text(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(module_dir)!r})

                import toy_kernel

                toy_kernel.run([1, 2, 3])
                toy_kernel.run([1, 2, 3])
                """
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _line_for(path: Path, needle: str) -> int:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if needle in line:
                return idx
        raise AssertionError(f"missing line containing {needle!r}")


if __name__ == "__main__":
    unittest.main()
