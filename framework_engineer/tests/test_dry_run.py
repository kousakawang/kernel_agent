"""Lightweight unit tests for the dry-run mechanism. CPU-only, no GPU/profiling."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.dry_run import cli as dr_cli
from framework_engineer.dry_run import kid_dryrun, locate_dryrun, templates
from framework_engineer.dry_run.fill_scan import scan_file


@contextlib.contextmanager
def _quiet():
    """Silence a CLI's stdout/stderr so unittest output stays clean."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _write_config(tmp: Path, kernels_per_backend: int = 3) -> Path:
    cfg = tmp / "cfg.py"
    cfg.write_text(
        "service_cmds = [{'backend_name': 'triton', 'cmd': 'x'}, "
        "{'backend_name': 'flashinfer', 'cmd': 'y'}]\n"
        "target = {'file': '/some/file.py', 'line': 78}\n"
        f"output_root = '{tmp / 'out'}'\n"
        f"kernels_per_backend = {kernels_per_backend}\n"
    )
    return cfg


class TestKidDryRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dr_ut_"))

    def test_generates_one_schema_per_backend(self) -> None:
        cfg = kid_dryrun.KidDryRunConfig.load(_write_config(self.tmp, 3))
        result = kid_dryrun.run(cfg)
        self.assertEqual(len(result.schemas), 2)
        for sp in result.schemas:
            s = json.loads(sp.read_text())
            self.assertTrue(s["dry_run"])
            self.assertEqual(len(s["kernels"]), 3)
            # target carried, not a placeholder
            self.assertEqual(s["target"]["line"], 78)
            # interface/archetype are placeholders needing fill
            self.assertIn("<FILL", s["kernels"][0]["interface"])
            self.assertIn("<FILL", s["kernels"][0]["archetype"])
            # no source_locations yet (locate adds it)
            self.assertNotIn("source_locations", s["kernels"][0])

    def test_fill_scan_finds_interface_and_archetype(self) -> None:
        cfg = kid_dryrun.KidDryRunConfig.load(_write_config(self.tmp, 1))
        sp = kid_dryrun.run(cfg).schemas[0]
        keys = {fp.key for fp in scan_file(sp)}
        self.assertIn("interface", keys)
        self.assertIn("archetype", keys)


class TestLocateDryRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dr_ut_"))

    def _kid_schema(self, archetype: str) -> Path:
        cfg = kid_dryrun.KidDryRunConfig.load(_write_config(self.tmp, 1))
        sp = kid_dryrun.run(cfg).schemas[0]
        s = json.loads(sp.read_text())
        k = s["kernels"][0]
        k["low_level_id"] = "k0"
        k["interface"] = "iface"
        k["archetype"] = archetype
        k["archetype_code"] = templates.ARCHETYPES[archetype]["code"]
        s["kernels"] = [k]
        sp.write_text(json.dumps(s, indent=2, ensure_ascii=False))
        return sp

    def test_null_rules_applied_for_triton(self) -> None:
        sp = self._kid_schema("sglang_triton")
        ok, frag = locate_dryrun.run_one(sp)
        self.assertTrue(ok)
        s = json.loads(sp.read_text())
        layers = s["kernels"][0]["source_locations"]["layers"]
        # a/b missed (need fill), c/d not_applicable (auto)
        self.assertEqual(layers["interface_definition"]["status"], "missed")
        self.assertEqual(layers["kernel_impl"]["status"], "missed")
        self.assertEqual(layers["py_cpp_binding"]["status"], "not_applicable")
        self.assertEqual(layers["kernel_header"]["status"], "not_applicable")

    def test_all_four_applicable_for_sgl_kernel(self) -> None:
        sp = self._kid_schema("sgl_kernel_builtin")
        locate_dryrun.run_one(sp)
        layers = json.loads(sp.read_text())["kernels"][0]["source_locations"]["layers"]
        for name in ("interface_definition", "kernel_impl", "py_cpp_binding", "kernel_header"):
            self.assertEqual(layers[name]["status"], "missed")

    def test_gate_blocks_when_kid_fields_unfilled(self) -> None:
        # Fresh KID schema with interface/archetype still placeholders.
        cfg = kid_dryrun.KidDryRunConfig.load(_write_config(self.tmp, 1))
        sp = kid_dryrun.run(cfg).schemas[0]
        ok, frag = locate_dryrun.run_one(sp)
        self.assertFalse(ok)
        self.assertIn("blocked_on", frag)


class TestDryRunCliGates(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dr_ut_"))

    def test_locate_cli_returns_2_on_gate(self) -> None:
        cfg = _write_config(self.tmp, 1)
        with _quiet():
            dr_cli.main(["kid", "--config", str(cfg), "--out", str(self.tmp / "out")])
            rc = dr_cli.main(["locate", "--workspace", str(self.tmp / "out" / "workspaces")])
        self.assertEqual(rc, 2)  # KID fields unfilled -> gate


if __name__ == "__main__":
    unittest.main(verbosity=2)
