from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from framework_engineer.third_party_solver import cloner, manifest as manifest_mod, version_resolver
from framework_engineer.third_party_solver.cmake_pins import (
    parse_cmake_text,
    parse_sgl_kernel_pins,
    read_sgl_kernel_src_version,
    sgl_kernel_src_root,
)
from framework_engineer.third_party_solver.flags import (
    annotate_universe,
    backends_from_service_cmds,
)
from framework_engineer.third_party_solver.registry import get_spec, iter_universe


SGLANG_ROOT = Path("/Users/bytedance/Desktop/infra_agent/sglang")


class CMakePinTests(unittest.TestCase):
    def test_parse_declare_block(self):
        text = """
        FetchContent_Declare(
            repo-flash-attention
            URL      https://${GITHUB_ARTIFACTORY}/sgl-project/sgl-attn/archive/bcf72ccc6816b36a5fae2c5a3c027604629785e0.tar.gz
            URL_HASH SHA256=2110d8ca1ed9b330b9f99c1d4088be12a37e44152c39fdf90dfb6526f532ba96
        )
        """
        pins = parse_cmake_text(text)
        self.assertIn("repo-flash-attention", pins)
        pin = pins["repo-flash-attention"]
        self.assertEqual(pin.owner, "sgl-project")
        self.assertEqual(pin.repo, "sgl-attn")
        self.assertEqual(pin.commit, "bcf72ccc6816b36a5fae2c5a3c027604629785e0")
        self.assertEqual(pin.url, "https://github.com/sgl-project/sgl-attn")
        self.assertTrue(pin.sha256)

    def test_parse_tag_ref(self):
        text = """
        FetchContent_Declare(
            repo-triton
            URL https://${GITHUB_ARTIFACTORY}/triton-lang/triton/archive/v3.6.0.tar.gz
        )
        """
        pin = parse_cmake_text(text)["repo-triton"]
        self.assertEqual(pin.commit, "v3.6.0")

    @unittest.skipUnless(SGLANG_ROOT.exists(), "needs real sglang tree")
    def test_real_sgl_kernel_pins(self):
        src = sgl_kernel_src_root(SGLANG_ROOT)
        pins = parse_sgl_kernel_pins(src)
        self.assertEqual(
            pins["repo-flash-attention"].commit,
            "bcf72ccc6816b36a5fae2c5a3c027604629785e0",
        )
        self.assertEqual(pins["repo-flashmla"].owner, "sgl-project")
        self.assertEqual(pins["repo-flashmla"].repo, "FlashMLA")
        self.assertEqual(pins["repo-cutlass"].owner, "NVIDIA")
        self.assertEqual(read_sgl_kernel_src_version(src), "0.4.3")


class FlagsTests(unittest.TestCase):
    def test_backend_values_parsed(self):
        cmds = [
            {"backend_name": "triton", "cmd": "x --linear-attn-backend triton"},
            {"backend_name": "fi", "cmd": "x --linear-attn-backend=flashinfer --attention-backend fa3"},
        ]
        backends = backends_from_service_cmds(cmds)
        self.assertEqual(backends["triton"], {"triton"})
        self.assertEqual(backends["fi"], {"flashinfer", "fa3"})

    def test_default_path_always_tagged_and_no_pruning(self):
        cmds = [{"backend_name": "fi", "cmd": "x --attention-backend flashinfer"}]
        ann = annotate_universe(cmds)
        # Every source-bearing repo present (no pruning by flags).
        source_names = {s.name for s in iter_universe(source_bearing_only=True)}
        self.assertEqual(set(ann.keys()), source_names)
        # Default-path libs carry the default_path tag even if not named.
        self.assertIn("default_path", ann["flashinfer"])
        self.assertIn("default_path", ann["cutlass"])
        # flashinfer flag attributes the flashinfer repo to that backend command.
        self.assertIn("fi", ann["flashinfer"])

    def test_f8_excluded_from_source_bearing(self):
        names = {s.name for s in iter_universe(source_bearing_only=True)}
        self.assertNotIn("flashinfer_cubin", names)


class VersionResolverTests(unittest.TestCase):
    def _fake_lookup(self, mapping):
        def lookup(dist):
            if dist in mapping:
                return mapping[dist]
            raise version_resolver.importlib_metadata.PackageNotFoundError(dist)

        return lookup

    @unittest.skipUnless(SGLANG_ROOT.exists(), "needs real sglang tree")
    def test_bucket_a_and_b(self):
        src = sgl_kernel_src_root(SGLANG_ROOT)
        lookup = self._fake_lookup(
            {"flashinfer_python": "0.6.12", "sgl_kernel": "0.4.3", "fla-core": "0.2.7"}
        )
        resolutions, meta = version_resolver.resolve_all(
            sgl_kernel_src=src,
            triggered_by_map={},
            version_lookup=lookup,
        )
        by_name = {r.name: r for r in resolutions}
        # Bucket A
        self.assertEqual(by_name["flashinfer"].version, "0.6.12")
        self.assertEqual(by_name["flashinfer"].ref, "v0.6.12")
        # Bucket B: pin commit, sgl_fork classification
        self.assertEqual(
            by_name["flash_attn"].ref, "bcf72ccc6816b36a5fae2c5a3c027604629785e0"
        )
        self.assertEqual(by_name["flash_attn"].clone_source, "sgl_fork")
        self.assertEqual(by_name["cutlass"].clone_source, "official")
        # alignment ok: src 0.4.3 == installed 0.4.3
        self.assertFalse(meta["sgl_kernel_version_mismatch"])

    @unittest.skipUnless(SGLANG_ROOT.exists(), "needs real sglang tree")
    def test_version_mismatch_reported_not_fixed(self):
        src = sgl_kernel_src_root(SGLANG_ROOT)
        lookup = self._fake_lookup({"sgl_kernel": "9.9.9"})  # != source 0.4.3
        resolutions, meta = version_resolver.resolve_all(
            sgl_kernel_src=src, triggered_by_map={}, version_lookup=lookup
        )
        self.assertTrue(meta["sgl_kernel_version_mismatch"])
        fa = next(r for r in resolutions if r.name == "flash_attn")
        self.assertTrue(fa.version_mismatch)
        # still resolves the pin (reports, does not block)
        self.assertTrue(fa.ref)

    def test_missing_package_sets_error(self):
        spec = get_spec("deep_gemm")
        res = version_resolver.resolve_repo(
            spec,
            sgl_kernel_pins={},
            version_lookup=self._fake_lookup({}),
            triggered_by=[],
            sgl_kernel_version_mismatch=False,
        )
        self.assertIsNone(res.version)
        self.assertIn("not installed", res.resolve_error)

    def test_f8_no_source(self):
        spec = get_spec("flashinfer_cubin")
        res = version_resolver.resolve_repo(
            spec,
            sgl_kernel_pins={},
            version_lookup=self._fake_lookup({}),
            triggered_by=[],
            sgl_kernel_version_mismatch=False,
        )
        self.assertFalse(res.has_source)
        self.assertIn("no source", res.resolve_error)

    def test_fa4_ref_derives_beta_tag(self):
        # flash_attn_4 pip 4.0.0b17 -> git tag fa4-v4.0.0.beta17 (derived, not None).
        spec = get_spec("flash_attn_4")
        res = version_resolver.resolve_repo(
            spec,
            sgl_kernel_pins={},
            version_lookup=self._fake_lookup({"flash-attn-4": "4.0.0b17"}),
            triggered_by=[],
            sgl_kernel_version_mismatch=False,
        )
        self.assertEqual(res.version, "4.0.0b17")
        self.assertEqual(res.ref, "fa4-v4.0.0.beta17")

    def test_deep_gemm_ref_uses_version_tag(self):
        # deep_gemm uses the default template -> v0.1.2 (a real tag on the sgl fork).
        res = version_resolver.resolve_repo(
            get_spec("deep_gemm"),
            sgl_kernel_pins={},
            version_lookup=self._fake_lookup({"sgl-deep-gemm": "0.1.2"}),
            triggered_by=[],
            sgl_kernel_version_mismatch=False,
        )
        self.assertEqual(res.ref, "v0.1.2")
        self.assertEqual(res.clone_source, "sgl_fork")


class ClonerTests(unittest.TestCase):
    def _res(self, **kw):
        base = dict(
            name="flash_attn",
            archetype="F3",
            version_source="cmake_pin",
            on_default_path=False,
            url_kind="sgl_fork",
            version="bcf72cc",
            url="https://github.com/sgl-project/sgl-attn",
            ref="bcf72ccc6816",
            clone_source="sgl_fork",
            has_source=True,
        )
        base.update(kw)
        return version_resolver.RepoResolution(**base)

    def test_dry_run_records_command_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = cloner.clone_repo(
                self._res(),
                cache_root=Path(tmp) / "cache",
                sgl_kernel_src=Path(tmp) / "nope",
                explicit_paths={},
                dry_run=True,
            )
        self.assertEqual(outcome.status, "clone_failed")
        self.assertIsNone(outcome.local_path)
        self.assertIn("git clone", outcome.clone_command)
        self.assertIn("bcf72ccc6816", outcome.clone_command)

    def test_real_clone_failure_records_only(self):
        # Unreachable URL -> failure must be recorded, not raised, not retried.
        res = self._res(url="https://invalid.invalid/nope/nope", ref="deadbeef")
        with tempfile.TemporaryDirectory() as tmp:
            outcome = cloner.clone_repo(
                res,
                cache_root=Path(tmp) / "cache",
                sgl_kernel_src=Path(tmp) / "nope",
                explicit_paths={},
                dry_run=False,
                clone_timeout=20,
            )
        self.assertEqual(outcome.status, "clone_failed")
        self.assertIsNone(outcome.local_path)
        self.assertTrue(outcome.clone_command)
        self.assertTrue(outcome.error)

    def test_explicit_path_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mysrc"
            src.mkdir()
            (src / "foo.py").write_text("x = 1\n")
            outcome = cloner.clone_repo(
                self._res(),
                cache_root=Path(tmp) / "cache",
                sgl_kernel_src=Path(tmp) / "nope",
                explicit_paths={"flash_attn": str(src)},
                dry_run=False,
            )
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.resolution, "explicit")
        self.assertEqual(outcome.local_path, str(src))

    def test_never_resolves_to_installed(self):
        # Option A: we always clone the pinned git source; a same-named installed
        # package (e.g. cutlass -> nvidia_cutlass_dsl) must never be used. With no
        # P1/P2 and dry-run, it falls through to a clone_command, not "installed".
        with tempfile.TemporaryDirectory() as tmp:
            res = self._res(name="cutlass", version_source="cmake_pin",
                            url="https://github.com/NVIDIA/cutlass", ref="57e3cfb")
            outcome = cloner.clone_repo(
                res,
                cache_root=Path(tmp) / "cache",
                sgl_kernel_src=Path(tmp) / "nope",
                explicit_paths={},
                dry_run=True,
            )
        self.assertNotEqual(outcome.resolution, "installed")
        self.assertEqual(outcome.status, "clone_failed")  # dry-run -> clone_command only
        self.assertIn("git clone", outcome.clone_command)


class ManifestTests(unittest.TestCase):
    def test_manifest_and_missing_render(self):
        res_ok = version_resolver.RepoResolution(
            name="fla", archetype="F6", version_source="importlib",
            on_default_path=False, url_kind="official", version="0.2.7",
            url="https://github.com/fla-org/flash-linear-attention", ref="v0.2.7",
            clone_source="official",
        )
        out_ok = cloner.CloneOutcome(status="ok", resolution="cloned", local_path="/c/fla/0.2.7")
        res_fail = version_resolver.RepoResolution(
            name="flash_attn", archetype="F3", version_source="cmake_pin",
            on_default_path=False, url_kind="sgl_fork", version="bcf72cc",
            url="https://github.com/sgl-project/sgl-attn", ref="bcf72cc",
            clone_source="sgl_fork",
        )
        out_fail = cloner.CloneOutcome(
            status="clone_failed", resolution="none",
            clone_command="git clone ...", error="network",
        )
        records = [
            manifest_mod.build_record(res_ok, out_ok),
            manifest_mod.build_record(res_fail, out_fail),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            m = manifest_mod.build_manifest(
                sglang_repo_root=Path("/repo"),
                third_party_cache=Path("/c"),
                meta={"sgl_kernel_installed_version": "0.4.3",
                      "sgl_kernel_source_version": "0.4.3",
                      "sgl_kernel_version_mismatch": False},
                records=records,
            )
            mpath = Path(tmp) / "third_party_manifest.json"
            manifest_mod.write_manifest(m, mpath)
            data = json.loads(mpath.read_text())
            self.assertEqual(len(data["repos"]), 2)
            self.assertEqual(data["repos"][0]["status"], "ok")
            self.assertEqual(data["repos"][0]["local_path"], "/c/fla/0.2.7")
            self.assertIsNone(data["repos"][1]["local_path"])
            self.assertEqual(len(data["failed"]), 1)

            missing = Path(tmp) / "missing_repos.md"
            wrote = manifest_mod.write_missing_repos(m, missing)
            self.assertTrue(wrote)
            body = missing.read_text()
            self.assertIn("flash_attn", body)
            self.assertIn("git clone", body)


if __name__ == "__main__":
    unittest.main()
