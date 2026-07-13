"""Hermetic UT for KID's instrumentation *capture* layer.

Goal (see .trae/documents/kid_instrumentation_ut_plan.md): verify — WITHOUT a GPU
and WITHOUT real sglang/torch/triton/flashinfer — that KID's runtime patches
correctly *capture* each kernel archetype. We feed fake module objects to the
real patch functions in ``runtime_instrumentation`` and read back the events
written to ``events/*.jsonl``.

Why this is CPU-only and hermetic:
  * ``_record_event`` is a pure file write.
  * ``_nvtx_push/_nvtx_pop`` import torch lazily and no-op when CUDA is absent,
    so event recording is unaffected.
  * A regular (non-target) wrapper only records while a target context
    (``_CURRENT``) is active — so tests that exercise wrap events run the fake
    kernel *inside* a target context.

Run (either form):
    cd kernel_agent
    python3 -m unittest framework_engineer.kernel_interface_decomposer.tests.test_instrumentation_capture -v
    # or by file path:
    python3 framework_engineer/kernel_interface_decomposer/tests/test_instrumentation_capture.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Import the KID instrumentation module. Normal path is the package import; the
# by-file-path fallback keeps the test runnable even when the package root is not
# on sys.path (e.g. invoked directly as a script from an odd cwd).
try:
    from framework_engineer.kernel_interface_decomposer import runtime_instrumentation as ri
except Exception:  # noqa: BLE001 — fall back to direct file load
    _KID_DIR = Path(__file__).resolve().parents[1]  # .../framework_engineer/kernel_interface_decomposer
    _spec = importlib.util.spec_from_file_location(
        "kid_runtime_instrumentation", _KID_DIR / "runtime_instrumentation.py"
    )
    assert _spec and _spec.loader
    ri = importlib.util.module_from_spec(_spec)
    sys.modules["kid_runtime_instrumentation"] = ri
    _spec.loader.exec_module(ri)


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------
class _CaptureBase(unittest.TestCase):
    """Temp output_dir + global-state reset + events reader + summary printer."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kid_ut_")
        # Reset the module-level singletons so tests do not leak into each other.
        ri._INSTALLED = False
        ri._CALL_COUNTER = 0
        ri._CURRENT.set(None)
        ri._CONFIG = {
            "output_dir": self.tmp,
            "workdir": self.tmp,
            "target": {"file": "", "line": 0},
            "resolution": {"third_party_prefixes": ["myflash"]},
        }

    def events(self) -> list[dict]:
        out: list[dict] = []
        events_dir = Path(self.tmp, "events")
        if not events_dir.exists():
            return out
        for p in sorted(events_dir.glob("*.jsonl")):
            out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return out

    def events_of(self, event_type: str) -> list[dict]:
        return [e for e in self.events() if e.get("event") == event_type]

    def dump(self, label: str) -> None:
        """Human-readable summary for eyeballing the log.

        ``target_wrap_failed`` is filtered out: most cases set no real target
        file, so ``_instrument_module`` records an expected, harmless failure
        that would otherwise clutter the review log.
        """
        print(f"\n=== [{label}] captured events ===")
        for e in self.events():
            if e.get("event") == "target_wrap_failed":
                continue
            impl = e.get("implementation") or {}
            print(
                "   {event:18} | category={category} | impl_kind={kind} | src={src}".format(
                    event=str(e.get("event")),
                    category=str(e.get("category")),
                    kind=str(impl.get("kind")),
                    src=str(impl.get("source_files")),
                )
            )

    @contextlib.contextmanager
    def target_ctx(self, call_id: int = 1, stage: str = "prefill"):
        """Enter a fake target context so regular wrappers record events."""
        token = ri._CURRENT.set(
            {"call_id": call_id, "stage": stage, "forward_mode": "EXTEND", "target_api": "t"}
        )
        try:
            yield
        finally:
            ri._CURRENT.reset(token)


def _load_module_from_source(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# (T) target wrap  — example: _layer_norm_fwd  (mirrors layernorm_gated.py:205)
# ---------------------------------------------------------------------------
class TestTargetWrap(_CaptureBase):
    def _make_target_module(self) -> tuple[ModuleType, int]:
        src = "def _layer_norm_fwd(x):\n    return x\n"
        path = Path(self.tmp, "faketarget.py")
        path.write_text(src)
        module = _load_module_from_source(path, "kid_ut_faketarget")
        target_line = 1  # `def _layer_norm_fwd` is on line 1
        ri._CONFIG["target"] = {"file": str(path.resolve()), "line": target_line}
        return module, target_line

    def test_target_is_wrapped_and_records_begin_end(self) -> None:
        module, _ = self._make_target_module()
        ri._instrument_module(module)

        self.assertTrue(
            getattr(module._layer_norm_fwd, "_kid_target_wrapped", False),
            "target function should be wrapped",
        )
        # calling it should emit target_begin/target_end
        module._layer_norm_fwd(1)
        self.dump("target-wrap")

        wrapped = self.events_of("target_wrapped")
        self.assertTrue(wrapped, "expected a target_wrapped event")
        self.assertFalse(self.events_of("target_wrap_failed"), "unexpected target_wrap_failed")

        begins = self.events_of("target_begin")
        ends = self.events_of("target_end")
        self.assertTrue(begins and ends, "expected target_begin and target_end")
        self.assertTrue(str(begins[0].get("api", "")).endswith("_layer_norm_fwd"))
        self.assertTrue(str(begins[0].get("file", "")).endswith("faketarget.py"))

    def test_known_fragility_reference_copied_before_hook(self) -> None:
        """Known limitation: a reference copied out BEFORE wrapping is not wrapped.

        This is a documented fragility (§7.1/§8): we assert the limitation holds
        so it stays visible, and print a warning rather than fail the suite.
        """
        module, _ = self._make_target_module()
        stale_ref = module._layer_norm_fwd  # copied out before instrumentation
        ri._instrument_module(module)

        module_wrapped = getattr(module._layer_norm_fwd, "_kid_target_wrapped", False)
        stale_wrapped = getattr(stale_ref, "_kid_target_wrapped", False)
        self.assertTrue(module_wrapped, "module attribute should be wrapped")
        if not stale_wrapped:
            print(
                "\n[WARN][known-fragility] a reference copied before the import hook "
                "runs is NOT instrumented (e.g. `from mod import fn`). Target must be "
                "wrapped before any such copy. This is expected."
            )
        self.assertFalse(stale_wrapped, "stale reference is expected to remain unwrapped")


# ---------------------------------------------------------------------------
# (F0) pytorch_native — example: torch.nn.functional.linear
# ---------------------------------------------------------------------------
class TestTorchFunctional(_CaptureBase):
    def _make_fake_functional(self) -> ModuleType:
        fake = ModuleType("torch.nn.functional")
        fake.linear = lambda *a, **k: 0
        fake.rms_norm = lambda *a, **k: 0
        fake.scaled_dot_product_attention = lambda *a, **k: 0
        return fake

    def test_linear_is_captured_as_pytorch_native(self) -> None:
        fake = self._make_fake_functional()
        ri._instrument_module(fake)  # name matches -> _patch_torch_functional
        self.assertTrue(getattr(fake.linear, "_kid_wrapper_wrapped", False))

        with self.target_ctx():
            fake.linear(1, 2)
        self.dump("F0-torch")

        wraps = self.events_of("wrap_begin")
        self.assertTrue(wraps, "expected a wrap_begin for torch.nn.functional.linear")
        e = next(e for e in wraps if str(e.get("api", "")).endswith("linear"))
        self.assertEqual(e.get("category"), "pytorch_native")
        self.assertEqual((e.get("implementation") or {}).get("kind"), "pytorch_native")
        self.assertEqual(e.get("api"), "torch.nn.functional.linear")


# ---------------------------------------------------------------------------
# (F1/F6) triton — example: _layer_norm_fwd_1pass_kernel (mirrors layernorm_gated.py:68)
#
# NOTE: F1 (sglang-owned) and F6 (third-party) go through the SAME triton patch
# and produce identical runtime events; the F1/F6 split is a later
# classification concern (kernel_file inside sglang tree or not), not a capture
# concern. So one triton capture case suffices here.
# ---------------------------------------------------------------------------
def _layer_norm_fwd_1pass_kernel():  # real fn -> gives real __code__ (file+line)
    return None


class TestTritonLaunch(_CaptureBase):
    def _make_fake_triton(self) -> ModuleType:
        fake = ModuleType("triton.runtime.jit")

        class JITFunction:
            def __init__(self, fn):
                self.fn = fn

            def __getitem__(self, grid):
                def launcher(*a, **k):
                    return None

                return launcher

        fake.JITFunction = JITFunction
        return fake

    def test_triton_launch_is_captured_with_kernel_source(self) -> None:
        fake = self._make_fake_triton()
        ri._instrument_module(fake)  # name matches -> _patch_triton_module
        self.assertTrue(getattr(fake.JITFunction, "_kid_getitem_patched", False))

        kernel_file = _layer_norm_fwd_1pass_kernel.__code__.co_filename
        kernel_line = _layer_norm_fwd_1pass_kernel.__code__.co_firstlineno

        with self.target_ctx():
            k = fake.JITFunction(_layer_norm_fwd_1pass_kernel)
            k[(1, 1)]()  # launch inside target ctx so it records
        self.dump("F1/F6-triton")

        wraps = self.events_of("wrap_begin")
        self.assertTrue(wraps, "expected a wrap_begin for the triton launch")
        e = wraps[0]
        self.assertEqual(e.get("category"), "triton_dsl")
        self.assertEqual(e.get("kernel"), "_layer_norm_fwd_1pass_kernel")
        impl = e.get("implementation") or {}
        self.assertEqual(impl.get("kind"), "triton_source")
        self.assertEqual(impl.get("source_files"), [kernel_file])
        self.assertEqual(impl.get("definition_line"), kernel_line)


# ---------------------------------------------------------------------------
# (F2/F3) sgl_kernel — example: torch.ops.sgl_kernel.fwd binding fn
# ---------------------------------------------------------------------------
def _reexported_from_elsewhere(*a, **k):
    return 0


_reexported_from_elsewhere.__module__ = "some.other.module"


class TestSglKernel(_CaptureBase):
    def _make_fake_sgl(self) -> ModuleType:
        fake = ModuleType("sgl_kernel")

        def fwd(*a, **k):
            return 0

        fwd.__module__ = "sgl_kernel"  # owner matches -> will be wrapped
        fake.fwd = fwd
        fake.reexported = _reexported_from_elsewhere  # owner mismatch -> skipped
        return fake

    def test_fwd_captured_as_sgl_kernel_without_source(self) -> None:
        fake = self._make_fake_sgl()
        ri._instrument_module(fake)  # name startswith sgl_kernel -> _wrap_module_functions
        self.assertTrue(getattr(fake.fwd, "_kid_wrapper_wrapped", False))

        with self.target_ctx():
            fake.fwd(1)
        self.dump("F2/F3-sgl_kernel")

        wraps = self.events_of("wrap_begin")
        self.assertTrue(wraps, "expected a wrap_begin for sgl_kernel.fwd")
        e = next(e for e in wraps if str(e.get("api", "")).endswith("fwd"))
        self.assertEqual(e.get("category"), "sgl_kernel")
        # AOT: no source files at runtime (locate fills them statically later)
        impl = e.get("implementation") or {}
        self.assertFalse(impl.get("source_files"), "sgl_kernel AOT should have no runtime source_files")

    def test_known_fragility_reexport_not_captured(self) -> None:
        """Documented fragility: re-exported fns (owner != module) are NOT wrapped."""
        fake = self._make_fake_sgl()
        ri._instrument_module(fake)

        self.assertFalse(
            getattr(fake.reexported, "_kid_wrapper_wrapped", False),
            "re-exported function is expected to remain unwrapped",
        )
        with self.target_ctx():
            fake.reexported(1)
        # No wrap event should mention the re-exported callable.
        self.assertFalse(
            [e for e in self.events_of("wrap_begin") if str(e.get("api", "")).endswith("reexported")],
            "re-export should not produce a wrap event",
        )
        print(
            "\n[WARN][known-fragility] sgl_kernel re-exports (owner != module.__name__) "
            "are NOT instrumented by _wrap_module_functions. Wrap at the defining module."
        )


# ---------------------------------------------------------------------------
# (F4) sglang-owned JIT — example: sglang.jit_kernel.utils.load_jit
# ---------------------------------------------------------------------------
class TestSglangLoadJit(_CaptureBase):
    def _make_fake_jit_utils(self) -> ModuleType:
        fake = ModuleType("sglang.jit_kernel.utils")
        fake.KERNEL_PATH = self.tmp
        fake.load_jit = lambda name, **kw: SimpleNamespace()
        return fake

    def test_load_jit_records_sources_and_wrappers(self) -> None:
        fake = self._make_fake_jit_utils()
        ri._instrument_module(fake)  # name matches -> _patch_sglang_jit_utils
        self.assertTrue(getattr(fake.load_jit, "_kid_load_jit_patched", False))

        # load_jit records directly (no target ctx needed).
        fake.load_jit(
            "m",
            cpp_files=["a.cpp"],
            cuda_files=["b.cu"],
            cpp_wrappers=[("fwd", "sym")],
        )
        self.dump("F4-sglang-jit")

        loaded = self.events_of("jit_module_loaded")
        self.assertTrue(loaded, "expected a jit_module_loaded event")
        meta = loaded[0]
        src = meta.get("source_files") or []
        self.assertTrue(any(str(s).endswith("a.cpp") for s in src), "cpp source missing")
        self.assertTrue(any(str(s).endswith("b.cu") for s in src), "cuda source missing")
        by_export = meta.get("wrappers_by_export") or {}
        self.assertIn("fwd", by_export)
        self.assertEqual(by_export["fwd"].get("symbols"), ["sym"])


# ---------------------------------------------------------------------------
# (F7) flashinfer JIT — example: flashinfer.jit.core.gen_jit_spec
#
# PENDING: _patch_flashinfer_jit is not implemented yet (KID upgrade target).
# This is a skip-spec: it documents the expected behavior so implementing the
# patch just means deleting the skip and making it green.
# ---------------------------------------------------------------------------
class TestFlashinferJit(_CaptureBase):
    @unittest.skip("pending _patch_flashinfer_jit — KID upgrade TDD target; drop skip once implemented")
    def test_gen_jit_spec_records_sources(self) -> None:
        fake = ModuleType("flashinfer.jit.core")
        fake.gen_jit_spec = lambda name, sources, **kw: SimpleNamespace(name=name, sources=sources)
        ri._instrument_module(fake)
        # Expected once implemented: gen_jit_spec is patched and a
        # `jit_spec_generated` event carries name + source_files.
        self.assertTrue(getattr(fake.gen_jit_spec, "_kid_gen_jit_spec_patched", False))
        fake.gen_jit_spec("gdn", ["/x/csrc/a.cu"])
        gen = self.events_of("jit_spec_generated")
        self.assertTrue(gen)
        self.assertEqual(gen[0].get("name"), "gdn")
        self.assertEqual((gen[0].get("implementation") or {}).get("source_files"), ["/x/csrc/a.cu"])


# ---------------------------------------------------------------------------
# install_from_env smoke — verify the loader entrypoint itself works
# ---------------------------------------------------------------------------
class TestInstallFromEnvSmoke(_CaptureBase):
    def test_install_from_env_writes_process_start(self) -> None:
        runtime_config = {
            "output_dir": self.tmp,
            "workdir": self.tmp,
            "target": {"file": "", "line": 0},
            "resolution": {"third_party_prefixes": []},
        }
        cfg_path = Path(self.tmp, "runtime_config.json")
        cfg_path.write_text(json.dumps(runtime_config))

        prev_enable = os.environ.get("KID_ENABLE")
        prev_cfg = os.environ.get("KID_RUNTIME_CONFIG")
        try:
            os.environ["KID_ENABLE"] = "1"
            os.environ["KID_RUNTIME_CONFIG"] = str(cfg_path)
            ri._INSTALLED = False
            ri.install_from_env()
            self.assertTrue(ri._INSTALLED, "install_from_env should mark installed")
            self.assertTrue(self.events_of("process_start"), "expected process_start event")
            self.dump("install-smoke")
        finally:
            # Clean up global + import-hook side effects so other tests stay hermetic.
            ri.sys.meta_path[:] = [
                m for m in ri.sys.meta_path if not isinstance(m, ri._InstrumentingFinder)
            ]
            ri._INSTALLED = False
            for key, prev in (("KID_ENABLE", prev_enable), ("KID_RUNTIME_CONFIG", prev_cfg)):
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev


# ---------------------------------------------------------------------------
# Tier-2: real-library probes (auto-skip when a library is unavailable).
# These do NOT launch kernels or need a GPU — they only assert KID's patch
# markers attach to the real module structure (catches upstream refactors).
# ---------------------------------------------------------------------------
class TestRealLibraryProbes(_CaptureBase):
    def test_real_torch_functional_marker(self) -> None:
        try:
            import torch.nn.functional as F
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"torch not importable: {exc}")
        ri._instrument_module(F)
        self.assertTrue(
            getattr(F.linear, "_kid_wrapper_wrapped", False),
            "real torch.nn.functional.linear should be wrapped",
        )

    def test_real_triton_getitem_marker(self) -> None:
        try:
            import triton.runtime.jit as tjit
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"triton not importable: {exc}")
        ri._instrument_module(tjit)
        jf = getattr(tjit, "JITFunction", None)
        if jf is None or not hasattr(jf, "__getitem__"):
            self.skipTest("triton JITFunction shape changed; no __getitem__ to patch")
        self.assertTrue(
            getattr(jf, "_kid_getitem_patched", False),
            "real triton JITFunction.__getitem__ should be patched",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
