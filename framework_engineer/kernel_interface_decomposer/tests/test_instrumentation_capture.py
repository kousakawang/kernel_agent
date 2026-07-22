from __future__ import annotations

import importlib
import functools
import inspect
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from framework_engineer.kernel_interface_decomposer import runtime_instrumentation as ri


def _import_patch_target(callback: object) -> object:
    return callback()  # type: ignore[operator]


def _transparent_api_wrapper(function: object) -> object:
    @functools.wraps(function)  # type: ignore[arg-type]
    def api_logging_wrapper(*args: object, **kwargs: object) -> object:
        return function(*args, **kwargs)  # type: ignore[operator]

    return api_logging_wrapper


@_transparent_api_wrapper
def _decorated_import_patch_target(callback: object) -> object:
    return callback()  # type: ignore[operator]


class _ImportPatchOwner:
    @classmethod
    def target(cls, callback: object) -> object:
        del cls
        return callback()  # type: ignore[operator]


class TestRuntimeInstrumentation(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.events_dir = Path(self.tempdir.name) / "capture_events"
        ri._CONFIG = {
            "events_dir": str(self.events_dir),
            "target": {"file": __file__, "line": 1},
        }
        ri._CALL_COUNTER = 0
        ri._CAPTURE_COUNTER = 0
        ri._CURRENT_HIGH.set(None)
        ri._CURRENT_CAPTURE.set(None)
        ri._ACTIVE_TARGET_FRAMES.clear()
        ri._TARGET_IMPORT_PATCH_ENABLED = False
        ri._TARGET_MODULE_NAMES = frozenset()
        ri._TARGET_PROFILER_INSTALLED = False
        self.nvtx_push = mock.patch.object(ri, "_nvtx_push", lambda text: None)
        self.nvtx_pop = mock.patch.object(ri, "_nvtx_pop", lambda: None)
        self.nvtx_push.start()
        self.nvtx_pop.start()

    def tearDown(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        self.nvtx_push.stop()
        self.nvtx_pop.stop()
        self.tempdir.cleanup()

    def events(self) -> list[dict[str, object]]:
        paths = list(self.events_dir.glob("events_*.jsonl"))
        return [
            json.loads(line)
            for path in paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _run_high(self, callback: object) -> object:
        def high() -> object:
            frame = inspect.currentframe()
            assert frame is not None
            with ri._high_scope(frame, "high"):
                return callback()  # type: ignore[operator]

        return high()

    def test_capture_outside_high_is_ignored(self) -> None:
        with ri.execution_capture(
            archetype="triton_launch", execution_interface="kernel"
        ):
            pass
        self.assertEqual(self.events(), [])

    def test_nested_capture_keeps_parent_and_full_stack(self) -> None:
        def workload() -> None:
            with ri.execution_capture(
                archetype="pytorch_dispatch", execution_interface="custom.default"
            ):
                with ri.execution_capture(
                    archetype="triton_launch", execution_interface="inner_kernel"
                ):
                    pass

        self._run_high(workload)
        outer, inner = self.events()
        self.assertIsNone(outer["parent_capture_id"])
        self.assertEqual(inner["parent_capture_id"], outer["capture_id"])
        self.assertEqual(inner["parent_call_id"], outer["parent_call_id"])
        stack = inner["python_stack"]
        self.assertTrue(stack)
        self.assertEqual(stack[0]["function"], "high")
        self.assertIn("call_site_to_next", stack[-1])

    def test_triton_cute_tilelang_and_inductor_adapters(self) -> None:
        launches: list[str] = []

        class JITFunction:
            def __init__(self) -> None:
                self.fn = lambda: None

            def __getitem__(self, grid: object) -> object:
                del grid
                return lambda: launches.append("triton")

        triton_module = types.ModuleType("triton.runtime.jit")
        triton_module.JITFunction = JITFunction
        ri._patch_triton(triton_module)

        cute_module = types.ModuleType("cutlass.cute")
        cute_module.compile = lambda kernel: (lambda: launches.append("cute"))
        ri._patch_cute(cute_module)

        class JITKernel:
            def __call__(self) -> None:
                launches.append("tilelang")

        tilelang_module = types.ModuleType("tilelang")
        tilelang_module.JITKernel = JITKernel
        ri._patch_tilelang(tilelang_module)

        class CachingAutotuner:
            fn = lambda self: None

            def run(self) -> None:
                launches.append("inductor")

        inductor_module = types.ModuleType("torch._inductor.runtime.triton_heuristics")
        inductor_module.CachingAutotuner = CachingAutotuner
        ri._patch_inductor(inductor_module)

        def workload() -> None:
            JITFunction()[1]()
            cute_module.compile(lambda: None)()
            JITKernel()()
            CachingAutotuner().run()

        self._run_high(workload)
        self.assertEqual(launches, ["triton", "cute", "tilelang", "inductor"])
        self.assertEqual(
            [event["archetype"] for event in self.events()],
            ["triton_launch", "cute_dsl_launch", "tilelang_launch", "inductor_launch"],
        )

    def test_tvm_factory_proxy_and_python_binding(self) -> None:
        calls: list[str] = []
        loaded = types.SimpleNamespace(run=lambda: calls.append("ffi"))
        proxy = ri._CapturedModuleProxy(loaded, "flashinfer", "flashinfer_jit.demo")

        binding = types.ModuleType("deep_gemm")
        binding.bf16_gemm_nt = lambda: calls.append("binding")
        ri._patch_python_bindings(binding)

        def workload() -> None:
            proxy.run()
            binding.bf16_gemm_nt()
            with ri.execution_capture(
                archetype="pytorch_dispatch",
                execution_interface="aten.mm.default",
                provider_hint="pytorch",
            ):
                calls.append("dispatch")

        self._run_high(workload)
        self.assertEqual(calls, ["ffi", "binding", "dispatch"])
        self.assertEqual(
            [event["archetype"] for event in self.events()],
            ["tvm_ffi_call", "python_binding", "pytorch_dispatch"],
        )

    def _profile_target(self) -> None:
        with ri.execution_capture(
            archetype="triton_launch", execution_interface="profiled_kernel"
        ):
            pass

    def test_target_code_profile_captures_direct_script_calls(self) -> None:
        function = type(self)._profile_target
        ri._CONFIG["target"] = {
            "file": function.__code__.co_filename,
            "line": function.__code__.co_firstlineno,
            "qualified_name": function.__qualname__,
        }
        ri._install_target_profiler()
        copied_reference = self._profile_target
        copied_reference()
        events = self.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["execution_interface"], "profiled_kernel")
        self.assertTrue(events[0]["parent_call_id"])

    def test_import_patch_wraps_module_target_without_global_profiler(self) -> None:
        module = types.ModuleType("fake_target_module")
        module.__file__ = __file__
        module._import_patch_target = _import_patch_target
        ri._CONFIG["target"] = {
            "file": __file__,
            "line": _import_patch_target.__code__.co_firstlineno,
            "qualified_name": "_import_patch_target",
        }

        self.assertTrue(ri._patch_high_target(module))
        self.assertIsNone(sys.getprofile())

        def workload() -> None:
            with ri.execution_capture(
                archetype="triton_launch", execution_interface="patched_kernel"
            ):
                pass

        module._import_patch_target(workload)
        events = self.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["execution_interface"], "patched_kernel")
        self.assertTrue(events[0]["parent_call_id"])
        self.assertEqual(events[0]["event"], "execution_capture")

    def test_service_patch_records_entry_and_unwraps_decorated_target(self) -> None:
        module = types.ModuleType("fake_decorated_target_module")
        module.__file__ = __file__
        module._decorated_import_patch_target = _decorated_import_patch_target
        original = _decorated_import_patch_target.__wrapped__
        ri._CONFIG.update(
            {
                "execution_mode": "service",
                "target": {
                    "file": __file__,
                    "line": original.__code__.co_firstlineno,
                    "qualified_name": "_decorated_import_patch_target",
                },
            }
        )

        self.assertTrue(ri._patch_high_target(module))

        def workload() -> None:
            with ri.execution_capture(
                archetype="triton_launch", execution_interface="decorated_kernel"
            ):
                pass

        module._decorated_import_patch_target(workload)
        high_event, execution_event = self.events()
        self.assertEqual(high_event["event"], "high_invocation")
        self.assertEqual(high_event["instrumentation_mode"], "import_patch")
        self.assertTrue(high_event["entry_python_stack"])
        self.assertEqual(
            high_event["entry_python_stack"][-1]["function"],
            "test_service_patch_records_entry_and_unwraps_decorated_target",
        )
        self.assertEqual(execution_event["event"], "execution_capture")
        functions = [
            frame["function"] for frame in execution_event["python_stack"]
        ]
        self.assertEqual(functions[0], "_decorated_import_patch_target")
        self.assertNotIn("api_logging_wrapper", functions)

    def test_import_patch_preserves_classmethod_descriptor(self) -> None:
        module = types.ModuleType("fake_class_target_module")
        module.__file__ = __file__
        module._ImportPatchOwner = _ImportPatchOwner
        ri._CONFIG["target"] = {
            "file": __file__,
            "line": _ImportPatchOwner.target.__func__.__code__.co_firstlineno,
            "qualified_name": "_ImportPatchOwner.target",
        }

        self.assertTrue(ri._patch_high_target(module))

        def workload() -> None:
            with ri.execution_capture(
                archetype="pytorch_dispatch", execution_interface="aten.add.Tensor"
            ):
                pass

        module._ImportPatchOwner.target(workload)
        self.assertEqual(len(self.events()), 1)

    def test_target_module_patch_failure_uses_profiler_fallback(self) -> None:
        module = types.ModuleType("fake_missing_target_module")
        module.__file__ = __file__
        ri._CONFIG["target"] = {
            "file": __file__,
            "line": _import_patch_target.__code__.co_firstlineno,
            "qualified_name": "missing_target",
        }
        ri._TARGET_IMPORT_PATCH_ENABLED = True
        with mock.patch.object(ri, "_install_target_profiler_fallback") as fallback:
            ri._instrument_module(module)
        fallback.assert_called_once()

    def test_target_module_name_is_derived_from_source_path(self) -> None:
        ri._CONFIG["target"] = {
            "file": __file__,
            "line": _import_patch_target.__code__.co_firstlineno,
            "qualified_name": "_import_patch_target",
        }
        self.assertIn(
            "framework_engineer.kernel_interface_decomposer.tests.test_instrumentation_capture",
            ri._target_module_names(),
        )

    def test_direct_script_and_module_entrypoints_use_profiler_path(self) -> None:
        ri._CONFIG["target"] = {
            "file": __file__,
            "line": _import_patch_target.__code__.co_firstlineno,
            "qualified_name": "_import_patch_target",
        }
        module_names = frozenset(ri._target_module_names())
        module_name = (
            "framework_engineer.kernel_interface_decomposer.tests."
            "test_instrumentation_capture"
        )
        with mock.patch.object(sys, "argv", [__file__]), mock.patch.object(
            sys, "orig_argv", [sys.executable, __file__], create=True
        ):
            self.assertTrue(ri._target_is_python_entrypoint(module_names))
        with mock.patch.object(sys, "argv", ["-m"]), mock.patch.object(
            sys,
            "orig_argv",
            [sys.executable, "-m", module_name],
            create=True,
        ):
            self.assertTrue(ri._target_is_python_entrypoint(module_names))

    def test_import_hook_patches_target_before_import_returns(self) -> None:
        package_root = Path(self.tempdir.name) / "packages"
        package = package_root / "toy_target_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        target_file = package / "ops.py"
        target_file.write_text(
            "def high_level(value):\n"
            "    return value + 1\n",
            encoding="utf-8",
        )
        ri._CONFIG["target"] = {
            "file": str(target_file),
            "line": 1,
            "qualified_name": "high_level",
        }
        original_meta_path = list(sys.meta_path)
        sys.path.insert(0, str(package_root))
        try:
            ri._TARGET_MODULE_NAMES = frozenset(ri._target_module_names())
            ri._TARGET_IMPORT_PATCH_ENABLED = True
            ri._install_import_hook()
            module = importlib.import_module("toy_target_package.ops")
            self.assertTrue(module.high_level._kid_high_target_wrapped)
            self.assertEqual(module.high_level(4), 5)
            self.assertIsNone(sys.getprofile())
        finally:
            sys.meta_path[:] = original_meta_path
            sys.path.remove(str(package_root))
            sys.modules.pop("toy_target_package.ops", None)
            sys.modules.pop("toy_target_package", None)

    def test_target_profiler_does_not_resolve_paths_for_unrelated_calls(self) -> None:
        function = type(self)._profile_target
        ri._CONFIG["target"] = {
            "file": function.__code__.co_filename,
            "line": function.__code__.co_firstlineno,
            "qualified_name": function.__qualname__,
        }
        ri._install_target_profiler()

        def unrelated() -> int:
            return 1

        with mock.patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("unrelated calls must not resolve paths"),
        ):
            self.assertEqual(unrelated(), 1)

    def test_recording_gate_excludes_startup_target_calls(self) -> None:
        gate = Path(self.tempdir.name) / "recording.enabled"
        active_dir = Path(self.tempdir.name) / "active_ranges"
        function = type(self)._profile_target
        ri._CONFIG.update(
            {
                "target": {
                    "file": function.__code__.co_filename,
                    "line": function.__code__.co_firstlineno,
                    "qualified_name": function.__qualname__,
                },
                "recording_gate_file": str(gate),
                "active_ranges_dir": str(active_dir),
            }
        )
        ri._install_target_profiler()
        copied_reference = self._profile_target

        copied_reference()
        self.assertEqual(self.events(), [])

        gate.touch()
        copied_reference()
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(list(active_dir.glob("*.active")), [])

        gate.unlink()
        copied_reference()
        self.assertEqual(len(self.events()), 1)


if __name__ == "__main__":
    unittest.main()
