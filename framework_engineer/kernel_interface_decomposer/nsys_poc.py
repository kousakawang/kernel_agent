#!/usr/bin/env python3
"""One-file Nsight Systems proof of concept for KID execution capture.

Remote/GPU run (``python`` is the workspace's remote runner)::

    python kernel_agent/framework_engineer/kernel_interface_decomposer/nsys_poc.py

Local parser/capture self-test (``python3`` is the local interpreter)::

    python3 kernel_agent/framework_engineer/kernel_interface_decomposer/nsys_poc.py --self-test

The normal launcher first probes packages/GPU capability and smoke-runs each
selected backend.  It then profiles a worker invocation of this same file,
exports the Nsight report to SQLite, and joins four event streams:

* KID high/execution NVTX ranges on the CPU thread;
* CUDA Runtime/Driver API calls made inside those ranges;
* GPU kernel activities connected to those API calls by correlation id.
* Python high-to-execution frame chains written by the capture adapters.

The NVTX durations are retained as CPU-side metadata.  Hotspot time always
comes from GPU kernel activities, never from the Python wrapper wall time.

Only the high-level target is explicitly decorated.  Seven common execution
boundaries cover PyTorch dispatch, Triton, CuTe DSL, TileLang, TVM FFI,
Inductor, and registered direct Python bindings.  Real SGLang/third-party
semantic functions remain unmodified; their appearance in captured frame and
callsite chains is the evidence a later resolver uses to recover the desired
interface.
"""

from __future__ import annotations

import argparse
import contextvars
import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Callable, ContextManager, Iterable, Iterator, Sequence


SCHEMA_VERSION = "kid-nsys-poc/v2"
LABEL_PREFIX = "KID:"
DEFAULT_OUTPUT_DIR = "nsys_poc_output"

_CURRENT_HIGH: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "kid_nsys_poc_current_high", default=None
)
_CURRENT_EXECUTION_CAPTURE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kid_nsys_poc_current_execution_capture", default=None
)
_CURRENT_WORKLOAD_CASE: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("kid_nsys_poc_current_workload_case", default=None)
)
_HIGH_CALL_COUNTER = 0
_EXECUTION_CAPTURE_COUNTER = 0
_CAPTURE_EVENTS_PATH: Path | None = None
_CAPTURE_EVENT_LOCK = threading.Lock()

# This is the minimal registry exercised by the one-file PoC.  The complete
# project registry lives in capture_registry.py; the values intentionally match.
POC_CAPTURE_REGISTRY = {
    "pytorch_dispatch": "torch.utils._python_dispatch.TorchDispatchMode.__torch_dispatch__",
    "triton_launch": "triton.runtime.jit.JITFunction.__getitem__ launcher",
    "cute_dsl_launch": "callable returned by cutlass.cute.compile",
    "tilelang_launch": "tilelang.JITKernel.__call__",
    "tvm_ffi_call": "exported callable on a tvm_ffi.module.Module",
    "inductor_launch": "torch._inductor.runtime.triton_heuristics.CachingAutotuner.run",
    "python_binding": "registered Python-visible extension/binding export callable",
}

MANDATORY_CASE_NAMES = (
    "pytorch_native",
    "sgl_kernel_builtin",
    "sgl_kernel_sgl_attn",
    "sglang_triton",
    "flashinfer_triton",
    "sglang_jit",
    "flashinfer_ffi",
    "sglang_cutedsl",
    "deepgemm_binding",
    "sglang_tilelang",
    "sglang_inductor",
)
OPTIONAL_CASE_NAMES = (
    "flash_attn4_cutedsl",
    "tokenspeed_mla_cutedsl",
)
VARIANT_ONLY_CASE_NAMES = ("pytorch_softmax",)
ALL_CASE_NAMES = MANDATORY_CASE_NAMES + OPTIONAL_CASE_NAMES + VARIANT_ONLY_CASE_NAMES
INVOCATION_VARIANTS = ("old", "softmax")

CASE_CONTRACTS: dict[str, dict[str, Any]] = {
    "pytorch_native": {
        "semantic_target": "torch.matmul",
        "expected_archetype": "pytorch_dispatch",
        "provider": "pytorch",
        "optional": False,
    },
    "pytorch_softmax": {
        "semantic_target": "torch.softmax",
        "expected_archetype": "pytorch_dispatch",
        "provider": "pytorch",
        "optional": False,
    },
    "sgl_kernel_builtin": {
        "semantic_target": "sgl_kernel.silu_and_mul",
        "expected_archetype": "pytorch_dispatch",
        "provider": "sgl-kernel",
        "optional": False,
    },
    "sgl_kernel_sgl_attn": {
        "semantic_target": "sgl_kernel.flash_attn.flash_attn_varlen_func",
        "expected_archetype": "pytorch_dispatch",
        "provider": "sgl-attn",
        "optional": False,
    },
    "sglang_triton": {
        "semantic_target": (
            "sglang.jit_kernel.diffusion.triton.rmsnorm_onepass."
            "triton_one_pass_rms_norm"
        ),
        "expected_archetype": "triton_launch",
        "provider": "sglang",
        "optional": False,
    },
    "flashinfer_triton": {
        "semantic_target": "flashinfer.triton.norm.rms_norm",
        "expected_archetype": "triton_launch",
        "provider": "flashinfer",
        "optional": False,
    },
    "sglang_jit": {
        "semantic_target": "sglang.jit_kernel.add_constant.add_constant",
        "expected_archetype": "tvm_ffi_call",
        "provider": "sglang",
        "optional": False,
    },
    "flashinfer_ffi": {
        "semantic_target": "flashinfer.sampling.min_p_sampling_from_probs",
        "expected_archetype": "tvm_ffi_call",
        "provider": "flashinfer",
        "optional": False,
    },
    "sglang_cutedsl": {
        "semantic_target": (
            "sglang.jit_kernel.diffusion.cutedsl."
            "scale_residual_norm_scale_shift.fused_norm_scale_shift"
        ),
        "expected_archetype": "cute_dsl_launch",
        "provider": "sglang",
        "optional": False,
    },
    "deepgemm_binding": {
        "semantic_target": "deep_gemm.bf16_gemm_nt",
        "expected_archetype": "python_binding",
        "provider": "deepgemm",
        "optional": False,
    },
    "sglang_tilelang": {
        "semantic_target": "sglang.srt.layers.mhc.hc_split_sinkhorn",
        "expected_archetype": "tilelang_launch",
        "provider": "sglang",
        "optional": False,
    },
    "sglang_inductor": {
        "semantic_target": (
            "sglang.srt.sampling.penaltylib.repetition_penalty."
            "apply_scaling_penalties"
        ),
        "expected_archetype": "inductor_launch",
        "provider": "sglang",
        "optional": False,
    },
    "flash_attn4_cutedsl": {
        "semantic_target": "flash_attn.cute.flash_attn_varlen_func",
        "expected_archetype": "cute_dsl_launch",
        "provider": "flash-attention",
        "optional": True,
        "minimum_compute_capability": (10, 0),
    },
    "tokenspeed_mla_cutedsl": {
        "semantic_target": "tokenspeed_mla.tokenspeed_mla_decode",
        "expected_archetype": "cute_dsl_launch",
        "provider": "tokenspeed-mla",
        "optional": True,
        "minimum_compute_capability": (10, 0),
    },
}


def _next_high_call_id() -> str:
    global _HIGH_CALL_COUNTER
    _HIGH_CALL_COUNTER += 1
    return str(_HIGH_CALL_COUNTER)


def _next_execution_capture_id() -> str:
    global _EXECUTION_CAPTURE_COUNTER
    _EXECUTION_CAPTURE_COUNTER += 1
    return str(_EXECUTION_CAPTURE_COUNTER)


def _sanitize_label_value(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def _make_label(kind: str, **fields: Any) -> str:
    parts = [f"{LABEL_PREFIX}type={kind}"]
    parts.extend(
        f"{key}={_sanitize_label_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    return "|".join(parts)


def _nvtx_push(label: str) -> None:
    # Imported lazily so --self-test works with the local CPU-only Python.
    import torch

    torch.cuda.nvtx.range_push(label)


def _nvtx_pop() -> None:
    import torch

    torch.cuda.nvtx.range_pop()


def _high_capture_mode() -> ContextManager[Any]:
    """Return the common-interface modes active for one high invocation."""

    return nullcontext()


def _frame_record(frame: FrameType) -> dict[str, Any]:
    code = frame.f_code
    return {
        "file": code.co_filename,
        "function": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "definition_line": code.co_firstlineno,
        "callsite": {"file": code.co_filename, "line": frame.f_lineno},
    }


def _capture_python_stack(high: dict[str, Any]) -> list[dict[str, Any]]:
    """Capture the high frame through the Python caller of a common interface."""

    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
    innermost_to_outermost: list[FrameType] = []
    boundary_code = high.get("boundary_code")
    found_boundary = False
    while frame is not None:
        innermost_to_outermost.append(frame)
        if frame.f_code is boundary_code:
            found_boundary = True
            break
        frame = frame.f_back

    if not found_boundary:
        return []

    # Instrumentation frames are represented by the synthetic execution leaf
    # instead of being mixed into the semantic candidate path.
    ignored_functions = {
        "_capture_python_stack",
        "execution_capture",
        "__enter__",
        "__torch_dispatch__",
        "wrapped_launcher",
    }
    records = [
        _frame_record(item)
        for item in reversed(innermost_to_outermost)
        if item.f_code.co_name not in ignored_functions
    ]
    return records


def _write_capture_event(event: dict[str, Any]) -> None:
    if _CAPTURE_EVENTS_PATH is None:
        return
    _CAPTURE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CAPTURE_EVENT_LOCK:
        with _CAPTURE_EVENTS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


@contextmanager
def workload_case_context(name: str) -> Iterator[None]:
    """Attach PoC-only case identity without adding a semantic NVTX range."""

    contract = CASE_CONTRACTS[name]
    token = _CURRENT_WORKLOAD_CASE.set({"name": name, **contract})
    try:
        yield
    finally:
        _CURRENT_WORKLOAD_CASE.reset(token)


def _active_provider(default: str | None = None) -> str | None:
    case = _CURRENT_WORKLOAD_CASE.get()
    return str(case["provider"]) if case and case.get("provider") else default


@contextmanager
def execution_capture(
    *,
    archetype: str,
    execution_interface: str,
    provider: str | None = None,
    implementation: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record one common-interface execution under the active high."""

    high = _CURRENT_HIGH.get()
    if high is None:
        yield
        return

    if archetype not in POC_CAPTURE_REGISTRY:
        raise ValueError(f"unknown PoC capture archetype: {archetype}")
    capture_id = _next_execution_capture_id()
    parent_capture_id = _CURRENT_EXECUTION_CAPTURE.get()
    token = _CURRENT_EXECUTION_CAPTURE.set(capture_id)
    stack = _capture_python_stack(high)
    workload_case = _CURRENT_WORKLOAD_CASE.get()
    event = {
        "event": "execution_capture",
        "capture_id": capture_id,
        "parent_capture_id": parent_capture_id,
        "high_call_id": high["call_id"],
        "archetype": archetype,
        "common_interface": POC_CAPTURE_REGISTRY[archetype],
        "execution_interface": execution_interface,
        "provider": _active_provider(provider),
        "workload_case": workload_case.get("name") if workload_case else None,
        "semantic_target_hint": (
            workload_case.get("semantic_target") if workload_case else None
        ),
        "python_stack": stack,
        "execution_leaf": {
            "kind": "common_interface",
            "archetype": archetype,
            "interface": execution_interface,
        },
        "implementation": implementation or {},
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        "cpu_capture_ns": time.monotonic_ns(),
    }
    _write_capture_event(event)
    _nvtx_push(
        _make_label(
            "execution",
            capture_id=capture_id,
            parent_capture_id=parent_capture_id,
            parent_call_id=high["call_id"],
            archetype=archetype,
            interface=execution_interface,
            provider=_active_provider(provider),
        )
    )
    try:
        yield
    finally:
        _nvtx_pop()
        _CURRENT_EXECUTION_CAPTURE.reset(token)


def high_level_target(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark one high-level invocation with a structured NVTX range."""

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        call_id = _next_high_call_id()
        high = {
            "call_id": call_id,
            "interface": func.__qualname__,
            "file": func.__code__.co_filename,
            "definition_line": func.__code__.co_firstlineno,
            "boundary_code": func.__code__,
        }
        token = _CURRENT_HIGH.set(high)
        _nvtx_push(
            _make_label(
                "high",
                call_id=call_id,
                interface=func.__qualname__,
                file=func.__code__.co_filename,
                line=func.__code__.co_firstlineno,
            )
        )
        try:
            with _high_capture_mode():
                return func(*args, **kwargs)
        finally:
            _nvtx_pop()
            _CURRENT_HIGH.reset(token)

    return wrapped

# Keep GPU-only imports and Triton source definitions out of the local
# ``python3 --self-test`` process.  The launcher starts this file with a GPU
# worker flag, so these definitions are present only in the target container.
_WORKER_MODE = "--worker" in sys.argv
_PROBE_WORKER_MODE = "--probe-worker" in sys.argv
_GPU_MODE = _WORKER_MODE or _PROBE_WORKER_MODE

if _GPU_MODE:
    import torch
    import triton
    import triton.language as tl
    from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily

    class _TorchExecutionCaptureMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            kwargs = kwargs or {}
            with execution_capture(
                archetype="pytorch_dispatch",
                execution_interface=str(func),
                provider="pytorch",
            ):
                return func(*args, **kwargs)

    def _high_capture_mode() -> ContextManager[Any]:
        return _TorchExecutionCaptureMode()

    @triton.jit
    def vector_add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    @triton.jit
    def vector_scale_kernel(
        x_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * 0.5, mask=mask)

    def low_level_gemm(a: Any, b: Any) -> Any:
        return torch.matmul(a, b)

    def low_level_softmax(x: Any) -> Any:
        return torch.softmax(x, dim=-1)

    def low_level_vector_pipeline(x: Any, y: Any) -> Any:
        """One semantic candidate that deliberately launches two GPU kernels."""

        intermediate = torch.empty_like(x)
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, 256),)
        vector_add_kernel[grid](
            x,
            y,
            intermediate,
            n_elements=n_elements,
            BLOCK_SIZE=256,
        )
        vector_scale_kernel[grid](
            intermediate,
            output,
            n_elements=n_elements,
            BLOCK_SIZE=256,
        )
        return output

    def _install_triton_capture() -> None:
        """Patch Triton's stable launcher boundary, not the toy low-level funcs."""

        classes: list[type[Any]] = []
        for module_name in ("triton.runtime.jit", "triton.runtime.autotuner"):
            module = __import__(module_name, fromlist=["*"])
            for class_name in ("JITFunction", "Autotuner", "Heuristics"):
                cls = getattr(module, class_name, None)
                if isinstance(cls, type) and cls not in classes:
                    classes.append(cls)

        for cls in classes:
            if getattr(cls, "_kid_poc_getitem_patched", False):
                continue
            original = getattr(cls, "__getitem__", None)
            if original is None:
                continue

            def make_getitem(orig: Callable[..., Any]) -> Callable[..., Any]:
                @functools.wraps(orig)
                def patched_getitem(self: Any, grid: Any) -> Any:
                    launcher = orig(self, grid)
                    kernel_fn = getattr(self, "fn", None)
                    kernel_name = getattr(
                        kernel_fn,
                        "__name__",
                        getattr(self, "__name__", type(self).__name__),
                    )
                    kernel_code = getattr(kernel_fn, "__code__", None)

                    @functools.wraps(launcher)
                    def wrapped_launcher(*args: Any, **kwargs: Any) -> Any:
                        with execution_capture(
                            archetype="triton_launch",
                            execution_interface=str(kernel_name),
                            provider="triton",
                            implementation={
                                "file": getattr(kernel_code, "co_filename", None),
                                "definition_line": getattr(
                                    kernel_code, "co_firstlineno", None
                                ),
                            },
                        ):
                            return launcher(*args, **kwargs)

                    return wrapped_launcher

                return patched_getitem

            setattr(cls, "__getitem__", make_getitem(original))
            setattr(cls, "_kid_poc_getitem_patched", True)

    def _callable_name(value: Any, fallback: str) -> str:
        for candidate in (
            getattr(value, "__qualname__", None),
            getattr(value, "__name__", None),
            getattr(getattr(value, "fn", None), "__qualname__", None),
            getattr(getattr(value, "fn", None), "__name__", None),
        ):
            if candidate:
                return str(candidate)
        return fallback

    def _callable_implementation(value: Any) -> dict[str, Any]:
        candidate = getattr(value, "fn", value)
        code = getattr(candidate, "__code__", None)
        if code is not None:
            return {
                "file": code.co_filename,
                "definition_line": code.co_firstlineno,
            }
        try:
            source_file = inspect.getsourcefile(type(value)) or inspect.getsourcefile(value)
        except (TypeError, OSError):
            source_file = None
        return {"file": source_file} if source_file else {}

    def _captured_callable(
        value: Callable[..., Any],
        *,
        archetype: str,
        interface: str,
        provider: str | None,
        implementation: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        if getattr(value, "_kid_poc_runtime_wrapped", False):
            return value

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with execution_capture(
                archetype=archetype,
                execution_interface=interface,
                provider=provider,
                implementation=implementation or _callable_implementation(value),
            ):
                return value(*args, **kwargs)

        try:
            functools.update_wrapper(wrapped, value)
        except (AttributeError, TypeError):
            pass
        setattr(wrapped, "_kid_poc_runtime_wrapped", True)
        return wrapped

    class _CapturedModuleProxy:
        """Proxy callable exports on a TVM-FFI module without mutating the module."""

        def __init__(self, module: Any, provider: str | None, origin: str) -> None:
            self._module = module
            self._provider = provider
            self._origin = origin
            self._wrapped_exports: dict[str, Any] = {}

        def __getattr__(self, name: str) -> Any:
            value = getattr(self._module, name)
            if not callable(value) or name.startswith("_"):
                return value
            if name not in self._wrapped_exports:
                self._wrapped_exports[name] = _captured_callable(
                    value,
                    archetype="tvm_ffi_call",
                    interface=f"{self._origin}.{name}",
                    provider=self._provider,
                    implementation={"factory": self._origin, "export": name},
                )
            return self._wrapped_exports[name]

        def __dir__(self) -> list[str]:
            return sorted(set(dir(self._module)) | set(self.__dict__))

        def __repr__(self) -> str:
            return f"_CapturedModuleProxy({self._module!r})"

    def _install_cute_capture() -> bool:
        cute = importlib.import_module("cutlass.cute")
        if getattr(cute, "_kid_poc_compile_patched", False):
            return True
        original_compile = getattr(cute, "compile", None)
        if not callable(original_compile):
            return False

        @functools.wraps(original_compile)
        def patched_compile(*args: Any, **kwargs: Any) -> Any:
            compiled = original_compile(*args, **kwargs)
            kernel = args[0] if args else compiled
            interface = _callable_name(kernel, type(kernel).__name__)
            return _captured_callable(
                compiled,
                archetype="cute_dsl_launch",
                interface=interface,
                provider=None,
                implementation=_callable_implementation(kernel),
            )

        cute.compile = patched_compile
        cute._kid_poc_compile_patched = True
        return True

    def _install_tilelang_capture() -> bool:
        tilelang = importlib.import_module("tilelang")
        cls = getattr(tilelang, "JITKernel", None)
        if not isinstance(cls, type):
            return False
        if getattr(cls, "_kid_poc_call_patched", False):
            return True
        original = getattr(cls, "__call__", None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            interface = _callable_name(self, type(self).__name__)
            with execution_capture(
                archetype="tilelang_launch",
                execution_interface=interface,
                provider=None,
                implementation=_callable_implementation(self),
            ):
                return original(self, *args, **kwargs)

        cls.__call__ = patched_call
        cls._kid_poc_call_patched = True
        return True

    def _install_sglang_tvm_ffi_capture() -> bool:
        utils = importlib.import_module("sglang.jit_kernel.utils")
        if getattr(utils, "_kid_poc_load_jit_patched", False):
            return True
        original = getattr(utils, "load_jit", None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def patched_load_jit(*args: Any, **kwargs: Any) -> Any:
            module = original(*args, **kwargs)
            module_name = str(args[0]) if args else "sglang.load_jit"
            return _CapturedModuleProxy(module, "sglang", f"sglang_jit.{module_name}")

        utils.load_jit = patched_load_jit
        utils._kid_poc_load_jit_patched = True
        return True

    def _install_flashinfer_tvm_ffi_capture() -> bool:
        core = importlib.import_module("flashinfer.jit.core")
        cls = getattr(core, "JitSpec", None)
        if not isinstance(cls, type):
            return False
        if getattr(cls, "_kid_poc_build_and_load_patched", False):
            return True
        original = getattr(cls, "build_and_load", None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def patched_build_and_load(self: Any, *args: Any, **kwargs: Any) -> Any:
            module = original(self, *args, **kwargs)
            name = str(getattr(self, "name", type(self).__name__))
            return _CapturedModuleProxy(module, "flashinfer", f"flashinfer_jit.{name}")

        cls.build_and_load = patched_build_and_load
        cls._kid_poc_build_and_load_patched = True
        return True

    def _install_deepgemm_capture() -> bool:
        module = importlib.import_module("deep_gemm")
        if getattr(module, "_kid_poc_binding_patched", False):
            return True
        wrapped_any = False
        for name in ("bf16_gemm_nt", "fp8_paged_mqa_logits"):
            original = getattr(module, name, None)
            if not callable(original):
                continue
            setattr(
                module,
                name,
                _captured_callable(
                    original,
                    archetype="python_binding",
                    interface=f"deep_gemm.{name}",
                    provider="deepgemm",
                    implementation={"module": "deep_gemm", "export": name},
                ),
            )
            wrapped_any = True
        module._kid_poc_binding_patched = True
        return wrapped_any

    def _install_inductor_capture() -> bool:
        heuristics = importlib.import_module(
            "torch._inductor.runtime.triton_heuristics"
        )
        cls = getattr(heuristics, "CachingAutotuner", None)
        if not isinstance(cls, type):
            return False
        if getattr(cls, "_kid_poc_run_patched", False):
            return True
        method_name = "run" if callable(getattr(cls, "run", None)) else "__call__"
        original = getattr(cls, method_name, None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def patched_run(self: Any, *args: Any, **kwargs: Any) -> Any:
            interface = _callable_name(
                getattr(self, "fn", self), "inductor.CachingAutotuner"
            )
            with execution_capture(
                archetype="inductor_launch",
                execution_interface=interface,
                provider="pytorch",
                implementation=_callable_implementation(getattr(self, "fn", self)),
            ):
                return original(self, *args, **kwargs)

        setattr(cls, method_name, patched_run)
        cls._kid_poc_run_patched = True
        cls._kid_poc_run_method = method_name
        return True

    def _install_all_capture_adapters() -> dict[str, bool]:
        _install_triton_capture()
        return {
            "pytorch_dispatch": True,
            "triton_launch": True,
            "cute_dsl_launch": _install_cute_capture(),
            "tilelang_launch": _install_tilelang_capture(),
            "tvm_ffi_call:sglang": _install_sglang_tvm_ffi_capture(),
            "tvm_ffi_call:flashinfer": _install_flashinfer_tvm_ffi_capture(),
            "inductor_launch": _install_inductor_capture(),
            "python_binding": _install_deepgemm_capture(),
        }

    @dataclass
    class WorkloadCase:
        name: str
        semantic_target: str
        expected_archetype: str
        provider: str
        run: Callable[[], Any]

    def _case_pytorch_native(size: int) -> Callable[[], Any]:
        matrix_size = max(128, min(size, 1024))
        a = torch.randn(
            matrix_size, matrix_size, device="cuda", dtype=torch.float16
        )
        b = torch.randn_like(a)

        def run() -> Any:
            return torch.matmul(a, b)

        return run

    def _case_pytorch_softmax(size: int) -> Callable[[], Any]:
        matrix_size = max(128, min(size, 1024))
        value = torch.randn(
            matrix_size, matrix_size, device="cuda", dtype=torch.float16
        )

        def run() -> Any:
            return torch.softmax(value, dim=-1)

        return run

    def _case_sgl_kernel_builtin() -> Callable[[], Any]:
        from sgl_kernel import silu_and_mul

        value = torch.randn(64, 4096, device="cuda", dtype=torch.float16)

        def run() -> Any:
            return silu_and_mul(value)

        return run

    def _case_sgl_kernel_sgl_attn() -> Callable[[], Any]:
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        q = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu_q = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

        def run() -> Any:
            return flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=4,
                cu_seqlens_q=cu_q,
                max_seqlen_k=4,
                cu_seqlens_k=cu_k,
                causal=True,
            )

        return run

    def _case_sglang_triton() -> Callable[[], Any]:
        from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import (
            triton_one_pass_rms_norm,
        )

        value = torch.randn(512, 2048, device="cuda", dtype=torch.float16)
        weight = torch.randn(2048, device="cuda", dtype=torch.float16)

        def run() -> Any:
            return triton_one_pass_rms_norm(value, weight)

        return run

    def _case_flashinfer_triton() -> Callable[[], Any]:
        from flashinfer.triton.norm import rms_norm

        value = torch.randn(64, 2048, device="cuda", dtype=torch.float16)
        weight = torch.randn(2048, device="cuda", dtype=torch.float16)
        output = torch.empty_like(value)

        def run() -> Any:
            rms_norm(value, weight, output, eps=1e-6)
            return output

        return run

    def _case_sglang_jit() -> Callable[[], Any]:
        from sglang.jit_kernel.add_constant import add_constant

        source = torch.arange(4096, dtype=torch.int32, device="cuda")

        def run() -> Any:
            return add_constant(source, constant=5)

        return run

    def _case_flashinfer_ffi() -> Callable[[], Any]:
        from flashinfer.sampling import min_p_sampling_from_probs

        probabilities = torch.softmax(
            torch.randn(4, 4096, device="cuda", dtype=torch.float32), dim=-1
        )

        def run() -> Any:
            return min_p_sampling_from_probs(
                probabilities, min_p=0.05, deterministic=True
            )

        return run

    def _case_sglang_cutedsl() -> Callable[[], Any]:
        from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
            fused_norm_scale_shift,
        )

        batch, sequence, hidden = 1, 4, 2048
        value = torch.randn(
            batch, sequence, hidden, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16)
        shift = torch.randn_like(scale)

        def run() -> Any:
            return fused_norm_scale_shift(
                value,
                weight,
                bias,
                scale,
                shift,
                norm_type="rms",
                eps=1e-5,
            )

        return run

    def _case_deepgemm_binding() -> Callable[[], Any]:
        import deep_gemm

        m = n = k = 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        def run() -> Any:
            deep_gemm.bf16_gemm_nt(a, b, output)
            return output

        return run

    def _case_sglang_tilelang() -> Callable[[], Any]:
        from sglang.srt.layers.mhc import hc_split_sinkhorn

        hc = 4
        mixes = torch.randn(2, 16, (2 + hc) * hc, device="cuda")
        scale = torch.ones(3, device="cuda")
        base = torch.zeros((2 + hc) * hc, device="cuda")

        def run() -> Any:
            return hc_split_sinkhorn(
                mixes, scale, base, hc_mult=hc, sinkhorn_iters=4
            )

        return run

    def _case_sglang_inductor() -> Callable[[], Any]:
        from sglang.srt.sampling.penaltylib.repetition_penalty import (
            apply_scaling_penalties,
        )

        logits = torch.randn(8, 4096, device="cuda", dtype=torch.float32)
        penalties = torch.full_like(logits, 1.1)

        def run() -> Any:
            apply_scaling_penalties(logits, penalties)
            return logits

        return run

    def _case_flash_attn4_cutedsl() -> Callable[[], Any]:
        from flash_attn.cute import flash_attn_varlen_func

        q = torch.randn(4, 2, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

        def run() -> Any:
            result = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=4,
                max_seqlen_k=4,
                causal=True,
                return_lse=False,
            )
            return result[0] if isinstance(result, tuple) else result

        return run

    def _case_tokenspeed_mla_cutedsl() -> Callable[[], Any]:
        import math

        import tokenspeed_mla

        batch, query_length, heads = 1, 1, 16
        kv_lora_rank, rope_dim, page_size = 512, 64, 64
        query = torch.randn(
            batch,
            query_length,
            heads,
            kv_lora_rank + rope_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        kv_cache = torch.randn(
            1,
            page_size,
            kv_lora_rank + rope_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.int8)
        block_tables = torch.zeros(batch, 1, device="cuda", dtype=torch.int32)
        sequence_lengths = torch.full(
            (batch,), 4, device="cuda", dtype=torch.int32
        )

        def run() -> Any:
            return tokenspeed_mla.tokenspeed_mla_decode(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=rope_dim,
                block_tables=block_tables,
                seq_lens=sequence_lengths,
                max_seq_len=4,
                softmax_scale=1.0 / math.sqrt(kv_lora_rank + rope_dim),
            )

        return run

    CASE_BUILDERS: dict[str, Callable[..., Callable[[], Any]]] = {
        "pytorch_native": _case_pytorch_native,
        "pytorch_softmax": _case_pytorch_softmax,
        "sgl_kernel_builtin": _case_sgl_kernel_builtin,
        "sgl_kernel_sgl_attn": _case_sgl_kernel_sgl_attn,
        "sglang_triton": _case_sglang_triton,
        "flashinfer_triton": _case_flashinfer_triton,
        "sglang_jit": _case_sglang_jit,
        "flashinfer_ffi": _case_flashinfer_ffi,
        "sglang_cutedsl": _case_sglang_cutedsl,
        "deepgemm_binding": _case_deepgemm_binding,
        "sglang_tilelang": _case_sglang_tilelang,
        "sglang_inductor": _case_sglang_inductor,
        "flash_attn4_cutedsl": _case_flash_attn4_cutedsl,
        "tokenspeed_mla_cutedsl": _case_tokenspeed_mla_cutedsl,
    }

    def _build_workload_cases(names: Sequence[str], size: int) -> list[WorkloadCase]:
        cases: list[WorkloadCase] = []
        for name in names:
            contract = CASE_CONTRACTS[name]
            builder = CASE_BUILDERS[name]
            run = (
                builder(size)
                if name in {"pytorch_native", "pytorch_softmax"}
                else builder()
            )
            cases.append(
                WorkloadCase(
                    name=name,
                    semantic_target=str(contract["semantic_target"]),
                    expected_archetype=str(contract["expected_archetype"]),
                    provider=str(contract["provider"]),
                    run=run,
                )
            )
        return cases

    def _run_cases(cases: Sequence[WorkloadCase]) -> list[Any]:
        outputs: list[Any] = []
        for case in cases:
            with workload_case_context(case.name):
                # An active Python dispatcher mode changes torch.compile's
                # execution path and can force eager redispatch.  Pop only the
                # PoC mode around the already-compiled Inductor case; its final
                # launcher remains captured by the Inductor adapter.
                mode_context = (
                    _pop_mode_temporarily()
                    if case.name == "sglang_inductor"
                    else nullcontext()
                )
                with mode_context:
                    outputs.append(case.run())
        return outputs

    def _first_tensor(value: Any) -> Any | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = _first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @((lambda func: func) if os.environ.get("KID_ENABLE") == "1" else high_level_target)
    def high_level(cases: Sequence[WorkloadCase]) -> list[Any]:
        return _run_cases(cases)


@dataclass(frozen=True)
class NvtxRange:
    start: int
    end: int
    fields: dict[str, str]
    global_tid: int | None
    pid: int | None

    @property
    def duration_us(self) -> float:
        return max(0, self.end - self.start) / 1000.0


@dataclass(frozen=True)
class ApiEvent:
    start: int
    end: int
    correlation_id: int
    name: str | None
    global_tid: int | None
    pid: int | None
    source_table: str


@dataclass(frozen=True)
class KernelEvent:
    start: int
    end: int
    correlation_id: int
    name: str
    pid: int | None
    device_id: int | None
    stream_id: int | None
    source_table: str

    @property
    def duration_us(self) -> float:
        return max(0, self.end - self.start) / 1000.0


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def _row_int(row: sqlite3.Row, candidates: Iterable[str]) -> int | None:
    keys = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        actual = keys.get(candidate.lower())
        if actual is None or row[actual] is None:
            continue
        try:
            return int(row[actual])
        except (TypeError, ValueError):
            return None
    return None


def _load_string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    table = next((name for name in _tables(conn) if name.lower() == "stringids"), None)
    if table is None:
        return {}
    columns = _columns(conn, table)
    id_column = _first(columns, ("id",))
    value_column = _first(columns, ("value", "string", "name"))
    if id_column is None or value_column is None:
        return {}
    strings: dict[int, str] = {}
    for row in conn.execute(f'SELECT "{id_column}", "{value_column}" FROM "{table}"'):
        try:
            strings[int(row[0])] = str(row[1])
        except (TypeError, ValueError):
            continue
    return strings


def _row_text(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    keys = {key.lower(): key for key in row.keys()}
    for candidate in ("text", "message", "name", "demangledName", "shortName", "mangledName"):
        actual = keys.get(candidate.lower())
        if actual is None or row[actual] is None:
            continue
        value = row[actual]
        if isinstance(value, int) and value in strings:
            return strings[value]
        return str(value)
    for candidate in (
        "textId",
        "messageId",
        "nameId",
        "demangledNameId",
        "shortNameId",
        "mangledNameId",
    ):
        actual = keys.get(candidate.lower())
        if actual is None or row[actual] is None:
            continue
        try:
            value = strings.get(int(row[actual]))
        except (TypeError, ValueError):
            value = None
        if value is not None:
            return value
    return None


def _parse_label(text: str) -> dict[str, str]:
    position = text.find(LABEL_PREFIX)
    if position < 0:
        return {}
    fields: dict[str, str] = {}
    for part in text[position + len(LABEL_PREFIX) :].split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def _load_nvtx_ranges(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> tuple[list[NvtxRange], list[str]]:
    ranges: list[NvtxRange] = []
    used_tables: list[str] = []
    for table in _tables(conn):
        if "nvtx" not in table.lower():
            continue
        columns = _columns(conn, table)
        start_column = _first(columns, ("start", "startNs", "startTime"))
        end_column = _first(columns, ("end", "endNs", "endTime"))
        if start_column is None or end_column is None:
            continue
        table_used = False
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if row[start_column] is None or row[end_column] is None:
                continue
            text = _row_text(row, strings)
            if not text or LABEL_PREFIX not in text:
                continue
            fields = _parse_label(text)
            if fields.get("type") not in {"high", "execution"}:
                continue
            ranges.append(
                NvtxRange(
                    start=int(row[start_column]),
                    end=int(row[end_column]),
                    fields=fields,
                    global_tid=_row_int(row, ("globalTid", "tid", "threadId")),
                    pid=_row_int(row, ("globalPid", "pid", "processId")),
                )
            )
            table_used = True
        if table_used:
            used_tables.append(table)
    ranges.sort(key=lambda item: (item.start, item.end))
    return ranges, used_tables


def _load_api_events(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> tuple[list[ApiEvent], list[str]]:
    events: list[ApiEvent] = []
    used_tables: list[str] = []
    for table in _tables(conn):
        lowered = table.lower()
        if "cupti" not in lowered or not ("runtime" in lowered or "driver" in lowered):
            continue
        columns = _columns(conn, table)
        start_column = _first(columns, ("start", "startNs", "startTime"))
        end_column = _first(columns, ("end", "endNs", "endTime"))
        correlation_column = _first(columns, ("correlationId", "correlation_id"))
        if start_column is None or end_column is None or correlation_column is None:
            continue
        table_used = False
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if row[start_column] is None or row[correlation_column] is None:
                continue
            start = int(row[start_column])
            end_value = row[end_column]
            events.append(
                ApiEvent(
                    start=start,
                    end=int(end_value) if end_value is not None else start,
                    correlation_id=int(row[correlation_column]),
                    name=_row_text(row, strings),
                    global_tid=_row_int(row, ("globalTid", "tid", "threadId")),
                    pid=_row_int(row, ("globalPid", "pid", "processId")),
                    source_table=table,
                )
            )
            table_used = True
        if table_used:
            used_tables.append(table)
    events.sort(key=lambda item: item.start)
    return events, used_tables


def _kernel_name(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    keys = {key.lower(): key for key in row.keys()}
    for candidate in (
        "demangledName",
        "shortName",
        "mangledName",
        "name",
        "demangledNameId",
        "shortNameId",
        "mangledNameId",
        "nameId",
    ):
        actual = keys.get(candidate.lower())
        if actual is None or row[actual] is None:
            continue
        value = row[actual]
        if isinstance(value, int):
            return strings.get(value, str(value))
        return str(value)
    return None


def _load_kernel_events(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> tuple[list[KernelEvent], list[str]]:
    events: list[KernelEvent] = []
    used_tables: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for table in _tables(conn):
        lowered = table.lower()
        if "cupti" not in lowered or "kernel" not in lowered or "runtime" in lowered:
            continue
        columns = _columns(conn, table)
        start_column = _first(columns, ("start", "startNs", "startTime"))
        end_column = _first(columns, ("end", "endNs", "endTime"))
        correlation_column = _first(columns, ("correlationId", "correlation_id"))
        if start_column is None or end_column is None or correlation_column is None:
            continue
        table_used = False
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if (
                row[start_column] is None
                or row[end_column] is None
                or row[correlation_column] is None
            ):
                continue
            name = _kernel_name(row, strings)
            if not name:
                continue
            event = KernelEvent(
                start=int(row[start_column]),
                end=int(row[end_column]),
                correlation_id=int(row[correlation_column]),
                name=name,
                pid=_row_int(row, ("globalPid", "pid", "processId")),
                device_id=_row_int(row, ("deviceId", "device", "device_id")),
                stream_id=_row_int(row, ("streamId", "stream", "stream_id")),
                source_table=table,
            )
            key = (
                event.start,
                event.end,
                event.correlation_id,
                event.name,
                event.pid,
                event.device_id,
                event.stream_id,
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
            table_used = True
        if table_used:
            used_tables.append(table)
    events.sort(key=lambda item: item.start)
    return events, used_tables


def _find_enclosing(
    ranges: Iterable[NvtxRange], timestamp: int, global_tid: int | None
) -> NvtxRange | None:
    candidates = [item for item in ranges if item.start <= timestamp <= item.end]
    if global_tid is not None:
        same_thread = [item for item in candidates if item.global_tid in {None, global_tid}]
        if same_thread:
            candidates = same_thread
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.end - item.start)


def _same_process(api: ApiEvent, kernel: KernelEvent) -> bool:
    return api.pid is None or kernel.pid is None or api.pid == kernel.pid


def _load_capture_events(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    events: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "execution_capture":
            continue
        capture_id = str(event.get("capture_id"))
        events[capture_id] = event
    return events


def analyze_sqlite(
    sqlite_path: Path, capture_events_path: Path | None = None
) -> dict[str, Any]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        strings = _load_string_ids(conn)
        ranges, nvtx_tables = _load_nvtx_ranges(conn, strings)
        api_events, api_tables = _load_api_events(conn, strings)
        kernel_events, kernel_tables = _load_kernel_events(conn, strings)
        table_names = _tables(conn)
    finally:
        conn.close()

    capture_events = _load_capture_events(capture_events_path)
    high_ranges = [item for item in ranges if item.fields.get("type") == "high"]
    execution_ranges = [
        item for item in ranges if item.fields.get("type") == "execution"
    ]
    if not high_ranges:
        raise RuntimeError(
            "No KID high-level NVTX ranges found. "
            f"SQLite tables: {', '.join(table_names)}"
        )
    if not kernel_events:
        raise RuntimeError(
            "No CUDA GPU kernel activities found. "
            f"SQLite tables: {', '.join(table_names)}"
        )

    apis_by_correlation: dict[int, list[ApiEvent]] = defaultdict(list)
    for event in api_events:
        apis_by_correlation[event.correlation_id].append(event)

    high_index = {id(item): index for index, item in enumerate(high_ranges)}
    execution_index = {
        id(item): index for index, item in enumerate(execution_ranges)
    }
    kernel_rows: list[dict[str, Any]] = []
    kernels_by_high: dict[int, list[dict[str, Any]]] = defaultdict(list)
    kernels_by_execution: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ordinal, kernel in enumerate(kernel_events, start=1):
        matches: list[tuple[NvtxRange, NvtxRange | None, ApiEvent]] = []
        for api in apis_by_correlation.get(kernel.correlation_id, []):
            if not _same_process(api, kernel):
                continue
            high = _find_enclosing(high_ranges, api.start, api.global_tid)
            if high is None:
                continue
            execution = _find_enclosing(
                execution_ranges, api.start, api.global_tid
            )
            if (
                execution is not None
                and execution.fields.get("parent_call_id")
                != high.fields.get("call_id")
            ):
                execution = None
            matches.append((high, execution, api))
        if not matches:
            # Warmup and post-high synchronize/correctness kernels land here.
            continue

        # Prefer a launch API in the deepest execution range; if both Runtime and
        # Driver records match, the latest launch timestamp is closest to GPU.
        high, execution, api = max(
            matches,
            key=lambda item: (item[1] is not None, item[2].start),
        )
        pid = kernel.pid if kernel.pid is not None else api.pid
        kernel_id = f"p{pid or 0}-c{kernel.correlation_id}-k{ordinal}"
        row = {
            "kernel_id": kernel_id,
            "correlation_id": kernel.correlation_id,
            "name": kernel.name,
            "pid": pid,
            "device_id": kernel.device_id,
            "stream_id": kernel.stream_id,
            "gpu_start_ns": kernel.start,
            "gpu_end_ns": kernel.end,
            "duration_us": kernel.duration_us,
            "launch_api": api.name,
            "launch_api_start_ns": api.start,
            "launch_api_table": api.source_table,
            "high_call_id": high.fields.get("call_id"),
            "execution_capture_id": (
                execution.fields.get("capture_id") if execution else None
            ),
        }
        kernel_rows.append(row)
        kernels_by_high[high_index[id(high)]].append(row)
        if execution is not None:
            kernels_by_execution[execution_index[id(execution)]].append(row)

    invocations: list[dict[str, Any]] = []
    for high_position, high in enumerate(high_ranges):
        high_kernels = kernels_by_high.get(high_position, [])
        high_total_us = sum(item["duration_us"] for item in high_kernels)
        child_ranges = [
            execution
            for execution in execution_ranges
            if execution.fields.get("parent_call_id") == high.fields.get("call_id")
            and high.start <= execution.start <= execution.end <= high.end
            and (
                high.global_tid is None
                or execution.global_tid is None
                or high.global_tid == execution.global_tid
            )
        ]
        child_ranges.sort(key=lambda item: item.start)

        execution_entries: list[dict[str, Any]] = []
        for execution in child_ranges:
            execution_kernels = kernels_by_execution.get(
                execution_index[id(execution)], []
            )
            execution_total_us = sum(
                item["duration_us"] for item in execution_kernels
            )
            capture_id = execution.fields.get("capture_id")
            event = capture_events.get(str(capture_id), {})
            execution_entries.append(
                {
                    "capture_id": capture_id,
                    "parent_capture_id": execution.fields.get(
                        "parent_capture_id", event.get("parent_capture_id")
                    ),
                    "parent_call_id": execution.fields.get("parent_call_id"),
                    "archetype": execution.fields.get(
                        "archetype", event.get("archetype", "unknown")
                    ),
                    "common_interface": event.get("common_interface"),
                    "execution_interface": execution.fields.get(
                        "interface", event.get("execution_interface", "unknown")
                    ),
                    "provider": execution.fields.get(
                        "provider", event.get("provider")
                    ),
                    "workload_case": event.get("workload_case"),
                    "semantic_target_hint": event.get("semantic_target_hint"),
                    "nvtx_cpu_duration_us": execution.duration_us,
                    "kernel_ids": [
                        item["kernel_id"] for item in execution_kernels
                    ],
                    "gpu_kernel_sum_us": execution_total_us,
                    "share_of_high_gpu": (
                        execution_total_us / high_total_us
                        if high_total_us > 0
                        else 0.0
                    ),
                    "python_stack": event.get("python_stack", []),
                    "execution_leaf": event.get("execution_leaf"),
                    "implementation": event.get("implementation", {}),
                    "child_capture_ids": [],
                    "start_ns": execution.start,
                }
            )

        by_capture_id = {
            str(entry["capture_id"]): entry for entry in execution_entries
        }
        for entry in execution_entries:
            parent = by_capture_id.get(str(entry.get("parent_capture_id")))
            if parent is not None:
                parent["child_capture_ids"].append(entry["capture_id"])

        kernel_by_id = {item["kernel_id"]: item for item in high_kernels}

        def populate_inclusive_metrics(
            entry: dict[str, Any], visiting: set[str]
        ) -> list[str]:
            capture_id = str(entry["capture_id"])
            if capture_id in visiting:
                return list(entry["kernel_ids"])
            inclusive_ids = list(entry["kernel_ids"])
            next_visiting = {*visiting, capture_id}
            for child_id in entry["child_capture_ids"]:
                child = by_capture_id.get(str(child_id))
                if child is None:
                    continue
                for kernel_id in populate_inclusive_metrics(child, next_visiting):
                    if kernel_id not in inclusive_ids:
                        inclusive_ids.append(kernel_id)
            entry["inclusive_kernel_ids"] = inclusive_ids
            inclusive_us = sum(
                kernel_by_id[kernel_id]["duration_us"]
                for kernel_id in inclusive_ids
                if kernel_id in kernel_by_id
            )
            entry["inclusive_gpu_kernel_sum_us"] = inclusive_us
            entry["inclusive_share_of_high_gpu"] = (
                inclusive_us / high_total_us if high_total_us > 0 else 0.0
            )
            entry["attribution_role"] = (
                "kernel_owner" if entry["kernel_ids"] else "ancestor_context"
            )
            return inclusive_ids

        for entry in execution_entries:
            populate_inclusive_metrics(entry, set())

        # Dispatcher also observes metadata/allocation ops such as empty_like.
        # They remain in capture_events.jsonl for debugging.  A capture with no
        # direct kernel is still retained when a nested child owns GPU work.
        capture_without_kernel_count = sum(
            not entry["inclusive_kernel_ids"] for entry in execution_entries
        )
        execution_entries = [
            entry for entry in execution_entries if entry["inclusive_kernel_ids"]
        ]

        rank_by_call = {
            id(entry): rank
            for rank, entry in enumerate(
                sorted(
                    execution_entries,
                    key=lambda item: item["gpu_kernel_sum_us"],
                    reverse=True,
                ),
                start=1,
            )
        }
        for entry in execution_entries:
            entry["hotspot_rank"] = rank_by_call[id(entry)]
            entry.pop("start_ns", None)

        attributed_ids = {
            kernel_id
            for entry in execution_entries
            for kernel_id in entry["kernel_ids"]
        }
        unattributed = [
            item for item in high_kernels if item["kernel_id"] not in attributed_ids
        ]
        attributed_us = sum(
            item["duration_us"]
            for item in high_kernels
            if item["kernel_id"] in attributed_ids
        )
        invocations.append(
            {
                "high_level": {
                    "call_id": high.fields.get("call_id"),
                    "interface": high.fields.get("interface", "unknown"),
                    "nvtx_cpu_duration_us": high.duration_us,
                    "kernel_ids": [item["kernel_id"] for item in high_kernels],
                    "gpu_kernel_sum_us": high_total_us,
                },
                "execution_targets": execution_entries,
                "capture_without_kernel_count": capture_without_kernel_count,
                "unattributed_kernel_ids": [item["kernel_id"] for item in unattributed],
                "coverage": attributed_us / high_total_us if high_total_us > 0 else 0.0,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_definition": {
            "gpu_kernel_sum_us": "sum of Nsight GPU kernel activity durations; not end-to-end wall time",
            "nvtx_cpu_duration_us": "CPU-side NVTX push-to-pop duration",
            "coverage": "execution-capture-attributed GPU duration / all high-level GPU duration",
        },
        "invocations": invocations,
        "kernels": kernel_rows,
        "diagnostics": {
            "nvtx_tables": nvtx_tables,
            "cuda_api_tables": api_tables,
            "kernel_tables": kernel_tables,
            "high_range_count": len(high_ranges),
            "execution_range_count": len(execution_ranges),
            "capture_event_count": len(capture_events),
            "cuda_api_event_count": len(api_events),
            "trace_kernel_count": len(kernel_events),
            "high_related_kernel_count": len(kernel_rows),
        },
    }


def _kernel_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["kernel_id"]: item for item in result.get("kernels", [])}


def format_summary(result: dict[str, Any], raw_paths: dict[str, Path] | None = None) -> str:
    kernels = _kernel_by_id(result)
    lines = [
        "KID Nsight Systems PoC",
        "========================",
        "GPU hotspot metric: sum of Nsight GPU kernel activity durations.",
        "NVTX durations below are CPU-side ranges and are not GPU execution time.",
        "",
    ]
    if raw_paths:
        lines.append("Artifacts:")
        for name, path in raw_paths.items():
            size = path.stat().st_size if path.exists() else 0
            lines.append(f"  {name}: {path} ({size / (1024 * 1024):.2f} MiB)")
        lines.append("")

    for invocation_number, invocation in enumerate(result.get("invocations", []), start=1):
        high = invocation["high_level"]
        lines.append(
            f"High invocation {invocation_number}: {high['interface']} "
            f"(call_id={high['call_id']})"
        )
        lines.append(
            f"  CPU NVTX={high['nvtx_cpu_duration_us']:.3f} us | "
            f"GPU kernel sum={high['gpu_kernel_sum_us']:.3f} us | "
            f"coverage={invocation['coverage']:.2%}"
        )
        if invocation.get("capture_without_kernel_count"):
            lines.append(
                "  Raw common-interface captures without a GPU kernel: "
                f"{invocation['capture_without_kernel_count']} "
                "(retained in capture_events.jsonl)"
            )
        execution_entries = sorted(
            invocation.get("execution_targets", []),
            key=lambda item: item["hotspot_rank"],
        )
        lines.append("  Execution captures (semantic target unresolved):")
        for index, execution in enumerate(execution_entries):
            branch = "└──" if index == len(execution_entries) - 1 else "├──"
            child_prefix = "    " if index == len(execution_entries) - 1 else "│   "
            lines.append(
                f"  {branch} #{execution['hotspot_rank']} "
                f"[{execution['archetype']}] {execution['execution_interface']} "
                f"GPU_direct={execution['gpu_kernel_sum_us']:.3f} us "
                f"share={execution['share_of_high_gpu']:.2%} "
                f"CPU_NVTX={execution['nvtx_cpu_duration_us']:.3f} us"
            )
            if execution.get("workload_case"):
                lines.append(
                    f"  {child_prefix}case={execution['workload_case']} | "
                    f"semantic_hint={execution.get('semantic_target_hint')} | "
                    f"provider={execution.get('provider')}"
                )
            if execution.get("child_capture_ids"):
                lines.append(
                    f"  {child_prefix}nested children={execution['child_capture_ids']} | "
                    "GPU_inclusive="
                    f"{execution['inclusive_gpu_kernel_sum_us']:.3f} us"
                )
            path = " -> ".join(
                frame.get("qualname") or frame.get("function", "unknown")
                for frame in execution.get("python_stack", [])
            )
            if path:
                lines.append(f"  {child_prefix}path: {path}")
            for kernel_id in execution["kernel_ids"]:
                kernel = kernels[kernel_id]
                name = kernel["name"]
                if len(name) > 160:
                    name = name[:157] + "..."
                lines.append(
                    f"  {child_prefix}└── {kernel_id} "
                    f"{kernel['duration_us']:.3f} us  {name}"
                )
        unattributed = invocation.get("unattributed_kernel_ids", [])
        if unattributed:
            lines.append("  Unattributed kernels:")
            for kernel_id in unattributed:
                kernel = kernels[kernel_id]
                name = kernel["name"]
                if len(name) > 160:
                    name = name[:157] + "..."
                lines.append(
                    f"    - {kernel_id} {kernel['duration_us']:.3f} us  {name}"
                )
        lines.append("")

    diagnostics = result.get("diagnostics", {})
    case_validation = result.get("case_validation")
    if case_validation:
        lines.append("Case validation:")
        for name, status in case_validation.get("cases", {}).items():
            lines.append(
                f"  {'PASS' if status.get('passed') else 'FAIL'} {name}: "
                f"expected={status.get('expected_archetype')} "
                f"observed={status.get('observed_owner_archetypes')} "
                f"kernels={len(status.get('kernel_ids', []))}"
            )
        lines.append("")
    lines.extend(
        [
            "Trace diagnostics:",
            f"  NVTX tables: {diagnostics.get('nvtx_tables')}",
            f"  CUDA API tables: {diagnostics.get('cuda_api_tables')}",
            f"  Kernel tables: {diagnostics.get('kernel_tables')}",
            f"  Trace kernels / high-related kernels: "
            f"{diagnostics.get('trace_kernel_count')} / "
            f"{diagnostics.get('high_related_kernel_count')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_worker(
    size: int,
    case_names: Sequence[str],
    worker_log: Path | None,
    capture_events_path: Path | None,
    invocation_variants: Sequence[str] | None = None,
) -> int:
    global _CAPTURE_EVENTS_PATH
    if not _WORKER_MODE:
        raise RuntimeError("worker workload was not initialized")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the worker")

    _CAPTURE_EVENTS_PATH = capture_events_path
    if _CAPTURE_EVENTS_PATH is not None:
        _CAPTURE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CAPTURE_EVENTS_PATH.unlink(missing_ok=True)
    hook_status = (
        {name: True for name in POC_CAPTURE_REGISTRY}
        if os.environ.get("KID_ENABLE") == "1"
        else _install_all_capture_adapters()
    )
    if not all(hook_status.values()):
        raise RuntimeError(f"capture adapter installation failed: {hook_status}")

    torch.manual_seed(0)
    variant_sequence = list(invocation_variants or ["old"])
    cases_by_variant: dict[str, list[WorkloadCase]] = {}
    for variant in dict.fromkeys(variant_sequence):
        names = (
            list(case_names)
            if invocation_variants is None
            else _variant_case_names(variant)
        )
        cases_by_variant[variant] = _build_workload_cases(names, size)

    # Warmup is outside any high-level context.  It compiles and initializes
    # every unique variant but emits no KID range or capture event.
    for variant in dict.fromkeys(variant_sequence):
        with _high_capture_mode():
            warm_outputs = _run_cases(cases_by_variant[variant])
        torch.cuda.synchronize()
        del warm_outputs

    all_results: list[Any] = []
    for variant in variant_sequence:
        results = high_level(cases_by_variant[variant])
        # Synchronize after high_level's NVTX pop.  This demonstrates why GPU
        # execution timestamps need correlation ids instead of range overlap.
        torch.cuda.synchronize()
        all_results.extend(results)
    checksum = 0.0
    for result in all_results:
        tensor = _first_tensor(result)
        if tensor is not None and tensor.numel():
            checksum += float(tensor.reshape(-1)[0].float().item())
    message = (
        f"worker_ok cases={','.join(case_names)} "
        f"invocation_variants={','.join(variant_sequence)} size={size} "
        f"device={torch.cuda.get_device_name(0)} "
        f"torch={torch.__version__} triton={triton.__version__} "
        f"checksum={checksum:.6f} hooks={json.dumps(hook_status, sort_keys=True)}\n"
    )
    print(message, end="", flush=True)
    if worker_log is not None:
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        worker_log.write_text(message, encoding="utf-8")
    return 0


def _run_logged(command: list[str], log_path: Path, *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}; "
            f"see {log_path}"
        )


def _find_report(output_dir: Path) -> Path:
    expected = output_dir / "profile.nsys-rep"
    if expected.exists():
        return expected
    candidates = sorted(output_dir.glob("profile*.nsys-rep")) + sorted(
        output_dir.glob("profile*.qdrep")
    )
    if not candidates:
        raise RuntimeError(f"nsys did not create a report under {output_dir}")
    return candidates[0]


def _find_sqlite(requested: Path) -> Path:
    if requested.exists():
        return requested
    alternatives = [
        requested.with_suffix(requested.suffix + ".sqlite"),
        requested.parent / (requested.stem + ".sqlite"),
    ]
    for candidate in alternatives:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"nsys export did not create SQLite output near {requested}")


PROBE_MODULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("torch", "torch", ("compile",)),
    ("triton", "triton", ("jit",)),
    ("sglang", "sglang", ()),
    ("sgl_kernel", "sglang-kernel", ("silu_and_mul",)),
    ("flashinfer", "flashinfer_python", ()),
    ("deep_gemm", "sgl-deep-gemm", ("bf16_gemm_nt",)),
    ("cutlass", "nvidia-cutlass-dsl", ()),
    ("tilelang", "tilelang", ("jit", "JITKernel")),
    ("quack", "quack-kernels", ()),
    ("tokenspeed_mla", "tokenspeed_mla", ("tokenspeed_mla_decode",)),
    ("flash_attn", "flash-attn-4", ()),
    ("tvm_ffi", "apache-tvm-ffi", ("Module", "load_module")),
)

CASE_MODULE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "pytorch_native": ("torch",),
    "pytorch_softmax": ("torch",),
    "sgl_kernel_builtin": ("torch", "sgl_kernel"),
    "sgl_kernel_sgl_attn": ("torch", "sgl_kernel"),
    "sglang_triton": ("torch", "triton", "sglang"),
    "flashinfer_triton": ("torch", "triton", "flashinfer"),
    "sglang_jit": ("torch", "sglang", "tvm_ffi"),
    "flashinfer_ffi": ("torch", "flashinfer", "tvm_ffi"),
    "sglang_cutedsl": ("torch", "sglang", "cutlass"),
    "deepgemm_binding": ("torch", "deep_gemm"),
    "sglang_tilelang": ("torch", "sglang", "tilelang"),
    "sglang_inductor": ("torch", "sglang", "triton"),
    "flash_attn4_cutedsl": ("torch", "flash_attn", "cutlass"),
    "tokenspeed_mla_cutedsl": ("torch", "tokenspeed_mla", "cutlass"),
}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _static_environment_probe(nsys_bin: str) -> dict[str, Any]:
    import torch

    packages: dict[str, dict[str, Any]] = {}
    for module_name, distribution, attrs in PROBE_MODULES:
        spec = importlib.util.find_spec(module_name)
        entry: dict[str, Any] = {
            "distribution": distribution,
            "version": _distribution_version(distribution),
            "present": spec is not None,
            "origin": getattr(spec, "origin", None),
            "required_attributes": list(attrs),
        }
        if spec is not None:
            try:
                module = importlib.import_module(module_name)
                entry["attributes"] = {
                    attr: hasattr(module, attr) for attr in attrs
                }
                entry["import_error"] = None
            except Exception as exc:
                entry["attributes"] = {attr: False for attr in attrs}
                entry["import_error"] = f"{type(exc).__name__}: {exc}"
        packages[module_name] = entry

    resolved_nsys = shutil.which(nsys_bin) if os.path.sep not in nsys_bin else nsys_bin
    nsys_version = None
    if resolved_nsys and Path(resolved_nsys).exists():
        version_result = subprocess.run(
            [resolved_nsys, "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        nsys_version = (version_result.stdout or version_result.stderr).strip()

    cuda_available = bool(torch.cuda.is_available())
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    result: dict[str, Any] = {
        "schema_version": "kid-nsys-poc-env/v1",
        "created_at_unix": time.time(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "gpu": {
            "cuda_available": cuda_available,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "compute_capability": list(capability) if capability else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        },
        "nsight_systems": {
            "requested": nsys_bin,
            "resolved": resolved_nsys,
            "version": nsys_version,
        },
        "packages": packages,
        "cases": {},
    }

    for name, contract in CASE_CONTRACTS.items():
        reasons: list[str] = []
        for module_name in CASE_MODULE_REQUIREMENTS[name]:
            package = packages.get(module_name, {})
            if not package.get("present"):
                reasons.append(f"missing module {module_name}")
            elif package.get("import_error"):
                reasons.append(
                    f"cannot import {module_name}: {package['import_error']}"
                )
        minimum = contract.get("minimum_compute_capability")
        if minimum and (capability is None or tuple(capability) < tuple(minimum)):
            reasons.append(
                "requires compute capability "
                f"{minimum[0]}.{minimum[1]} or newer"
            )
        result["cases"][name] = {
            **contract,
            "static_supported": not reasons,
            "static_reasons": reasons,
            "smoke": None,
        }
    return result


def _parse_requested_cases(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(ALL_CASE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown --cases values: {unknown}; allowed={list(ALL_CASE_NAMES)}"
        )
    return list(dict.fromkeys(names))


def _parse_invocation_variants(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(INVOCATION_VARIANTS))
    if unknown:
        raise ValueError(
            "unknown --invocation-variants values: "
            f"{unknown}; allowed={list(INVOCATION_VARIANTS)}"
        )
    if not variants:
        raise ValueError("--invocation-variants must contain at least one variant")
    return variants


def _variant_case_names(variant: str) -> list[str]:
    if variant == "old":
        return list(MANDATORY_CASE_NAMES)
    if variant == "softmax":
        return [
            "pytorch_softmax" if name == "pytorch_native" else name
            for name in MANDATORY_CASE_NAMES
        ]
    raise ValueError(f"unknown invocation variant: {variant}")


def _requested_case_names(args: argparse.Namespace) -> list[str] | None:
    variants = _parse_invocation_variants(args.invocation_variants)
    if variants is None:
        return _parse_requested_cases(args.cases)
    if args.include_optional:
        raise ValueError(
            "--include-optional cannot be combined with --invocation-variants"
        )
    return list(
        dict.fromkeys(
            name for variant in variants for name in _variant_case_names(variant)
        )
    )


def _select_cases(
    probe: dict[str, Any],
    requested: list[str] | None,
    include_optional: bool,
) -> list[str]:
    candidates = (
        requested
        if requested is not None
        else list(MANDATORY_CASE_NAMES)
        + (list(OPTIONAL_CASE_NAMES) if include_optional else [])
    )
    selected: list[str] = []
    failures: list[str] = []
    for name in candidates:
        status = probe["cases"][name]
        if status["static_supported"]:
            selected.append(name)
        elif not CASE_CONTRACTS[name]["optional"]:
            failures.append(f"{name}: {', '.join(status['static_reasons'])}")
    if failures:
        raise RuntimeError("mandatory environment probe failed: " + "; ".join(failures))
    if not selected:
        raise RuntimeError("no supported workload cases selected")
    return selected


def _run_probe_worker(
    size: int, case_names: Sequence[str], probe_json: Path
) -> int:
    if not _PROBE_WORKER_MODE:
        raise RuntimeError("probe worker workload was not initialized")
    probe = json.loads(probe_json.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the probe worker")
    hook_status = _install_all_capture_adapters()
    probe["capture_adapters"] = hook_status
    failed_mandatory: list[str] = []
    torch.manual_seed(0)
    for name in case_names:
        started = time.perf_counter()
        try:
            cases = _build_workload_cases([name], size)
            with _high_capture_mode():
                output = _run_cases(cases)[0]
            torch.cuda.synchronize()
            tensor = _first_tensor(output)
            smoke = {
                "status": "passed",
                "elapsed_s": time.perf_counter() - started,
                "output_shape": list(tensor.shape) if tensor is not None else None,
                "output_dtype": str(tensor.dtype) if tensor is not None else None,
            }
        except Exception as exc:
            smoke = {
                "status": "failed",
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if not CASE_CONTRACTS[name]["optional"]:
                failed_mandatory.append(name)
        probe["cases"][name]["smoke"] = smoke
        probe_json.write_text(
            json.dumps(probe, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[probe] {name}: {smoke}", flush=True)
    return 1 if failed_mandatory else 0


def _prepare_environment(
    args: argparse.Namespace, output_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    probe_path = output_dir / "environment_probe.json"
    probe_log = output_dir / "probe.log"
    probe = _static_environment_probe(args.nsys_bin)
    requested = _requested_case_names(args)
    selected = _select_cases(probe, requested, args.include_optional)
    probe["selected_cases"] = selected
    probe_path.write_text(
        json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--probe-worker",
        "--size",
        str(args.size),
        "--cases",
        ",".join(selected),
        "--probe-json",
        str(probe_path),
    ]
    _run_logged(command, probe_log)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    return probe, selected


def _preflight(nsys_bin: str) -> str:
    resolved_nsys = shutil.which(nsys_bin) if os.path.sep not in nsys_bin else nsys_bin
    if not resolved_nsys or not Path(resolved_nsys).exists():
        raise RuntimeError(f"Nsight Systems binary not found: {nsys_bin}")
    check_code = (
        "import torch, triton; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "print(f'torch={torch.__version__} triton={triton.__version__} '"
        "+ f'device={torch.cuda.get_device_name(0)}')"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "GPU Python preflight failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
    return result.stdout.strip()


def _validate_profile_cases(
    result: dict[str, Any], selected_cases: Sequence[str]
) -> dict[str, Any]:
    entries = [
        entry
        for invocation in result.get("invocations", [])
        for entry in invocation.get("execution_targets", [])
    ]
    validation: dict[str, Any] = {"passed": True, "cases": {}}
    for name in selected_cases:
        contract = CASE_CONTRACTS[name]
        case_entries = [entry for entry in entries if entry.get("workload_case") == name]
        owners = [entry for entry in case_entries if entry.get("kernel_ids")]
        archetypes = sorted({str(entry.get("archetype")) for entry in owners})
        kernel_ids = sorted(
            {
                kernel_id
                for entry in owners
                for kernel_id in entry.get("kernel_ids", [])
            }
        )
        expected = str(contract["expected_archetype"])
        passed = bool(owners and kernel_ids and expected in archetypes)
        case_result = {
            "passed": passed,
            "expected_archetype": expected,
            "observed_owner_archetypes": archetypes,
            "capture_count": len(case_entries),
            "kernel_ids": kernel_ids,
            "semantic_target": contract["semantic_target"],
            "provider": contract["provider"],
        }
        validation["cases"][name] = case_result
        if not passed and not contract["optional"]:
            validation["passed"] = False
    for invocation in result.get("invocations", []):
        if invocation.get("coverage") != 1.0:
            validation["passed"] = False
            validation.setdefault("coverage_failures", []).append(
                {
                    "call_id": invocation["high_level"].get("call_id"),
                    "coverage": invocation.get("coverage"),
                    "unattributed_kernel_ids": invocation.get(
                        "unattributed_kernel_ids", []
                    ),
                }
            )
    return validation


def run_launcher(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    script_path = Path(__file__).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    preflight = _preflight(args.nsys_bin)
    print(f"[preflight] {preflight}", flush=True)
    print("[probe] inspecting packages and prewarming selected cases...", flush=True)
    environment_probe, selected_cases = _prepare_environment(args, output_dir)
    print(f"[probe] selected cases: {', '.join(selected_cases)}", flush=True)

    nsys_log = output_dir / "nsys.log"
    worker_log = output_dir / "workload.log"
    capture_events_path = output_dir / "capture_events.jsonl"
    profile_base = output_dir / "profile"
    profile_command = [
        args.nsys_bin,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        f"--output={profile_base}",
        sys.executable,
        "-u",
        str(script_path),
        "--worker",
        "--size",
        str(args.size),
    ]
    if args.invocation_variants:
        profile_command.extend(
            ["--invocation-variants", str(args.invocation_variants)]
        )
    else:
        profile_command.extend(["--cases", ",".join(selected_cases)])
    profile_command.extend([
        "--worker-log",
        str(worker_log),
        "--capture-events",
        str(capture_events_path),
    ])
    print("[1/3] profiling SGLang backend cases with Nsight Systems...", flush=True)
    _run_logged(profile_command, nsys_log)
    report_path = _find_report(output_dir)

    requested_sqlite = output_dir / "profile.sqlite"
    export_command = [
        args.nsys_bin,
        "export",
        "--force-overwrite=true",
        "--type=sqlite",
        f"--output={requested_sqlite}",
        str(report_path),
    ]
    print("[2/3] exporting Nsight report to SQLite...", flush=True)
    _run_logged(export_command, nsys_log, append=True)
    sqlite_path = _find_sqlite(requested_sqlite)

    print("[3/3] joining NVTX, CUDA API, and GPU kernel events...", flush=True)
    result = analyze_sqlite(sqlite_path, capture_events_path)
    result["case_validation"] = _validate_profile_cases(result, selected_cases)
    result["run"] = {
        "script": str(script_path),
        "size": args.size,
        "preflight": preflight,
        "selected_cases": selected_cases,
        "environment_probe_schema": environment_probe.get("schema_version"),
        "created_at_unix": time.time(),
        "nsys_report": str(report_path),
        "nsys_sqlite": str(sqlite_path),
    }
    decomposition_path = output_dir / "decomposition.json"
    decomposition_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raw_paths = {
        "summary": output_dir / "summary.log",
        "decomposition": decomposition_path,
        "workload_log": worker_log,
        "capture_events": capture_events_path,
        "nsys_log": nsys_log,
        "probe_log": output_dir / "probe.log",
        "environment_probe": output_dir / "environment_probe.json",
        "nsys_report": report_path,
        "nsys_sqlite": sqlite_path,
    }
    summary = format_summary(result, raw_paths)
    summary_path = output_dir / "summary.log"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary, end="", flush=True)
    print(f"\nCompact results: {summary_path} and {decomposition_path}", flush=True)
    if not result["case_validation"]["passed"]:
        raise RuntimeError(
            "profile case validation failed; inspect decomposition.json "
            "case_validation and summary.log"
        )
    return 0


def _create_synthetic_sqlite(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, text TEXT, globalTid INTEGER, globalPid INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, correlationId INTEGER,
                nameId INTEGER, globalTid INTEGER, globalPid INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, correlationId INTEGER,
                shortName INTEGER, globalPid INTEGER, deviceId INTEGER, streamId INTEGER
            );
            """
        )
        strings = [
            (1, "cudaLaunchKernel"),
            (101, "kernel_a"),
            (102, "kernel_b"),
            (103, "kernel_c"),
            (104, "kernel_d"),
            (105, "unattributed_kernel"),
        ]
        conn.executemany("INSERT INTO StringIds VALUES (?, ?)", strings)
        ranges = [
            (0, 70_000, "KID:type=high|call_id=1|interface=high_level", 77, 7),
            (
                5_000,
                30_000,
                "KID:type=execution|capture_id=1|parent_call_id=1|archetype=pytorch_dispatch|interface=custom_op.default|provider=torch",
                77,
                7,
            ),
            (
                8_000,
                25_000,
                "KID:type=execution|capture_id=2|parent_capture_id=1|parent_call_id=1|archetype=triton_launch|interface=nested_kernel|provider=triton",
                77,
                7,
            ),
            (
                35_000,
                60_000,
                "KID:type=execution|capture_id=3|parent_call_id=1|archetype=triton_launch|interface=vector_pipeline|provider=triton",
                77,
                7,
            ),
        ]
        conn.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)", ranges)
        api_rows = [
            (10_000, 10_100, 201, 1, 77, 7),
            (20_000, 20_100, 202, 1, 77, 7),
            (40_000, 40_100, 203, 1, 77, 7),
            (50_000, 50_100, 204, 1, 77, 7),
            (65_000, 65_100, 205, 1, 77, 7),
        ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?, ?)",
            api_rows,
        )
        # Every GPU kernel starts after the high NVTX range ended at 70 us.
        kernel_rows = [
            (120_000, 170_000, 201, 101, 7, 0, 1),
            (170_000, 250_000, 202, 102, 7, 0, 1),
            (250_000, 270_000, 203, 103, 7, 0, 1),
            (270_000, 310_000, 204, 104, 7, 0, 1),
            (310_000, 340_000, 205, 105, 7, 0, 1),
        ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
            kernel_rows,
        )
        conn.commit()
    finally:
        conn.close()


def _test_capture_without_torch(capture_events_path: Path) -> None:
    global _nvtx_push, _nvtx_pop, _HIGH_CALL_COUNTER
    global _EXECUTION_CAPTURE_COUNTER, _CAPTURE_EVENTS_PATH
    original_push = _nvtx_push
    original_pop = _nvtx_pop
    original_capture_events_path = _CAPTURE_EVENTS_PATH
    events: list[tuple[str, str | None]] = []
    _HIGH_CALL_COUNTER = 0
    _EXECUTION_CAPTURE_COUNTER = 0
    _CAPTURE_EVENTS_PATH = capture_events_path
    _nvtx_push = lambda label: events.append(("push", label))  # type: ignore[assignment]
    _nvtx_pop = lambda: events.append(("pop", None))  # type: ignore[assignment]
    try:
        archetypes = list(POC_CAPTURE_REGISTRY)

        def nested_common_interface(index: int) -> int:
            if index == len(archetypes):
                return 1
            archetype = archetypes[index]
            with execution_capture(
                archetype=archetype,
                execution_interface=f"fake.{archetype}",
                provider="fake-provider",
            ):
                return nested_common_interface(index + 1)

        def fake_common_interface() -> int:
            return nested_common_interface(0)

        def semantic_candidate() -> int:
            return fake_common_interface()

        @high_level_target
        def high() -> int:
            return semantic_candidate()

        assert fake_common_interface() == 1
        assert events == [], "capture outside high must not emit NVTX"
        assert high() == 1
        labels = [label for kind, label in events if kind == "push"]
        assert len(labels) == 1 + len(archetypes)
        assert labels[0] and "type=high" in labels[0]
        assert labels[1] and "type=execution" in labels[1]
        assert "parent_call_id=1" in labels[1]
        assert labels[2] and "type=execution" in labels[2]
        assert "parent_capture_id=1" in labels[2]
        assert [kind for kind, _ in events] == (
            ["push"] * (1 + len(archetypes))
            + ["pop"] * (1 + len(archetypes))
        )

        capture_events = _load_capture_events(capture_events_path)
        assert list(capture_events) == [str(i) for i in range(1, 8)]
        assert capture_events["1"]["parent_capture_id"] is None
        assert capture_events["2"]["parent_capture_id"] == "1"
        assert [capture_events[str(i)]["archetype"] for i in range(1, 8)] == archetypes
        event = capture_events["1"]
        functions = [item["function"] for item in event["python_stack"]]
        assert functions[0] == "high"
        assert "semantic_candidate" in functions
        semantic_frame = next(
            item for item in event["python_stack"]
            if item["function"] == "semantic_candidate"
        )
        assert semantic_frame["callsite"]["line"] > 0
    finally:
        _nvtx_push = original_push
        _nvtx_pop = original_pop
        _CAPTURE_EVENTS_PATH = original_capture_events_path
        _CURRENT_HIGH.set(None)
        _CURRENT_EXECUTION_CAPTURE.set(None)
        _CURRENT_WORKLOAD_CASE.set(None)


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="kid_nsys_poc_test_") as tmp:
        capture_events_path = Path(tmp) / "capture_events.jsonl"
        _test_capture_without_torch(capture_events_path)
        # The synthetic SQLite below has its own capture ids/tree encoded in
        # NVTX labels; do not merge the decorator-only test events into it.
        capture_events_path.unlink()
        sqlite_path = Path(tmp) / "synthetic.sqlite"
        _create_synthetic_sqlite(sqlite_path)
        result = analyze_sqlite(sqlite_path, capture_events_path)
    assert result["schema_version"] == SCHEMA_VERSION
    assert len(result["invocations"]) == 1
    invocation = result["invocations"][0]
    assert invocation["high_level"]["gpu_kernel_sum_us"] == 220.0
    assert len(invocation["execution_targets"]) == 3
    outer, inner, independent = invocation["execution_targets"]
    assert outer["gpu_kernel_sum_us"] == 0.0
    assert outer["inclusive_gpu_kernel_sum_us"] == 130.0
    assert outer["child_capture_ids"] == ["2"]
    assert outer["attribution_role"] == "ancestor_context"
    assert inner["gpu_kernel_sum_us"] == 130.0
    assert inner["inclusive_gpu_kernel_sum_us"] == 130.0
    assert inner["parent_capture_id"] == "1"
    assert inner["attribution_role"] == "kernel_owner"
    assert independent["gpu_kernel_sum_us"] == 60.0
    assert outer["archetype"] == "pytorch_dispatch"
    assert inner["archetype"] == "triton_launch"
    assert len(invocation["unattributed_kernel_ids"]) == 1
    assert abs(invocation["coverage"] - (190.0 / 220.0)) < 1e-12
    assert all(kernel["gpu_start_ns"] > 70_000 for kernel in result["kernels"])
    print("SELF-TEST PASS")
    print(format_summary(result), end="")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-file NVTX + Nsight Systems KID decomposition proof of concept"
    )
    parser.add_argument("--size", type=int, default=2048, help="square matrix size")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--self-test", action="store_true", help="run local synthetic tests")
    parser.add_argument(
        "--probe-env",
        action="store_true",
        help="probe the remote GPU environment and smoke selected cases without nsys",
    )
    workload_group = parser.add_mutually_exclusive_group()
    workload_group.add_argument(
        "--cases",
        default=None,
        help="comma-separated workload case names (default: all mandatory cases)",
    )
    workload_group.add_argument(
        "--invocation-variants",
        default=None,
        help=(
            "comma-separated high-level variants; supported values are old and "
            "softmax"
        ),
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="include hardware-supported optional FA4/TokenSpeed cases",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-log", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--capture-events", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.self_test:
            return run_self_test()
        if args.probe_worker:
            names = _parse_requested_cases(args.cases) or list(MANDATORY_CASE_NAMES)
            if not args.probe_json:
                raise ValueError("--probe-worker requires --probe-json")
            return _run_probe_worker(args.size, names, Path(args.probe_json).resolve())
        if args.worker:
            variants = _parse_invocation_variants(args.invocation_variants)
            names = (
                _requested_case_names(args)
                if variants is not None
                else _parse_requested_cases(args.cases)
            ) or list(MANDATORY_CASE_NAMES)
            return _run_worker(
                args.size,
                names,
                Path(args.worker_log).resolve() if args.worker_log else None,
                Path(args.capture_events).resolve() if args.capture_events else None,
                variants,
            )
        if args.probe_env:
            output_dir = Path(args.output_dir).expanduser().resolve()
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            probe, selected = _prepare_environment(args, output_dir)
            print(
                json.dumps(
                    {
                        "selected_cases": selected,
                        "gpu": probe.get("gpu"),
                        "capture_adapters": probe.get("capture_adapters"),
                        "cases": {
                            name: probe["cases"][name] for name in selected
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        return run_launcher(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"KID Nsight PoC failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
