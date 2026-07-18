from __future__ import annotations

import ast
import contextvars
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import threading
import time
import urllib.parse
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable, ContextManager, Iterator

from .capture_registry import CAPTURE_BY_ARCHETYPE


_INSTALLED = False
_CONFIG: dict[str, Any] = {}
_EVENT_LOCK = threading.Lock()
_COUNTER_LOCK = threading.Lock()
_CALL_COUNTER = 0
_CAPTURE_COUNTER = 0
_WRITER_PID = os.getpid()
_ACTIVE_TARGET_FRAMES: dict[tuple[int, int], ContextManager[Any]] = {}
_TORCH_MODE_CLASS: type[Any] | None = None
_CURRENT_HIGH: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "kid_current_high", default=None
)
_CURRENT_CAPTURE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kid_current_capture", default=None
)


def install_from_env() -> None:
    """Install capture hooks from a sitecustomize-injected runtime config."""

    global _INSTALLED, _CONFIG
    if _INSTALLED or os.environ.get("KID_ENABLE") != "1":
        return
    config_path = os.environ.get("KID_RUNTIME_CONFIG")
    if not config_path:
        raise RuntimeError("KID_RUNTIME_CONFIG is required when KID_ENABLE=1")
    _CONFIG = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _INSTALLED = True
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork)
    _install_import_hook()
    _patch_already_imported_modules()
    _install_target_profiler()


def _after_fork() -> None:
    global _EVENT_LOCK, _COUNTER_LOCK, _CALL_COUNTER, _CAPTURE_COUNTER, _WRITER_PID
    _EVENT_LOCK = threading.Lock()
    _COUNTER_LOCK = threading.Lock()
    _CALL_COUNTER = 0
    _CAPTURE_COUNTER = 0
    _WRITER_PID = os.getpid()
    _ACTIVE_TARGET_FRAMES.clear()
    _CURRENT_HIGH.set(None)
    _CURRENT_CAPTURE.set(None)


def _target() -> dict[str, Any]:
    return dict(_CONFIG.get("target") or {})


def _target_file() -> Path:
    return Path(str(_target().get("file", ""))).expanduser().resolve()


def _events_dir() -> Path:
    value = _CONFIG.get("events_dir") or Path(_CONFIG.get("output_dir", ".")) / "capture_events"
    return Path(str(value)).expanduser().resolve()


def _next_id(kind: str) -> str:
    global _CALL_COUNTER, _CAPTURE_COUNTER
    with _COUNTER_LOCK:
        if kind == "high":
            _CALL_COUNTER += 1
            counter = _CALL_COUNTER
            prefix = "h"
        else:
            _CAPTURE_COUNTER += 1
            counter = _CAPTURE_COUNTER
            prefix = "c"
    return f"p{os.getpid()}-{prefix}{counter}"


def _label(kind: str, **fields: Any) -> str:
    components = [f"KID:type={kind}"]
    for key, value in fields.items():
        if value is None:
            continue
        encoded = urllib.parse.quote(str(value).replace("\n", " ")[:1000], safe="/.:_-<>")
        components.append(f"{key}={encoded}")
    return "|".join(components)


def _nvtx_push(text: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(text)
    except Exception:
        pass


def _nvtx_pop() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _write_capture_event(event: dict[str, Any]) -> None:
    global _WRITER_PID
    if _WRITER_PID != os.getpid():
        _after_fork()
    out = _events_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"events_{os.getpid()}.jsonl"
    record = {
        **event,
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
    }
    with _EVENT_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _locate_target_definition() -> tuple[str | None, int | None]:
    explicit = _target().get("qualified_name")
    path = _target_file()
    requested_line = int(_target().get("line", 0))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return (str(explicit) if explicit else None, None)
    candidates: list[tuple[int, str, int]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = int(getattr(node, "lineno", 0))
            end = int(getattr(node, "end_lineno", start))
            next_parents = [*parents, node.name]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and start <= requested_line <= end:
                candidates.append((end - start, ".".join(next_parents), start))
            for child in ast.iter_child_nodes(node):
                visit(child, next_parents)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, [])
    if explicit:
        explicit_text = str(explicit)
        matching = [item for item in candidates if item[1] == explicit_text or item[1].endswith("." + explicit_text)]
        if matching:
            best = min(matching)
            return explicit_text, best[2]
        return explicit_text, None
    if not candidates:
        return None, None
    best = min(candidates)
    return best[1], best[2]


def _find_forward_mode(value: Any, depth: int = 0) -> Any:
    if depth > 2 or value is None:
        return None
    try:
        if hasattr(value, "forward_mode"):
            return value.forward_mode
    except Exception:
        return None
    values = value.values() if isinstance(value, dict) else value if isinstance(value, (tuple, list)) else ()
    for item in values:
        found = _find_forward_mode(item, depth + 1)
        if found is not None:
            return found
    return None


def _stage_from_frame(frame: FrameType) -> tuple[str, str | None]:
    mode = None
    for value in frame.f_locals.values():
        mode = _find_forward_mode(value)
        if mode is not None:
            break
    if mode is None:
        return "unknown", None
    name = str(getattr(mode, "name", mode))
    for method, stage in (
        ("is_mixed", "mixed"),
        ("is_prefill", "prefill"),
        ("is_decode", "decode"),
        ("is_idle", "idle"),
    ):
        try:
            if bool(getattr(mode, method)()):
                return stage, name
        except Exception:
            continue
    return "unknown", name


def _high_capture_mode() -> ContextManager[Any]:
    return _TORCH_MODE_CLASS() if _TORCH_MODE_CLASS is not None else nullcontext()


@contextmanager
def _high_scope(frame: FrameType, interface: str) -> Iterator[None]:
    call_id = _next_id("high")
    stage, forward_mode = _stage_from_frame(frame)
    high = {
        "call_id": call_id,
        "interface": interface,
        "file": frame.f_code.co_filename,
        "definition_line": frame.f_code.co_firstlineno,
        "boundary_code": frame.f_code,
        "stage": stage,
        "forward_mode": forward_mode,
    }
    token = _CURRENT_HIGH.set(high)
    _nvtx_push(
        _label(
            "high",
            call_id=call_id,
            interface=interface,
            file=frame.f_code.co_filename,
            line=frame.f_code.co_firstlineno,
            stage=stage,
            forward_mode=forward_mode,
        )
    )
    try:
        with _high_capture_mode():
            yield
    finally:
        _nvtx_pop()
        _CURRENT_HIGH.reset(token)


def _install_target_profiler() -> None:
    target_path = _target_file()
    qualname, definition_line = _locate_target_definition()
    if not qualname:
        raise RuntimeError(f"cannot resolve target function at {target_path}:{_target().get('line')}")
    short_name = qualname.rsplit(".", 1)[-1]
    previous = sys.getprofile()

    def profile(frame: FrameType, event: str, arg: Any) -> Any:
        key = (threading.get_ident(), id(frame))
        if event == "call":
            try:
                same_file = Path(frame.f_code.co_filename).resolve() == target_path
            except Exception:
                same_file = False
            code_qualname = str(getattr(frame.f_code, "co_qualname", frame.f_code.co_name))
            name_matches = code_qualname == qualname or code_qualname.endswith("." + qualname) or (
                frame.f_code.co_name == short_name
                and (definition_line is None or frame.f_code.co_firstlineno == definition_line)
            )
            if same_file and name_matches:
                scope = _high_scope(frame, qualname)
                scope.__enter__()
                _ACTIVE_TARGET_FRAMES[key] = scope
        elif event == "return":
            scope = _ACTIVE_TARGET_FRAMES.pop(key, None)
            if scope is not None:
                scope.__exit__(None, None, None)
        if previous is not None:
            previous(frame, event, arg)
        return profile

    sys.setprofile(profile)
    threading.setprofile(profile)


def _frame_record(frame: FrameType) -> dict[str, Any]:
    return {
        "file": frame.f_code.co_filename,
        "definition_line": frame.f_code.co_firstlineno,
        "function": frame.f_code.co_name,
        "qualname": str(getattr(frame.f_code, "co_qualname", frame.f_code.co_name)),
        "call_site_to_next": {
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
        },
    }


def _capture_python_stack(high: dict[str, Any]) -> list[dict[str, Any]]:
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
    inner_to_outer: list[FrameType] = []
    boundary = high.get("boundary_code")
    found = False
    while frame is not None:
        if frame.f_code is boundary:
            inner_to_outer.append(frame)
            found = True
            break
        inner_to_outer.append(frame)
        frame = frame.f_back
    if not found:
        return []
    ignored = {
        "_capture_python_stack",
        "execution_capture",
        "__enter__",
        "__torch_dispatch__",
        "wrapped",
        "wrapped_launcher",
    }
    return [
        _frame_record(item)
        for item in reversed(inner_to_outer)
        if item.f_code.co_name not in ignored
    ]


@contextmanager
def execution_capture(
    *,
    archetype: str,
    execution_interface: str,
    provider_hint: str | None = None,
    implementation_hint: dict[str, Any] | None = None,
) -> Iterator[None]:
    high = _CURRENT_HIGH.get()
    if high is None:
        yield
        return
    if archetype not in CAPTURE_BY_ARCHETYPE:
        raise ValueError(f"unknown KID capture archetype: {archetype}")
    capture_id = _next_id("capture")
    parent_capture_id = _CURRENT_CAPTURE.get()
    token = _CURRENT_CAPTURE.set(capture_id)
    stack = _capture_python_stack(high)
    event = {
        "event": "execution_capture",
        "capture_id": capture_id,
        "parent_capture_id": parent_capture_id,
        "parent_call_id": high["call_id"],
        "archetype": archetype,
        "common_interface": CAPTURE_BY_ARCHETYPE[archetype].common_interfaces[0],
        "execution_interface": execution_interface,
        "provider_hint": provider_hint,
        "execution_leaf": {
            "archetype": archetype,
            "interface": execution_interface,
            "kind": "common_interface",
        },
        "implementation_hint": implementation_hint or {},
        "cpu_capture_ns": time.monotonic_ns(),
        "python_stack": stack,
    }
    _write_capture_event(event)
    _nvtx_push(
        _label(
            "execution",
            capture_id=capture_id,
            parent_capture_id=parent_capture_id,
            parent_call_id=high["call_id"],
            archetype=archetype,
            interface=execution_interface,
            provider_hint=provider_hint,
        )
    )
    try:
        yield
    finally:
        _nvtx_pop()
        _CURRENT_CAPTURE.reset(token)


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


def _callable_hint(value: Any) -> dict[str, Any]:
    candidate = getattr(value, "fn", value)
    code = getattr(candidate, "__code__", None)
    if code is not None:
        return {"file": code.co_filename, "definition_line": code.co_firstlineno}
    try:
        source = inspect.getsourcefile(type(value)) or inspect.getsourcefile(value)
    except (OSError, TypeError):
        source = None
    return {"file": source} if source else {}


def _provider_from_path(path: str | None) -> str | None:
    if not path:
        return None
    lowered = path.replace("\\", "/").lower()
    for marker, provider in (
        ("/flashinfer/", "flashinfer"),
        ("/sglang/", "sglang"),
        ("/sgl_kernel/", "sgl-kernel"),
        ("/deep_gemm/", "deepgemm"),
    ):
        if marker in lowered:
            return provider
    return None


def _captured_callable(
    value: Callable[..., Any],
    *,
    archetype: str,
    interface: str,
    provider_hint: str | None = None,
    implementation_hint: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    if getattr(value, "_kid_runtime_wrapped", False):
        return value

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with execution_capture(
            archetype=archetype,
            execution_interface=interface,
            provider_hint=provider_hint,
            implementation_hint=implementation_hint or _callable_hint(value),
        ):
            return value(*args, **kwargs)

    try:
        functools.update_wrapper(wrapped, value)
    except (AttributeError, TypeError):
        pass
    setattr(wrapped, "_kid_runtime_wrapped", True)
    return wrapped


class _CapturedModuleProxy:
    def __init__(self, module: Any, provider_hint: str | None, origin: str):
        self._kid_module = module
        self._kid_provider_hint = provider_hint
        self._kid_origin = origin
        self._kid_wrapped: dict[str, Any] = {}

    def _wrap(self, name: str, value: Any) -> Any:
        if not callable(value) or name.startswith("_"):
            return value
        if name not in self._kid_wrapped:
            self._kid_wrapped[name] = _captured_callable(
                value,
                archetype="tvm_ffi_call",
                interface=f"{self._kid_origin}.{name}",
                provider_hint=self._kid_provider_hint,
                implementation_hint={"factory": self._kid_origin, "export": name},
            )
        return self._kid_wrapped[name]

    def __getattr__(self, name: str) -> Any:
        return self._wrap(name, getattr(self._kid_module, name))

    def __getitem__(self, name: str) -> Any:
        return self._wrap(str(name), self._kid_module[name])

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._kid_module)) | set(self.__dict__))

    def __repr__(self) -> str:
        return repr(self._kid_module)


def _install_torch_dispatch(module: ModuleType) -> None:
    global _TORCH_MODE_CLASS
    if _TORCH_MODE_CLASS is not None:
        return
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:
        return

    class KidTorchDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            text = str(func)
            namespace = text.split(".", 1)[0]
            provider = "pytorch" if namespace in {"aten", "prims"} else (
                "sgl-kernel" if namespace == "sgl_kernel" else None
            )
            with execution_capture(
                archetype="pytorch_dispatch",
                execution_interface=text,
                provider_hint=provider,
            ):
                return func(*args, **(kwargs or {}))

    _TORCH_MODE_CLASS = KidTorchDispatchMode
    _patch_torch_compile(module)


def _patch_torch_compile(module: ModuleType) -> None:
    original = getattr(module, "compile", None)
    if not callable(original) or getattr(original, "_kid_compile_patched", False):
        return

    def suspend_mode(value: Callable[..., Any]) -> Callable[..., Any]:
        if getattr(value, "_kid_compiled_callable_wrapped", False):
            return value

        @functools.wraps(value)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _CURRENT_HIGH.get() is None:
                return value(*args, **kwargs)
            try:
                from torch.utils._python_dispatch import _pop_mode_temporarily

                with _pop_mode_temporarily():
                    return value(*args, **kwargs)
            except RuntimeError:
                return value(*args, **kwargs)

        setattr(wrapped, "_kid_compiled_callable_wrapped", True)
        return wrapped

    @functools.wraps(original)
    def patched_compile(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if args and callable(args[0]):
            return suspend_mode(result)

        @functools.wraps(result)
        def decorator(func: Callable[..., Any]) -> Any:
            return suspend_mode(result(func))

        return decorator

    setattr(patched_compile, "_kid_compile_patched", True)
    module.compile = patched_compile


def _patch_triton(module: ModuleType) -> None:
    for class_name in ("JITFunction", "Autotuner", "Heuristics"):
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type) or getattr(cls, "_kid_getitem_patched", False):
            continue
        original = getattr(cls, "__getitem__", None)
        if not callable(original):
            continue

        def make_getitem(orig: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(orig)
            def patched(self: Any, grid: Any) -> Any:
                launcher = orig(self, grid)
                kernel = getattr(self, "fn", self)
                interface = _callable_name(kernel, type(self).__name__)
                hint = _callable_hint(kernel)
                return _captured_callable(
                    launcher,
                    archetype="triton_launch",
                    interface=interface,
                    provider_hint=_provider_from_path(hint.get("file")),
                    implementation_hint=hint,
                )

            return patched

        cls.__getitem__ = make_getitem(original)
        cls._kid_getitem_patched = True


def _patch_cute(module: ModuleType) -> None:
    original = getattr(module, "compile", None)
    if not callable(original) or getattr(module, "_kid_compile_patched", False):
        return

    @functools.wraps(original)
    def patched_compile(*args: Any, **kwargs: Any) -> Any:
        compiled = original(*args, **kwargs)
        kernel = args[0] if args else compiled
        hint = _callable_hint(kernel)
        return _captured_callable(
            compiled,
            archetype="cute_dsl_launch",
            interface=_callable_name(kernel, type(kernel).__name__),
            provider_hint=_provider_from_path(hint.get("file")),
            implementation_hint=hint,
        )

    module.compile = patched_compile
    module._kid_compile_patched = True


def _patch_tilelang(module: ModuleType) -> None:
    cls = getattr(module, "JITKernel", None)
    if not isinstance(cls, type) or getattr(cls, "_kid_call_patched", False):
        return
    original = getattr(cls, "__call__", None)
    if not callable(original):
        return

    @functools.wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        hint = _callable_hint(self)
        with execution_capture(
            archetype="tilelang_launch",
            execution_interface=_callable_name(self, type(self).__name__),
            provider_hint=_provider_from_path(hint.get("file")),
            implementation_hint=hint,
        ):
            return original(self, *args, **kwargs)

    cls.__call__ = patched
    cls._kid_call_patched = True


def _patch_sglang_jit(module: ModuleType) -> None:
    original = getattr(module, "load_jit", None)
    if not callable(original) or getattr(original, "_kid_load_jit_patched", False):
        return

    @functools.wraps(original)
    def patched(*args: Any, **kwargs: Any) -> Any:
        loaded = original(*args, **kwargs)
        name = str(args[0]) if args else "load_jit"
        return _CapturedModuleProxy(loaded, "sglang", f"sglang_jit.{name}")

    patched._kid_load_jit_patched = True
    module.load_jit = patched


def _patch_flashinfer_jit(module: ModuleType) -> None:
    cls = getattr(module, "JitSpec", None)
    if not isinstance(cls, type) or getattr(cls, "_kid_build_load_patched", False):
        return
    original = getattr(cls, "build_and_load", None)
    if not callable(original):
        return

    @functools.wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        loaded = original(self, *args, **kwargs)
        name = str(getattr(self, "name", type(self).__name__))
        return _CapturedModuleProxy(loaded, "flashinfer", f"flashinfer_jit.{name}")

    cls.build_and_load = patched
    cls._kid_build_load_patched = True


PYTHON_BINDING_EXPORTS: dict[str, tuple[str, ...]] = {
    "deep_gemm": ("bf16_gemm_nt", "fp8_paged_mqa_logits"),
}


def _patch_python_bindings(module: ModuleType) -> None:
    exports = PYTHON_BINDING_EXPORTS.get(module.__name__, ())
    for name in exports:
        value = getattr(module, name, None)
        if not callable(value):
            continue
        setattr(
            module,
            name,
            _captured_callable(
                value,
                archetype="python_binding",
                interface=f"{module.__name__}.{name}",
                provider_hint="deepgemm" if module.__name__ == "deep_gemm" else module.__name__,
                implementation_hint={"module": module.__name__, "export": name},
            ),
        )


def _patch_inductor(module: ModuleType) -> None:
    cls = getattr(module, "CachingAutotuner", None)
    if not isinstance(cls, type) or getattr(cls, "_kid_run_patched", False):
        return
    method_name = "run" if callable(getattr(cls, "run", None)) else "__call__"
    original = getattr(cls, method_name, None)
    if not callable(original):
        return

    @functools.wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = getattr(self, "fn", self)
        with execution_capture(
            archetype="inductor_launch",
            execution_interface=_callable_name(kernel, "inductor.CachingAutotuner"),
            implementation_hint=_callable_hint(kernel),
        ):
            return original(self, *args, **kwargs)

    setattr(cls, method_name, patched)
    cls._kid_run_patched = True


def _instrument_module(module: ModuleType) -> None:
    name = getattr(module, "__name__", "")
    if name == "torch":
        _install_torch_dispatch(module)
    elif name in {"triton.runtime.jit", "triton.runtime.autotuner"}:
        _patch_triton(module)
    elif name == "cutlass.cute":
        _patch_cute(module)
    elif name == "tilelang":
        _patch_tilelang(module)
    elif name == "sglang.jit_kernel.utils":
        _patch_sglang_jit(module)
    elif name == "flashinfer.jit.core":
        _patch_flashinfer_jit(module)
    elif name in PYTHON_BINDING_EXPORTS:
        _patch_python_bindings(module)
    elif name == "torch._inductor.runtime.triton_heuristics":
        _patch_inductor(module)


_WATCHED_MODULES = frozenset(
    {
        "torch",
        "triton.runtime.jit",
        "triton.runtime.autotuner",
        "cutlass.cute",
        "tilelang",
        "sglang.jit_kernel.utils",
        "flashinfer.jit.core",
        "deep_gemm",
        "torch._inductor.runtime.triton_heuristics",
    }
)


class _InstrumentingLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader):
        self.wrapped = wrapped

    def create_module(self, spec: Any) -> Any:
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)  # type: ignore[attr-defined]
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)  # type: ignore[attr-defined]
        _instrument_module(module)


class _InstrumentingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del target
        if fullname not in _WATCHED_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _InstrumentingLoader):
            return None
        spec.loader = _InstrumentingLoader(spec.loader)
        return spec


def _install_import_hook() -> None:
    if not any(isinstance(item, _InstrumentingFinder) for item in sys.meta_path):
        sys.meta_path.insert(0, _InstrumentingFinder())


def _patch_already_imported_modules() -> None:
    for name in _WATCHED_MODULES:
        module = sys.modules.get(name)
        if isinstance(module, ModuleType):
            _instrument_module(module)
