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
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


_INSTALLED = False
_CONFIG: dict[str, Any] = {}
_EVENT_LOCK = threading.Lock()
_CALL_COUNTER = 0
_CURRENT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "kid_current_invocation", default=None
)


def install_from_env() -> None:
    global _INSTALLED, _CONFIG
    if _INSTALLED or os.environ.get("KID_ENABLE") != "1":
        return
    config_path = os.environ.get("KID_RUNTIME_CONFIG")
    if not config_path:
        return
    try:
        _CONFIG = json.loads(Path(config_path).read_text())
    except Exception as exc:
        print(f"[kernel-interface-decomposer] failed to load runtime config: {exc}", file=sys.stderr)
        return
    _INSTALLED = True
    _record_event("process_start", {})
    _install_import_hook()
    _patch_already_imported_modules()


def _output_dir() -> Path:
    return Path(_CONFIG.get("output_dir", ".")).resolve()


def _target_file() -> Path:
    return Path((_CONFIG.get("target") or {}).get("file", "")).resolve()


def _target_line() -> int:
    return int((_CONFIG.get("target") or {}).get("line", 0))


def _third_party_prefixes() -> tuple[str, ...]:
    resolution = _CONFIG.get("resolution") or {}
    prefixes = resolution.get("third_party_prefixes") or []
    return tuple(str(prefix) for prefix in prefixes)


def _source_roots() -> list[str]:
    roots = []
    for item in (_CONFIG.get("resolution") or {}).get("source_roots") or []:
        path = Path(str(item))
        if not path.is_absolute():
            path = Path(_CONFIG.get("workdir", ".")).resolve() / path
        roots.append(str(path.resolve()))
    return roots


def _sanitize(value: Any) -> str:
    text = str(value)
    return text.replace("|", "/").replace("\n", " ")[:500]


def _label(kind: str, fields: dict[str, Any]) -> str:
    parts = [f"PYGPU:type={kind}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_sanitize(value)}")
    return "|".join(parts)


def _nvtx_push(text: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(text)
    except Exception:
        return


def _nvtx_pop() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
    except Exception:
        return


def _record_event(event_type: str, payload: dict[str, Any]) -> None:
    out = _output_dir() / "events"
    try:
        out.mkdir(parents=True, exist_ok=True)
        event = {
            "event": event_type,
            "ts_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            **payload,
        }
        path = out / f"events_{os.getpid()}.jsonl"
        with _EVENT_LOCK:
            with path.open("a") as f:
                f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except Exception:
        return


def _locate_target_qualname() -> str | None:
    explicit = (_CONFIG.get("target") or {}).get("qualified_name")
    if explicit:
        return str(explicit)
    path = _target_file()
    line = _target_line()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return None

    best: tuple[int, str] | None = None

    def visit(node: ast.AST, parents: list[str]) -> None:
        nonlocal best
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            if start <= line <= end:
                name = ".".join([*parents, node.name])
                span = end - start
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if best is None or span <= best[0]:
                        best = (span, name)
                for child in ast.iter_child_nodes(node):
                    visit(child, [*parents, node.name])
                return
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, [])
    return best[1] if best else None


def _find_forward_mode_from_value(value: Any, depth: int = 0) -> Any:
    if depth > 2 or value is None:
        return None
    if hasattr(value, "forward_mode"):
        try:
            return getattr(value, "forward_mode")
        except Exception:
            return None
    if isinstance(value, dict):
        for item in value.values():
            found = _find_forward_mode_from_value(item, depth + 1)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_forward_mode_from_value(item, depth + 1)
            if found is not None:
                return found
    return None


def _stage_from_mode(mode: Any) -> tuple[str, str | None]:
    if mode is None:
        return "unknown", None
    mode_name = getattr(mode, "name", str(mode))
    try:
        if mode.is_mixed():
            return "mixed", mode_name
        if mode.is_prefill():
            return "prefill", mode_name
        if mode.is_decode():
            return "decode", mode_name
        if mode.is_idle():
            return "idle", mode_name
    except Exception:
        pass
    return "unknown", mode_name


def _infer_stage(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str | None]:
    for value in (*args, kwargs):
        mode = _find_forward_mode_from_value(value)
        if mode is not None:
            return _stage_from_mode(mode)
    try:
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        while frame is not None:
            for key in ("forward_batch", "batch", "schedule_batch"):
                if key in frame.f_locals:
                    mode = _find_forward_mode_from_value(frame.f_locals[key])
                    if mode is not None:
                        return _stage_from_mode(mode)
            frame = frame.f_back
    except Exception:
        pass
    return "unknown", None


def _wrap_target_callable(func: Callable[..., Any], qualname: str, file: str, line: int) -> Callable[..., Any]:
    if getattr(func, "_kid_target_wrapped", False):
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _CALL_COUNTER
        _CALL_COUNTER += 1
        call_id = _CALL_COUNTER
        stage, forward_mode = _infer_stage(args, kwargs)
        ctx = {
            "call_id": call_id,
            "stage": stage,
            "forward_mode": forward_mode,
            "target_api": qualname,
        }
        token = _CURRENT.set(ctx)
        fields = {
            "call_id": call_id,
            "api": qualname,
            "file": file,
            "line": line,
            "stage": stage,
            "forward_mode": forward_mode,
            "pid": os.getpid(),
        }
        text = _label("target", fields)
        _record_event("target_begin", fields)
        _nvtx_push(text)
        try:
            return func(*args, **kwargs)
        finally:
            _nvtx_pop()
            _record_event("target_end", fields)
            _CURRENT.reset(token)

    setattr(wrapper, "_kid_target_wrapped", True)
    return wrapper


def _wrap_regular_callable(
    func: Callable[..., Any],
    *,
    api: str,
    category: str,
    implementation: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    if getattr(func, "_kid_wrapper_wrapped", False):
        return func

    try:
        file = inspect.getsourcefile(func)
        line = inspect.getsourcelines(func)[1]
    except Exception:
        file = None
        line = None

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = _CURRENT.get()
        if ctx is None:
            return func(*args, **kwargs)
        fields = {
            "call_id": ctx.get("call_id"),
            "api": api,
            "category": category,
            "file": file,
            "line": line,
            "stage": ctx.get("stage"),
            "forward_mode": ctx.get("forward_mode"),
        }
        payload = {**fields}
        if implementation:
            payload["implementation"] = implementation
        text = _label("wrap", fields)
        _record_event("wrap_begin", payload)
        _nvtx_push(text)
        try:
            return func(*args, **kwargs)
        finally:
            _nvtx_pop()
            _record_event("wrap_end", payload)

    setattr(wrapper, "_kid_wrapper_wrapped", True)
    return wrapper


def _wrap_target_in_module(module: ModuleType) -> None:
    target_path = _target_file()
    try:
        module_path = Path(getattr(module, "__file__", "")).resolve()
    except Exception:
        return
    if module_path != target_path:
        return
    qualname = _locate_target_qualname()
    if not qualname:
        _record_event("target_wrap_failed", {"reason": "qualname_not_found", "file": str(target_path)})
        return
    parts = qualname.split(".")
    obj: Any = module
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            _record_event("target_wrap_failed", {"reason": "owner_not_found", "qualname": qualname})
            return
    leaf = parts[-1]
    func = getattr(obj, leaf, None)
    if func is None:
        _record_event("target_wrap_failed", {"reason": "callable_not_found", "qualname": qualname})
        return
    setattr(obj, leaf, _wrap_target_callable(func, qualname, str(target_path), _target_line()))
    _record_event("target_wrapped", {"api": qualname, "file": str(target_path), "line": _target_line()})


def _wrap_module_functions(module: ModuleType, *, category: str) -> None:
    for name, value in list(vars(module).items()):
        if name.startswith("_") or not callable(value):
            continue
        if inspect.isclass(value) or inspect.ismodule(value):
            continue
        owner = getattr(value, "__module__", "")
        if owner != module.__name__:
            continue
        api = f"{module.__name__}.{getattr(value, '__qualname__', name)}"
        try:
            setattr(module, name, _wrap_regular_callable(value, api=api, category=category))
        except Exception:
            continue


def _patch_torch_functional(module: ModuleType) -> None:
    names = [
        "linear",
        "conv1d",
        "conv2d",
        "layer_norm",
        "rms_norm",
        "scaled_dot_product_attention",
        "silu",
        "gelu",
        "softmax",
        "embedding",
    ]
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            setattr(
                module,
                name,
                _wrap_regular_callable(
                    value,
                    api=f"torch.nn.functional.{name}",
                    category="pytorch_native",
                    implementation={
                        "kind": "pytorch_native",
                        "source_status": "external_documented",
                        "symbols": [f"torch.nn.functional.{name}"],
                    },
                ),
            )


def _caller_location() -> tuple[str | None, int | None, str | None]:
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
    while frame is not None:
        filename = frame.f_code.co_filename
        if "kernel_interface_decomposer" not in filename:
            return filename, frame.f_lineno, frame.f_code.co_name
        frame = frame.f_back
    return None, None, None


def _patch_triton_module(module: ModuleType) -> None:
    for class_name in ("JITFunction", "Autotuner", "Heuristics"):
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, "_kid_getitem_patched", False):
            continue
        original = getattr(cls, "__getitem__", None)
        if original is None:
            continue

        def make_getitem(orig: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(orig)
            def patched_getitem(self: Any, grid: Any) -> Any:
                launcher = orig(self, grid)
                if getattr(launcher, "_kid_triton_launcher_wrapped", False):
                    return launcher
                kernel_fn = getattr(self, "fn", None)
                kernel_name = getattr(kernel_fn, "__name__", getattr(self, "__name__", type(self).__name__))
                kernel_file = getattr(getattr(kernel_fn, "__code__", None), "co_filename", None)
                kernel_line = getattr(getattr(kernel_fn, "__code__", None), "co_firstlineno", None)

                @functools.wraps(launcher)
                def wrapped_launcher(*args: Any, **kwargs: Any) -> Any:
                    ctx = _CURRENT.get()
                    if ctx is None:
                        return launcher(*args, **kwargs)
                    launch_file, launch_line, launch_func = _caller_location()
                    fields = {
                        "call_id": ctx.get("call_id"),
                        "api": launch_func,
                        "category": "triton_dsl",
                        "kernel": kernel_name,
                        "file": launch_file,
                        "line": launch_line,
                        "stage": ctx.get("stage"),
                        "forward_mode": ctx.get("forward_mode"),
                    }
                    implementation = {
                        "kind": "triton_source",
                        "source_status": "resolved" if kernel_file else "unknown",
                        "source_files": [kernel_file] if kernel_file else [],
                        "symbols": [kernel_name],
                        "definition_line": kernel_line,
                    }
                    payload = {**fields, "implementation": implementation}
                    _record_event("wrap_begin", payload)
                    _nvtx_push(_label("wrap", fields))
                    try:
                        return launcher(*args, **kwargs)
                    finally:
                        _nvtx_pop()
                        _record_event("wrap_end", payload)

                setattr(wrapped_launcher, "_kid_triton_launcher_wrapped", True)
                return wrapped_launcher

            return patched_getitem

        setattr(cls, "__getitem__", make_getitem(original))
        setattr(cls, "_kid_getitem_patched", True)


class _JitModuleProxy:
    def __init__(self, module: Any, metadata: dict[str, Any]):
        self._kid_module = module
        self._kid_metadata = metadata
        self._kid_wrapped: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._kid_module, name)
        if not callable(value):
            return value
        if name in self._kid_wrapped:
            return self._kid_wrapped[name]
        wrappers = self._kid_metadata.get("wrappers_by_export", {})
        implementation = wrappers.get(name, self._kid_metadata)
        wrapped = _wrap_regular_callable(
            value,
            api=f"{self._kid_metadata.get('module_name', 'jit_module')}.{name}",
            category="runtime_jit",
            implementation=implementation,
        )
        self._kid_wrapped[name] = wrapped
        return wrapped

    def __repr__(self) -> str:
        return repr(self._kid_module)


def _patch_sglang_jit_utils(module: ModuleType) -> None:
    original = getattr(module, "load_jit", None)
    if original is None or getattr(original, "_kid_load_jit_patched", False):
        return

    @functools.wraps(original)
    def patched_load_jit(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        kernel_path = Path(getattr(module, "KERNEL_PATH", ".")).resolve()
        cpp_files = [str((kernel_path / "csrc" / item).resolve()) for item in kwargs.get("cpp_files") or []]
        cuda_files = [str((kernel_path / "csrc" / item).resolve()) for item in kwargs.get("cuda_files") or []]
        cpp_wrappers = kwargs.get("cpp_wrappers") or []
        cuda_wrappers = kwargs.get("cuda_wrappers") or []
        module_name = "sgl_kernel_jit_" + "_".join(str(arg) for arg in args)
        wrappers_by_export = {}
        for export_name, symbol in [*cpp_wrappers, *cuda_wrappers]:
            wrappers_by_export[str(export_name)] = {
                "kind": "runtime_jit_source",
                "source_status": "resolved",
                "source_files": cpp_files + cuda_files,
                "symbols": [str(symbol)],
                "export_name": str(export_name),
                "compile_flags": {
                    "extra_cflags": kwargs.get("extra_cflags") or [],
                    "extra_cuda_cflags": kwargs.get("extra_cuda_cflags") or [],
                    "extra_ldflags": kwargs.get("extra_ldflags") or [],
                },
            }
        metadata = {
            "kind": "runtime_jit_source",
            "source_status": "resolved" if (cpp_files or cuda_files) else "unknown",
            "module_name": module_name,
            "source_files": cpp_files + cuda_files,
            "wrappers_by_export": wrappers_by_export,
        }
        _record_event("jit_module_loaded", metadata)
        return _JitModuleProxy(result, metadata)

    setattr(patched_load_jit, "_kid_load_jit_patched", True)
    setattr(module, "load_jit", patched_load_jit)


def _instrument_module(module: ModuleType) -> None:
    name = getattr(module, "__name__", "")
    _wrap_target_in_module(module)
    if name == "torch.nn.functional":
        _patch_torch_functional(module)
    if name.startswith("sgl_kernel"):
        _wrap_module_functions(module, category="sgl_kernel")
    if name in {"sglang.jit_kernel.utils"}:
        _patch_sglang_jit_utils(module)
    if name in {"triton.runtime.jit", "triton.runtime.autotuner"}:
        _patch_triton_module(module)
    third_party = _third_party_prefixes()
    if third_party and name.startswith(third_party):
        _wrap_module_functions(module, category="third_party")


class _InstrumentingLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, fullname: str):
        self.wrapped = wrapped
        self.fullname = fullname

    def create_module(self, spec: Any) -> Any:
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)  # type: ignore[attr-defined]
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)  # type: ignore[attr-defined]
        _instrument_module(module)


class _InstrumentingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname.startswith("framework_engineer.kernel_interface_decomposer"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not _should_consider_module(fullname, spec):
            return None
        spec.loader = _InstrumentingLoader(spec.loader, fullname)
        return spec


def _should_consider_module(fullname: str, spec: Any) -> bool:
    if fullname in {"torch.nn.functional", "sglang.jit_kernel.utils", "triton.runtime.jit", "triton.runtime.autotuner"}:
        return True
    if fullname.startswith("sgl_kernel"):
        return True
    third_party = _third_party_prefixes()
    if third_party and fullname.startswith(third_party):
        return True
    origin = getattr(spec, "origin", None)
    if origin:
        try:
            return Path(origin).resolve() == _target_file()
        except Exception:
            return False
    return False


def _install_import_hook() -> None:
    if not any(isinstance(item, _InstrumentingFinder) for item in sys.meta_path):
        sys.meta_path.insert(0, _InstrumentingFinder())


def _patch_already_imported_modules() -> None:
    for module in list(sys.modules.values()):
        if isinstance(module, ModuleType):
            try:
                _instrument_module(module)
            except Exception:
                continue
