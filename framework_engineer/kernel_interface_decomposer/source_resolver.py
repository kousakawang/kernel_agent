from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DecomposerConfig


@dataclass
class FunctionLocation:
    qualname: str
    file: str
    line: int


class SourceResolver:
    def __init__(self, config: DecomposerConfig):
        self.config = config
        self.workdir = config.workdir
        self.source_roots = self._resolve_source_roots()
        self.events = self._load_events()
        self._sgl_registry: dict[str, dict[str, Any]] | None = None
        self._triton_def_cache: dict[str, FunctionLocation | None] = {}
        self.target_qualified_name = config.target_qualified_name or self.locate_function_at(
            config.target_file, config.target_line
        ).qualname

    def _resolve_source_roots(self) -> list[Path]:
        roots = []
        for item in self.config.resolution.get("source_roots") or []:
            path = Path(str(item))
            if not path.is_absolute():
                path = self.workdir / path
            roots.append(path.resolve())
        if not roots:
            roots = [self.workdir.resolve()]
        return roots

    def _load_events(self) -> list[dict[str, Any]]:
        events_dir = self.config.output_dir / "events"
        events: list[dict[str, Any]] = []
        if not events_dir.exists():
            return events
        for path in sorted(events_dir.glob("*.jsonl")):
            try:
                for line in path.read_text().splitlines():
                    if line.strip():
                        events.append(json.loads(line))
            except Exception:
                continue
        return events

    def locate_function_at(self, file: Path, line: int) -> FunctionLocation:
        try:
            tree = ast.parse(file.read_text())
        except Exception:
            return FunctionLocation(file.stem, str(file), line)
        best: tuple[int, str, int] | None = None

        def visit(node: ast.AST, parents: list[str]) -> None:
            nonlocal best
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                if start <= line <= end:
                    qualname = ".".join([*parents, node.name])
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        span = end - start
                        if best is None or span <= best[0]:
                            best = (span, qualname, start)
                    for child in ast.iter_child_nodes(node):
                        visit(child, [*parents, node.name])
                    return
            for child in ast.iter_child_nodes(node):
                visit(child, parents)

        visit(tree, [])
        if best is None:
            return FunctionLocation(file.stem, str(file), line)
        return FunctionLocation(best[1], str(file), best[2])

    def resolve(
        self,
        *,
        raw_kernel_name: str,
        normalized_kernel_name: str,
        wrapper: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        wrapper = wrapper or {}
        category = self._infer_category(raw_kernel_name, wrapper)
        wrapper_info = self._resolve_wrapper(wrapper)
        implementation = self._resolve_implementation(category, raw_kernel_name, normalized_kernel_name, wrapper, wrapper_info)
        return category, wrapper_info, implementation

    def _resolve_wrapper(self, wrapper: dict[str, Any]) -> dict[str, Any]:
        api = wrapper.get("api")
        file = wrapper.get("file")
        line = _to_int(wrapper.get("line"))
        launch_line = line
        if file and line:
            loc = self.locate_function_at(Path(file), line)
            if not api or api == loc.qualname.split(".")[-1]:
                api = loc.qualname
            return {"api": api, "file": loc.file, "line": loc.line, "launch_line": launch_line}
        return {"api": api or "unknown", "file": file, "line": line, "launch_line": launch_line}

    def _infer_category(self, raw_kernel_name: str, wrapper: dict[str, Any]) -> str:
        category = wrapper.get("category")
        if category:
            return str(category)
        api = str(wrapper.get("api") or "")
        if api.startswith("torch.nn.functional") or api.startswith("aten::"):
            return "pytorch_native"
        if api.startswith("sgl_kernel") or "torch.ops.sgl_kernel" in api:
            return "sgl_kernel"
        if self._find_triton_definition(raw_kernel_name):
            return "triton_dsl"
        for prefix in self.config.resolution.get("third_party_prefixes") or []:
            if api.startswith(str(prefix)):
                return "third_party"
        return "unknown"

    def _resolve_implementation(
        self,
        category: str,
        raw_kernel_name: str,
        normalized_kernel_name: str,
        wrapper: dict[str, Any],
        wrapper_info: dict[str, Any],
    ) -> dict[str, Any]:
        event_impl = self._implementation_from_events(raw_kernel_name, wrapper)
        if event_impl:
            return event_impl
        if category == "triton_dsl":
            return self._resolve_triton(raw_kernel_name)
        if category == "runtime_jit":
            return {
                "kind": "runtime_jit_source",
                "source_status": "unknown",
                "source_files": [],
                "symbols": [normalized_kernel_name],
            }
        if category == "sgl_kernel":
            return self._resolve_sgl_kernel(wrapper_info, normalized_kernel_name)
        if category == "pytorch_native":
            symbol = wrapper_info.get("api") or normalized_kernel_name
            return {
                "kind": "pytorch_native",
                "source_status": "external_documented",
                "source_files": [],
                "symbols": [symbol],
            }
        if category == "third_party":
            file = wrapper_info.get("file")
            return {
                "kind": "third_party_source" if file else "third_party_binary",
                "source_status": "resolved" if file else "unavailable_binary",
                "source_files": [file] if file else [],
                "symbols": [wrapper_info.get("api") or normalized_kernel_name],
            }
        return {
            "kind": "unknown",
            "source_status": "unknown",
            "source_files": [],
            "symbols": [normalized_kernel_name],
        }

    def _implementation_from_events(self, raw_kernel_name: str, wrapper: dict[str, Any]) -> dict[str, Any] | None:
        wrapper_api = wrapper.get("api")
        for event in reversed(self.events):
            impl = event.get("implementation")
            if not impl:
                continue
            if event.get("kernel") and event.get("kernel") == raw_kernel_name:
                return impl
            if wrapper_api and event.get("api") == wrapper_api:
                return impl
        return None

    def _resolve_triton(self, raw_kernel_name: str) -> dict[str, Any]:
        found = self._find_triton_definition(raw_kernel_name)
        if found:
            return {
                "kind": "triton_source",
                "source_status": "resolved",
                "source_files": [found.file],
                "symbols": [raw_kernel_name],
                "definition_line": found.line,
            }
        return {
            "kind": "triton_source",
            "source_status": "unknown",
            "source_files": [],
            "symbols": [raw_kernel_name],
        }

    def _find_triton_definition(self, name: str) -> FunctionLocation | None:
        if name in self._triton_def_cache:
            return self._triton_def_cache[name]
        if not name or not name.replace("_", "").isalnum():
            self._triton_def_cache[name] = None
            return None
        for root in self.source_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                if f"def {name}" not in text:
                    continue
                try:
                    tree = ast.parse(text)
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                        found = FunctionLocation(name, str(path), node.lineno)
                        self._triton_def_cache[name] = found
                        return found
        self._triton_def_cache[name] = None
        return None

    def _resolve_sgl_kernel(self, wrapper_info: dict[str, Any], normalized_kernel_name: str) -> dict[str, Any]:
        api = str(wrapper_info.get("api") or "")
        op_name = api.rsplit(".", 1)[-1] if api and api != "unknown" else normalized_kernel_name
        registry = self._sgl_kernel_registry()
        entry = registry.get(op_name)
        if not entry:
            for key, value in registry.items():
                if key in api or key in normalized_kernel_name:
                    op_name, entry = key, value
                    break
        if not entry:
            return {
                "kind": "sgl_kernel_source",
                "source_status": "unknown",
                "source_files": [],
                "symbols": [op_name],
            }
        symbol = entry.get("symbol") or op_name
        candidates = self._find_symbol_sources(symbol)
        source_files = list(dict.fromkeys([entry["registration_file"], *candidates]))
        return {
            "kind": "sgl_kernel_source",
            "source_status": "resolved" if source_files else "unknown",
            "source_files": source_files,
            "symbols": [symbol],
            "torch_op": f"torch.ops.sgl_kernel.{op_name}",
            "registration": {
                "file": entry["registration_file"],
                "line": entry["registration_line"],
            },
        }

    def _sgl_kernel_registry(self) -> dict[str, dict[str, Any]]:
        if self._sgl_registry is not None:
            return self._sgl_registry
        registry: dict[str, dict[str, Any]] = {}
        roots = [self.workdir / "sglang" / "sgl-kernel" / "csrc"]
        for root in roots:
            if not root.exists():
                continue
            for path in list(root.rglob("*.cc")) + list(root.rglob("*.cpp")):
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                line_offsets = _line_offsets(text)
                for match in re.finditer(r'm\.impl\(\s*"([^"]+)"\s*,(?P<body>.*?)\);', text, re.DOTALL):
                    op = match.group(1)
                    body = match.group("body")
                    sym_match = re.search(r"&\s*([A-Za-z_][A-Za-z0-9_]*)", body)
                    registry[op] = {
                        "symbol": sym_match.group(1) if sym_match else op,
                        "registration_file": str(path),
                        "registration_line": _line_for_offset(line_offsets, match.start()),
                    }
        self._sgl_registry = registry
        return registry

    def _find_symbol_sources(self, symbol: str) -> list[str]:
        out: list[str] = []
        if not symbol:
            return out
        roots = [self.workdir / "sglang" / "sgl-kernel" / "csrc", self.workdir / "sglang" / "sgl-kernel" / "include"]
        pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.suffix not in {".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                if pattern.search(text):
                    out.append(str(path))
        return out[:12]


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _line_for_offset(offsets: list[int], offset: int) -> int:
    lo, hi = 0, len(offsets)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= offset:
            lo = mid
        else:
            hi = mid
    return lo + 1


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None
