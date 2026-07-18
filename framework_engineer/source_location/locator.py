"""Layer-1 deterministic schema enrichment.

This module intentionally does only two things:

* add the fixed ``source_locations`` skeleton to every KID kernel entry;
* enumerate deterministic Python interface-definition candidates.

It does not inspect implementation call chains and it does not locate bindings.
Those remain Layer-2 work (and future registered binding-provider helpers).
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .contracts import (
    STATUS_AMBIGUOUS,
    STATUS_MISSED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_FOUND,
    STATUS_RESOLVED,
)

SOURCE_LAYER1 = "locate_layer1"
_PROTECTED_SOURCES = {"locate_layer2_agent", "manual"}
_UNRESOLVED_STATUSES = {
    STATUS_AMBIGUOUS,
    STATUS_MISSED,
    STATUS_NOT_FOUND,
}


class LocateError(ValueError):
    """Invalid global input that prevents the locate pass from running."""


@dataclass(frozen=True, order=True)
class Candidate:
    file: str
    def_line: int

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "def_line": self.def_line}


@dataclass(frozen=True)
class SearchRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class ImportBinding:
    local_name: str
    module: str | None
    symbol: str | None
    level: int
    line: int


@dataclass
class InterfaceSearchResult:
    status: str
    candidates: list[Candidate]
    repo_hint: str | None
    evidence: str
    diagnostics: list[str] = field(default_factory=list)

    def to_layer(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hits": [candidate.to_dict() for candidate in self.candidates],
            "repo_hint": self.repo_hint,
            "source": SOURCE_LAYER1,
        }


@dataclass
class KernelLocateResult:
    low_level_id: str
    interface: str
    status: str
    candidates: list[Candidate]
    repo_hint: str | None
    evidence: str
    diagnostics: list[str]
    needs_agent: bool
    unresolved_layers: list[str]


@dataclass
class LocateRunResult:
    schema: Path
    output: Path
    report_path: Path
    kernels: list[KernelLocateResult]
    skipped_roots: list[dict[str, str]]

    def summary(self) -> dict[str, Any]:
        counts = {
            STATUS_RESOLVED: 0,
            STATUS_AMBIGUOUS: 0,
            STATUS_NOT_FOUND: 0,
            STATUS_NOT_APPLICABLE: 0,
        }
        for kernel in self.kernels:
            counts[kernel.status] = counts.get(kernel.status, 0) + 1
        return {
            "schema": str(self.schema),
            "output": str(self.output),
            "report": str(self.report_path),
            "total": len(self.kernels),
            "interface_resolved": counts[STATUS_RESOLVED],
            "interface_ambiguous": counts[STATUS_AMBIGUOUS],
            "interface_not_found": counts[STATUS_NOT_FOUND],
            "interface_not_applicable": counts[STATUS_NOT_APPLICABLE],
        }


def iter_kernel_entries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Return KID kernel entries from the flat or real nested schema shape."""

    kernels = schema.get("kernels")
    if isinstance(kernels, list):
        return [entry for entry in kernels if isinstance(entry, dict)]

    out: list[dict[str, Any]] = []
    invocations = schema.get("invocations")
    if isinstance(invocations, list):
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            selected = invocation.get("selected_kernels")
            if isinstance(selected, list):
                out.extend(entry for entry in selected if isinstance(entry, dict))
    return out


def _kernel_id(entry: dict[str, Any], index: int) -> str:
    for key in ("low_level_id", "kernel_id", "id"):
        value = entry.get(key)
        if value:
            return str(value)
    kernel = entry.get("kernel")
    if isinstance(kernel, dict):
        for key in ("normalized_name", "raw_name"):
            value = kernel.get(key)
            if value:
                return str(value)
    return f"kernel_{index}"


def _placeholder_hit(_layer: str) -> dict[str, str]:
    return {
        "file": (
            "<FILL: layer-2 agent 定位 "
            "(kernel_impl 调用链 / kernel_header 对应头)>"
        ),
        "def_line": "<FILL: 定义起始行号>",
    }


def _layer(
    status: str,
    *,
    hits: list[dict[str, Any]] | None = None,
    repo_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "hits": list(hits or []),
        "repo_hint": repo_hint,
        "source": SOURCE_LAYER1,
    }


def build_source_locations_template(entry: dict[str, Any]) -> dict[str, Any]:
    """Build the first-round Layer-1 skeleton without binding resolution."""

    archetype = str(entry.get("archetype") or "")
    code = str(entry.get("archetype_code") or "")
    binding_provider = entry.get("binding_provider")

    if code == "F0" or archetype == "pytorch_native":
        layers = {
            name: _layer(STATUS_NOT_APPLICABLE)
            for name in (
                "interface_definition",
                "kernel_impl",
                "py_cpp_binding",
                "kernel_header",
            )
        }
    else:
        layers = {
            "interface_definition": _layer(STATUS_NOT_FOUND),
            "kernel_impl": _layer(
                STATUS_MISSED,
                hits=[_placeholder_hit("kernel_impl")],
            ),
            "py_cpp_binding": (
                _layer(STATUS_NOT_APPLICABLE)
                if binding_provider is None
                else _layer(STATUS_NOT_FOUND)
            ),
            "kernel_header": (
                _layer(STATUS_NOT_APPLICABLE)
                if code in {"F1", "F6"}
                else _layer(
                    STATUS_MISSED,
                    hits=[_placeholder_hit("kernel_header")],
                )
            ),
        }

    needs_agent = any(
        layer["status"] in _UNRESOLVED_STATUSES for layer in layers.values()
    )
    return {
        "archetype": archetype,
        "archetype_code": code,
        "source": SOURCE_LAYER1,
        "needs_agent": needs_agent,
        "layers": layers,
    }


def _contains_protected_source(source_locations: Any) -> bool:
    if not isinstance(source_locations, dict):
        return False
    if source_locations.get("source") in _PROTECTED_SOURCES:
        return True
    layers = source_locations.get("layers")
    if not isinstance(layers, dict):
        return False
    return any(
        isinstance(layer, dict) and layer.get("source") in _PROTECTED_SOURCES
        for layer in layers.values()
    )


def _definition_line(node: ast.AST, *, include_decorators: bool) -> int:
    line = int(getattr(node, "lineno", 1))
    if include_decorators and isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        line = min(
            [line]
            + [
                int(getattr(decorator, "lineno", line))
                for decorator in node.decorator_list
            ]
        )
    return line


def _node_end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)))


def _absolute(path: Path) -> Path:
    """Make a path absolute without replacing a manifest symlink spelling."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_identifier(value: str) -> bool:
    return bool(value) and value.isidentifier()


def _callable_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _callable_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _module_scope_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield module-level statements, descending through control flow only."""

    def visit(statements: Iterable[ast.stmt]) -> Iterator[ast.AST]:
        for statement in statements:
            yield statement
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if isinstance(nested, list):
                    yield from visit(item for item in nested if isinstance(item, ast.stmt))
            handlers = getattr(statement, "handlers", None)
            if isinstance(handlers, list):
                for handler in handlers:
                    nested = getattr(handler, "body", None)
                    if isinstance(nested, list):
                        yield from visit(
                            item for item in nested if isinstance(item, ast.stmt)
                        )

    yield from visit(tree.body)


def _scope_import_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield imports in one function scope, skipping nested definitions."""

    def visit(node: ast.AST, *, root: bool = False) -> Iterator[ast.AST]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            return
        if not root and isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    yield from visit(scope, root=True)


class PythonSourceIndex:
    """Read-only AST/search index over explicit repository roots."""

    def __init__(self, roots: list[SearchRoot]):
        self.roots = roots
        files: set[Path] = set()
        for root in roots:
            if root.path.exists():
                files.update(_absolute(path) for path in root.path.rglob("*.py"))
        self.python_files = sorted(files)
        self._tree_cache: dict[Path, ast.Module | None] = {}
        self._text_cache: dict[Path, str | None] = {}
        self._module_cache: dict[str, list[Path]] = {}

    def text(self, path: Path) -> str | None:
        path = _absolute(path)
        if path not in self._text_cache:
            try:
                self._text_cache[path] = path.read_text(errors="ignore")
            except OSError:
                self._text_cache[path] = None
        return self._text_cache[path]

    def tree(self, path: Path) -> ast.Module | None:
        path = _absolute(path)
        if path not in self._tree_cache:
            text = self.text(path)
            if text is None:
                self._tree_cache[path] = None
            else:
                try:
                    self._tree_cache[path] = ast.parse(text, filename=str(path))
                except SyntaxError:
                    self._tree_cache[path] = None
        return self._tree_cache[path]

    def owner(self, path: Path) -> SearchRoot | None:
        path = _absolute(path)
        real_path = path.resolve()
        owners: list[SearchRoot] = []
        for root in self.roots:
            try:
                path.relative_to(root.path)
            except ValueError:
                try:
                    real_path.relative_to(root.path.resolve())
                except ValueError:
                    continue
            owners.append(root)
        return max(owners, key=lambda root: len(root.path.parts), default=None)

    def repo_hint(self, candidates: list[Candidate]) -> str | None:
        owners = {
            str(owner.path)
            for candidate in candidates
            if (owner := self.owner(Path(candidate.file))) is not None
        }
        return next(iter(owners)) if len(owners) == 1 else None

    def module_files(self, module: str) -> list[Path]:
        if module in self._module_cache:
            return self._module_cache[module]
        parts = tuple(part for part in module.split(".") if part)
        if not parts:
            return []
        file_suffix = (*parts[:-1], f"{parts[-1]}.py")
        package_suffix = (*parts, "__init__.py")
        matches = [
            path
            for path in self.python_files
            if tuple(path.parts[-len(file_suffix) :]) == file_suffix
            or tuple(path.parts[-len(package_suffix) :]) == package_suffix
        ]
        self._module_cache[module] = matches
        return matches

    def relative_module_files(
        self, current_file: Path, module: str | None, level: int
    ) -> list[Path]:
        if level <= 0:
            return self.module_files(module or "")
        base = _absolute(current_file).parent
        for _ in range(level - 1):
            base = base.parent
        if module:
            base = base.joinpath(*module.split("."))
        candidates = [base.with_suffix(".py"), base / "__init__.py"]
        return sorted(_absolute(path) for path in candidates if path.is_file())

    def _top_level_definitions(
        self, path: Path, symbol: str
    ) -> list[Candidate]:
        tree = self.tree(path)
        if tree is None:
            return []
        out: list[Candidate] = []
        for node in _module_scope_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    out.append(Candidate(str(_absolute(path)), int(node.lineno)))
        return out

    def _class_methods(
        self, path: Path, class_name: str, method_name: str
    ) -> list[Candidate]:
        tree = self.tree(path)
        if tree is None:
            return []
        out: list[Candidate] = []
        for node in _module_scope_nodes(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            methods = [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method_name
            ]
            # A run of @overload declarations plus its implementation describes
            # one interface. Its first declaration is the deterministic anchor.
            if methods:
                out.append(Candidate(str(_absolute(path)), int(methods[0].lineno)))
        return out

    def _module_imports(self, path: Path) -> list[ImportBinding]:
        tree = self.tree(path)
        if tree is None:
            return []
        return _bindings_from_nodes(
            _module_scope_nodes(tree), source_text=self.text(path)
        )

    def resolve_module_symbol(
        self,
        module: str | None,
        symbol: str,
        *,
        current_file: Path | None = None,
        level: int = 0,
        visited: set[tuple[str, str]] | None = None,
    ) -> list[Candidate]:
        visited = set(visited or set())
        key = (f"{current_file}:{level}:{module}", symbol)
        if key in visited or len(visited) > 20:
            return []
        visited.add(key)

        if current_file is not None and level > 0:
            files = self.relative_module_files(current_file, module, level)
        else:
            files = self.module_files(module or "")
        if not files:
            return []

        direct: list[Candidate] = []
        for path in files:
            direct.extend(self._top_level_definitions(path, symbol))
        if direct:
            return _dedupe_candidates(direct)

        reexports: list[Candidate] = []
        for path in files:
            for binding in self._module_imports(path):
                if binding.local_name != symbol:
                    continue
                target_symbol = binding.symbol or symbol
                target_files = (
                    self.relative_module_files(path, binding.module, binding.level)
                    if binding.level > 0
                    else self.module_files(binding.module or "")
                )
                if target_files:
                    reexports.extend(
                        self.resolve_module_symbol(
                            binding.module,
                            target_symbol,
                            current_file=path,
                            level=binding.level,
                            visited=visited,
                        )
                    )
                else:
                    # Binary-backed Python APIs (for example DeepGEMM's _C)
                    # have no Python def. Their explicit re-export is the only
                    # deterministic Python interface location.
                    reexports.append(Candidate(str(_absolute(path)), binding.line))
        return _dedupe_candidates(reexports)

    def find_class_method(self, class_name: str, method_name: str) -> list[Candidate]:
        marker = f"class {class_name}"
        out: list[Candidate] = []
        for path in self.python_files:
            text = self.text(path)
            if text is None or marker not in text or f"def {method_name}" not in text:
                continue
            out.extend(self._class_methods(path, class_name, method_name))
        return _dedupe_candidates(out)

    def find_named_definitions(self, name: str) -> list[Candidate]:
        if not _is_identifier(name):
            return []
        marker = re.compile(rf"\b{re.escape(name)}\b")
        out: list[Candidate] = []
        for path in self.python_files:
            text = self.text(path)
            if text is None or marker.search(text) is None:
                continue
            tree = self.tree(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) and node.name == name:
                    out.append(Candidate(str(_absolute(path)), int(node.lineno)))
            for binding in self._module_imports(path):
                if binding.local_name != name:
                    continue
                target = self.resolve_module_symbol(
                    binding.module,
                    binding.symbol or name,
                    current_file=path,
                    level=binding.level,
                )
                if target:
                    out.extend(target)
                else:
                    out.append(Candidate(str(_absolute(path)), binding.line))
        return _dedupe_candidates(out)


def _import_alias_line(
    node: ast.Import | ast.ImportFrom,
    alias: ast.alias,
    source_text: str | None,
) -> int:
    line = int(getattr(alias, "lineno", node.lineno))
    if source_text is None or node.end_lineno is None:
        return line
    lines = source_text.splitlines()
    pattern = re.compile(rf"\b{re.escape(alias.name)}\b")
    for line_number in range(int(node.lineno), int(node.end_lineno) + 1):
        if line_number <= len(lines) and pattern.search(lines[line_number - 1]):
            return line_number
    return line


def _bindings_from_nodes(
    nodes: Iterable[ast.AST], *, source_text: str | None = None
) -> list[ImportBinding]:
    bindings: list[ImportBinding] = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bindings.append(
                    ImportBinding(
                        local_name=alias.asname or alias.name,
                        module=node.module,
                        symbol=alias.name,
                        level=int(node.level or 0),
                        line=_import_alias_line(node, alias, source_text),
                    )
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings.append(
                    ImportBinding(
                        local_name=alias.asname or alias.name.split(".")[0],
                        module=alias.name,
                        symbol=None,
                        level=0,
                        line=_import_alias_line(node, alias, source_text),
                    )
                )
    return bindings


def _dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(set(candidates), key=lambda candidate: (candidate.file, candidate.def_line))


def _find_call(tree: ast.Module, line: int) -> ast.Call | None:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and int(getattr(node, "lineno", 0)) <= line <= _node_end_line(node)
    ]
    if not calls:
        return None
    return min(
        calls,
        key=lambda node: (
            _node_end_line(node) - int(node.lineno),
            (
                int(getattr(node, "end_col_offset", 0))
                - int(getattr(node, "col_offset", 0))
                if _node_end_line(node) == int(node.lineno)
                else 0
            ),
            -int(getattr(node, "col_offset", 0)),
        ),
    )


def _containing_scope(tree: ast.Module, line: int) -> ast.AST | None:
    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and int(node.lineno) <= line <= _node_end_line(node)
    ]
    if not scopes:
        return None
    return min(scopes, key=lambda node: _node_end_line(node) - int(node.lineno))


def _visible_imports(tree: ast.Module, line: int) -> dict[str, ImportBinding]:
    nodes: list[ast.AST] = [
        node
        for node in _module_scope_nodes(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and int(node.lineno) <= line
    ]
    scope = _containing_scope(tree, line)
    if scope is not None:
        nodes.extend(
            node
            for node in _scope_import_nodes(scope)
            if int(getattr(node, "lineno", line + 1)) <= line
        )
    bindings = _bindings_from_nodes(nodes)
    return {binding.local_name: binding for binding in bindings}


def _runtime_candidates(
    entry: dict[str, Any], index: PythonSourceIndex
) -> tuple[list[Candidate], list[str]]:
    diagnostics: list[str] = []
    runtime_event = entry.get("runtime_event")
    if not isinstance(runtime_event, dict):
        return [], diagnostics
    implementation = runtime_event.get("implementation")
    if not isinstance(implementation, dict):
        return [], diagnostics
    line = implementation.get("definition_line")
    try:
        definition_line = int(line)
    except (TypeError, ValueError):
        return [], diagnostics
    interface_name = str(entry.get("interface") or "").rsplit(".", 1)[-1]
    if not _is_identifier(interface_name):
        return [], diagnostics

    out: list[Candidate] = []
    for raw_path in implementation.get("source_files") or []:
        path = Path(str(raw_path)).expanduser()
        if path.suffix != ".py":
            continue
        if not path.is_file():
            diagnostics.append(f"runtime source file not found: {path}")
            continue
        tree = index.tree(path)
        if tree is None:
            diagnostics.append(f"runtime source is not parseable Python: {path}")
            continue
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == interface_name
            and min(
                [int(node.lineno)]
                + [int(getattr(d, "lineno", node.lineno)) for d in node.decorator_list]
            )
            <= definition_line
            <= _node_end_line(node)
        ]
        for node in matches:
            out.append(
                Candidate(
                    str(_absolute(path)),
                    _definition_line(node, include_decorators=True),
                )
            )
    return _dedupe_candidates(out), diagnostics


def _callsite_context(
    entry: dict[str, Any], index: PythonSourceIndex
) -> tuple[Path | None, str | None, dict[str, ImportBinding], list[str]]:
    diagnostics: list[str] = []
    runtime_event = entry.get("runtime_event")
    call_site = runtime_event.get("call_site") if isinstance(runtime_event, dict) else None
    if not isinstance(call_site, dict):
        diagnostics.append("runtime_event.call_site is missing")
        return None, None, {}, diagnostics
    path = Path(str(call_site.get("file") or "")).expanduser()
    try:
        line = int(call_site.get("line"))
    except (TypeError, ValueError):
        diagnostics.append("runtime_event.call_site.line is invalid")
        return path, None, {}, diagnostics
    if not path.is_file():
        diagnostics.append(f"call-site file not found: {path}")
        return path, None, {}, diagnostics
    tree = index.tree(path)
    if tree is None:
        diagnostics.append(f"call-site is not parseable Python: {path}")
        return path, None, {}, diagnostics
    call = _find_call(tree, line)
    if call is None:
        diagnostics.append(f"no call expression covers call-site line {line}: {path}")
        return path, None, _visible_imports(tree, line), diagnostics
    return path, _callable_chain(call.func), _visible_imports(tree, line), diagnostics


def _resolve_callsite_import(
    path: Path | None,
    callable_name: str | None,
    bindings: dict[str, ImportBinding],
    index: PythonSourceIndex,
) -> list[Candidate]:
    if path is None or not callable_name:
        return []
    parts = callable_name.split(".")
    binding = bindings.get(parts[0])
    if binding is None:
        return []
    if binding.symbol is not None:
        symbol = binding.symbol
        if len(parts) > 1:
            # Imported class/module followed by an attribute. Class methods are
            # resolved separately from the interface qualname.
            symbol = parts[-1]
        return index.resolve_module_symbol(
            binding.module,
            symbol,
            current_file=path,
            level=binding.level,
        )
    if len(parts) < 2:
        return []
    return index.resolve_module_symbol(binding.module, parts[-1])


def _interface_class_method(
    interface: str, index: PythonSourceIndex
) -> list[Candidate]:
    parts = interface.split(".")
    if len(parts) == 2 and parts[0][:1].isupper():
        return index.find_class_method(parts[0], parts[1])
    return []


def _interface_module_symbol(
    interface: str, index: PythonSourceIndex
) -> list[Candidate]:
    if interface.startswith("torch.ops."):
        return []
    parts = interface.split(".")
    if len(parts) < 2 or parts[0][:1].isupper():
        return []
    symbol = parts[-1]
    if not _is_identifier(symbol):
        return []
    # Try the conventional module.symbol form first. Some interfaces name a
    # module and its exported function identically, so also try the full dotted
    # value as a module path.
    out = index.resolve_module_symbol(".".join(parts[:-1]), symbol)
    if out:
        return out
    return index.resolve_module_symbol(interface, symbol)


def _fallback_names(
    entry: dict[str, Any], callable_name: str | None
) -> list[str]:
    interface = str(entry.get("interface") or "")
    names: list[str] = []
    leaf = interface.rsplit(".", 1)[-1]
    if _is_identifier(leaf):
        names.append(leaf)
    if callable_name:
        call_leaf = callable_name.rsplit(".", 1)[-1]
        if _is_identifier(call_leaf):
            names.append(call_leaf)
    kernel = entry.get("kernel")
    if isinstance(kernel, dict):
        normalized = str(kernel.get("normalized_name") or "")
        if _is_identifier(normalized):
            names.append(normalized)
    return list(dict.fromkeys(names))


def _hint_for_entry(entry: dict[str, Any], index: PythonSourceIndex) -> str | None:
    interface = str(entry.get("interface") or "")
    provider = str(entry.get("binding_provider") or "")
    selectors: list[str]
    if interface.startswith("torch.ops.sgl_kernel") or interface.startswith("sglang"):
        selectors = ["sgl_kernel", "sglang"]
    elif interface.startswith("torch.ops.flashinfer") or interface.startswith("flashinfer"):
        selectors = ["flashinfer"]
    elif interface.startswith("deep_gemm") or provider == "deepgemm":
        selectors = ["deep_gemm", "deepgemm"]
    else:
        selectors = []
    for selector in selectors:
        matches = [root for root in index.roots if selector in root.name.lower()]
        if matches:
            return str(max(matches, key=lambda root: len(root.path.parts)).path)
    return None


def locate_interface_definition(
    entry: dict[str, Any], index: PythonSourceIndex
) -> InterfaceSearchResult:
    code = str(entry.get("archetype_code") or "")
    archetype = str(entry.get("archetype") or "")
    if code == "F0" or archetype == "pytorch_native":
        return InterfaceSearchResult(
            status=STATUS_NOT_APPLICABLE,
            candidates=[],
            repo_hint=None,
            evidence="archetype_null_rule",
        )

    diagnostics: list[str] = []
    candidates, runtime_diagnostics = _runtime_candidates(entry, index)
    diagnostics.extend(runtime_diagnostics)
    if candidates:
        return _result_from_candidates(
            candidates,
            index=index,
            evidence="runtime_event.implementation",
            diagnostics=diagnostics,
        )

    call_path, callable_name, bindings, call_diagnostics = _callsite_context(entry, index)
    diagnostics.extend(call_diagnostics)

    candidates = _resolve_callsite_import(call_path, callable_name, bindings, index)
    if candidates:
        return _result_from_candidates(
            candidates,
            index=index,
            evidence="call_site_import",
            diagnostics=diagnostics,
        )

    interface = str(entry.get("interface") or "")
    candidates = _interface_class_method(interface, index)
    if not candidates:
        candidates = _interface_module_symbol(interface, index)
    if candidates:
        return _result_from_candidates(
            candidates,
            index=index,
            evidence="interface_qualified_name",
            diagnostics=diagnostics,
        )

    for name in _fallback_names(entry, callable_name):
        candidates = index.find_named_definitions(name)
        if candidates:
            return _result_from_candidates(
                candidates,
                index=index,
                evidence=f"exact_name_search:{name}",
                diagnostics=diagnostics,
            )

    diagnostics.append("no Python interface definition candidate found")
    return InterfaceSearchResult(
        status=STATUS_NOT_FOUND,
        candidates=[],
        repo_hint=_hint_for_entry(entry, index),
        evidence="not_found",
        diagnostics=diagnostics,
    )


def _result_from_candidates(
    candidates: list[Candidate],
    *,
    index: PythonSourceIndex,
    evidence: str,
    diagnostics: list[str],
) -> InterfaceSearchResult:
    candidates = _dedupe_candidates(candidates)
    return InterfaceSearchResult(
        status=STATUS_RESOLVED if len(candidates) == 1 else STATUS_AMBIGUOUS,
        candidates=candidates,
        repo_hint=index.repo_hint(candidates),
        evidence=evidence,
        diagnostics=diagnostics,
    )


def load_search_roots(
    manifest_path: Path, sglang_repo_root: Path
) -> tuple[list[SearchRoot], list[dict[str, str]]]:
    sglang_repo_root = _absolute(sglang_repo_root)
    if not sglang_repo_root.is_dir():
        raise LocateError(f"sglang repo root not found: {sglang_repo_root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LocateError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise LocateError(f"invalid manifest JSON: {manifest_path}: {exc}") from exc
    except UnicodeError as exc:
        raise LocateError(f"manifest is not valid UTF-8: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise LocateError("manifest root must be a JSON object")

    roots = [SearchRoot("sglang", sglang_repo_root)]
    sgl_kernel = sglang_repo_root / "sgl-kernel"
    if sgl_kernel.is_dir():
        roots.append(SearchRoot("sgl_kernel", _absolute(sgl_kernel)))

    raw_repos = manifest.get("repos", [])
    if isinstance(raw_repos, dict):
        repos = [
            {"name": name, **record}
            for name, record in raw_repos.items()
            if isinstance(record, dict)
        ]
    elif isinstance(raw_repos, list):
        repos = [record for record in raw_repos if isinstance(record, dict)]
    else:
        raise LocateError("manifest.repos must be a list or object")

    skipped: list[dict[str, str]] = []
    seen = {root.path for root in roots}
    for record in repos:
        name = str(record.get("name") or "unknown")
        if record.get("status") != "ok":
            continue
        raw_path = record.get("local_path")
        if not raw_path:
            skipped.append({"name": name, "reason": "local_path is empty"})
            continue
        path = _absolute(Path(str(raw_path)))
        if not path.is_dir():
            skipped.append({"name": name, "reason": f"path not found: {path}"})
            continue
        if path not in seen:
            roots.append(SearchRoot(name, path))
            seen.add(path)
    return roots, skipped


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LocateError(f"schema not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LocateError(f"invalid schema JSON: {path}: {exc}") from exc
    except UnicodeError as exc:
        raise LocateError(f"schema is not valid UTF-8: {path}") from exc
    if not isinstance(data, dict):
        raise LocateError("schema root must be a JSON object")
    if not iter_kernel_entries(data):
        raise LocateError("schema contains no kernel entries")
    return data


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _report_payload(result: LocateRunResult) -> dict[str, Any]:
    needs_agent: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for kernel in result.kernels:
        if kernel.needs_agent:
            needs_agent.append(
                {
                    "low_level_id": kernel.low_level_id,
                    "interface": kernel.interface,
                    "status": kernel.status,
                    "hits": [candidate.to_dict() for candidate in kernel.candidates],
                    "repo_hint": kernel.repo_hint,
                    "unresolved_layers": kernel.unresolved_layers,
                    "diagnostics": kernel.diagnostics,
                }
            )
        if kernel.diagnostics:
            diagnostics.append(
                {
                    "low_level_id": kernel.low_level_id,
                    "interface": kernel.interface,
                    "evidence": kernel.evidence,
                    "messages": kernel.diagnostics,
                }
            )
    return {
        **result.summary(),
        "search_roots_skipped": result.skipped_roots,
        "needs_agent": needs_agent,
        "diagnostics": diagnostics,
    }


def locate_schema(
    schema_path: Path,
    *,
    manifest_path: Path,
    sglang_repo_root: Path,
    output_path: Path | None = None,
) -> LocateRunResult:
    schema_path = schema_path.expanduser().resolve()
    output_path = (output_path or schema_path).expanduser().resolve()
    data = _load_schema(schema_path)
    entries = iter_kernel_entries(data)
    for index, entry in enumerate(entries, 1):
        if _contains_protected_source(entry.get("source_locations")):
            raise LocateError(
                f"refusing to overwrite Layer-2/manual source_locations for "
                f"{_kernel_id(entry, index)}"
            )

    roots, skipped_roots = load_search_roots(
        manifest_path.expanduser().resolve(), sglang_repo_root
    )
    source_index = PythonSourceIndex(roots)
    kernel_results: list[KernelLocateResult] = []
    for index, entry in enumerate(entries, 1):
        source_locations = build_source_locations_template(entry)
        interface_result = locate_interface_definition(entry, source_index)
        source_locations["layers"]["interface_definition"] = interface_result.to_layer()
        unresolved_layers = [
            name
            for name, layer in source_locations["layers"].items()
            if layer.get("status") in _UNRESOLVED_STATUSES
        ]
        source_locations["needs_agent"] = bool(unresolved_layers)
        entry["source_locations"] = source_locations
        kernel_results.append(
            KernelLocateResult(
                low_level_id=_kernel_id(entry, index),
                interface=str(entry.get("interface") or ""),
                status=interface_result.status,
                candidates=interface_result.candidates,
                repo_hint=interface_result.repo_hint,
                evidence=interface_result.evidence,
                diagnostics=interface_result.diagnostics,
                needs_agent=bool(unresolved_layers),
                unresolved_layers=unresolved_layers,
            )
        )

    report_path = output_path.parent / "ref" / "locate_report.json"
    result = LocateRunResult(
        schema=schema_path,
        output=output_path,
        report_path=report_path,
        kernels=kernel_results,
        skipped_roots=skipped_roots,
    )
    _write_json_atomic(output_path, data)
    _write_json_atomic(report_path, _report_payload(result))
    return result
