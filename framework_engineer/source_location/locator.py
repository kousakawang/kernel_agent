"""Deterministic Python interface candidate locator for KID v3 schemas.

The locator never decides the final four source layers.  It resolves imports,
qualified Python interfaces, class methods, and explicit re-exports inside the
configured source repositories, then writes a transient ``locate_candidates``
block for the source-locate Agent to verify or replace.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .contracts import (
    CANDIDATE_AMBIGUOUS,
    CANDIDATE_NOT_FOUND,
    CANDIDATE_RESOLVED,
    ContractError,
    kernel_entries,
    load_json_object,
    validate_kid_schema,
    write_json_atomic,
)


class LocateError(ContractError):
    """A global input error that prevents candidate location."""


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


@dataclass(frozen=True)
class ClassDefinition:
    file: Path
    node: ast.ClassDef


@dataclass
class InterfaceSearchResult:
    status: str
    candidates: list[Candidate]
    repo_hint: str | None
    evidence: str
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "repo_hint": self.repo_hint,
            "evidence": self.evidence,
            "diagnostics": self.diagnostics,
        }


@dataclass
class KernelLocateResult:
    low_level_id: str
    interface: str
    result: InterfaceSearchResult


@dataclass
class LocateRunResult:
    schema: Path
    output: Path
    kernels: list[KernelLocateResult]
    skipped_roots: list[dict[str, str]]

    def summary(self) -> dict[str, Any]:
        counts = {
            CANDIDATE_RESOLVED: 0,
            CANDIDATE_AMBIGUOUS: 0,
            CANDIDATE_NOT_FOUND: 0,
        }
        for kernel in self.kernels:
            counts[kernel.result.status] += 1
        return {
            "schema": str(self.schema),
            "output": str(self.output),
            "total": len(self.kernels),
            "interface_resolved": counts[CANDIDATE_RESOLVED],
            "interface_ambiguous": counts[CANDIDATE_AMBIGUOUS],
            "interface_not_found": counts[CANDIDATE_NOT_FOUND],
            "search_roots_skipped": self.skipped_roots,
        }


def _absolute(path: Path) -> Path:
    """Make a path absolute without replacing a manifest symlink spelling."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _dedupe(candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(set(candidates), key=lambda item: (item.file, item.def_line))


def _is_identifier(value: str) -> bool:
    return bool(value) and value.isidentifier()


def _node_end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)))


def _module_scope_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield module statements, descending through control flow but not defs."""

    def visit(statements: Iterable[ast.stmt]) -> Iterator[ast.AST]:
        for statement in statements:
            yield statement
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            for field_name in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field_name, None)
                if isinstance(nested, list):
                    yield from visit(
                        child for child in nested if isinstance(child, ast.stmt)
                    )
            handlers = getattr(statement, "handlers", None)
            if isinstance(handlers, list):
                for handler in handlers:
                    body = getattr(handler, "body", None)
                    if isinstance(body, list):
                        yield from visit(
                            child for child in body if isinstance(child, ast.stmt)
                        )

    yield from visit(tree.body)


def _scope_import_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield imports visible in one function scope, excluding nested defs."""

    def visit(node: ast.AST, *, root: bool = False) -> Iterator[ast.AST]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            return
        if not root and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    yield from visit(scope, root=True)


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


class PythonSourceIndex:
    """Read-only AST index over explicit repository roots."""

    def __init__(self, roots: list[SearchRoot]):
        self.roots = roots
        files: set[Path] = set()
        for root in roots:
            files.update(
                _absolute(path)
                for path in root.path.rglob("*.py")
                if ".git" not in path.parts and "__pycache__" not in path.parts
            )
        self.python_files = sorted(files)
        self._text_cache: dict[Path, str | None] = {}
        self._tree_cache: dict[Path, ast.Module | None] = {}
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

    def provider_hint(self, provider: str | None) -> str | None:
        if not provider:
            return None
        needle = provider.lower().replace("-", "_")
        matches = [
            root
            for root in self.roots
            if needle in root.name.lower().replace("-", "_")
        ]
        if len(matches) != 1:
            return None
        return str(matches[0].path)

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
        base = _absolute(current_file).parent
        for _ in range(max(0, level - 1)):
            base = base.parent
        if module:
            base = base.joinpath(*module.split("."))
        candidates = [base.with_suffix(".py"), base / "__init__.py"]
        return sorted(_absolute(path) for path in candidates if path.is_file())

    def _module_imports(self, path: Path) -> list[ImportBinding]:
        tree = self.tree(path)
        if tree is None:
            return []
        return _bindings_from_nodes(
            _module_scope_nodes(tree), source_text=self.text(path)
        )

    def _direct_symbols(self, path: Path, symbol: str) -> list[Candidate]:
        tree = self.tree(path)
        if tree is None:
            return []
        return [
            Candidate(str(_absolute(path)), int(node.lineno))
            for node in _module_scope_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ]

    def resolve_symbol(
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
        if key in visited or len(visited) > 30:
            return []
        visited.add(key)
        files = (
            self.relative_module_files(current_file, module, level)
            if current_file is not None and level > 0
            else self.module_files(module or "")
        )
        if not files:
            return []

        direct: list[Candidate] = []
        for path in files:
            direct.extend(self._direct_symbols(path, symbol))
        if direct:
            return _dedupe(direct)

        reexports: list[Candidate] = []
        for path in files:
            for binding in self._module_imports(path):
                if binding.local_name != symbol:
                    continue
                target_symbol = binding.symbol or symbol
                nested = self.resolve_symbol(
                    binding.module,
                    target_symbol,
                    current_file=path,
                    level=binding.level,
                    visited=visited,
                )
                if nested:
                    reexports.extend(nested)
                else:
                    # A binary module has no parseable Python definition.  Its
                    # explicit public re-export is the deterministic anchor.
                    reexports.append(Candidate(str(_absolute(path)), binding.line))
        return _dedupe(reexports)

    def resolve_class(
        self,
        module: str | None,
        class_name: str,
        *,
        current_file: Path | None = None,
        level: int = 0,
        visited: set[tuple[str, str]] | None = None,
    ) -> list[ClassDefinition]:
        visited = set(visited or set())
        key = (f"{current_file}:{level}:{module}", class_name)
        if key in visited or len(visited) > 30:
            return []
        visited.add(key)
        files = (
            self.relative_module_files(current_file, module, level)
            if current_file is not None and level > 0
            else self.module_files(module or "")
        )
        direct: list[ClassDefinition] = []
        for path in files:
            tree = self.tree(path)
            if tree is None:
                continue
            direct.extend(
                ClassDefinition(_absolute(path), node)
                for node in _module_scope_nodes(tree)
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        if direct:
            return direct

        reexports: list[ClassDefinition] = []
        for path in files:
            for binding in self._module_imports(path):
                if binding.local_name != class_name:
                    continue
                reexports.extend(
                    self.resolve_class(
                        binding.module,
                        binding.symbol or class_name,
                        current_file=path,
                        level=binding.level,
                        visited=visited,
                    )
                )
        return reexports

    def resolve_class_method(
        self,
        module: str,
        class_name: str,
        method_name: str,
        *,
        current_file: Path | None = None,
        level: int = 0,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for definition in self.resolve_class(
            module,
            class_name,
            current_file=current_file,
            level=level,
        ):
            methods = [
                child
                for child in definition.node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method_name
            ]
            # Overload declarations plus their implementation form one public
            # interface; anchor the first declaration deterministically.
            if methods:
                candidates.append(
                    Candidate(str(definition.file), int(methods[0].lineno))
                )
        return _dedupe(candidates)


def _callable_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _callable_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


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
            int(getattr(node, "end_col_offset", 0))
            - int(getattr(node, "col_offset", 0)),
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
    return {binding.local_name: binding for binding in _bindings_from_nodes(nodes)}


def _callsite_context(
    entry: dict[str, Any], index: PythonSourceIndex
) -> tuple[Path | None, str | None, dict[str, ImportBinding], list[str]]:
    diagnostics: list[str] = []
    call_site = entry["runtime_event"]["call_site"]
    path = Path(call_site["file"]).expanduser()
    line = int(call_site["line"])
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
        if len(parts) == 1:
            return index.resolve_symbol(
                binding.module,
                binding.symbol,
                current_file=path,
                level=binding.level,
            )
        # A directly imported class followed by a class attribute/method.
        module = binding.module or ""
        return index.resolve_class_method(
            module,
            binding.symbol,
            parts[-1],
            current_file=path,
            level=binding.level,
        )
    if len(parts) < 2:
        return []
    return index.resolve_symbol(binding.module, parts[-1])


def _resolve_qualified_interface(
    interface: str, index: PythonSourceIndex
) -> list[Candidate]:
    if interface.startswith("torch.ops."):
        return []
    parts = interface.split(".")
    if len(parts) < 2 or not all(_is_identifier(part) for part in parts):
        return []

    # module.function or module.reexported_symbol
    candidates = index.resolve_symbol(".".join(parts[:-1]), parts[-1])
    if candidates:
        return candidates

    # module.ReExportedClass.method
    if len(parts) >= 3:
        candidates = index.resolve_class_method(
            ".".join(parts[:-2]), parts[-2], parts[-1]
        )
        if candidates:
            return candidates
    return []


def _result_from_candidates(
    candidates: list[Candidate],
    *,
    index: PythonSourceIndex,
    evidence: str,
    diagnostics: list[str],
) -> InterfaceSearchResult:
    candidates = _dedupe(candidates)
    return InterfaceSearchResult(
        status=(
            CANDIDATE_RESOLVED
            if len(candidates) == 1
            else CANDIDATE_AMBIGUOUS
        ),
        candidates=candidates,
        repo_hint=index.repo_hint(candidates),
        evidence=evidence,
        diagnostics=diagnostics,
    )


def locate_interface_definition(
    entry: dict[str, Any], index: PythonSourceIndex
) -> InterfaceSearchResult:
    diagnostics: list[str] = []
    call_path, callable_name, bindings, call_diagnostics = _callsite_context(
        entry, index
    )
    diagnostics.extend(call_diagnostics)

    candidates = _resolve_callsite_import(
        call_path, callable_name, bindings, index
    )
    if candidates:
        return _result_from_candidates(
            candidates,
            index=index,
            evidence="call_site_import",
            diagnostics=diagnostics,
        )

    interface = entry["interface"]
    candidates = _resolve_qualified_interface(interface, index)
    if candidates:
        return _result_from_candidates(
            candidates,
            index=index,
            evidence="interface_qualified_name",
            diagnostics=diagnostics,
        )

    diagnostics.append(
        "no interface definition found in the configured source roots; "
        "global leaf-name fallback is intentionally disabled"
    )
    return InterfaceSearchResult(
        status=CANDIDATE_NOT_FOUND,
        candidates=[],
        repo_hint=index.provider_hint(entry.get("provider")),
        evidence="not_found",
        diagnostics=diagnostics,
    )


def load_search_roots(
    manifest_path: Path, sglang_repo_root: Path
) -> tuple[list[SearchRoot], list[dict[str, str]]]:
    sglang_repo_root = _absolute(sglang_repo_root)
    if not sglang_repo_root.is_dir():
        raise LocateError(f"sglang repo root not found: {sglang_repo_root}")
    try:
        manifest = load_json_object(manifest_path, label="manifest")
    except ContractError as exc:
        raise LocateError(str(exc)) from exc

    roots = [SearchRoot("sglang", sglang_repo_root)]
    nested_sgl_kernel = sglang_repo_root / "sgl-kernel"
    if nested_sgl_kernel.is_dir():
        roots.append(SearchRoot("sgl_kernel", _absolute(nested_sgl_kernel)))

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
        raise LocateError("manifest.repos must be an array or object")

    skipped: list[dict[str, str]] = []
    seen = {root.path for root in roots}
    for record in repos:
        name = str(record.get("name") or "unknown")
        if record.get("status") != "ok":
            skipped.append({"name": name, "reason": "status is not ok"})
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


def locate_schema(
    schema_path: Path,
    *,
    manifest_path: Path,
    sglang_repo_root: Path,
    output_path: Path,
) -> LocateRunResult:
    """Locate interface candidates and write an enriched copy of KID v3."""

    schema_path = _absolute(schema_path)
    output_path = _absolute(output_path)
    if schema_path == output_path:
        raise LocateError("--out must differ from --schema")
    try:
        data = load_json_object(schema_path, label="schema")
        entries = validate_kid_schema(data, allow_locate_candidates=True)
    except ContractError as exc:
        raise LocateError(str(exc)) from exc

    roots, skipped_roots = load_search_roots(
        _absolute(manifest_path), sglang_repo_root
    )
    source_index = PythonSourceIndex(roots)
    kernel_results: list[KernelLocateResult] = []
    for entry in entries:
        result = locate_interface_definition(entry, source_index)
        entry["locate_candidates"] = {
            "interface_definition": result.to_dict()
        }
        kernel_results.append(
            KernelLocateResult(
                low_level_id=entry["low_level_id"],
                interface=entry["interface"],
                result=result,
            )
        )

    write_json_atomic(output_path, data)
    return LocateRunResult(
        schema=schema_path,
        output=output_path,
        kernels=kernel_results,
        skipped_roots=skipped_roots,
    )
