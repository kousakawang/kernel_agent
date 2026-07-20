"""Mechanical extraction of the source-locate Agent's four-layer result.

``resolved`` and ``best_effort`` hits are copied as whole files.  Python AST or
C-family brace scanning computes only the focus range recorded in
``read_hints.txt``; no ``end_line`` is added to the schema.  ``missed`` and
``not_applicable`` are valid final states and produce explicit placeholders.
"""

from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    DIRECTORY_LAYERS,
    EXTRACTABLE_STATUSES,
    LAYER_FILENAME,
    LAYER_PLACEHOLDER_FILENAME,
    LAYERS,
    LayerHit,
    LayerResult,
    ORDERED_DIRECTORY_LAYERS,
    STATUS_BEST_EFFORT,
    STATUS_NOT_APPLICABLE,
    agent_layers,
    load_json_object,
    validate_agent_schema,
    write_json_atomic,
)

# Range-completion caps + file-type sets.
_MAX_SPAN = 4000  # safety cap on how many lines a single definition may span
_C_SUFFIXES = {".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hh"}
_COMMENT_PREFIX = {
    ".py": "# ",
    ".cc": "// ",
    ".cpp": "// ",
    ".cu": "// ",
    ".cuh": "// ",
    ".h": "// ",
    ".hpp": "// ",
}


class ExtractError(Exception):
    """A filesystem or hit error that prevents safe extraction."""


@dataclass
class KernelExtract:
    kernel_id: str
    layers_written: list[str] = field(default_factory=list)
    layers_placeholder: list[str] = field(default_factory=list)
    kernel_sources_dir: str = ""


@dataclass
class ExtractReport:
    schema: str
    workspace_out: str
    kernels: list[KernelExtract] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        written = sum(len(k.layers_written) for k in self.kernels)
        placeholders = sum(len(k.layers_placeholder) for k in self.kernels)
        return {
            "schema": self.schema,
            "workspace_out": self.workspace_out,
            "kernels": len(self.kernels),
            "written": written,
            "placeholders": placeholders,
        }


def _comment(ext: str, text: str) -> str:
    prefix = _COMMENT_PREFIX.get(ext, "# ")
    return f"{prefix}{text}\n"


def _directory_layer_filename(layer: str, index: int, hit_file: str | None) -> str:
    """Name for one hit inside a directory layer's ``<layer>/`` subdir.

    Ordered layers (kernel_impl call chain; py_cpp_binding py->cpp bridge order)
    get a numeric prefix; all use the source basename so the origin is obvious
    (e.g. ``1_activation.cu``). kernel_header is a 1:1 correspondence with impl
    files, so it is NOT numbered.
    """
    if hit_file:
        base = Path(hit_file).name
    else:
        base = LAYER_PLACEHOLDER_FILENAME[layer]
    if layer in ORDERED_DIRECTORY_LAYERS:
        return f"{index + 1}_{base}"
    return base


def _single_file_name(layer: str, hit_file: str | None) -> str:
    """Filename for the single-file interface_definition layer; keeps .py
    (or its source suffix if it's a known code extension)."""
    default = LAYER_FILENAME[layer]
    if hit_file:
        suffix = Path(hit_file).suffix
        if suffix in {".py", ".cu", ".cpp", ".cc", ".cuh", ".h", ".hpp"}:
            stem = Path(default).stem
            return f"{stem}{suffix}"
    return default


def _end_line_python(lines: list[str], def_line: int) -> int:
    """End line (1-based, inclusive) of the def/class starting at ``def_line``.

    Prefer AST (robust to nested blocks/strings); fall back to indentation when
    the file/snippet does not parse.
    """
    text = "".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _end_line_by_indent(lines, def_line)
    best: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", None)
            # Decorators sit above node.lineno; def_line may point at the first
            # decorator, so match if def_line falls in [first_decorator, lineno].
            deco_start = min(
                [getattr(d, "lineno", start) for d in node.decorator_list] + [start]
            )
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                continue
            if deco_start <= def_line <= start:
                if best is None or end > best:
                    best = end
    if best is not None:
        return best
    return _end_line_by_indent(lines, def_line)


def _end_line_by_indent(lines: list[str], def_line: int) -> int:
    """Indentation fallback: body ends when indentation returns to <= the def's."""
    n = len(lines)
    i = def_line - 1
    if i >= n:
        return def_line
    base_indent = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    last = i
    while j < n:
        line = lines[j]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                break
            last = j
        j += 1
    return last + 1


def _end_line_c_family(lines: list[str], def_line: int) -> int:
    """End line for a C/C++/CUDA definition starting at ``def_line``.

    Scans forward to the matching closing brace of the first ``{``; if no brace
    appears before a terminating ``;`` (a bare declaration, e.g. a header
    prototype), the statement ends at that ``;``.

    The scanner is comment/string/char aware: ``{``/``}``/``;`` inside line
    comments (``//``), block comments (``/* */``), string literals, char
    literals (``'{'``), and raw strings (``R"(...)"``) are ignored. Without this,
    a commented-out or quoted brace — very common in real kernels — unbalances
    the count and runs the range far past the true end (e.g. sgl-attn's
    ``mha_fwd`` overshooting by ~570 lines off a ``// ... {`` in the body).
    """
    n = len(lines)
    depth = 0
    seen_brace = False
    # State that must persist ACROSS lines. Block comments and raw strings can
    # span multiple lines; plain string/char literals cannot (C forbids a raw
    # newline in them), so those reset per line — which also defends against a
    # stray apostrophe (e.g. a C++14 digit separator) swallowing a whole line.
    in_block_comment = False
    in_raw_string = False
    raw_delim = ""

    last = min(n, def_line - 1 + _MAX_SPAN)
    for idx in range(def_line - 1, last):
        line = lines[idx]
        in_string = False
        in_char = False
        i = 0
        width = len(line)
        while i < width:
            ch = line[i]
            nxt = line[i + 1] if i + 1 < width else ""

            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if in_raw_string:
                closing = ")" + raw_delim + '"'
                if line.startswith(closing, i):
                    in_raw_string = False
                    i += len(closing)
                    continue
                i += 1
                continue
            if in_string:
                if ch == "\\":
                    i += 2  # escaped char (e.g. \" ) — skip both
                    continue
                if ch == '"':
                    in_string = False
                i += 1
                continue
            if in_char:
                if ch == "\\":
                    i += 2
                    continue
                if ch == "'":
                    in_char = False
                i += 1
                continue

            # --- NORMAL state ---------------------------------------------
            if ch == "/" and nxt == "/":
                break  # line comment: ignore the rest of this line
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch == "R" and nxt == '"':  # raw string R"delim( ... )delim"
                j = i + 2
                delim = ""
                while j < width and line[j] != "(" and len(delim) < 16:
                    delim += line[j]
                    j += 1
                if j < width and line[j] == "(":
                    in_raw_string = True
                    raw_delim = delim
                    i = j + 1
                    continue
                # not a real raw-string opener; fall through
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == "'":
                # Char literal — unless it's a C++14 digit separator (1'000),
                # which is always wedged between two alphanumerics.
                prev = line[i - 1] if i > 0 else ""
                if not prev.isalnum():
                    in_char = True
                i += 1
                continue

            if ch == "{":
                depth += 1
                seen_brace = True
            elif ch == "}":
                depth -= 1
                if seen_brace and depth == 0:
                    return idx + 1
            elif ch == ";" and not seen_brace:
                return idx + 1  # bare declaration/prototype
            i += 1
    return last if n else def_line


def _end_line_for(src: Path, lines: list[str], def_line: int) -> int:
    """Dispatch range-completion by file type; clamp to file bounds + cap."""
    n = len(lines)
    if def_line < 1 or def_line > n:
        return min(max(def_line, 1), n) if n else def_line
    suffix = src.suffix.lower()
    if suffix == ".py":
        end = _end_line_python(lines, def_line)
    elif suffix in _C_SUFFIXES:
        end = _end_line_c_family(lines, def_line)
    else:
        end = def_line
    end = max(def_line, min(end, n, def_line + _MAX_SPAN))
    return end


def _hit_problem(hit: LayerHit) -> str | None:
    """Filesystem/validity check for a single hit."""
    path = Path(hit.file)
    if not path.exists():
        return f"file not found: {hit.file}"
    if not path.is_file():
        return f"not a file: {hit.file}"
    try:
        total = len(path.read_text(errors="ignore").splitlines())
    except OSError as exc:
        return f"unreadable: {exc}"
    if hit.def_line > total:
        return f"def_line {hit.def_line} out of range (1-{total})"
    return None


def _range_hint(src: Path, def_line: int) -> str:
    """Compute the focus range while leaving copied source bytes untouched."""

    text = src.read_text(errors="ignore")
    lines = text.splitlines(keepends=True)
    end = _end_line_for(src, lines, def_line)
    return f"read lines {def_line}-{end}"


def _validate_source_hits(entries: list[dict[str, Any]]) -> None:
    """Fail before cleaning old output when any declared hit is unusable."""

    problems: list[str] = []
    for entry in entries:
        kernel_id = entry["low_level_id"]
        for layer_name, result in agent_layers(entry).items():
            if not result.is_extractable:
                continue
            for index, hit in enumerate(result.hits):
                problem = _hit_problem(hit)
                if problem:
                    problems.append(
                        f"{kernel_id}/{layer_name}/hit[{index}]: {problem}"
                    )
    if problems:
        raise ExtractError("invalid source hits:\n  - " + "\n  - ".join(problems))


def extract_workspace(
    schema_path: Path,
    workspace_out: Path | None = None,
) -> ExtractReport:
    schema_path = Path(schema_path).resolve()
    schema = load_json_object(schema_path, label="schema")
    entries = validate_agent_schema(schema)
    _validate_source_hits(entries)

    workspace_out = (
        Path(workspace_out).resolve() if workspace_out else schema_path.parent
    )
    kernels_root = workspace_out / "kernel_sources"

    # --- Extraction ----------------------------------------------------------
    # Regenerate from a clean slate: wipe the whole kernel_sources/ tree so a
    # re-run (after the user changed config/paths/ids) leaves no residue — e.g. a
    # previous run's kernel_impl.py sitting next to the new kernel_impl.cu, or an
    # orphaned subdir from a renamed/removed kernel. Done only after structural
    # and filesystem preflight, so invalid input preserves previous good output.
    if kernels_root.exists():
        shutil.rmtree(kernels_root)

    report = ExtractReport(schema=str(schema_path), workspace_out=str(workspace_out))
    for entry in entries:
        kid = entry["low_level_id"]
        dest = kernels_root / kid
        dest.mkdir(parents=True, exist_ok=True)
        layers = agent_layers(entry)

        ke = KernelExtract(kernel_id=kid, kernel_sources_dir=str(dest))
        read_hints: list[str] = []

        for name in LAYERS:
            written, hints = _emit_layer(dest, name, layers[name])
            read_hints.extend(hints)
            if written == "written":
                ke.layers_written.append(name)
            else:
                ke.layers_placeholder.append(name)

        (dest / "read_hints.txt").write_text("\n".join(read_hints) + "\n")
        entry["kernel_sources_dir"] = str(dest)
        report.kernels.append(ke)

    write_json_atomic(schema_path, schema)
    return report


def _emit_layer(
    dest: Path,
    layer: str,
    result: LayerResult,
) -> tuple[str, list[str]]:
    """Write one layer's file(s); return (status, read_hint_lines).

    The single-file interface_definition layer writes ``dest/<layer_filename>``;
    directory layers (kernel_impl / kernel_header / py_cpp_binding) write into
    ``dest/<layer>/`` — one file per hit (kernel_impl + py_cpp_binding numbered by
    order). Returns one hint line per file.
    """
    is_dir = layer in DIRECTORY_LAYERS
    if result.status in EXTRACTABLE_STATUSES:
        if is_dir:
            return _emit_directory_resolved(dest, layer, result)
        return _emit_single_resolved(dest, layer, result)

    # missed and not_applicable are valid Agent conclusions.
    if is_dir:
        return _emit_directory_placeholder(dest, layer, result)
    return _emit_single_placeholder(dest, layer, result)


def _emit_single_resolved(dest: Path, layer: str, result: LayerResult) -> tuple[str, list[str]]:
    hit = result.hits[0]
    filename = _single_file_name(layer, hit.file)
    source = Path(hit.file)
    hint = _range_hint(source, hit.def_line)
    shutil.copyfile(source, dest / filename)
    qualifier = " [best_effort]" if result.status == STATUS_BEST_EFFORT else ""
    return "written", [f"{filename}: {hint}{qualifier}  (from {hit.file})"]


def _emit_directory_resolved(dest: Path, layer: str, result: LayerResult) -> tuple[str, list[str]]:
    subdir = dest / layer
    subdir.mkdir(parents=True, exist_ok=True)
    hints: list[str] = []
    for i, hit in enumerate(result.hits):
        filename = _directory_layer_filename(layer, i, hit.file)
        source = Path(hit.file)
        hint = _range_hint(source, hit.def_line)
        shutil.copyfile(source, subdir / filename)
        qualifier = " [best_effort]" if result.status == STATUS_BEST_EFFORT else ""
        hints.append(
            f"{layer}/{filename}: {hint}{qualifier}  (from {hit.file})"
        )
    return "written", hints


def _emit_single_placeholder(
    dest: Path, layer: str, result: LayerResult
) -> tuple[str, list[str]]:
    filename = LAYER_FILENAME[layer]
    if result.status == STATUS_NOT_APPLICABLE:
        (dest / filename).write_text(
            _comment(Path(filename).suffix, "该层不适用。")
        )
        return "placeholder", [f"{filename}: N/A (not applicable)"]

    reason = "status=missed"
    (dest / filename).write_text(
        _comment(
            Path(filename).suffix,
            f"该层未定位（{reason}），见 ref/locate_agent_notes.md。",
        )
    )
    return "placeholder", [f"{filename}: MISSING ({reason})"]


def _emit_directory_placeholder(
    dest: Path, layer: str, result: LayerResult
) -> tuple[str, list[str]]:
    subdir = dest / layer
    subdir.mkdir(parents=True, exist_ok=True)
    placeholder = LAYER_PLACEHOLDER_FILENAME[layer]
    if result.status == STATUS_NOT_APPLICABLE:
        (subdir / placeholder).write_text(
            _comment(Path(placeholder).suffix, "该层不适用。")
        )
        return "placeholder", [f"{layer}/: N/A (not applicable)"]

    reason = "status=missed"
    (subdir / placeholder).write_text(
        _comment(
            Path(placeholder).suffix,
            f"该层未定位（{reason}），见 ref/locate_agent_notes.md。",
        )
    )
    return "placeholder", [f"{layer}/: MISSING ({reason})"]
