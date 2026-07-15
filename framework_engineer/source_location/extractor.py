"""Layer 3 extraction (formerly `import-decomposition`).

Reads a locate-enriched ``decomposition_<backend>.schema.json`` and, for each
kernel, copies the four source layers into ``kernel_sources/<id>/`` files plus a
``read_hints.txt``, then backfills ``kernel_sources_dir`` into the schema.

Purely mechanical + idempotent (locate standard §1/§6):
  * resolved       -> copy the WHOLE source file; a range-completion helper (py:
                      AST/indent; c-family: brace/`;` matching) computes the
                      definition's end line only to record a ``read lines X-Y``
                      focus hint in read_hints.txt. Content is NEVER truncated —
                      filtering is left to a later translate_problem filter.
  * not_applicable -> empty file + comment (form-decided null, e.g. triton c/d)
  * missed/blank   -> placeholder empty file + comment (only with allow_empty)

Layer shapes (locate standard §2): ``interface_definition``/``py_cpp_binding``
are single-file (exactly 1 hit; >1 = ambiguous). ``kernel_impl``/``kernel_header``
are *directory* layers whose ``hits`` may hold multiple entries — each is copied
into a ``<layer>/`` subdirectory (kernel_impl preserves call order via index).

A ``missed``/unfilled REQUIRED layer is a hard stop unless ``allow_empty`` is set.
"""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    DIRECTORY_LAYERS,
    LAYER_FILENAME,
    LAYER_PLACEHOLDER_FILENAME,
    LAYERS,
    LayerHit,
    LayerResult,
    ORDERED_DIRECTORY_LAYERS,
    REQUIRED_LAYERS,
    SINGLE_FILE_LAYERS,
    STATUS_NOT_APPLICABLE,
    STATUS_RESOLVED,
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
    """Hard-stop condition (e.g. a required layer is unresolved)."""


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
    stopped: bool = False
    missing: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        written = sum(len(k.layers_written) for k in self.kernels)
        placeholders = sum(len(k.layers_placeholder) for k in self.kernels)
        return {
            "schema": self.schema,
            "workspace_out": self.workspace_out,
            "kernels": len(self.kernels),
            "written": written,
            "placeholders": placeholders,
            "stopped": self.stopped,
        }


def _iter_kernel_entries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of kernel entries regardless of schema nesting.

    KID's real schema nests kernels under invocations[].selected_kernels[];
    dry-run uses a flat ``kernels`` list. Support both.
    """
    if isinstance(schema.get("kernels"), list):
        return [k for k in schema["kernels"] if isinstance(k, dict)]
    out: list[dict[str, Any]] = []
    for inv in schema.get("invocations") or []:
        if isinstance(inv, dict):
            for k in inv.get("selected_kernels") or []:
                if isinstance(k, dict):
                    out.append(k)
    return out


def _kernel_id(entry: dict[str, Any], index: int) -> str:
    for key in ("low_level_id", "kernel_id", "id"):
        val = entry.get(key)
        if val and "<FILL" not in str(val):
            return str(val)
    kernel = entry.get("kernel") or {}
    name = kernel.get("normalized_name") or kernel.get("raw_name")
    if name and "<FILL" not in str(name):
        return str(name)
    return f"kernel_{index}"


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


def _layer_problem(result: LayerResult) -> str | None:
    """Return a reason string if a layer cannot be extracted, else None.

    Covers data-level gaps (status / unfilled placeholder / no hits), the
    single-file-vs-directory hit-count rule (single-file layers must have
    exactly one hit; >1 is ambiguous), AND filesystem gaps (a filled path that
    points nowhere / an out-of-range def_line). A wrong/nonexistent path is
    treated *exactly like "not filled"* (handoff contract).
    """
    if result.status == STATUS_NOT_APPLICABLE:
        return None  # legitimately empty; not a problem
    if result.status != STATUS_RESOLVED:
        return f"status={result.status}"
    if not result.hits:
        return "no hits"
    if result.name in SINGLE_FILE_LAYERS and len(result.hits) > 1:
        return f"ambiguous: {len(result.hits)} hits for single-file layer"
    for i, hit in enumerate(result.hits):
        prob = _hit_problem(hit)
        if prob is not None:
            # Point at which hit failed for directory layers.
            return prob if len(result.hits) == 1 else f"hit[{i}] {prob}"
    return None


def _hit_problem(hit: LayerHit) -> str | None:
    """Filesystem/validity check for a single hit."""
    if hit.is_fillable():
        return "unfilled placeholder"
    if not hit.file:
        return "empty file path"
    path = Path(hit.file)
    if not path.exists():
        return f"file not found: {hit.file}"
    if not path.is_file():
        return f"not a file: {hit.file}"
    if hit.def_line is not None:
        try:
            total = len(path.read_text(errors="ignore").splitlines())
        except OSError as exc:
            return f"unreadable: {exc}"
        if hit.def_line < 1 or hit.def_line > total:
            return f"def_line {hit.def_line} out of range (1-{total})"
    return None


def _copy_and_range(src: Path, def_line: int | None) -> tuple[str, str]:
    """Copy ``src`` verbatim; compute a *focus hint* range for read_hints.txt.

    Layer 3 is pure "copy the file + record where to look" — it does NOT truncate
    content (content filtering is left to a later translate_problem filter). So we
    return the WHOLE file; ``def_line`` + the computed end line only produce the
    ``read lines X-Y`` hint that tells the reader which span is the definition.
    """
    text = src.read_text(errors="ignore")
    if def_line is None:
        return text, "whole file"
    lines = text.splitlines(keepends=True)
    end = _end_line_for(src, lines, def_line)
    return text, f"read lines {def_line}-{end}"


def extract_workspace(
    schema_path: Path,
    workspace_out: Path | None = None,
    *,
    allow_empty: bool = False,
) -> ExtractReport:
    schema_path = Path(schema_path).resolve()
    schema = json.loads(schema_path.read_text())
    workspace_out = Path(workspace_out).resolve() if workspace_out else schema_path.parent
    kernels_root = workspace_out / "kernel_sources"

    entries = _iter_kernel_entries(schema)

    # --- Pre-flight hard-stop gate: any unresolved REQUIRED layer? -----------
    # A required layer is "missing" if it has no usable location OR the location
    # the user filled does not exist on disk (wrong/nonexistent path == not filled).
    missing: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        kid = _kernel_id(entry, idx)
        layers = _entry_layers(entry)
        for name in REQUIRED_LAYERS:
            result = layers.get(name)
            if result is None:
                missing.append({"kernel": kid, "layer": name, "repo_hint": None, "reason": "layer block absent"})
                continue
            problem = _layer_problem(result)
            if problem is not None:
                missing.append(
                    {"kernel": kid, "layer": name, "repo_hint": result.repo_hint, "reason": problem}
                )
    if missing and not allow_empty:
        report = ExtractReport(
            schema=str(schema_path),
            workspace_out=str(workspace_out),
            stopped=True,
            missing=missing,
        )
        return report

    # --- Extraction ----------------------------------------------------------
    # Regenerate from a clean slate: wipe the whole kernel_sources/ tree so a
    # re-run (after the user changed config/paths/ids) leaves no residue — e.g. a
    # previous run's kernel_impl.py sitting next to the new kernel_impl.cu, or an
    # orphaned subdir from a renamed/removed kernel. Done only *after* the gate,
    # so a hard-stopped re-run never destroys the previous good outputs.
    if kernels_root.exists():
        shutil.rmtree(kernels_root)

    report = ExtractReport(schema=str(schema_path), workspace_out=str(workspace_out))
    for idx, entry in enumerate(entries):
        kid = _kernel_id(entry, idx)
        dest = kernels_root / kid
        dest.mkdir(parents=True, exist_ok=True)
        layers = _entry_layers(entry)
        archetype = _entry_archetype(entry)

        ke = KernelExtract(kernel_id=kid, kernel_sources_dir=str(dest))
        read_hints: list[str] = []

        for name in LAYERS:
            result = layers.get(name) or LayerResult(name=name, status="not_found")
            written, hints = _emit_layer(dest, name, result, archetype, allow_empty)
            read_hints.extend(hints)
            if written == "written":
                ke.layers_written.append(name)
            else:
                ke.layers_placeholder.append(name)

        (dest / "read_hints.txt").write_text("\n".join(read_hints) + "\n")
        entry["kernel_sources_dir"] = str(dest)
        report.kernels.append(ke)

    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    return report


def _entry_layers(entry: dict[str, Any]) -> dict[str, LayerResult]:
    sl = entry.get("source_locations") or {}
    layers_raw = sl.get("layers") or {}
    out: dict[str, LayerResult] = {}
    for name in LAYERS:
        d = layers_raw.get(name)
        if isinstance(d, dict):
            out[name] = LayerResult.from_dict(name, d)
    return out


def _entry_archetype(entry: dict[str, Any]) -> str:
    sl = entry.get("source_locations") or {}
    return str(sl.get("archetype") or entry.get("archetype") or "unknown")


def _emit_layer(
    dest: Path,
    layer: str,
    result: LayerResult,
    archetype: str,
    allow_empty: bool,
) -> tuple[str, list[str]]:
    """Write one layer's file(s); return (status, read_hint_lines).

    The single-file interface_definition layer writes ``dest/<layer_filename>``;
    directory layers (kernel_impl / kernel_header / py_cpp_binding) write into
    ``dest/<layer>/`` — one file per hit (kernel_impl + py_cpp_binding numbered by
    order). Returns one hint line per file.
    """
    is_dir = layer in DIRECTORY_LAYERS
    problem = _layer_problem(result)

    if problem is None and result.status == STATUS_RESOLVED:
        if is_dir:
            return _emit_directory_resolved(dest, layer, result)
        return _emit_single_resolved(dest, layer, result)

    # not_applicable / missed / not_found / ambiguous / invalid -> placeholder.
    if is_dir:
        return _emit_directory_placeholder(dest, layer, result, archetype, problem)
    return _emit_single_placeholder(dest, layer, result, archetype, problem)


def _emit_single_resolved(dest: Path, layer: str, result: LayerResult) -> tuple[str, list[str]]:
    hit = result.hits[0]
    filename = _single_file_name(layer, hit.file)
    content, rng = _copy_and_range(Path(hit.file), hit.def_line)
    (dest / filename).write_text(content)
    return "written", [f"{filename}: {rng}  (from {hit.file})"]


def _emit_directory_resolved(dest: Path, layer: str, result: LayerResult) -> tuple[str, list[str]]:
    subdir = dest / layer
    subdir.mkdir(parents=True, exist_ok=True)
    hints: list[str] = []
    for i, hit in enumerate(result.hits):
        filename = _directory_layer_filename(layer, i, hit.file)
        content, rng = _copy_and_range(Path(hit.file), hit.def_line)
        (subdir / filename).write_text(content)
        hints.append(f"{layer}/{filename}: {rng}  (from {hit.file})")
    return "written", hints


def _emit_single_placeholder(
    dest: Path, layer: str, result: LayerResult, archetype: str, problem: str | None
) -> tuple[str, list[str]]:
    if result.status == STATUS_NOT_APPLICABLE:
        filename = LAYER_FILENAME[layer]
        (dest / filename).write_text(
            _comment(Path(filename).suffix, f"该层形态不适用（archetype={archetype}）。")
        )
        return "placeholder", [f"{filename}: N/A (not applicable for {archetype})"]

    # Preserve the filled source extension even when the path is wrong/missing.
    hit_file = result.hits[0].file if (result.hits and not result.hits[0].is_fillable()) else None
    filename = _single_file_name(layer, hit_file)
    reason = problem or "not located"
    (dest / filename).write_text(
        _comment(Path(filename).suffix, f"该层未定位（{reason}），见 ref/locate_agent_notes.md，用户已知风险。")
    )
    return "placeholder", [f"{filename}: MISSING ({reason})"]


def _emit_directory_placeholder(
    dest: Path, layer: str, result: LayerResult, archetype: str, problem: str | None
) -> tuple[str, list[str]]:
    subdir = dest / layer
    subdir.mkdir(parents=True, exist_ok=True)
    placeholder = LAYER_PLACEHOLDER_FILENAME[layer]
    if result.status == STATUS_NOT_APPLICABLE:
        (subdir / placeholder).write_text(
            _comment(Path(placeholder).suffix, f"该层形态不适用（archetype={archetype}）。")
        )
        return "placeholder", [f"{layer}/: N/A (not applicable for {archetype})"]

    reason = problem or "not located"
    (subdir / placeholder).write_text(
        _comment(Path(placeholder).suffix, f"该层未定位（{reason}），见 ref/locate_agent_notes.md，用户已知风险。")
    )
    return "placeholder", [f"{layer}/: MISSING ({reason})"]
