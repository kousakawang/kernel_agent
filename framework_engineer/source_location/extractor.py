"""Layer 3 extraction (formerly `import-decomposition`).

Reads a locate-enriched ``decomposition_<backend>.schema.json`` and, for each
kernel, slices the four source layers into ``kernel_sources/<id>/`` files plus a
``read_hints.txt``, then backfills ``kernel_sources_dir`` into the schema.

Purely mechanical + idempotent (KID_and_locate §5.6, handoff contract §4):
  * resolved       -> slice [line_start, line_end] (+padding) into the file
  * not_applicable -> empty file + comment (form-decided null, e.g. triton c/d)
  * missed/blank   -> placeholder empty file + comment (only with allow_empty)

A ``missed``/unfilled REQUIRED layer is a hard stop unless ``allow_empty`` is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    LAYER_FILENAME,
    LAYERS,
    LayerResult,
    REQUIRED_LAYERS,
    STATUS_NOT_APPLICABLE,
    STATUS_RESOLVED,
)

PADDING_LINES = 200
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


def _target_filename(layer: str, hit_file: str | None) -> str:
    """Refine kernel_impl's extension from the resolved source suffix."""
    default = LAYER_FILENAME[layer]
    if layer == "kernel_impl" and hit_file:
        suffix = Path(hit_file).suffix
        if suffix in {".py", ".cu", ".cpp", ".cc"}:
            return f"kernel_impl{suffix}"
    return default


def _layer_problem(result: LayerResult) -> str | None:
    """Return a reason string if a layer cannot be extracted, else None.

    Covers both data-level gaps (status / unfilled placeholder / no hits) AND
    filesystem-level gaps (path filled but points nowhere, or an incoherent line
    range). The latter is treated *exactly like "not filled"* — per the handoff
    contract, a user who fills a wrong/nonexistent path must be caught the same
    as if the layer were never located.
    """
    if result.status == STATUS_NOT_APPLICABLE:
        return None  # legitimately empty; not a problem
    if result.status != STATUS_RESOLVED:
        return f"status={result.status}"
    if not result.hits:
        return "no hits"
    hit = result.hits[0]
    if hit.is_fillable():
        return "unfilled placeholder"
    if not hit.file:
        return "empty file path"
    path = Path(hit.file)
    if not path.exists():
        return f"file not found: {hit.file}"
    if not path.is_file():
        return f"not a file: {hit.file}"
    if (
        hit.line_start is not None
        and hit.line_end is not None
        and hit.line_end < hit.line_start
    ):
        return f"invalid line range {hit.line_start}-{hit.line_end}"
    return None


def _slice_source(src: Path, line_start: int | None, line_end: int | None) -> str:
    text = src.read_text(errors="ignore")
    if line_start is None:
        return text
    lines = text.splitlines(keepends=True)
    start = max(1, line_start) - 1
    end = line_end if line_end is not None else line_start
    start = max(0, start - PADDING_LINES)
    end = min(len(lines), end + PADDING_LINES)
    return "".join(lines[start:end])


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
            written, hint = _emit_layer(dest, name, result, archetype, allow_empty)
            read_hints.append(hint)
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
) -> tuple[str, str]:
    """Write one layer file; return (status, read_hint_line)."""
    problem = _layer_problem(result)
    if problem is None and result.status == STATUS_RESOLVED:
        hit = result.hits[0]
        filename = _target_filename(layer, hit.file)
        content = _slice_source(Path(hit.file), hit.line_start, hit.line_end)
        (dest / filename).write_text(content)
        rng = (
            f"read lines {hit.line_start}-{hit.line_end}"
            if hit.line_start is not None
            else "whole file"
        )
        return "written", f"{filename}: {rng}  (from {hit.file})"

    filename = LAYER_FILENAME[layer]
    ext = Path(filename).suffix
    if result.status == STATUS_NOT_APPLICABLE:
        (dest / filename).write_text(
            _comment(ext, f"该层形态不适用（archetype={archetype}）。")
        )
        return "placeholder", f"{filename}: N/A (not applicable for {archetype})"

    # missed / not_found / ambiguous / invalid-path -> placeholder.
    # `problem` distinguishes a plain "not located" from a filled-but-invalid
    # path (e.g. file not found), so the user sees WHY it was treated as missing.
    reason = problem or "not located"
    (dest / filename).write_text(
        _comment(ext, f"该层未定位（{reason}），见 locate_agent_notes.md，用户已知风险。")
    )
    return "placeholder", f"{filename}: MISSING ({reason})"
