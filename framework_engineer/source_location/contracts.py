"""Data contracts for source-location layers (KID_and_locate §5.5).

The Layer 3 `extract` stage only *reads* the ``source_locations`` block that
Layer 1/2 wrote into a kernel's schema entry, so this module keeps the shapes
minimal and parse-oriented. The full producer-side contracts (LayerResolution
with dispatch) live with the future locator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The four source layers, in the canonical order used across the pipeline.
LAYERS: tuple[str, ...] = (
    "interface_definition",  # a
    "kernel_impl",           # b
    "py_cpp_binding",        # c
    "kernel_header",         # d
)

# Layers that must resolve to a real location (never legitimately null).
REQUIRED_LAYERS: tuple[str, ...] = ("interface_definition", "kernel_impl")

# Directory layers may hold *multiple* hits (locate standard §2): kernel_impl is
# the ordered call chain (launcher -> ... -> __global__); py_cpp_binding is the
# py<->cpp bridge whose form varies by AOT/JIT (a .py load_jit line + a C++
# *_binding.cu FFI export may BOTH be needed, so multi-file/multi-format);
# kernel_header is the per-impl-file headers. Only interface_definition is
# single-file (exactly 1 hit; >1 hit is ambiguous). Layer 3 extracts a directory
# layer into a <layer>/ subdirectory.
DIRECTORY_LAYERS: tuple[str, ...] = ("kernel_impl", "kernel_header", "py_cpp_binding")
SINGLE_FILE_LAYERS: tuple[str, ...] = ("interface_definition",)

# Directory layers whose hits carry a meaningful ORDER, so Layer 3 numbers their
# files (`<n>_<basename>`): kernel_impl = call chain (launcher -> __global__);
# py_cpp_binding = py-side bridge -> cpp-side registration. kernel_header is a
# one-to-one correspondence with impl files, so it is NOT numbered.
ORDERED_DIRECTORY_LAYERS: tuple[str, ...] = ("kernel_impl", "py_cpp_binding")

# Single-file layers' output filename (+ default extension). Directory layers do
# NOT use this — their files are named per-hit from each source basename under a
# <layer>/ subdirectory (see extractor._directory_layer_filename).
LAYER_FILENAME: dict[str, str] = {
    "interface_definition": "interface_definition.py",
}

# Placeholder filenames used only for empty/placeholder directory layers (the
# real files are named per-hit). Keeps not_applicable/missed dirs discoverable.
LAYER_PLACEHOLDER_FILENAME: dict[str, str] = {
    "kernel_impl": "kernel_impl.py",
    "kernel_header": "kernel_header.h",
    "py_cpp_binding": "py_cpp_binding.cc",
}

# Statuses a layer may carry (KID_and_locate §5.5). ``missed`` is added by the
# Layer 2 agent when it also fails; treated like not_found by extract.
STATUS_RESOLVED = "resolved"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NOT_FOUND = "not_found"
STATUS_MISSED = "missed"

# Placeholder sentinel used by dry-run skeletons; extract must refuse to treat a
# layer as resolved while any of its fields still contains this.
FILL_SENTINEL = "<FILL"


@dataclass
class LayerHit:
    file: str
    def_line: int | None = None
    _raw_def: Any = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerHit":
        raw_def = d.get("def_line")
        return cls(
            file=str(d.get("file", "")),
            def_line=_opt_int(raw_def),
            _raw_def=raw_def,
        )

    def is_fillable(self) -> bool:
        """True if this hit still carries an unfilled placeholder."""
        return FILL_SENTINEL in str(self.file) or _has_fill(self._raw_def)



@dataclass
class LayerResult:
    name: str
    status: str
    hits: list[LayerHit] = field(default_factory=list)
    repo_hint: str | None = None

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "LayerResult":
        hits_raw = d.get("hits") or []
        hits = [LayerHit.from_dict(h) for h in hits_raw if isinstance(h, dict)]
        return cls(
            name=name,
            status=str(d.get("status", STATUS_NOT_FOUND)),
            hits=hits,
            repo_hint=d.get("repo_hint"),
        )

    def is_required(self) -> bool:
        return self.name in REQUIRED_LAYERS

    def is_directory_layer(self) -> bool:
        return self.name in DIRECTORY_LAYERS

    def is_effectively_missing(self) -> bool:
        """A layer that yields no usable location for extraction.

        Directory layers (kernel_impl/kernel_header) may carry multiple hits and
        are missing only if *any* hit is unusable; single-file layers need
        exactly one usable hit.
        """
        if self.status == STATUS_NOT_APPLICABLE:
            return False
        if self.status != STATUS_RESOLVED:
            return True
        if not self.hits:
            return True
        return any(h.is_fillable() for h in self.hits)


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and FILL_SENTINEL in value:
        # Preserve the placeholder marker so is_fillable() can detect it.
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_fill(value: Any) -> bool:
    return isinstance(value, str) and FILL_SENTINEL in value
