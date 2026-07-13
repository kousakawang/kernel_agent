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

# Per-layer output file stem + default extension (b's extension is refined from
# the resolved source file's suffix at extraction time).
LAYER_FILENAME: dict[str, str] = {
    "interface_definition": "interface_definition.py",
    "kernel_impl": "kernel_impl.py",
    "py_cpp_binding": "py_cpp_binding.cc",
    "kernel_header": "kernel_header.h",
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
    line_start: int | None = None
    line_end: int | None = None
    _raw_start: Any = None
    _raw_end: Any = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerHit":
        raw_start = d.get("line_start")
        raw_end = d.get("line_end")
        return cls(
            file=str(d.get("file", "")),
            line_start=_opt_int(raw_start),
            line_end=_opt_int(raw_end),
            _raw_start=raw_start,
            _raw_end=raw_end,
        )

    def is_fillable(self) -> bool:
        """True if this hit still carries an unfilled placeholder."""
        return (
            FILL_SENTINEL in str(self.file)
            or _has_fill(self._raw_start)
            or _has_fill(self._raw_end)
        )



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

    def is_effectively_missing(self) -> bool:
        """A layer that yields no usable location for extraction."""
        if self.status == STATUS_NOT_APPLICABLE:
            return False
        if self.status != STATUS_RESOLVED:
            return True
        if not self.hits:
            return True
        return self.hits[0].is_fillable()


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
