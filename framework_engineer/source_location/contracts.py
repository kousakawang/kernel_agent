"""Shared contracts for the source-locate tools.

The tools consume only ``kernel-interface-decomposition/v3``.  ``locate`` adds
transient interface candidates; the source-locate Agent replaces those with the
four-layer ``source_locations`` result; ``extract`` consumes that final result.
Profiling metrics and coverage are deliberately opaque to this package.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KID_SCHEMA_VERSION = "kernel-interface-decomposition/v3"

CAPTURE_ARCHETYPES: frozenset[str] = frozenset(
    {
        "pytorch_dispatch",
        "triton_launch",
        "cute_dsl_launch",
        "tilelang_launch",
        "tvm_ffi_call",
        "inductor_launch",
        "python_binding",
    }
)

# Canonical four-layer order used by the Agent contract and extract output.
LAYERS: tuple[str, ...] = (
    "interface_definition",
    "kernel_impl",
    "py_cpp_binding",
    "kernel_header",
)
DIRECTORY_LAYERS: tuple[str, ...] = (
    "kernel_impl",
    "py_cpp_binding",
    "kernel_header",
)
SINGLE_FILE_LAYERS: tuple[str, ...] = ("interface_definition",)
ORDERED_DIRECTORY_LAYERS: tuple[str, ...] = ("kernel_impl", "py_cpp_binding")
# Every semantic low-level target has a Python boundary and an implementation.
# Missing source for either is ``missed``, never ``not_applicable``.
ALWAYS_APPLICABLE_LAYERS: frozenset[str] = frozenset(
    {"interface_definition", "kernel_impl"}
)

LAYER_FILENAME: dict[str, str] = {
    "interface_definition": "interface_definition.py",
}
LAYER_PLACEHOLDER_FILENAME: dict[str, str] = {
    "kernel_impl": "kernel_impl.py",
    "py_cpp_binding": "py_cpp_binding.cc",
    "kernel_header": "kernel_header.h",
}

# locate-only statuses.  These are candidates, not Agent conclusions.
CANDIDATE_RESOLVED = "resolved"
CANDIDATE_AMBIGUOUS = "ambiguous"
CANDIDATE_NOT_FOUND = "not_found"
CANDIDATE_STATUSES: frozenset[str] = frozenset(
    {CANDIDATE_RESOLVED, CANDIDATE_AMBIGUOUS, CANDIDATE_NOT_FOUND}
)

# Final Agent statuses accepted by extract.
STATUS_RESOLVED = "resolved"
STATUS_BEST_EFFORT = "best_effort"
STATUS_MISSED = "missed"
STATUS_NOT_APPLICABLE = "not_applicable"
FINAL_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_RESOLVED,
        STATUS_BEST_EFFORT,
        STATUS_MISSED,
        STATUS_NOT_APPLICABLE,
    }
)
EXTRACTABLE_STATUSES: frozenset[str] = frozenset(
    {STATUS_RESOLVED, STATUS_BEST_EFFORT}
)

_SAFE_LOW_LEVEL_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_OLD_TOP_LEVEL_FIELDS = {"dry_run"}
_OLD_KERNEL_FIELDS = {"archetype_code", "binding_provider"}
_FORBIDDEN_RUNTIME_FIELDS = {"implementation", "source_files", "symbols"}
_TRANSIENT_KERNEL_FIELDS = {
    "locate_candidates",
    "source_locations",
    "kernel_sources_dir",
}


class ContractError(ValueError):
    """The JSON shape is incompatible with the source-locate contract."""


@dataclass(frozen=True)
class LayerHit:
    file: str
    def_line: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LayerHit":
        return cls(file=str(value["file"]), def_line=int(value["def_line"]))


@dataclass
class LayerResult:
    name: str
    status: str
    hits: list[LayerHit] = field(default_factory=list)
    repo_hint: str | None = None

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "LayerResult":
        return cls(
            name=name,
            status=str(value["status"]),
            hits=[LayerHit.from_dict(hit) for hit in value["hits"]],
            repo_hint=value["repo_hint"],
        )

    @property
    def is_extractable(self) -> bool:
        return self.status in EXTRACTABLE_STATUSES


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one UTF-8 JSON object and normalize user-facing errors."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid {label} JSON: {path}: {exc}") from exc
    except UnicodeError as exc:
        raise ContractError(f"{label} is not valid UTF-8: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} root must be a JSON object")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a partially-written schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def kernel_entries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the flat v3 ``kernels`` list or raise a contract error."""

    entries = schema.get("kernels")
    if not isinstance(entries, list) or not entries:
        raise ContractError("schema.kernels must be a non-empty array")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ContractError("every schema.kernels entry must be an object")
    return entries


def validate_kid_schema(
    schema: dict[str, Any],
    *,
    allow_locate_candidates: bool = False,
    allow_source_locations: bool = False,
    allow_kernel_sources_dir: bool = False,
) -> list[dict[str, Any]]:
    """Validate only the KID fields needed by source location.

    Metrics, measurement values, rank ordering, and coverage intentionally are
    not validated here.  KID owns those semantics; this stage only preserves
    them.
    """

    if schema.get("schema_version") != KID_SCHEMA_VERSION:
        raise ContractError(
            f"schema_version must be {KID_SCHEMA_VERSION!r}"
        )
    old_top = sorted(_OLD_TOP_LEVEL_FIELDS.intersection(schema))
    if old_top:
        raise ContractError(f"legacy top-level fields are forbidden: {old_top}")

    entries = kernel_entries(schema)
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"kernels[{index}]"
        old_fields = sorted(_OLD_KERNEL_FIELDS.intersection(entry))
        if old_fields:
            raise ContractError(f"{where} contains legacy fields: {old_fields}")
        if "locate_candidates" in entry and not allow_locate_candidates:
            raise ContractError(f"{where}.locate_candidates is not allowed here")
        if "source_locations" in entry and not allow_source_locations:
            raise ContractError(f"{where}.source_locations is not allowed here")
        if "kernel_sources_dir" in entry and not allow_kernel_sources_dir:
            raise ContractError(f"{where}.kernel_sources_dir is not allowed here")

        low_level_id = entry.get("low_level_id")
        if not isinstance(low_level_id, str) or not _SAFE_LOW_LEVEL_ID.fullmatch(
            low_level_id
        ):
            raise ContractError(
                f"{where}.low_level_id must be a non-empty safe path segment"
            )
        if low_level_id in seen_ids:
            raise ContractError(f"duplicate low_level_id: {low_level_id}")
        seen_ids.add(low_level_id)

        interface = entry.get("interface")
        if not isinstance(interface, str) or not interface.strip():
            raise ContractError(f"{where}.interface must be a non-empty string")
        archetype = entry.get("archetype")
        if archetype not in CAPTURE_ARCHETYPES:
            raise ContractError(f"{where}.archetype is not a supported capture category")
        if "provider" not in entry or not (
            entry["provider"] is None or isinstance(entry["provider"], str)
        ):
            raise ContractError(f"{where}.provider must be a string or null")

        kernel = entry.get("kernel")
        if not isinstance(kernel, dict):
            raise ContractError(f"{where}.kernel must be an object")
        for name in ("raw_name", "normalized_name"):
            if not isinstance(kernel.get(name), str) or not kernel[name]:
                raise ContractError(f"{where}.kernel.{name} must be a string")

        runtime_event = entry.get("runtime_event")
        if not isinstance(runtime_event, dict):
            raise ContractError(f"{where}.runtime_event must be an object")
        forbidden_runtime = sorted(_FORBIDDEN_RUNTIME_FIELDS.intersection(runtime_event))
        if forbidden_runtime:
            raise ContractError(
                f"{where}.runtime_event contains forbidden source fields: "
                f"{forbidden_runtime}"
            )
        call_site = runtime_event.get("call_site")
        if not isinstance(call_site, dict):
            raise ContractError(f"{where}.runtime_event.call_site must be an object")
        if not isinstance(call_site.get("file"), str) or not call_site["file"]:
            raise ContractError(
                f"{where}.runtime_event.call_site.file must be a string"
            )
        line = call_site.get("line")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise ContractError(
                f"{where}.runtime_event.call_site.line must be a positive integer"
            )
    return entries


def validate_agent_schema(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the Agent's final four-layer result before extraction."""

    entries = validate_kid_schema(
        schema,
        allow_source_locations=True,
        allow_kernel_sources_dir=True,
    )
    for index, entry in enumerate(entries):
        where = f"kernels[{index}].source_locations"
        source_locations = entry.get("source_locations")
        if not isinstance(source_locations, dict):
            raise ContractError(f"{where} must be an object")
        if set(source_locations) != {"layers"}:
            raise ContractError(f"{where} must contain only the 'layers' field")
        layers = source_locations.get("layers")
        if not isinstance(layers, dict) or set(layers) != set(LAYERS):
            raise ContractError(
                f"{where}.layers must contain exactly: {', '.join(LAYERS)}"
            )
        for layer_name in LAYERS:
            _validate_layer(layers[layer_name], f"{where}.layers.{layer_name}", layer_name)
    return entries


def _validate_layer(value: Any, where: str, layer_name: str) -> None:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    if set(value) != {"status", "hits", "repo_hint"}:
        raise ContractError(
            f"{where} must contain exactly status, hits, and repo_hint"
        )
    status = value.get("status")
    if status not in FINAL_STATUSES:
        raise ContractError(f"{where}.status is invalid: {status!r}")
    if layer_name in ALWAYS_APPLICABLE_LAYERS and status == STATUS_NOT_APPLICABLE:
        raise ContractError(
            f"{where}.status cannot be not_applicable for {layer_name}"
        )
    hits = value.get("hits")
    if not isinstance(hits, list):
        raise ContractError(f"{where}.hits must be an array")
    if status in EXTRACTABLE_STATUSES and not hits:
        raise ContractError(f"{where} status={status} requires at least one hit")
    if status in {STATUS_MISSED, STATUS_NOT_APPLICABLE} and hits:
        raise ContractError(f"{where} status={status} requires an empty hits array")
    if layer_name in SINGLE_FILE_LAYERS and len(hits) > 1:
        raise ContractError(f"{where} accepts at most one hit")
    repo_hint = value.get("repo_hint")
    if repo_hint is not None and not isinstance(repo_hint, str):
        raise ContractError(f"{where}.repo_hint must be a string or null")
    for index, hit in enumerate(hits):
        hit_where = f"{where}.hits[{index}]"
        if not isinstance(hit, dict) or set(hit) != {"file", "def_line"}:
            raise ContractError(
                f"{hit_where} must contain exactly file and def_line"
            )
        file = hit.get("file")
        if not isinstance(file, str) or not file or not Path(file).is_absolute():
            raise ContractError(f"{hit_where}.file must be an absolute path")
        def_line = hit.get("def_line")
        if isinstance(def_line, bool) or not isinstance(def_line, int) or def_line < 1:
            raise ContractError(f"{hit_where}.def_line must be a positive integer")


def agent_layers(entry: dict[str, Any]) -> dict[str, LayerResult]:
    """Parse already-validated layer dictionaries."""

    raw = entry["source_locations"]["layers"]
    return {name: LayerResult.from_dict(name, raw[name]) for name in LAYERS}


def kid_projection(schema: dict[str, Any]) -> dict[str, Any]:
    """Return KID-owned data with all source-locate additions removed."""

    projected = copy.deepcopy(schema)
    for entry in kernel_entries(projected):
        for field in _TRANSIENT_KERNEL_FIELDS:
            entry.pop(field, None)
    return projected
