"""Private contracts for source-locate Agent decisions.

The autonomous Agent records semantic reasoning in a scratch decisions file.
``agent_helper finalize`` validates that richer representation, strips the
reasoning-only fields, and writes the deliberately small downstream schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import (
    ALWAYS_APPLICABLE_LAYERS,
    EXTRACTABLE_STATUSES,
    FINAL_STATUSES,
    LAYERS,
    SINGLE_FILE_LAYERS,
    STATUS_BEST_EFFORT,
    STATUS_MISSED,
    STATUS_NOT_APPLICABLE,
    ContractError,
)

DECISIONS_SCHEMA_VERSION = "source-locate-agent-decisions/v1"

_ROOT_FIELDS = {"schema_version", "kernels"}
_KERNEL_FIELDS = {
    "low_level_id",
    "summary",
    "layers",
    "gaps",
    "manual_followup",
}
_LAYER_FIELDS = {"status", "rationale", "hits"}
_HIT_FIELDS = {"file", "def_line", "symbol", "reason"}
def _non_empty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    raise ContractError(f"{where} has invalid fields ({', '.join(details)})")


def validate_decisions(
    payload: dict[str, Any],
    *,
    expected_kernel_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate the private decisions document and index it by kernel id."""

    _exact_fields(payload, _ROOT_FIELDS, "decisions")
    if payload.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ContractError(
            f"decisions.schema_version must be {DECISIONS_SCHEMA_VERSION!r}"
        )
    kernels = payload.get("kernels")
    if not isinstance(kernels, list):
        raise ContractError("decisions.kernels must be an array")

    by_id: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(kernels):
        where = f"decisions.kernels[{index}]"
        if not isinstance(decision, dict):
            raise ContractError(f"{where} must be an object")
        _exact_fields(decision, _KERNEL_FIELDS, where)
        kernel_id = _non_empty_string(
            decision.get("low_level_id"), f"{where}.low_level_id"
        )
        if kernel_id in by_id:
            raise ContractError(f"duplicate decision low_level_id: {kernel_id}")
        _non_empty_string(decision.get("summary"), f"{where}.summary")

        layers = decision.get("layers")
        if not isinstance(layers, dict) or set(layers) != set(LAYERS):
            raise ContractError(
                f"{where}.layers must contain exactly: {', '.join(LAYERS)}"
            )
        statuses: list[str] = []
        for layer_name in LAYERS:
            status = _validate_decision_layer(
                layers[layer_name], f"{where}.layers.{layer_name}", layer_name
            )
            statuses.append(status)

        gaps = decision.get("gaps")
        if not isinstance(gaps, list):
            raise ContractError(f"{where}.gaps must be an array")
        for gap_index, gap in enumerate(gaps):
            _non_empty_string(gap, f"{where}.gaps[{gap_index}]")
        if any(
            status in {STATUS_BEST_EFFORT, STATUS_MISSED}
            for status in statuses
        ) and not gaps:
            raise ContractError(
                f"{where}.gaps must explain every best_effort or missed result"
            )

        followup = decision.get("manual_followup")
        if followup is not None:
            _non_empty_string(followup, f"{where}.manual_followup")
        if STATUS_MISSED in statuses and followup is None:
            raise ContractError(
                f"{where}.manual_followup is required when a layer is missed"
            )
        by_id[kernel_id] = decision

    expected = set(expected_kernel_ids)
    actual = set(by_id)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ContractError(
            "decisions must contain every KID low_level_id exactly once "
            f"({', '.join(details)})"
        )
    return by_id


def _validate_decision_layer(value: Any, where: str, layer_name: str) -> str:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    _exact_fields(value, _LAYER_FIELDS, where)
    status = value.get("status")
    if status not in FINAL_STATUSES:
        raise ContractError(f"{where}.status is invalid: {status!r}")
    if (
        layer_name in ALWAYS_APPLICABLE_LAYERS
        and status == STATUS_NOT_APPLICABLE
    ):
        raise ContractError(
            f"{where}.status cannot be not_applicable for {layer_name}"
        )
    _non_empty_string(value.get("rationale"), f"{where}.rationale")

    hits = value.get("hits")
    if not isinstance(hits, list):
        raise ContractError(f"{where}.hits must be an array")
    if status in EXTRACTABLE_STATUSES and not hits:
        raise ContractError(f"{where} status={status} requires at least one hit")
    if status in {STATUS_MISSED, STATUS_NOT_APPLICABLE} and hits:
        raise ContractError(f"{where} status={status} requires an empty hits array")
    if layer_name in SINGLE_FILE_LAYERS and len(hits) > 1:
        raise ContractError(f"{where} accepts at most one hit")

    seen_hits: set[tuple[str, int]] = set()
    for index, hit in enumerate(hits):
        hit_where = f"{where}.hits[{index}]"
        if not isinstance(hit, dict):
            raise ContractError(f"{hit_where} must be an object")
        _exact_fields(hit, _HIT_FIELDS, hit_where)
        file = _non_empty_string(hit.get("file"), f"{hit_where}.file")
        if not Path(file).is_absolute():
            raise ContractError(f"{hit_where}.file must be an absolute path")
        def_line = hit.get("def_line")
        if isinstance(def_line, bool) or not isinstance(def_line, int) or def_line < 1:
            raise ContractError(f"{hit_where}.def_line must be a positive integer")
        _non_empty_string(hit.get("symbol"), f"{hit_where}.symbol")
        _non_empty_string(hit.get("reason"), f"{hit_where}.reason")
        key = (file, def_line)
        if key in seen_hits:
            raise ContractError(f"{where} contains duplicate hit: {file}:{def_line}")
        seen_hits.add(key)
    return str(status)


def stripped_layers(decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the minimal downstream layer shape, excluding Agent reasoning."""

    result: dict[str, dict[str, Any]] = {}
    for layer_name in LAYERS:
        layer = decision["layers"][layer_name]
        result[layer_name] = {
            "status": layer["status"],
            "hits": [
                {"file": hit["file"], "def_line": hit["def_line"]}
                for hit in layer["hits"]
            ],
            "repo_hint": None,
        }
    return result
