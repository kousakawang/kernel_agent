from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any


Invocation = dict[str, Any]
Sampler = Callable[[Sequence[Invocation], int], list[Invocation]]


def _all(invocations: Sequence[Invocation], count: int) -> list[Invocation]:
    del count
    return list(invocations)


def _last_n(invocations: Sequence[Invocation], count: int) -> list[Invocation]:
    by_stage: dict[str, list[Invocation]] = defaultdict(list)
    for invocation in invocations:
        stage = str(invocation.get("high_level", {}).get("stage", "unknown"))
        by_stage[stage].append(invocation)
    selected_ids = {
        id(invocation)
        for stage_invocations in by_stage.values()
        for invocation in stage_invocations[-count:]
    }
    return [invocation for invocation in invocations if id(invocation) in selected_ids]


def _capture_depth(
    capture: dict[str, Any], captures_by_id: dict[str, dict[str, Any]]
) -> int:
    depth = 0
    current = capture
    seen = {str(capture.get("capture_id"))}
    while current.get("parent_capture_id") is not None:
        parent_id = str(current["parent_capture_id"])
        if parent_id in seen:
            raise ValueError(f"capture parent cycle detected at {parent_id}")
        parent = captures_by_id.get(parent_id)
        if parent is None:
            break
        seen.add(parent_id)
        current = parent
        depth += 1
    return depth


def decomposition_signature(invocation: Invocation) -> dict[str, Any]:
    """Return the stable execution decomposition used to group invocations.

    Runtime identities, timings, kernel names/counts, provider hints, and stage
    are deliberately absent. Repeated owner captures with the same execution
    boundary and Python call path collapse to one set member.
    """

    captures = list(invocation.get("execution_captures") or [])
    captures_by_id = {
        str(capture.get("capture_id")): capture for capture in captures
    }
    owner_rows: dict[str, dict[str, Any]] = {}
    for capture in captures:
        if not capture.get("kernel_ids"):
            continue
        stack = []
        for frame in capture.get("python_stack") or []:
            edge = frame.get("call_site_to_next") or {}
            stack.append(
                {
                    "file": frame.get("file"),
                    "definition_line": frame.get("definition_line"),
                    "qualname": frame.get("qualname") or frame.get("function"),
                    "call_site_to_next": {
                        "file": edge.get("file"),
                        "line": edge.get("line"),
                    },
                }
            )
        row = {
            "archetype": capture.get("archetype"),
            "common_interface": capture.get("common_interface"),
            "execution_interface": capture.get("execution_interface"),
            "capture_depth": _capture_depth(capture, captures_by_id),
            "python_call_path": stack,
        }
        canonical_row = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        owner_rows[canonical_row] = row

    return {
        "kernel_owner_captures": [owner_rows[key] for key in sorted(owner_rows)],
        "unattributed_kernel_count": len(
            invocation.get("unattributed_kernel_ids") or []
        ),
    }


def decomposition_signature_hash(invocation: Invocation) -> str:
    canonical = json.dumps(
        decomposition_signature(invocation),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique_decomposition(
    invocations: Sequence[Invocation], count: int
) -> list[Invocation]:
    del count
    representatives: dict[str, Invocation] = {}
    for invocation in invocations:
        canonical = json.dumps(
            decomposition_signature(invocation),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        # Input is chronological, so assignment retains the last representative.
        representatives[canonical] = invocation
    selected_ids = {id(invocation) for invocation in representatives.values()}
    return [invocation for invocation in invocations if id(invocation) in selected_ids]


SAMPLERS: dict[str, Sampler] = {
    "all": _all,
    "last_n": _last_n,
    "single": _last_n,
    "unique_decomposition": _unique_decomposition,
}


def register_sampler(name: str, sampler: Sampler) -> None:
    """Register an in-process sampling policy without allowing config code execution."""

    if not name or name in SAMPLERS:
        raise ValueError(f"sampling strategy already registered or invalid: {name!r}")
    SAMPLERS[name] = sampler


def select_invocations(
    invocations: Sequence[Invocation], selection: dict[str, Any]
) -> tuple[list[Invocation], dict[str, Any]]:
    chronological = sorted(
        invocations,
        key=lambda item: int(item.get("_nvtx_start_ns", 0)),
    )
    skip = int(selection.get("skip_invocations", 0))
    after_skip = chronological[skip:]
    allowed_stages = list(selection.get("stages") or [])
    eligible = [
        item
        for item in after_skip
        if not allowed_stages
        or str(item.get("high_level", {}).get("stage", "unknown")) in allowed_stages
    ]
    strategy = str(selection.get("sampling", "unique_decomposition"))
    try:
        sampler = SAMPLERS[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown sampling strategy: {strategy}") from exc
    count = int(selection.get("sample_count_per_stage", 1))
    selected = sampler(eligible, count)
    selected.sort(key=lambda item: int(item.get("_nvtx_start_ns", 0)))

    def call_ids(items: Sequence[Invocation]) -> list[str]:
        return [str(item.get("high_level", {}).get("call_id")) for item in items]

    selected_identity = {id(item) for item in selected}
    diagnostics = {
        "observed_invocation_count": len(chronological),
        "skipped_invocation_count": min(skip, len(chronological)),
        "eligible_invocation_count": len(eligible),
        "selected_invocation_count": len(selected),
        "discarded_invocation_count": len(eligible) - len(selected),
        "observed_call_ids": call_ids(chronological),
        "eligible_call_ids": call_ids(eligible),
        "selected_call_ids": call_ids(selected),
        "discarded_call_ids": call_ids(
            [item for item in eligible if id(item) not in selected_identity]
        ),
    }
    if strategy == "unique_decomposition":
        groups: dict[str, dict[str, Any]] = {}
        for item in eligible:
            signature = decomposition_signature(item)
            canonical = json.dumps(
                signature,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            group = groups.setdefault(
                canonical,
                {
                    "signature_hash": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest(),
                    "member_call_ids": [],
                    "observed_stages": [],
                    "representative_call_id": None,
                    "_representative_start_ns": 0,
                },
            )
            call_id = str(item.get("high_level", {}).get("call_id"))
            stage = str(item.get("high_level", {}).get("stage", "unknown"))
            group["member_call_ids"].append(call_id)
            if stage not in group["observed_stages"]:
                group["observed_stages"].append(stage)
            group["representative_call_id"] = call_id
            group["_representative_start_ns"] = int(item.get("_nvtx_start_ns", 0))
        ordered_groups = sorted(
            groups.values(), key=lambda group: group["_representative_start_ns"]
        )
        for group in ordered_groups:
            group.pop("_representative_start_ns", None)
        diagnostics["unique_decomposition_count"] = len(ordered_groups)
        diagnostics["decomposition_groups"] = ordered_groups
    for item in selected:
        item.pop("_nvtx_start_ns", None)
    return selected, diagnostics
