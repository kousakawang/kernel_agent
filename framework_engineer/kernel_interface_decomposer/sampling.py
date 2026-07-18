from __future__ import annotations

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


SAMPLERS: dict[str, Sampler] = {
    "all": _all,
    "last_n": _last_n,
    "single": _last_n,
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
    strategy = str(selection.get("sampling", "last_n"))
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
    for item in selected:
        item.pop("_nvtx_start_ns", None)
    return selected, diagnostics
