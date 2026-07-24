"""Select representative snapshot groups and samples from raw captures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..task_pack_layout import task_dir
from .models import SCHEMA_VERSION, SnapshotSample
from .store import SnapshotStore


class SnapshotSelector:
    def __init__(self, store: SnapshotStore):
        self.store = store

    def select(
        self,
        *,
        max_groups: int | None = None,
        max_samples_per_group: int = 8,
    ) -> dict[str, Any]:
        raw_index = self.store.read_raw_index()
        raw_groups = list(raw_index.get("groups", {}).values())
        ranked = sorted(
            raw_groups,
            key=lambda group: (-int(group.get("total_hit_count", 0)), group.get("group_id", "")),
        )
        if max_groups is not None:
            ranked = ranked[:max_groups]

        if self.store.selected_dir.exists():
            shutil.rmtree(self.store.selected_dir)
        self.store.selected_dir.mkdir(parents=True, exist_ok=True)

        selected_groups = []
        selected_sample_count = 0
        for idx, group in enumerate(ranked, start=1):
            group_id = group["group_id"]
            selected_samples = []
            for sample_ref in group.get("samples", [])[:max_samples_per_group]:
                sample_id = sample_ref["sample_id"]
                raw_meta = self.store.read_sample_meta(self.store.raw_sample_dir(group_id, sample_id))
                selected_sample = SnapshotSample(
                    task_id=raw_meta.task_id,
                    group_id=group_id,
                    sample_id=sample_id,
                    target=raw_meta.target,
                    interface=raw_meta.interface,
                    files=raw_meta.files,
                    mutation=raw_meta.mutation,
                    hashes=raw_meta.hashes,
                    capture=raw_meta.capture,
                    tolerance=raw_meta.tolerance,
                    schema_version=raw_meta.schema_version,
                )
                self.store.copy_raw_sample_to_selected(group_id, sample_id, group_id, sample_id, selected_sample)
                selected_samples.append(
                    {
                        "sample_id": sample_id,
                        "sample_dir": f"snapshots/selected/{group_id}/samples/{sample_id}",
                        "hashes": selected_sample.hashes,
                        "capture": selected_sample.capture,
                        "files": selected_sample.files,
                    }
                )
                selected_sample_count += 1

            selected_group = {
                "schema_version": SCHEMA_VERSION,
                "task_id": group.get("task_id"),
                "group_id": group_id,
                "group_key": group.get("group_key"),
                "shape_hash": group.get("shape_hash"),
                "target": group.get("target", {}),
                "interface": group.get("interface", {}),
                "mutation": group.get("mutation", {}),
                "tolerance": group.get("tolerance", {}),
                "original_call_timing_summary": group.get("original_call_timing_summary", {}),
                "selection": {
                    "priority": "required",
                    "reason": "top_frequency" if idx == 1 else "high_frequency_group",
                    "rank": idx,
                    "total_hit_count": int(group.get("total_hit_count", 0)),
                    "forward_hit_count": int(group.get("forward_hit_count", 0)),
                    "raw_sample_count": int(group.get("sample_count", 0)),
                    "selected_sample_count": len(selected_samples),
                },
                "samples": selected_samples,
            }
            group_dir = self.store.selected_group_dir(group_id)
            group_dir.mkdir(parents=True, exist_ok=True)
            (group_dir / "group_meta.json").write_text(
                json.dumps(selected_group, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            selected_groups.append(selected_group)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "selection_policy": "rank_groups_by_total_hit_count_keep_bounded_samples",
            "raw_group_count": int(raw_index.get("raw_group_count", len(raw_groups))),
            "raw_sample_count": int(raw_index.get("raw_sample_count", 0)),
            "raw_total_hit_count": int(raw_index.get("total_hit_count", 0)),
            "selected_group_count": len(selected_groups),
            "selected_sample_count": selected_sample_count,
            "case_groups": selected_groups,
        }
        self.store.write_manifest(manifest)
        return manifest


def write_shape_list_summary(task_pack: Path, manifest: dict[str, Any]) -> None:
    shape_groups = []
    for group in manifest.get("case_groups", []):
        tensors = []
        args_tree = group.get("interface", {}).get("args_tree")
        kwargs_tree = group.get("interface", {}).get("kwargs_tree")
        for tree_name, tree in (("args", args_tree), ("kwargs", kwargs_tree)):
            tensors.extend(_collect_tensors(tree, tree_name))
        shape_groups.append(
            {
                "group_id": group["group_id"],
                "priority": group.get("selection", {}).get("priority", "required"),
                "total_hit_count": group.get("selection", {}).get("total_hit_count", 0),
                "forward_hit_count": group.get("selection", {}).get("forward_hit_count", 0),
                "selected_sample_count": group.get("selection", {}).get("selected_sample_count", 0),
                "original_call_timing_summary": group.get("original_call_timing_summary", {}),
                "group_key": group.get("group_key"),
                "shape_hash": group.get("shape_hash"),
                "snapshot_dir": f"snapshots/selected/{group['group_id']}",
                "tensors": tensors,
            }
        )
    payload = {
        "schema_version": "phase1.shape_group_summary.v1",
        "source": "snapshots/manifest.json",
        "note": "Selected snapshot samples are the replay source; this file is only a group index/summary.",
        "shape_groups": shape_groups,
        "shape_cases": shape_groups,
    }
    (task_dir(task_pack) / "shape_list.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _collect_tensors(tree: Any, prefix: str) -> list[dict[str, Any]]:
    if not tree:
        return []
    kind = tree.get("kind")
    if kind == "tensor":
        meta = dict(tree["meta"])
        meta["path"] = prefix if not meta.get("path") else meta["path"]
        return [meta]
    if kind in ("tuple", "list"):
        out = []
        for idx, item in enumerate(tree.get("items", [])):
            out.extend(_collect_tensors(item, f"{prefix}.{idx}"))
        return out
    if kind == "dict":
        out = []
        for key, item in tree.get("items", {}).items():
            out.extend(_collect_tensors(item, f"{prefix}.{key}"))
        return out
    return []
