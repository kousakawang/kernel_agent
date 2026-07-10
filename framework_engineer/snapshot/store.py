"""Filesystem storage for Phase 1 snapshots."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, SnapshotCase, SnapshotSample


def _torch():
    try:
        import torch
    except Exception:  # pragma: no cover - torch may be absent on local dev hosts.
        return None
    return torch


class SnapshotStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.selected_dir = self.root / "selected"
        self.manifest_path = self.root / "manifest.json"
        self.raw_index_path = self.root / "raw_index.json"

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.selected_dir.mkdir(parents=True, exist_ok=True)

    def raw_case_dir(self, call_id: str) -> Path:
        return self.raw_dir / call_id

    def selected_case_dir(self, case_id: str) -> Path:
        return self.selected_dir / case_id

    def raw_group_dir(self, group_id: str) -> Path:
        return self.raw_dir / group_id

    def raw_sample_dir(self, group_id: str, sample_id: str) -> Path:
        return self.raw_group_dir(group_id) / sample_id

    def selected_group_dir(self, group_id: str) -> Path:
        return self.selected_dir / group_id

    def selected_sample_dir(self, group_id: str, sample_id: str) -> Path:
        return self.selected_group_dir(group_id) / "samples" / sample_id

    def write_case_meta(self, case_dir: Path, case: SnapshotCase) -> None:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "meta.json").write_text(json.dumps(case.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def write_sample_meta(self, sample_dir: Path, sample: SnapshotSample) -> None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "meta.json").write_text(json.dumps(sample.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def read_sample_meta(self, sample_dir: Path) -> SnapshotSample:
        data = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
        return SnapshotSample.from_dict(data)

    def save_payload(self, value: Any, path: Path) -> str:
        """Save payloads with torch when available, pickle otherwise.

        Real SGLang captures will use torch.save. The pickle fallback keeps local
        non-PyTorch tests useful for primitive-only toy functions.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch = _torch()
        if torch is not None:
            torch.save(value, path)
            return "torch"
        with path.open("wb") as f:
            pickle.dump(value, f)
        return "pickle"

    def load_payload(self, path: Path) -> Any:
        torch = _torch()
        if torch is not None:
            try:
                return torch.load(path, map_location="cpu")
            except Exception:
                pass
        with path.open("rb") as f:
            return pickle.load(f)

    def read_case_meta(self, case_dir: Path) -> SnapshotCase:
        data = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        return SnapshotCase.from_dict(data)

    def list_raw_cases(self) -> list[SnapshotCase]:
        return self._list_cases(self.raw_dir)

    def list_selected_cases(self) -> list[SnapshotCase]:
        return self._list_cases(self.selected_dir)

    def _list_cases(self, root: Path) -> list[SnapshotCase]:
        if not root.exists():
            return []
        cases = []
        for meta in sorted(root.glob("*/meta.json")):
            cases.append(SnapshotCase.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
        return cases

    def copy_raw_to_selected(self, raw_call_id: str, selected_case_id: str, updated_case: SnapshotCase) -> Path:
        src = self.raw_case_dir(raw_call_id)
        dst = self.selected_case_dir(selected_case_id)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        self.write_case_meta(dst, updated_case)
        return dst

    def copy_raw_sample_to_selected(
        self,
        group_id: str,
        sample_id: str,
        selected_group_id: str,
        selected_sample_id: str,
        updated_sample: SnapshotSample,
    ) -> Path:
        src = self.raw_sample_dir(group_id, sample_id)
        dst = self.selected_sample_dir(selected_group_id, selected_sample_id)
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        self.write_sample_meta(dst, updated_sample)
        return dst

    def read_raw_index(self) -> dict[str, Any]:
        if not self.raw_index_path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "index_type": "raw_group_index",
                "raw_group_count": 0,
                "raw_sample_count": 0,
                "total_hit_count": 0,
                "groups": {},
            }
        return json.loads(self.raw_index_path.read_text(encoding="utf-8"))

    def write_raw_index(self, index: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        groups = index.get("groups", {})
        index["raw_group_count"] = len(groups)
        index["raw_sample_count"] = sum(int(group.get("sample_count", 0)) for group in groups.values())
        index["total_hit_count"] = sum(int(group.get("total_hit_count", 0)) for group in groups.values())
        self.raw_index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))
