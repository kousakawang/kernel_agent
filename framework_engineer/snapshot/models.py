"""Data models for Phase 1 snapshot replay.

The models intentionally stay close to JSON-compatible dictionaries because
snapshot metadata must be copied into standalone task packs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "phase1.snapshot.v1"


@dataclass
class SnapshotTensorMeta:
    path: str
    dtype: str
    shape: list[int]
    stride: list[int]
    device_type: str
    device_index: int | None
    layout: str
    is_contiguous: bool
    requires_grad: bool
    numel: int
    storage_offset: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotTensorMeta":
        return cls(**data)


@dataclass
class SnapshotCase:
    task_id: str
    case_id: str
    target: dict[str, Any]
    interface: dict[str, Any]
    files: dict[str, str]
    mutation: dict[str, Any]
    hashes: dict[str, str]
    selection: dict[str, Any]
    tolerance: dict[str, float]
    raw_call_ids: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "case_id": self.case_id,
            "raw_call_ids": self.raw_call_ids,
            "target": self.target,
            "interface": self.interface,
            "files": self.files,
            "mutation": self.mutation,
            "hashes": self.hashes,
            "selection": self.selection,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotCase":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            task_id=data["task_id"],
            case_id=data["case_id"],
            raw_call_ids=list(data.get("raw_call_ids", [])),
            target=dict(data.get("target", {})),
            interface=dict(data.get("interface", {})),
            files=dict(data.get("files", {})),
            mutation=dict(data.get("mutation", {})),
            hashes=dict(data.get("hashes", {})),
            selection=dict(data.get("selection", {})),
            tolerance=dict(data.get("tolerance", {})),
        )


@dataclass
class SnapshotSample:
    task_id: str
    group_id: str
    sample_id: str
    target: dict[str, Any]
    interface: dict[str, Any]
    files: dict[str, str]
    mutation: dict[str, Any]
    hashes: dict[str, str]
    capture: dict[str, Any]
    tolerance: dict[str, float]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "sample_id": self.sample_id,
            "target": self.target,
            "interface": self.interface,
            "files": self.files,
            "mutation": self.mutation,
            "hashes": self.hashes,
            "capture": self.capture,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotSample":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            task_id=data["task_id"],
            group_id=data["group_id"],
            sample_id=data["sample_id"],
            target=dict(data.get("target", {})),
            interface=dict(data.get("interface", {})),
            files=dict(data.get("files", {})),
            mutation=dict(data.get("mutation", {})),
            hashes=dict(data.get("hashes", {})),
            capture=dict(data.get("capture", {})),
            tolerance=dict(data.get("tolerance", {})),
        )
