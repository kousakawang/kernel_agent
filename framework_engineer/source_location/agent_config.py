"""Configuration contract for the Prompt-driven source-locate workflow.

The user supplies one small config file.  Output paths are derived from its
workspace so the entry Prompt can orchestrate locate, Agent decisions/finalize,
and extract without asking the user to assemble individual CLI commands.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, load_json_object


AGENT_CONFIG_SCHEMA_VERSION = "source-locate-agent-config/v1"

_CONFIG_FIELDS = {
    "schema_version",
    "testcase_id",
    "kid_schema",
    "third_party_manifest",
    "sglang_repo_root",
    "workspace",
}
_SAFE_TESTCASE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _exact_fields(value: dict[str, Any]) -> None:
    actual = set(value)
    if actual == _CONFIG_FIELDS:
        return
    missing = sorted(_CONFIG_FIELDS - actual)
    extra = sorted(actual - _CONFIG_FIELDS)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    raise ContractError(
        "source-locate Agent config has invalid fields "
        f"({', '.join(details)})"
    )


def _path(value: Any, *, field: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Agent config {field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return _absolute(path)


@dataclass(frozen=True)
class AgentRunPaths:
    """Canonical per-testcase workspace layout."""

    workspace: Path
    locate_dir: Path
    candidate_schema: Path
    agent_dir: Path
    decisions: Path
    located_schema: Path
    notes: Path
    extract_dir: Path
    extracted_schema: Path
    kernel_sources: Path

    @classmethod
    def from_workspace(cls, workspace: Path) -> "AgentRunPaths":
        workspace = _absolute(workspace)
        locate_dir = workspace / "locate"
        agent_dir = workspace / "agent"
        extract_dir = workspace / "extract"
        return cls(
            workspace=workspace,
            locate_dir=locate_dir,
            candidate_schema=locate_dir / "locate_candidates.schema.json",
            agent_dir=agent_dir,
            decisions=agent_dir / "source_locate_decisions.json",
            located_schema=agent_dir / "located.schema.json",
            notes=agent_dir / "ref" / "locate_agent_notes.md",
            extract_dir=extract_dir,
            extracted_schema=extract_dir / "decomposition.extracted.schema.json",
            kernel_sources=extract_dir / "kernel_sources",
        )

    def create_directories(self) -> None:
        self.locate_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.notes.parent.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "workspace": str(self.workspace),
            "candidate_schema": str(self.candidate_schema),
            "decisions": str(self.decisions),
            "located_schema": str(self.located_schema),
            "notes": str(self.notes),
            "extracted_schema": str(self.extracted_schema),
            "kernel_sources": str(self.kernel_sources),
        }


@dataclass(frozen=True)
class SourceLocateAgentConfig:
    path: Path
    testcase_id: str
    kid_schema: Path
    third_party_manifest: Path
    sglang_repo_root: Path
    run: AgentRunPaths

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT_CONFIG_SCHEMA_VERSION,
            "config": str(self.path),
            "testcase_id": self.testcase_id,
            "kid_schema": str(self.kid_schema),
            "third_party_manifest": str(self.third_party_manifest),
            "sglang_repo_root": str(self.sglang_repo_root),
            "artifacts": self.run.to_dict(),
        }


def load_agent_config(path: Path) -> SourceLocateAgentConfig:
    """Load the exact v1 config and resolve relative paths beside the config."""

    path = _absolute(path)
    payload = load_json_object(path, label="source-locate Agent config")
    _exact_fields(payload)
    if payload.get("schema_version") != AGENT_CONFIG_SCHEMA_VERSION:
        raise ContractError(
            "Agent config schema_version must be "
            f"{AGENT_CONFIG_SCHEMA_VERSION!r}"
        )
    testcase_id = payload.get("testcase_id")
    if (
        not isinstance(testcase_id, str)
        or not testcase_id
        or not _SAFE_TESTCASE_ID.fullmatch(testcase_id)
    ):
        raise ContractError(
            "Agent config testcase_id must be a non-empty safe path segment"
        )

    base = path.parent
    kid_schema = _path(payload.get("kid_schema"), field="kid_schema", base=base)
    manifest = _path(
        payload.get("third_party_manifest"),
        field="third_party_manifest",
        base=base,
    )
    sglang = _path(
        payload.get("sglang_repo_root"), field="sglang_repo_root", base=base
    )
    workspace = _path(payload.get("workspace"), field="workspace", base=base)

    for field, candidate in (
        ("kid_schema", kid_schema),
        ("third_party_manifest", manifest),
    ):
        if not candidate.is_file():
            raise ContractError(f"Agent config {field} not found: {candidate}")
    if not sglang.is_dir():
        raise ContractError(f"Agent config sglang_repo_root not found: {sglang}")
    if workspace.exists() and not workspace.is_dir():
        raise ContractError(f"Agent config workspace is not a directory: {workspace}")

    protected = {path, kid_schema, manifest}
    run = AgentRunPaths.from_workspace(workspace)
    for label, output in run.to_dict().items():
        if Path(output) in protected:
            raise ContractError(
                f"Agent config derived {label} must differ from config inputs: {output}"
            )

    return SourceLocateAgentConfig(
        path=path,
        testcase_id=testcase_id,
        kid_schema=kid_schema,
        third_party_manifest=manifest,
        sglang_repo_root=sglang,
        run=run,
    )
