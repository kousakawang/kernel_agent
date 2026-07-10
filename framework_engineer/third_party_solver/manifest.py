"""Build and write third_party_manifest.json + missing_repos.md.

Merges each repo's version resolution (version_resolver) with its clone outcome
(cloner) into a manifest record with three-state ``status``:

    ok           -> local_path filled with the on-disk source path
    clone_failed -> local_path null, clone_command provided (re-runnable)
    failed       -> no source / could not locate repo; listed in missing_repos.md
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cloner import CloneOutcome
from .version_resolver import RepoResolution


@dataclass
class RepoRecord:
    name: str
    archetype: str
    version: str | None
    version_source: str
    clone_source: str
    resolution: str
    local_path: str | None
    url: str | None
    ref: str | None
    triggered_by: list[str]
    on_default_path: bool
    version_mismatch: bool
    status: str
    clone_command: str | None = None
    error: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    schema_version: int
    generated_at: str
    sglang_repo_root: str
    sgl_kernel_installed_version: str | None
    sgl_kernel_source_version: str | None
    sgl_kernel_version_mismatch: bool
    third_party_cache: str
    repos: list[RepoRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "sglang_repo_root": self.sglang_repo_root,
            "sgl_kernel_installed_version": self.sgl_kernel_installed_version,
            "sgl_kernel_source_version": self.sgl_kernel_source_version,
            "sgl_kernel_version_mismatch": self.sgl_kernel_version_mismatch,
            "third_party_cache": self.third_party_cache,
            "repos": [r.to_dict() for r in self.repos],
            "failed": [
                {"name": r.name, "reason": r.error, "clone_command": r.clone_command}
                for r in self.repos
                if r.status in ("failed", "clone_failed")
            ],
        }
        return data


def _default_evidence(res: RepoResolution, outcome: CloneOutcome) -> str:
    parts: list[str] = []
    if res.version_source == "importlib":
        parts.append(f"importlib version {res.version}")
    elif res.version_source == "cmake_pin":
        parts.append(f"sgl-kernel CMake pin {res.version}")
    if outcome.resolution and outcome.resolution != "none":
        parts.append(f"resolution={outcome.resolution}")
    if res.on_default_path:
        parts.append("default-path library")
    if res.version_mismatch:
        parts.append("sgl-kernel version mismatch")
    return "; ".join(parts) if parts else ""


def build_record(res: RepoResolution, outcome: CloneOutcome) -> RepoRecord:
    return RepoRecord(
        name=res.name,
        archetype=res.archetype,
        version=res.version,
        version_source=res.version_source,
        clone_source=res.clone_source,
        resolution=outcome.resolution,
        local_path=outcome.local_path,
        url=res.url,
        ref=res.ref,
        triggered_by=res.triggered_by,
        on_default_path=res.on_default_path,
        version_mismatch=res.version_mismatch,
        status=outcome.status,
        clone_command=outcome.clone_command,
        error=outcome.error or res.resolve_error,
        evidence=_default_evidence(res, outcome),
    )


def build_manifest(
    *,
    sglang_repo_root: Path,
    third_party_cache: Path,
    meta: dict[str, Any],
    records: list[RepoRecord],
) -> Manifest:
    return Manifest(
        schema_version=1,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sglang_repo_root=str(sglang_repo_root),
        sgl_kernel_installed_version=meta.get("sgl_kernel_installed_version"),
        sgl_kernel_source_version=meta.get("sgl_kernel_source_version"),
        sgl_kernel_version_mismatch=bool(meta.get("sgl_kernel_version_mismatch")),
        third_party_cache=str(third_party_cache),
        repos=records,
    )


def write_manifest(manifest: Manifest, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")


def render_missing_repos(manifest: Manifest) -> str | None:
    """Render missing_repos.md; return None when there is nothing to report."""

    problem = [r for r in manifest.repos if r.status in ("failed", "clone_failed")]
    if not problem:
        return None

    lines = ["# Missing / unresolved third-party repos", ""]
    lines.append(
        "These repos could not be made available locally. Resolving them is out of "
        "scope for `resolve-third-party`; fix manually and re-run, or run the "
        "provided clone command.\n"
    )
    for r in problem:
        lines.append(f"## {r.name}  ({r.status}, {r.archetype})")
        lines.append(f"- version: `{r.version}`")
        if r.url:
            lines.append(f"- repo: {r.url}  (ref `{r.ref}`)")
        if r.error:
            lines.append(f"- reason: {r.error}")
        if r.clone_command:
            lines.append("- suggested clone command:")
            lines.append(f"  ```bash\n  {r.clone_command}\n  ```")
        else:
            lines.append("- suggested path: (no source available; manual investigation)")
        lines.append("")
    return "\n".join(lines)


def write_missing_repos(manifest: Manifest, path: Path) -> bool:
    """Write missing_repos.md if needed. Returns True when a file was written."""

    text = render_missing_repos(manifest)
    if text is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True
