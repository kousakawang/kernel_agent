"""Clone missing third-party repos into a (name, version) cache.

Resolution precedence per repo (first hit wins):
    P1 explicit_paths[name]                     -> resolution="explicit"
    P2 embedded git tree in sgl-kernel/3rdparty -> resolution="embedded"
    P3 clone the pinned git source into third_party_cache/<name>/<version>
                                                -> resolution="cloned"

Option A: we do NOT resolve to installed site-packages. A wheel is not equivalent
to the git source tree (different path layout for JIT csrc; tests/benchmarks usually
stripped; same-named package may be unrelated). Since these repos back a lot of
downstream work and disk is cheap vs models, we always clone the pinned git source
for a uniform, complete tree.

Clone-failure policy (see plan): we do NOT retry, switch mirrors, or otherwise try
to fix a failed clone. We record ``status="clone_failed"``, leave ``local_path``
empty, and emit a re-runnable ``clone_command`` (with the version checkout). Solving
the failure is out of scope for this skill.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .version_resolver import RepoResolution


@dataclass
class CloneOutcome:
    status: str  # ok | clone_failed | failed
    resolution: str  # explicit | embedded | cloned | none
    local_path: str | None = None
    clone_command: str | None = None
    error: str | None = None


def _cache_dir(cache_root: Path, name: str, version: str | None) -> Path:
    return cache_root / name / (version or "unknown")


def build_clone_command(
    url: str, ref: str | None, dest: Path, *, https_proxy: str | None = None
) -> str:
    """A single re-runnable command that clones + checks out the pinned ref.

    Shallow-fetches the exact ref so it works for both tags and commit hashes.
    When ``https_proxy`` is set, it is embedded inline on every git invocation so
    the emitted command stays copy-paste re-runnable.
    """

    prefix = f"https_proxy={https_proxy} " if https_proxy else ""
    if ref:
        return (
            f"{prefix}git clone --filter=blob:none {url} {dest} && "
            f"{prefix}git -C {dest} fetch --depth=1 origin {ref} && "
            f"git -C {dest} checkout {ref}"
        )
    return f"{prefix}git clone --depth=1 {url} {dest}"


def _looks_like_source(path: Path) -> bool:
    """Heuristic: a usable source tree has code, not just compiled artifacts."""

    if not path.is_dir():
        return False
    for pattern in ("*.py", "*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"):
        if next(path.rglob(pattern), None) is not None:
            return True
    return False


def _find_embedded(sgl_kernel_src: Path, name: str) -> Path | None:
    candidate = sgl_kernel_src / "3rdparty" / name
    if _looks_like_source(candidate):
        return candidate
    return None


def _clone_steps(url: str, ref: str | None, dest: Path) -> list[list[str]]:
    """The git argv steps to clone + checkout, run without a shell."""

    if ref:
        return [
            ["git", "clone", "--filter=blob:none", url, str(dest)],
            ["git", "-C", str(dest), "fetch", "--depth=1", "origin", ref],
            ["git", "-C", str(dest), "checkout", ref],
        ]
    return [["git", "clone", "--depth=1", url, str(dest)]]


def _run_clone(
    url: str,
    ref: str | None,
    dest: Path,
    timeout: int,
    *,
    https_proxy: str | None = None,
) -> tuple[bool, str | None]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = None
    if https_proxy:
        import os

        env = {**os.environ, "https_proxy": https_proxy, "HTTPS_PROXY": https_proxy}
    for argv in _clone_steps(url, ref, dest):
        try:
            # shell=False (argv list) — no shell interpolation, no injection surface.
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, f"clone timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return False, f"clone raised: {exc}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, tail[-1] if tail else f"git exited {proc.returncode}"
    return True, None


def clone_repo(
    res: RepoResolution,
    *,
    cache_root: Path,
    sgl_kernel_src: Path,
    explicit_paths: dict[str, str],
    dry_run: bool,
    clone_timeout: int = 600,
    https_proxy: str | None = None,
) -> CloneOutcome:
    # No source to clone (F8) or version resolution already failed.
    if not res.has_source:
        return CloneOutcome(status="failed", resolution="none", error=res.resolve_error)
    if res.resolve_error:
        return CloneOutcome(status="failed", resolution="none", error=res.resolve_error)

    # P1 explicit user-provided path (trusted as-is; user opts in).
    explicit = explicit_paths.get(res.name)
    if explicit:
        p = Path(explicit).expanduser()
        if _looks_like_source(p):
            return CloneOutcome(status="ok", resolution="explicit", local_path=str(p))

    # P2 embedded git checkout under sgl-kernel/3rdparty (already a full source tree).
    embedded = _find_embedded(sgl_kernel_src, res.name)
    if embedded is not None:
        return CloneOutcome(status="ok", resolution="embedded", local_path=str(embedded))

    # NOTE (Option A): we intentionally do NOT resolve to installed site-packages.
    # A wheel is not equivalent to the git source tree:
    #   * wheel path layout differs (e.g. flashinfer csrc lives at flashinfer/data/csrc/
    #     in a wheel vs top-level csrc/ in git) — JIT source tracing would look in the
    #     wrong place;
    #   * wheels usually omit tests/ benchmarks/ examples/ — the exact material
    #     problem_translate needs for L4 references;
    #   * a same-named package may be an unrelated lib (cutlass -> nvidia_cutlass_dsl).
    # Since these repos back a lot of downstream work and disk is cheap vs models,
    # we always clone the pinned git source for a uniform, complete tree.

    # P3 clone the pinned git source into the (name, version) cache.
    dest = _cache_dir(cache_root, res.name, res.version)
    command = build_clone_command(res.url or "", res.ref, dest, https_proxy=https_proxy)

    if _looks_like_source(dest):  # cache hit
        return CloneOutcome(status="ok", resolution="cloned", local_path=str(dest))

    if not res.url:
        return CloneOutcome(
            status="failed", resolution="none", error="no clone url in registry"
        )

    if dry_run:
        # Do not touch the network; report what WOULD run.
        return CloneOutcome(
            status="clone_failed",
            resolution="none",
            clone_command=command,
            error="dry-run: clone not attempted",
        )

    # Clean a partial/broken dir before attempting.
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    ok, err = _run_clone(res.url, res.ref, dest, clone_timeout, https_proxy=https_proxy)
    if ok:
        return CloneOutcome(status="ok", resolution="cloned", local_path=str(dest))
    # Failure: record only, do not retry or fix.
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    return CloneOutcome(
        status="clone_failed",
        resolution="none",
        clone_command=command,
        error=err,
    )
