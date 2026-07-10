"""Resolve the correct version for each universe repo (strict two-bucket).

Bucket A (``version_source == "importlib"``)
    Independent pip package. Version = ``importlib.metadata.version(dist_name)``,
    read from the *installed* environment (authoritative). clone ref = ``v{version}``.

Bucket B (``version_source == "cmake_pin"``)
    Compiled into sgl_kernel's .so; no independent version. Version = the commit
    pinned by sgl-kernel's FetchContent (cmake_pins.py). We first verify the
    sgl-kernel *source tree* version matches the *installed* sgl_kernel version;
    on mismatch we only REPORT (``version_mismatch=True``) and never checkout.

This module performs no network or git work; it only decides
``(name, version, url, ref, clone_source, status hints)`` for the cloner.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import cmake_pins
from .registry import RepoSpec, iter_universe


@dataclass
class RepoResolution:
    name: str
    archetype: str
    version_source: str
    on_default_path: bool
    url_kind: str
    import_name: str | None = None
    version: str | None = None
    url: str | None = None
    ref: str | None = None
    clone_source: str = "official"  # official | sgl_fork | embedded | installed | explicit
    triggered_by: list[str] = field(default_factory=list)
    version_mismatch: bool = False
    has_source: bool = True
    resolve_error: str | None = None
    """Set when version resolution itself failed (before any clone attempt)."""
    notes: list[str] = field(default_factory=list)


VersionLookup = Callable[[str], str]


def _default_version_lookup(dist_name: str) -> str:
    return importlib_metadata.version(dist_name)


def _short(commit: str) -> str:
    # Keep tags (v3.6.0) intact; shorten long hex commit hashes for the cache dir.
    if len(commit) >= 12 and all(c in "0123456789abcdef" for c in commit.lower()):
        return commit[:12]
    return commit


def resolve_repo(
    spec: RepoSpec,
    *,
    sgl_kernel_pins: dict[str, cmake_pins.CMakePin],
    version_lookup: VersionLookup,
    triggered_by: list[str],
    sgl_kernel_version_mismatch: bool,
) -> RepoResolution:
    res = RepoResolution(
        name=spec.name,
        archetype=spec.archetype,
        version_source=spec.version_source,
        on_default_path=spec.on_default_path,
        url_kind=spec.url_kind,
        import_name=spec.import_name,
        url=spec.url,
        triggered_by=triggered_by,
        has_source=spec.has_source,
    )
    res.clone_source = "sgl_fork" if spec.url_kind == "sgl_fork" else "official"

    if not spec.has_source:
        # F8: downloaded cubin. Still record the runtime version for diagnostics.
        if spec.dist_name:
            try:
                res.version = version_lookup(spec.dist_name)
            except Exception:  # noqa: BLE001
                res.version = None
        res.resolve_error = "no source (downloaded cubin / F8)"
        return res

    if spec.version_source == "importlib":
        if not spec.dist_name:
            res.resolve_error = "importlib source but no dist_name in registry"
            return res
        try:
            res.version = version_lookup(spec.dist_name)
        except importlib_metadata.PackageNotFoundError:
            res.resolve_error = f"package not installed: {spec.dist_name}"
            return res
        except Exception as exc:  # noqa: BLE001
            res.resolve_error = f"version lookup failed: {exc}"
            return res
        res.ref = spec.ref_template.format(version=res.version) if spec.ref_template else None
        return res

    if spec.version_source == "cmake_pin":
        pin = sgl_kernel_pins.get(spec.cmake_target or "")
        if pin is None:
            res.resolve_error = (
                f"cmake target not found in sgl-kernel: {spec.cmake_target}"
            )
            return res
        res.version = _short(pin.commit)
        res.ref = pin.commit
        # Prefer the URL owner/repo parsed from the pin (authoritative fork info),
        # falling back to the registry URL.
        if pin.url:
            res.url = pin.url
            res.clone_source = (
                "sgl_fork" if pin.owner and "sgl" in pin.owner.lower() else res.clone_source
            )
        res.version_mismatch = sgl_kernel_version_mismatch
        if sgl_kernel_version_mismatch:
            res.notes.append(
                "sgl-kernel source tree version != installed sgl_kernel version; "
                "pin may not match the running .so"
            )
        return res

    res.resolve_error = f"unknown version_source: {spec.version_source}"
    return res


def check_sgl_kernel_version_alignment(
    sgl_kernel_src: Path,
    *,
    version_lookup: VersionLookup = _default_version_lookup,
) -> tuple[bool, str | None, str | None]:
    """Return (mismatch, installed_version, source_version).

    mismatch is True when both versions are known and differ. When either side is
    unknown we return mismatch=False (nothing to contradict) and let notes carry it.
    """

    src_version = cmake_pins.read_sgl_kernel_src_version(sgl_kernel_src)
    # The installed wheel's distribution name is "sglang-kernel" (verified in env);
    # try that first, then a couple of historical/alternate spellings.
    installed_version = None
    for dist in ("sglang-kernel", "sgl-kernel", "sgl_kernel"):
        try:
            installed_version = version_lookup(dist)
            break
        except Exception:  # noqa: BLE001
            continue
    if src_version and installed_version and src_version != installed_version:
        return True, installed_version, src_version
    return False, installed_version, src_version


def resolve_all(
    *,
    sgl_kernel_src: Path,
    triggered_by_map: dict[str, list[str]],
    version_lookup: VersionLookup = _default_version_lookup,
) -> tuple[list[RepoResolution], dict[str, object]]:
    """Resolve versions for the entire universe.

    Returns (resolutions, meta) where meta carries sgl-kernel alignment info.
    """

    pins = cmake_pins.parse_sgl_kernel_pins(sgl_kernel_src)
    mismatch, installed_ver, src_ver = check_sgl_kernel_version_alignment(
        sgl_kernel_src, version_lookup=version_lookup
    )

    resolutions: list[RepoResolution] = []
    for spec in iter_universe():
        resolutions.append(
            resolve_repo(
                spec,
                sgl_kernel_pins=pins,
                version_lookup=version_lookup,
                triggered_by=triggered_by_map.get(spec.name, []),
                sgl_kernel_version_mismatch=mismatch,
            )
        )

    meta = {
        "sgl_kernel_installed_version": installed_ver,
        "sgl_kernel_source_version": src_ver,
        "sgl_kernel_version_mismatch": mismatch,
    }
    return resolutions, meta
