"""Parse sgl-kernel FetchContent pins (Bucket B version source).

sgl-kernel compiles several third-party libraries directly into its ``.so`` and
pins each via CMake ``FetchContent_Declare`` in either ``CMakeLists.txt`` or
``cmake/flashmla.cmake``. The installed ``sgl_kernel`` wheel does NOT ship these
CMake files, so we read the pin from a matching sgl-kernel *source tree*.

Real pin shape (from sgl-kernel/CMakeLists.txt)::

    FetchContent_Declare(
        repo-flash-attention
        URL      https://${GITHUB_ARTIFACTORY}/sgl-project/sgl-attn/archive/bcf72cc...tar.gz
        URL_HASH SHA256=2110d8...
    )

We extract ``{cmake_target: CMakePin(url, commit, sha256, owner, repo)}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CMakePin:
    target: str
    commit: str
    """Commit hash or tag ref parsed from the archive URL (e.g. v3.6.0)."""
    url: str
    """Normalized clone URL (github.com/<owner>/<repo>), mirror-stripped."""
    owner: str | None
    repo: str | None
    sha256: str | None
    raw_url: str
    """Original URL as written in the cmake file (may contain ${GITHUB_ARTIFACTORY})."""


# FetchContent_Declare( <target> ... URL <url> ... [URL_HASH SHA256=<hash>] ... )
_DECLARE_RE = re.compile(
    r"FetchContent_Declare\s*\(\s*(?P<target>[A-Za-z0-9_\-]+)(?P<body>.*?)\)",
    re.DOTALL,
)
_URL_RE = re.compile(r"\bURL\s+(?P<url>\S+)")
_SHA_RE = re.compile(r"URL_HASH\s+SHA256=(?P<sha>[0-9a-fA-F]+)")
# .../<owner>/<repo>/archive/<ref>.tar.gz   where <ref> is a commit hash or a tag.
_ARCHIVE_RE = re.compile(
    r"/(?P<owner>[^/]+)/(?P<repo>[^/]+)/archive/(?P<ref>[^/]+?)\.tar\.gz",
)


def _normalize_url(raw_url: str, owner: str | None, repo: str | None) -> str:
    if owner and repo:
        return f"https://github.com/{owner}/{repo}"
    # Fall back to stripping a ${GITHUB_ARTIFACTORY}-style mirror placeholder.
    stripped = re.sub(r"https?://\$\{[^}]+\}/", "https://github.com/", raw_url)
    return re.sub(r"/archive/.*$", "", stripped)


def parse_cmake_text(text: str) -> dict[str, CMakePin]:
    """Parse all FetchContent_Declare blocks in one cmake file's text."""

    pins: dict[str, CMakePin] = {}
    for decl in _DECLARE_RE.finditer(text):
        target = decl.group("target")
        body = decl.group("body")
        url_match = _URL_RE.search(body)
        if not url_match:
            continue
        raw_url = url_match.group("url")
        sha_match = _SHA_RE.search(body)
        archive = _ARCHIVE_RE.search(raw_url)
        owner = archive.group("owner") if archive else None
        repo = archive.group("repo") if archive else None
        ref = archive.group("ref") if archive else ""
        pins[target] = CMakePin(
            target=target,
            commit=ref,
            url=_normalize_url(raw_url, owner, repo),
            owner=owner,
            repo=repo,
            sha256=sha_match.group("sha") if sha_match else None,
            raw_url=raw_url,
        )
    return pins


def _iter_cmake_files(sgl_kernel_src: Path):
    top = sgl_kernel_src / "CMakeLists.txt"
    if top.exists():
        yield top
    cmake_dir = sgl_kernel_src / "cmake"
    if cmake_dir.is_dir():
        for path in sorted(cmake_dir.glob("*.cmake")):
            yield path


def parse_sgl_kernel_pins(sgl_kernel_src: Path) -> dict[str, CMakePin]:
    """Parse pins across sgl-kernel's CMakeLists.txt + cmake/*.cmake.

    Later files override earlier ones on target-name collision, but targets are
    distinct in practice (repo-flashmla lives only in flashmla.cmake).
    """

    pins: dict[str, CMakePin] = {}
    for path in _iter_cmake_files(sgl_kernel_src):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        pins.update(parse_cmake_text(text))
    return pins


def sgl_kernel_src_root(sglang_repo_root: Path) -> Path:
    """Return the sgl-kernel source tree inside a sglang checkout."""

    return sglang_repo_root / "sgl-kernel"


def read_sgl_kernel_src_version(sgl_kernel_src: Path) -> str | None:
    """Read sgl-kernel's declared version from its source tree.

    Path: sgl-kernel/python/sgl_kernel/version.py -> ``__version__ = "0.4.3"``.
    """

    version_file = sgl_kernel_src / "python" / "sgl_kernel" / "version.py"
    if not version_file.exists():
        return None
    match = re.search(
        r"__version__\s*=\s*['\"]([^'\"]+)['\"]", version_file.read_text(errors="ignore")
    )
    return match.group(1) if match else None
