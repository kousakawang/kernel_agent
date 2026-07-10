"""Fixed universe of GPU-inference third-party repositories.

Design decisions (see .trae/documents/resolve_third_party_skill_plan.md):

* The set of GPU-inference-relevant kernel/comm libraries is finite, so we hardcode
  a *universe* instead of discovering it at runtime.
* `on_default_path` marks libraries that are used even when the launch command does
  NOT name them explicitly (e.g. flashinfer / deep_gemm sit on the default path).
  Confirmed once here; not re-derived by an agent every run.
* Backend flags only *annotate* `triggered_by`; they NEVER prune the clone set.
  We resolve + clone the whole source-bearing universe (archetypes F0..F7), because
  flag-only gating would miss default-path libraries.

Version source per archetype (strict bucketing):
  * ``importlib``  — independent pip package, version from importlib.metadata.
  * ``cmake_pin``  — compiled into sgl_kernel's .so, version from sgl-kernel's
    FetchContent pin (cmake_pins.py). The dist wheel has no CMakeLists, so the pin
    is read from the matching sgl-kernel source tree.

The ``cmake_target`` field names the FetchContent declare target inside
sgl-kernel (see CMakeLists.txt / cmake/flashmla.cmake) so version_resolver can map
a registry entry to its pinned commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoSpec:
    """One entry in the fixed universe."""

    name: str
    """Canonical repo name used as the manifest key and cache dir name."""

    archetype: str
    """Kernel form: F0..F8 (see design doc 8.1.0)."""

    version_source: str
    """"importlib" (Bucket A) or "cmake_pin" (Bucket B)."""

    dist_name: str | None = None
    """PyPI distribution name for importlib.metadata (Bucket A only)."""

    import_name: str | None = None
    """Python import name if it differs from ``name`` (diagnostics only)."""

    ref_template: str | None = "v{version}"
    """How to form the git ref for a Bucket A clone from the resolved version.

    Default ``v{version}`` suits standard release tags. Set to ``"{version}"`` for
    repos whose tags omit the ``v`` prefix, or ``None`` to clone the default branch
    when no matching tag exists (e.g. beta wheels like flash-attn-4 4.0.0b17).
    """

    cmake_target: str | None = None
    """FetchContent target name inside sgl-kernel (Bucket B only)."""

    url: str | None = None
    """Clone URL. For sgl forks this points at the sgl-project mirror."""

    url_kind: str = "official"
    """"official" | "sgl_fork" — where the correct source lives."""

    on_default_path: bool = True
    """True if used even when launch cmd does not name it explicitly."""

    backend_flags: tuple[str, ...] = field(default_factory=tuple)
    """Backend selector values that, when present, attribute this repo to a path.

    Used ONLY to populate ``triggered_by`` annotations, never to prune.
    """

    has_source: bool = True
    """False for F8 (downloaded cubin) — nothing to clone, always FAILED."""

    note: str = ""


# ---------------------------------------------------------------------------
# The universe. Values marked (verify-in-env) are best-known defaults that the
# implementer should confirm via importlib in the target GPU container.
# ---------------------------------------------------------------------------
UNIVERSE: tuple[RepoSpec, ...] = (
    # --- Bucket A: independent pip packages (runtime-loaded) --------------
    RepoSpec(
        name="flashinfer",
        archetype="F7",
        version_source="importlib",
        dist_name="flashinfer_python",
        url="https://github.com/flashinfer-ai/flashinfer",
        url_kind="official",
        on_default_path=True,
        backend_flags=("flashinfer",),
        note="C++ JIT + cuteDSL; runtime GDN/attention path. csrc ships in wheel.",
    ),
    RepoSpec(
        name="deep_gemm",
        archetype="F7",
        version_source="importlib",
        dist_name="sgl-deep-gemm",  # verified: import deep_gemm -> dist sgl-deep-gemm 0.1.2
        ref_template="release",  # sgl fork ships from the `release` branch (dist METADATA)
        url="https://github.com/sgl-project/DeepGEMM",  # verified: METADATA Project-URL Repository
        url_kind="sgl_fork",
        on_default_path=True,
        backend_flags=("deep_gemm", "deepgemm"),
        note="JIT/NVRTC GEMM; templates ship in package. Default MoE/GEMM path. "
        "sgl fork tracked via the `release` branch (no version tag).",
    ),
    RepoSpec(
        name="flash_attn_4",
        archetype="F5",
        version_source="importlib",
        dist_name="flash-attn-4",  # verified: pip dist flash-attn-4 4.0.0b17
        import_name="flash_attn",  # installed under dist-packages/flash_attn
        ref_template=None,  # beta wheel has no matching git tag -> clone default branch
        url="https://github.com/Dao-AILab/flash-attention",
        url_kind="official",
        on_default_path=False,
        backend_flags=("fa4", "flashattention4"),
        note="FA4 ships as a beta wheel with no matching git tag; clone default "
        "branch for the full source tree (version recorded from importlib).",
    ),
    # NOTE: `fla` and `causal_conv1d` are intentionally NOT in this universe.
    # They are NOT external pip packages in this stack:
    #   * fla           -> sglang-owned (sglang.srt.layers.attention.fla.*, F1/F2);
    #                      external `fla` only appears in docstrings. `import fla`
    #                      fails in the target env.
    #   * causal_conv1d -> triton (sglang-owned) or sgl_kernel .so (F2,
    #                      csrc/mamba/causal_conv1d.cu). No standalone package.
    # Both are located in-place by the `locate-kernel-source` skill (F1/F2), so
    # resolve-third-party (which only clones EXTERNAL repos) must not list them.
    # --- Bucket B: compiled into sgl_kernel .so (pin from FetchContent) ----
    RepoSpec(
        name="flash_attn",
        archetype="F3",
        version_source="cmake_pin",
        cmake_target="repo-flash-attention",
        url="https://github.com/sgl-project/sgl-attn",
        url_kind="sgl_fork",
        on_default_path=False,
        backend_flags=("fa3", "fa"),
        note="sgl fork of flash-attention; kernels compiled into sgl_kernel's .so.",
    ),
    RepoSpec(
        name="flash_mla",
        archetype="F3",
        version_source="cmake_pin",
        cmake_target="repo-flashmla",
        url="https://github.com/sgl-project/FlashMLA",
        url_kind="sgl_fork",
        on_default_path=False,
        backend_flags=("flashmla", "flash_mla"),
        note="sgl fork of FlashMLA; pinned in cmake/flashmla.cmake.",
    ),
    RepoSpec(
        name="cutlass",
        archetype="F3",
        version_source="cmake_pin",
        cmake_target="repo-cutlass",
        url="https://github.com/NVIDIA/cutlass",
        url_kind="official",
        on_default_path=True,
        backend_flags=(),
        note="Header-only templates included by sgl-kernel and flash_mla.",
    ),
    RepoSpec(
        name="mscclpp",
        archetype="F3",
        version_source="cmake_pin",
        cmake_target="repo-mscclpp",
        url="https://github.com/microsoft/mscclpp",
        url_kind="official",
        on_default_path=True,
        backend_flags=(),
        note="Communication lib compiled into sgl-kernel.",
    ),
    RepoSpec(
        name="flashinfer_embedded",
        archetype="F3",
        version_source="cmake_pin",
        cmake_target="repo-flashinfer",
        url="https://github.com/flashinfer-ai/flashinfer",
        url_kind="official",
        on_default_path=True,
        backend_flags=(),
        note="Only norm.cu/renorm.cu compiled into sgl-kernel; distinct pin from "
        "runtime flashinfer (F7). Kept separate so (name,version) never collide.",
    ),
    # NOTE: `triton_kernels` is intentionally NOT in this universe. sgl-kernel's
    # CMake merely `install(DIRECTORY .../python/triton_kernels/ ...)` — it copies
    # a pure-python dir into the wheel, nothing is compiled into the .so. It ships
    # as the installed `triton_kernels` package (F1/F6, pure python), so its source
    # is read in-place by `locate-kernel-source`. Cloning the whole (huge) triton
    # repo by a pinned tag would be pure waste. sglang's own `@triton.jit` kernels
    # are F1 and out of scope here too.
    # --- F8: downloaded pre-compiled cubin — NO source, always FAILED -----
    RepoSpec(
        name="flashinfer_cubin",
        archetype="F8",
        version_source="importlib",
        dist_name="flashinfer_cubin",
        url=None,
        url_kind="official",
        on_default_path=False,
        backend_flags=("trtllm",),
        has_source=False,
        note="Downloaded pre-compiled cubin blob; no source to clone -> FAILED.",
    ),
)


_BY_NAME: dict[str, RepoSpec] = {spec.name: spec for spec in UNIVERSE}


def get_spec(name: str) -> RepoSpec | None:
    return _BY_NAME.get(name)


def iter_universe(*, source_bearing_only: bool = False):
    """Iterate the universe.

    When ``source_bearing_only`` is True, skip F8 (no source to clone).
    """

    for spec in UNIVERSE:
        if source_bearing_only and not spec.has_source:
            continue
        yield spec
