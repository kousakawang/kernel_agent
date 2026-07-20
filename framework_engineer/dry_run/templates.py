"""Skeleton templates + placeholder constants for the dry-run mechanism.

The dry-run produces artifacts that are *byte-for-byte structurally identical*
to what the real pipeline emits — it only replaces values that real KID
profiling / real locate would compute with either:
  * an auto-derived value (archetype null-rules, plaintext names), or
  * a ``<FILL: ...>`` placeholder (things a human must judge/locate).

Archetype plaintext names (handoff contract / plan §2). ``code`` is the F0-F8
attribute kept as an auxiliary field; configs/products never use the bare code.
"""

from __future__ import annotations

from typing import Any

LAYERS: tuple[str, ...] = (
    "interface_definition",
    "kernel_impl",
    "py_cpp_binding",
    "kernel_header",
)
DIRECTORY_LAYERS: tuple[str, ...] = (
    "kernel_impl",
    "py_cpp_binding",
    "kernel_header",
)

FILL = "<FILL"  # legacy dry-run sentinel prefix


def fill(hint: str) -> str:
    return f"<FILL: {hint}>"


# --- source roles -----------------------------------------------------------
# `source` = the role that LAST updated a given layer's location. Lives per-layer
# (each layer may be filled by a different role); the top-level source_locations
# carries a *derived aggregate* (agent > manual > layer1 > dry_run).
SOURCE_DRY_RUN = "dry_run"            # dry-run skeleton (placeholder, unfilled)
SOURCE_LAYER1 = "locate_layer1"       # deterministic CLI located it
SOURCE_LAYER2_AGENT = "locate_layer2_agent"  # Layer 2 agent updated it
SOURCE_MANUAL = "manual"              # a human filled it by hand

# Precedence for deriving the top-level aggregate: if any layer was touched by an
# agent/human, that dominates the "still just CLI/dry-run" default.
_SOURCE_PRECEDENCE = [
    SOURCE_LAYER2_AGENT,
    SOURCE_MANUAL,
    SOURCE_LAYER1,
    SOURCE_DRY_RUN,
]


def aggregate_source(layer_sources: list[str]) -> str:
    """Derive the top-level source from the per-layer sources (highest precedence)."""
    present = [s for s in layer_sources if s]
    for role in _SOURCE_PRECEDENCE:
        if role in present:
            return role
    return SOURCE_DRY_RUN


# --- archetype table --------------------------------------------------------
# name -> {code, na_layers}  where na_layers are the four-layer names that are
# form-decided not_applicable, and (for F8) a special missing kernel_impl.
ARCHETYPES: dict[str, dict[str, Any]] = {
    "pytorch_native":         {"code": "F0", "na_layers": ["kernel_impl", "py_cpp_binding", "kernel_header"]},
    "sglang_triton":          {"code": "F1", "na_layers": ["py_cpp_binding", "kernel_header"]},
    "sgl_kernel_builtin":     {"code": "F2", "na_layers": []},
    "sgl_kernel_thirdparty":  {"code": "F3", "na_layers": []},
    "sglang_jit":             {"code": "F4", "na_layers": []},
    "thirdparty_aot":         {"code": "F5", "na_layers": []},
    "thirdparty_triton_dsl":  {"code": "F6", "na_layers": ["py_cpp_binding", "kernel_header"]},
    "thirdparty_cpp_jit":     {"code": "F7", "na_layers": []},
    "downloaded_cubin":       {"code": "F8", "na_layers": ["py_cpp_binding", "kernel_header"]},
}

ARCHETYPE_NAMES = tuple(ARCHETYPES.keys())


def kid_kernel_template(index: int) -> dict[str, Any]:
    """One kernel slot in a KID dry-run schema (no source_locations yet)."""
    return {
        "rank": index + 1,
        "kernel": {
            "raw_name": fill("GPU kernel 名, 如 _layer_norm_fwd_1pass_kernel; 或直接删该槽=放弃该 kernel"),
            "normalized_name": fill("规范化 kernel 名, 可与 raw_name 相同"),
        },
        "low_level_id": fill("该 low_level 的稳定 id, 如 layernorm_fwd; 用作 kernel_sources 子目录名"),
        "interface": fill("运行时接口名, 如 torch.ops.sgl_kernel.fwd / triton fn 名 / get_xxx_module().op"),
        "archetype": fill("明文类别名, 9 选 1: " + " / ".join(ARCHETYPE_NAMES)),
        "archetype_code": fill("对应 F0-F8, 见 dry_run.example.py 顶部对照表"),
        "metrics": {
            "duration_us": fill("可选: 该 kernel 耗时 us; 不关心可删本行"),
            "share_in_invocation": fill("可选: 占本次 forward 比例; 不关心可删本行"),
        },
        "runtime_event": {
            "call_site": {
                "file": fill("该接口被调用处的源文件绝对路径 (KID 运行时抓调用栈)"),
                "line": fill("该接口被调用处的行号 (整数)"),
            },
            "implementation": {
                "source_files": [],
                "note": "真实 KID 由运行时插桩填充; dry-run 留空, 源码位置在下一步 locate 填",
            },
            "attribution": {"method": "dry_run", "confidence": "n/a"},
        },
    }


def kid_schema_skeleton(backend_name: str, target: dict[str, Any], num_kernels: int) -> dict[str, Any]:
    return {
        "schema_version": "kernel-interface-decomposition/dry-run-v1",
        "backend_name": backend_name,
        "dry_run": True,
        "target": {
            "file": target.get("file"),
            "line": target.get("line"),
            "note": "分解起点(用户指定的 high_level 模块); 不是分解出来的 low_level target",
        },
        "coverage_report": {
            "note": fill("真实 KID 在此报漏检覆盖率; dry-run 不统计"),
        },
        "kernels": [kid_kernel_template(i) for i in range(num_kernels)],
    }


def _missed_hit(is_directory: bool, layer: str) -> dict[str, Any]:
    """One placeholder hit. Only ``file`` + ``def_line`` (locate standard §1:
    end line is computed by the Layer 3 CLI, never filled here)."""
    if is_directory and layer == "kernel_impl":
        file_hint = (
            "调用链上一个文件的绝对路径; kernel_impl 是目录层, 可加多个 hit, "
            "按调用顺序: launcher -> ... -> 真正的 __global__ kernel"
        )
    elif is_directory and layer == "py_cpp_binding":
        file_hint = (
            "py<->cpp 绑定处的绝对路径; py_cpp_binding 是目录层, 可加多个 hit(多格式), "
            "如 .py 的 load_jit/build_and_load 行 + C++ 的 *_binding.cc/*_binding.cu 导出"
        )
    elif is_directory:
        file_hint = "对应源文件的头文件绝对路径; kernel_header 是目录层, 可加多个 hit(与 kernel_impl 源文件一一对应)"
    else:
        file_hint = "源文件绝对路径, 如 /sgl-workspace/sglang/.../foo.py 或 third_party_cache 内 clone 文件"
    return {
        "file": fill(file_hint),
        "def_line": fill("定义起始行号(整数); 结束行由 Layer 3 CLI 自动补, 不要填"),
    }


def source_locations_skeleton(archetype: str) -> dict[str, Any]:
    """Build the source_locations block for a kernel given its (filled) archetype.

    Applies the null-rules: form-decided not_applicable layers get status
    ``not_applicable`` with no placeholder; every other layer starts as
    ``missed`` with a ``{file, def_line}`` placeholder (simulating "agent could
    not locate, hand to human") — this is exactly the real "missing" shape.

    Layer shapes (locate standard §2): ``kernel_impl``/``kernel_header``/
    ``py_cpp_binding`` are *directory* layers whose ``hits`` may hold multiple
    entries (kernel_impl is an ordered call chain; py_cpp_binding may need both a
    .py bridge line and a C++ *_binding.cu export); ``interface_definition`` is
    single-file (exactly one hit). The skeleton seeds one placeholder hit either
    way — a human adds more hits to a directory layer as needed.

    ``source`` lives *per layer* (= role that last updated that layer). In the
    dry-run skeleton every layer starts as ``dry_run``; after a human fills a
    layer they should set it to ``manual``. The top-level ``source`` is a derived
    aggregate over the per-layer values.
    """
    meta = ARCHETYPES.get(archetype)
    code = meta["code"] if meta else fill("未知 archetype, 请回上一步修正")
    na_layers = set(meta["na_layers"]) if meta else set()

    layers: dict[str, Any] = {}
    needs_agent = False
    for name in LAYERS:
        if name in na_layers:
            layers[name] = {
                "status": "not_applicable",
                "hits": [],
                "repo_hint": None,
                "source": SOURCE_DRY_RUN,
            }
            continue
        # F8: kernel_impl is genuinely source-less -> mark missed (no source).
        layers[name] = {
            "status": "missed",
            "hits": [_missed_hit(name in DIRECTORY_LAYERS, name)],
            "repo_hint": fill("可选: 该库仓库根; 无则留空/删"),
            "source": SOURCE_DRY_RUN,
        }
        needs_agent = True

    return {
        "archetype": archetype if meta else fill("明文类别名"),
        "archetype_code": code,
        "source": aggregate_source([lyr.get("source") for lyr in layers.values()]),
        "needs_agent": needs_agent,
        "layers": layers,
    }
