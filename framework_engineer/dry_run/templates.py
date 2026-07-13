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

FILL = "<FILL"  # sentinel prefix; must match source_location.contracts.FILL_SENTINEL


def fill(hint: str) -> str:
    return f"<FILL: {hint}>"


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

LAYERS = ("interface_definition", "kernel_impl", "py_cpp_binding", "kernel_header")
REQUIRED_LAYERS = ("interface_definition", "kernel_impl")


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
            "wrapper": {
                "api": fill("可选: wrapper API 名; 真实 KID 由 profiling 填, dry-run 可留空/删"),
                "file": fill("可选: wrapper 源文件; dry-run 可留空/删"),
                "line": fill("可选: wrapper 行号; dry-run 可留空/删"),
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


def source_locations_skeleton(archetype: str) -> dict[str, Any]:
    """Build the source_locations block for a kernel given its (filled) archetype.

    Applies the null-rules: form-decided not_applicable layers get status
    ``not_applicable`` with no placeholder; every other layer starts as
    ``missed`` with file/line placeholders (simulating "agent could not locate,
    hand to human") — this is exactly the real "missing" shape.
    """
    meta = ARCHETYPES.get(archetype)
    code = meta["code"] if meta else fill("未知 archetype, 请回上一步修正")
    na_layers = set(meta["na_layers"]) if meta else set()

    layers: dict[str, Any] = {}
    needs_agent = False
    for name in LAYERS:
        if name in na_layers:
            layers[name] = {"status": "not_applicable", "hits": [], "repo_hint": None}
            continue
        # F8: kernel_impl is genuinely source-less -> mark missed (no source).
        layers[name] = {
            "status": "missed",
            "hits": [
                {
                    "file": fill("源文件绝对路径, 如 /sgl-workspace/sglang/.../foo.py 或 third_party_cache 内 clone 文件"),
                    "line_start": fill("起始行号(整数)"),
                    "line_end": fill("结束行号(整数)"),
                }
            ],
            "repo_hint": fill("可选: 该库仓库根; 无则留空/删"),
        }
        needs_agent = True

    return {
        "archetype": archetype if meta else fill("明文类别名"),
        "archetype_code": code,
        "source": "dry_run",
        "needs_agent": needs_agent,
        "layers": layers,
    }
