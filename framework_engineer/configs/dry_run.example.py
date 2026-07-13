"""Example config for the KID dry-run (Step 0.5 -> Step 1 delivery validation).

Structurally identical to what the real (V2) KID will consume: multi-backend
`service_cmds` + a unified high_level `target`. The dry-run does NOT run these
commands; they document the backend paths whose schemas get generated.

Run:
    python -m framework_engineer.dry_run.cli kid --config <this file>
    # fill interface/archetype in each generated schema, then:
    python -m framework_engineer.dry_run.cli locate --workspace <output_root>/workspaces
    # fill file/line for missed layers, then:
    python -m framework_engineer.dry_run.cli extract --workspace <output_root>/workspaces

archetype 明文类别名对照（配置/产物里只用明文名，禁用裸 F* 代号）：
    pytorch_native        (F0)  torch/aten/cuBLAS API
    sglang_triton         (F1)  sglang 自带 triton
    sgl_kernel_builtin    (F2)  sgl-kernel 内实现 (AOT)
    sgl_kernel_thirdparty (F3)  sgl-kernel FetchContent 编入的三方
    sglang_jit            (F4)  sglang-owned JIT
    thirdparty_aot        (F5)  三方 C++/cuda AOT
    thirdparty_triton_dsl (F6)  三方 triton/cuteDSL
    thirdparty_cpp_jit    (F7)  三方 C++ JIT (flashinfer/deep_gemm)
    downloaded_cubin      (F8)  下载预编译 cubin (无源)
"""

# One entry per backend launch path (mirrors resolve_third_party.example.py).
service_cmds = [
    {
        "backend_name": "triton",
        "cmd": (
            "python3 -m sglang.launch_server --model-path /data/models/Qwen3.5-9B/ "
            "--linear-attn-backend triton --disable-cuda-graph"
        ),
    },
    {
        "backend_name": "flashinfer",
        "cmd": (
            "python3 -m sglang.launch_server --model-path /data/models/Qwen3.5-9B/ "
            "--linear-attn-backend flashinfer --disable-cuda-graph"
        ),
    },
]

# Unified high_level target (the decomposition START point, not a low_level).
target = {
    "file": "/sgl-workspace/sglang/python/sglang/srt/layers/radix_linear_attention.py",
    "line": 78,
}

# sglang checkout root (used by real locate; optional for dry-run).
sglang_repo_root = "/sgl-workspace/sglang"

# Where dry-run writes workspaces/.
output_root = "/tmp/kid_dry_run_out"

# Number of kernel template slots per backend. Fill all to validate multiple
# low_level targets; delete extras to model selective drop.
kernels_per_backend = 3
