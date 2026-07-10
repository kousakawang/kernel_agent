"""Example config for the resolve-third-party skill.

Copy to your GPU/SGLang environment, edit values, then run:

    python -m framework_engineer.third_party_solver.cli resolve --config /path/to/this.py --dry-run
    python -m framework_engineer.third_party_solver.cli resolve --config /path/to/this.py

The skill's agent (skills/resolve_third_party.md) orchestrates these two calls and
reviews the manifest between them.
"""

# One entry per backend launch path. `backend_name` labels the path in
# `triggered_by`; `cmd` is the exact sglang launch command.
service_cmds = [
    {
        "backend_name": "triton",
        "cmd": (
            "python3 -m sglang.launch_server --model-path /data/models/Qwen3.5-9B/ "
            "--host 127.0.0.1 --port 8080 --linear-attn-backend triton "
            "--tensor-parallel-size 1 --disable-radix-cache"
        ),
    },
    {
        "backend_name": "flashinfer",
        "cmd": (
            "python3 -m sglang.launch_server --model-path /data/models/Qwen3.5-9B/ "
            "--host 127.0.0.1 --port 8080 --linear-attn-backend flashinfer "
            "--attention-backend fa3 --tensor-parallel-size 1 --disable-radix-cache"
        ),
    },
    {
        "backend_name": "cutedsl",
        "cmd": (
            "python3 -m sglang.launch_server --model-path /data/models/Qwen3.5-9B/ "
            "--host 127.0.0.1 --port 8080 --linear-attn-backend cutedsl "
            "--tensor-parallel-size 1 --disable-radix-cache"
        ),
    },
]

# Optional: workload/test commands (used only if the agent needs runtime confirmation).
workload_cmds = [
    "python3 -m sglang.bench_serving --backend sglang --num-prompts 1 "
    "--host 127.0.0.1 --port 8080"
]

# sglang checkout root; must contain the sgl-kernel/ source tree (for CMake pins).
sglang_repo_root = "/sgl-workspace/sglang"

# Clone destination, keyed internally by (name, version).
third_party_cache = "/data/third_party_cache"

# Where third_party_manifest.json + missing_repos.md are written.
output_root = "/data/step0_5_out"

# Optional: user-provided P1 paths that override auto-resolution, e.g.
# {"flash_mla": "/some/local/FlashMLA"}
explicit_paths = {}

# Optional: env for any runtime confirmation the agent performs.
extra_env = {
    "PYTHONPATH": "/sgl-workspace/sglang/python",
}
