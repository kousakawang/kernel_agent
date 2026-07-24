"""Framework Engineer Phase 1 target config.

Copy this file to your GPU/SGLang environment, edit values, then run:

    python -m framework_engineer.cli validate-config --config /path/to/config.py
    python -m framework_engineer.cli run-phase1 --config /path/to/config.py
"""

# Group-level metadata.
task_group_id = "linear_attention_targets_h20"
output_root = "/tmp/linear_attention_targets"

target_model = "Qwen3.5-9B"
target_framework = "SGLang"
target_hardware = "H20"
objective = "Generate snapshot task packs for framework-owned target interfaces."

# Required shared service/workload controls.
service_cmd = """
CUDA_VISIBLE_DEVICES=7 SGLANG_VLM_CACHE_SIZE_MB=0 python3 -m sglang.launch_server --model-path /data01/models/Qwen3.5-9B/ --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1  --disable-radix-cache
""".strip()

workload_cmd = """
python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts 1 --apply-chat-template --random-output-len 32 --random-input-len 16 --image-resolution 480x720 --image-format jpeg --image-count 1 --image-content random --random-range-ratio 1 --host=127.0.0.1 --port=8080
""".strip()

# Required shared forward boundary.
forward_boundary_file = "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py"
forward_boundary_line = 1148

# Optional service controls.
non_cudagraph_service_cmd = None
health_url = "http://127.0.0.1:8080/health"
startup_timeout = 240
workload_timeout = 1200
extra_env = {
    # "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONPATH": "/sgl-workspace/sglang/python",
}

# Optional source-locate extract directory. When set, run-phase1 matches each
# configured target_file + target_line against interface_definition.hits, then
# copies only the matched low_level_id directory and matching JSON manifest(s)
# into <task_pack>/task/kernel_source_package/.
kernel_source_package_path = None
# kernel_source_package_path = "/path/to/source_locate/workspace/extract"

# Optional capture/selection controls.
signature = "candidate(*args, **kwargs)"
max_capture_groups = 64
max_samples_per_group = 8
max_samples_per_forward_per_group = 4
max_selected_groups = 8
max_selected_samples_per_group = 8

# Optional validation controls.
run_baseline = True
run_probe_env = True
skip_env_check = True
run_benchmark_smoke = False
validate_device = "cuda"
validate_warmup = 3
validate_repeat = 5
force = False

# Multi-target recommended form. Each target becomes one independent task_pack.
# For an installed third-party package, target_file/target_line may point to a
# matching local checkout. run-phase1 resolves the effective site-packages file
# and definition line with the current Python interpreter before instrumentation.
targets = [
    {
        "task_id": "target_1",
        "target_file": "/sgl-workspace/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py",
        "target_line": 339,
        "drop_first_arg": False,
    },
    # {
    #     "task_id": "target_2",
    #     "target_file": "/path/to/target_file.py",
    #     "target_line": 200,
    #     "drop_first_arg": False,
    # },
]

# Single-target compatibility form. Use this instead of `targets` if desired:
#
# task_id = "single_target"
# task_pack = "/tmp/single_target_task_pack"
# target_file = "/path/to/target_file.py"
# target_line = 100
# drop_first_arg = False
