"""Example config for test_real_sglang_phase1.py.

Copy this file to a server-local path, edit values, then run:

    KA_REAL_SGLANG_CONFIG=/path/to/real_sglang_phase1_config.py \
      python3 -m unittest framework_engineer.tests.test_real_sglang_phase1
"""

# Required.
service_cmd = """
python -m sglang.launch_server \
  --model-path /path/to/model \
  --host 127.0.0.1 \
  --port 30000
""".strip()

workload_cmd = """
python /path/to/run_your_workload.py \
  --endpoint http://127.0.0.1:30000
""".strip()

target_file = "/path/to/sglang/python/sglang/srt/path/to/target_file.py"
target_line = None  # Preferred: line inside the target function or on its `def`.
function_name = None  # Legacy fallback when target_line is not set.
target_name = None  # Optional legacy override when function_name is used.

# Optional but recommended: a higher-level forward/module/backend function that
# wraps repeated target kernel calls for one model forward window.
forward_boundary_file = None
forward_boundary_line = None  # Preferred: line inside the boundary function or on its `def`.
forward_boundary_function = None
forward_boundary_name = None  # Optional override.

# Optional task output.
task_id = "real_sglang_target_phase1"
task_pack = "/tmp/real_sglang_target_task_pack"
keep_task_pack = True
skip_baseline = False

# Optional service controls.
health_url = "http://127.0.0.1:30000/health"
startup_timeout = 240
workload_timeout = 1200
test_timeout = 3600

# If appending --disable-cuda-graph to service_cmd is wrong for your setup,
# provide the exact non-cudagraph launch command here. The CLI will also dedupe
# duplicate --disable-cuda-graph flags.
non_cudagraph_service_cmd = service_cmd + " --disable-cuda-graph"

# Target ABI/capture controls.
signature = "candidate(*args, **kwargs)"
target_mode = "extend"
target_backend = "triton"
target_layer_id = ""

# Set False if target function is not an instance/class method.
drop_first_arg = True

# Free-function targets usually use:
#
# target_file = "/path/to/sglang/python/sglang/srt/path/to/target_file.py"
# target_line = 123
# drop_first_arg = False
#
# If no forward boundary is available yet, calls_per_forward can be used as a
# fragile fallback after probe-target-calls confirms a stable call count.
calls_per_forward = None
max_capture_groups = 64
max_samples_per_group = 8
max_samples_per_forward_per_group = 3
max_raw_cases = None  # Deprecated alias for max_capture_groups.
max_selected_groups = 8
max_selected_samples_per_group = 8
max_selected_cases = None  # Deprecated alias for max_selected_groups.
candidate_function = "candidate"

# Validation controls.
run_probe_env = False
skip_env_check = True
run_benchmark = False
validate_device = "cuda"
validate_warmup = 3
validate_repeat = 5

# Per-CLI switches. Keys are exact CLI subcommand names.
# Missing keys default to True. You are responsible for dependency correctness
# when skipping steps; for example, skipping capture-snapshots requires existing
# raw snapshots if select-snapshots remains enabled.
cli_tests = {
    "scaffold-task-pack": True,
    "run-baseline": True,
    "probe-target-calls": True,
    "capture-snapshots": True,
    "select-snapshots": True,
    "generate-harness": True,
    "probe-env": True,
    "validate-task-pack": True,
}

# Extra env vars passed to service/workload/CLI subprocesses.
extra_env = {
    # "CUDA_VISIBLE_DEVICES": "0",
    # "PYTHONPATH": "/path/to/sglang/python",
}
