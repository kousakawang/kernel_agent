# KID Runtime Capture CLI

This package implements the deterministic first stage of Kernel Interface
Decomposition. It captures execution-level events and correlates them with GPU
kernels. Semantic interface selection is performed later by the Semantic
Resolver Agent.

## Configuration

Each `kid-runtime-config/v2` file describes one backend and one test command:

```json
{
  "schema_version": "kid-runtime-config/v2",
  "backend_name": "triton",
  "workdir": "/workspace",
  "output_dir": "/workspace/kid/cli_log/triton",
  "target": {
    "file": "/workspace/sglang/model_runner.py",
    "line": 500,
    "qualified_name": "ModelRunner.forward"
  },
  "cmd": "python -m sglang.launch_server ...",
  "test_cmd": "python test_request.py",
  "ready": {
    "type": "http",
    "url": "http://127.0.0.1:30000/health",
    "timeout_sec": 300
  },
  "stop": {"signal": "SIGINT", "grace_sec": 30},
  "env": {},
  "selection": {
    "skip_invocations": 0,
    "stages": ["prefill", "decode", "unknown"],
    "sample_count_per_stage": 1,
    "sampling": "last_n",
    "aggregation": "single"
  },
  "profiling": {
    "nsys_bin": "nsys",
    "max_runtime_sec": 1800,
    "disable_cuda_graph": true,
    "min_capture_coverage": 1.0
  }
}
```

Set `cmd` to `null` to profile `test_cmd` directly. Warmup belongs inside the
test workload and should run outside the selected high-level invocation.

Sampling supports `all`, `last_n`, and `single`. `last_n` selects the final N
invocations independently per stage; `single` is the golden-compatible alias
for `last_n` with N=1. Raw JSONL and SQLite always retain every invocation.

## Commands

```bash
python3 -m framework_engineer.kernel_interface_decomposer capture config.json

python3 -m framework_engineer.kernel_interface_decomposer analyze config.json \
  --sqlite path/to/profile.sqlite \
  --events-dir path/to/capture_events
```

`capture` runs the service/test lifecycle and Nsight Systems. `analyze` only
rebuilds normalized Runtime Capture data from existing evidence. The old `run`
command and direct semantic/source resolution are intentionally unsupported.

## Output

```text
<output_dir>/
├── environment_probe.json
├── runtime_capture.schema.json
├── capture_events/events_<pid>.jsonl
├── trace/profile.sqlite
└── logs/{probe,nsys,test,summary}.log
```

`profile.nsys-rep` is temporary and is removed after SQLite export. The Runtime
schema contains complete high→execution stacks, nested capture relationships,
CUDA launch correlation, direct/inclusive GPU duration, and attribution
coverage. It must not contain semantic targets or source-location results.

## Validation and tests

```bash
PYTHONPATH=kernel_agent python3 -m \
  framework_engineer.kernel_interface_decomposer.artifact_validator \
  --runtime-only kernel_agent/example_kernels/nsys_poc_kid_golden

PYTHONPATH=kernel_agent python3 -m unittest discover \
  -s kernel_agent/framework_engineer/kernel_interface_decomposer/tests -v
```

`test_cli_analyze_golden.py` is the formal offline end-to-end CLI regression:
it invokes `python -m ... analyze` in a subprocess, validates the generated
artifact directory, and compares the normalized Runtime schema with the
retained Nsight PoC golden. Run it alone with:

```bash
PYTHONPATH=kernel_agent python3 -m unittest \
  framework_engineer.kernel_interface_decomposer.tests.test_cli_analyze_golden -v
```

The full `capture` regression requires CUDA and Nsight Systems and is opt-in so
normal local test discovery stays GPU-free. With the remote Python runner, use:

```bash
cd kernel_agent
KID_RUN_GPU_E2E=1 python -m unittest \
  framework_engineer.kernel_interface_decomposer.tests.test_cli_capture_golden -v
```

It runs all 11 mandatory PoC cases through the formal `capture` command, runs
the Runtime artifact validator, and compares stable capture/kernel structure
with the golden. Runtime IDs and measured durations are intentionally not fixed.

Capture categories and adapter boundaries are defined in
[CAPTURE_MECHANISMS.md](CAPTURE_MECHANISMS.md) and `capture_registry.py`.
