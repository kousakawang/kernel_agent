# KID Implementation Reference

The public entry point is the prompt-driven KID Agent documented in
[`KID_README.md`](../../KID_README.md). This package implements its deterministic
Runtime Capture CLI and Semantic Resolution helper commands. Users normally
start `framework_engineer/prompts/start_kid.md` with one backend config
directory instead of invoking the two stages as separate tasks.

## Configuration

Each `kid-runtime-config/v3` file describes one backend and one test command:

```json
{
  "schema_version": "kid-runtime-config/v3",
  "backend_name": "triton",
  "workdir": "/workspace",
  "output_dir": "/workspace/kid",
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
    "sample_count_per_stage": 1,
    "sampling": "unique_decomposition",
    "aggregation": "single"
  },
  "profiling": {
    "nsys_bin": "nsys",
    "max_runtime_sec": 1800,
    "disable_cuda_graph": true,
    "min_capture_coverage": 1.0,
    "trace_retention": "on_failure"
  }
}
```

Every capture follows `launch → warmup/ready → nsys start → test → nsys stop`.
When `cmd` starts a service, KID launches it in a paused Nsight Systems session,
waits for `ready`, then enables collection immediately before `test_cmd`. When
`cmd=null`, an Nsight-owned helper runs the same `test_cmd` twice: the first run
warms caches with collection and Runtime recording disabled, and the second is
the formal test. A direct `test_cmd` must therefore be repeatable and idempotent.
Only `cuda,nvtx` are traced; OS runtime tracing is intentionally disabled.

Sampling supports `unique_decomposition`, `all`, `last_n`, and `single`.
`unique_decomposition` is the default: it groups invocations by kernel-owner
execution boundaries, nesting, high→execution call paths, and unattributed
kernel count, then keeps the final invocation in each group. Stage, runtime
IDs, timings, provider hints, kernel names/counts, and repeated identical
execution calls do not split a group. `last_n` selects the final N invocations
independently per stage; `single` is the golden-compatible alias for `last_n`
with N=1. JSONL retains every invocation. SQLite is a temporary analysis input
by default and is retained only when capture fails.
Stage is detected from Runtime state and emitted as diagnostic evidence; v3
does not allow a user-supplied stage filter.

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
└── <backend_name>/
    ├── cli_log/
    │   ├── environment_probe.json
    │   ├── runtime_capture.schema.json
    │   ├── capture_events/events_<pid>.jsonl
    │   └── logs/{probe,nsys,warmup,test,summary}.log
    ├── ref/
    └── output/
```

Both `profile.nsys-rep` and the exported SQLite are temporary on success. Set
`profiling.trace_retention=always` for parser development or golden generation;
`on_failure` (the default) keeps SQLite only for failed capture diagnostics, and
`never` removes it even on failure. Standalone `analyze` never deletes its
explicit external `--sqlite` input. The Runtime
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
python kernel_agent/framework_engineer/kernel_interface_decomposer/tests/test_cli_capture_golden.py \
  --gpu-e2e -v

python kernel_agent/framework_engineer/kernel_interface_decomposer/tests/test_cli_capture_convergence.py \
  --gpu-e2e -v

python kernel_agent/framework_engineer/kernel_interface_decomposer/tests/test_cli_capture_window.py \
  --gpu-e2e -v
```

The first test runs all 11 mandatory PoC cases once. The convergence test runs
`old,old` and `old,softmax`, verifies that raw traces retain both calls, and
checks that Runtime output contains one and two unique decomposition groups,
respectively. The capture-window test compares direct execution with a service
that deliberately calls the target three times before readiness; only the one
test-triggered call may appear in either trace. All three run the Runtime
artifact validator and compare stable
structure with golden data. Runtime IDs, durations, and optional
`provider_hint` values are intentionally not fixed.

Capture categories and adapter boundaries are defined in
[CAPTURE_MECHANISMS.md](CAPTURE_MECHANISMS.md) and `capture_registry.py`.

## KID Agent semantic phase

The KID Agent performs the second stage after Runtime capture. Internally it
uses one `kid-semantic-resolver-config/v3` and the following deterministic
commands:

```bash
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  prepare semantic_resolver_config.json
# Agent reads context and writes decisions + notes.
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  finalize semantic_resolver_config.json
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  validate semantic_resolver_config.json
```

`prepare` exposes only Runtime evidence and source snippets. Agent decisions
must assign every direct kernel owner exactly once to a semantic interface.
`finalize` computes all metrics and publishes `kernel-interface-decomposition/v2`
atomically. Context, decisions, and notes are internal evidence; source_locate
consumes only the final decomposition.

The Semantic config contains only `backend_name`, the third-party manifest,
and optional Runtime-to-local path mappings. It loads the sibling
`runtime_capture_config.json` and derives Runtime/context/decisions/notes/final
paths from its top-level `output_dir`.

Run its CPU-only regression with:

```bash
PYTHONPATH=kernel_agent python3 -m unittest \
  framework_engineer.kernel_interface_decomposer.tests.test_semantic_resolver -v
```
