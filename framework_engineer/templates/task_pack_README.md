# Framework Engineer Task Pack

This task pack is a self-contained delivery for Kernel Translate and Kernel
Engineer. The outer directory intentionally contains only this README, the
delivery validator, and the `task/` payload.

## Directory layout

```text
<task_pack>/
├── README.md
├── validate_task_pack.py
└── task/
    ├── task.yaml
    ├── env_manifest.yaml
    ├── shape_list.json
    ├── candidate_impl.py
    ├── original_impl.py
    ├── reference_impl.py
    ├── snapshot_runtime.py
    ├── correctness_test.py
    ├── benchmark.py
    ├── kernel_translate/
    │   └── README.md
    ├── kernel_engineer_ws/
    │   └── README.md
    ├── scripts/
    │   ├── run_correctness.py
    │   ├── run_benchmark.py
    │   └── run_ncu.py
    ├── env_probe/
    ├── snapshots/
    ├── kernel_source_package/
    └── docs/
```

## Contents

- `task/task.yaml`: task contract, target ABI, commands, and writable-path policy.
- `task/env_manifest.yaml`: captured Python, GPU, DSL, compiler, and profiler availability.
- `task/shape_list.json`: summary of selected snapshot groups; it is not the replay source.
- `task/snapshots/manifest.json`: selected replay groups and samples.
- `task/snapshots/selected/`: immutable correctness and benchmark inputs.
- `task/snapshots/raw/`: raw captures retained for audit and reselection.
- `task/snapshot_runtime.py`: standalone snapshot loading and comparison runtime.
- `task/original_impl.py`: linked replay of the captured runtime target.
- `task/reference_impl.py`: reference replay plus snapshot-golden fallback.
- `task/candidate_impl.py`: final candidate implementation entry.
- `task/correctness_test.py`: immutable correctness harness and tolerances.
- `task/benchmark.py`: immutable benchmark and timing rules.
- `task/scripts/`: stable Python entry points for correctness, benchmark, and NCU.
- `task/env_probe/`: Python probes for Triton, CuTe DSL, CUDA extension, and NCU.
- `task/kernel_source_package/`: optional, read-only extracted original kernel sources.
- `task/docs/`: read-only Phase 1 evidence, reports, and validation output.

## Writable workspaces

### Kernel Translate

Kernel Translate may create and modify files only under:

```text
task/kernel_translate/
```

Translated implementations, mapping notes, helper code, and intermediate
artifacts all belong there. Kernel Translate may read the rest of the task pack,
but it must not modify `candidate_impl.py`, snapshots, test/benchmark harnesses,
environment files, scripts, contracts, or files outside its workspace.

### Kernel Engineer

Kernel Engineer may:

- create, modify, and delete files under `task/kernel_engineer_ws/`;
- modify `task/candidate_impl.py`.

`task/kernel_engineer_ws/` is where custom `.py`, `.cu`, `.cpp`, `.cuh`, build
files, compiled artifacts, profiler output, and temporary iteration files
belong. The iteration log should be written to
`task/kernel_engineer_ws/iteration_log.md`. `task/kernel_translate/`,
`task/kernel_source_package/`, and all files under `task/docs/` are read-only
references.

Kernel Engineer must not modify snapshots, `shape_list.json`,
`snapshot_runtime.py`, original/reference implementations, correctness or
benchmark harnesses, scripts, environment probes, `task.yaml`, the outer README,
or `validate_task_pack.py`.

## Validate the delivery

Run the complete acceptance gate from the outer task-pack directory:

```bash
python validate_task_pack.py
```

The default gate checks layout, workspace policy, Python syntax, selected
snapshot integrity, optional kernel source packaging, environment compatibility,
and correctness. Benchmark is intentionally opt-in:

```bash
python validate_task_pack.py --run-benchmark
```

Diagnostic-only flags are `--skip-env-check`, `--skip-correctness`, `--device`,
and `--timeout`. The report is written to
`task/docs/task_pack_validation_report.json`.

## Run task commands

Run the following commands from the outer task-pack directory.

### Check candidate correctness

This command replays the selected snapshots, runs `candidate_impl.py`, and
compares its outputs and mutated inputs with the captured golden results. Run it
after every implementation change. For each testcase, the terminal prints a
separate, clearly delimited block. Input tensors are listed one per line with
their path, shape, dtype, and stride. The block ends with `[correctness] PASS`
when the case succeeds. Raw Python or JSON data structures are not printed in
the default human-readable mode.

```bash
python task/scripts/run_correctness.py
```

Use `--device cpu|cuda`, `--mode snapshot-golden|reference-replay`,
`--group-id`, or `--sample-id` to narrow or change the check.

### Measure performance

This command benchmarks the selected snapshots. By default it attempts both the
linked original implementation and the candidate so that it can report
speedup. A `[benchmark] RUN` line identifies each testcase and prints its input
tensor shapes one per line before measurement. The matching result block reports
median, mean, minimum, and maximum time for each available implementation.
Separate delimiters make the boundary between testcases explicit.

```bash
python task/scripts/run_benchmark.py
```

Use `--target candidate` when only the candidate is executable, or
`--target reference|both` to select the comparison target. `--warmup`,
`--repeat`, `--group-id`, and `--sample-id` control the benchmark run.

Both commands default to clean human-readable logs. Machine consumers may
explicitly request JSONL with `--output-format json`; this is the only mode that
prints the structured `case_shape` field.

### Profile one snapshot group with NCU

This command launches Nsight Compute for one selected snapshot group. Find valid
group IDs in `task/shape_list.json`; the sample ID is optional.

```bash
python task/scripts/run_ncu.py <group_id> [sample_id]
```

These three files under `task/scripts/` are the task pack's stable command
entries. In addition to the command-line options above, they accept `DEVICE`,
`CORRECTNESS_MODE`, `TARGET`, `WARMUP`, `REPEAT`, `OUTPUT_FORMAT`, and `PYTHON`
as optional environment-variable overrides.

`task/docs/original_capture_benchmark_summary.json` contains advisory timing
observed during capture and excludes snapshot serialization. When linked
original replay is available, use the reference timing produced by
`task/benchmark.py` as the speedup baseline.
