# Framework Engineer TODO

## 1. Persist Reference Benchmark Baseline

Current state:

- `correctness_test.py` uses captured snapshot outputs/post-state as golden by default.
- `benchmark.py --target both` can compare linked original (`reference`) against `candidate`.
- Initial `candidate_impl.py` delegates to linked original when available, otherwise falls back to snapshot-golden.
- `run-baseline` records service/workload-level baseline, not per-snapshot target microbenchmark baseline.
- Snapshot capture records advisory original framework call timing in sample metadata and
  `docs/original_capture_benchmark_summary.json`; this excludes snapshot dump time but is not a speedup baseline.

Problem:

- If linked original is available, the generated task pack can produce a valid original-vs-candidate benchmark, but Framework Engineer does not persist the original per-snapshot benchmark at task-pack generation time.
- If linked original is unavailable, snapshot fallback can keep correctness runnable but cannot provide meaningful performance baseline or speedup.
- Capture-time original call timing is useful as a fallback reference, but it may differ from linked-original benchmark timing because of cuda-graph mode, service context, synchronization, and instrumentation effects.

Planned work:

- After `generate-harness`, run a reference benchmark when possible:

  ```bash
  TARGET=reference bash scripts/run_benchmark.sh
  ```

- Persist outputs into task pack docs:

  ```text
  docs/reference_benchmark.jsonl
  docs/reference_benchmark_summary.json
  docs/reference_benchmark_report.md
  ```

- Record explicit status:

  ```json
  {
    "reference_available": true,
    "baseline_kind": "linked_original_microbenchmark",
    "group_count": 0,
    "sample_count": 0
  }
  ```

- If linked original is unavailable, persist:

  ```json
  {
    "reference_available": false,
    "baseline_kind": "unavailable",
    "reason": "<original_impl error>",
    "candidate_only_benchmark_allowed": true,
    "speedup_available": false
  }
  ```

- Update `validate-task-pack` to report whether a reference benchmark exists, is unavailable, or was skipped.
- Update task-pack README to tell Kernel Engineer whether speedup can be computed against linked original.

Notes:

- Snapshot fallback is correctness-only. It must not be presented as performance baseline.
- Capture-time original call timing is advisory only. It must not replace linked-original benchmark timing when `original_impl.py` can replay.
- If linked original cannot replay because the original target is an instance method or depends on framework-owned state, Framework Engineer may later add framework-side timing capture, but that is separate from task-pack-local benchmark.

## 2. Split Upper-level Interface Into Multiple Optimization Targets

Current state:

- Phase 1.2 supports a basic multi-target launcher.
- User provides `targets = [...]`, and each target independently generates one task pack.
- Framework Engineer does not yet infer target list from an upper-level module/backend/interface.

Problem:

- Real optimization requests are often phrased at a higher level, such as `linear_attention.extend`.
- That upper-level interface may call multiple core functions/kernels.
- Some targets should remain single-kernel task packs; others may later be candidates for fusion or merged task packs.

Planned work:

- Add an analysis feature that accepts:

  ```python
  upper_target_file = "/path/to/file.py"
  upper_target_line = 123
  ```

- Resolve the upper-level interface by AST.
- Inspect direct and selected nested calls inside that interface.
- Classify candidate optimization targets:

  - Python-visible free functions.
  - Python-visible class/static methods.
  - Triton/CUDA/CuTe DSL launch wrappers.
  - Calls that are not suitable for Phase 1 snapshot task packs.

- Produce a target proposal report:

  ```text
  docs/target_decomposition_report.md
  docs/target_decomposition_report.json
  ```

- Generate a suggested `targets = [...]` config patch rather than silently modifying user config.
- Keep the first version conservative: only propose targets; do not perform fusion planning or task-pack merge.

Out of scope for first version:

- Automatic fusion planner.
- Splitting a captured task pack into smaller task packs.
- Merging multiple task packs into one fused task pack.
- Proving that a target is performance-critical without profiling data.

Notes:

- This feature must not violate the role boundary: Framework Engineer can identify Python-visible optimization target candidates and generate task packs, but it should not decide low-level fusion strategy alone.
- Fusion planning can be added later as a Kernel Engineer capability or a separate planner role.

## 3. Integrate Optimized Candidate Back Into Framework

Current state:

- Framework Engineer can generate task packs for Kernel Engineer.
- Kernel Engineer is expected to modify `candidate_impl.py` / `kernel_sources/` and eventually deliver a `KernelDeliveryPackage`.
- Phase 1.2 does not yet consume the optimized delivery or patch the original framework.

Problem:

- A faster task-pack candidate is not useful until it is wired back into the serving framework.
- Integration may require feature flags, fallback paths, layout/workspace/metadata preparation, and e2e correctness/performance validation.
- Some Kernel Engineer deliveries may also include a `FrameworkChangeRequest`, such as weight prepacking, contiguous layout requirements, workspace allocation, metadata precompute, or prefill/decode path split.

Planned work:

- Define a delivery intake command or workflow:

  ```bash
  python -m framework_engineer.cli review-kernel-delivery \
    --task-pack <task_pack> \
    --delivery <kernel_delivery_package>
  ```

- Validate delivery contents:

  - candidate implementation path or compiled artifact.
  - supported dtype/shape/layout.
  - correctness results.
  - benchmark/profile evidence.
  - unsupported cases and fallback requirements.
  - whether a `FrameworkChangeRequest` is included.

- Generate an integration plan:

  ```text
  docs/framework_integration_plan.md
  docs/framework_integration_plan.json
  ```

- Patch framework code behind a feature flag where possible:

  ```text
  enable_<task_id>_optimized_kernel
  ```

- Keep a fallback to the original framework path for unsupported shapes/dtypes/layouts or runtime errors.
- Run task-pack correctness again after integration artifact selection.
- Run framework-level e2e validation:

  - workload correctness / output consistency.
  - latency/throughput comparison against saved baseline.
  - precision tolerance report.
  - fallback coverage report.

- Produce final integration report:

  ```text
  docs/framework_integration_report.md
  docs/framework_integration_report.json
  docs/e2e_verification_report.md
  ```

Out of scope for first version:

- Fully automatic framework refactor when the delivery requires broad model/backend changes.
- Automatically accepting invasive `FrameworkChangeRequest` items.
- Multi-target fusion integration unless a fusion planner/task-pack merge flow exists.

Notes:

- Framework Engineer owns integration and e2e validation; Kernel Engineer owns the optimized candidate and kernel-level evidence.
- If single-kernel benchmark improves but e2e does not improve, the integration report must explain likely causes: target not hot enough, launch/sync overhead, memory bandwidth bottleneck, fallback too frequent, framework glue overhead, or benchmark mismatch.
- Integration should be reversible and guarded by a feature flag until e2e validation is stable.
