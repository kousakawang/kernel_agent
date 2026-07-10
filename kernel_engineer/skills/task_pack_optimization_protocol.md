# Skill: Task Pack Optimization Protocol

此 skill 定义 Kernel Engineer 如何处理 Framework Engineer 交付的 Phase 1 task pack。

## 输入目录

Kernel Engineer 只以 task pack 为原始输入：

```text
task_pack/
  task.yaml
  shape_list.json
  env_manifest.yaml
  snapshots/
    manifest.json
    selected/
  original_source/
    manifest.json
    <copied_target_source>
  original_impl.py
  reference_impl.py
  candidate_impl.py
  correctness_test.py
  benchmark.py
  scripts/
```

## 第一动作

1. 读 `task.yaml`，确认 ABI、目标、禁止修改项。
2. 读 `shape_list.json`，确认 required/hot shape。
3. 读 `snapshots/manifest.json`，确认 selected group/sample。
4. 读 `original_source/manifest.json` 和原始源码参考。
5. 读 `env_manifest.yaml`，确认可用实现路径。
6. 运行 `bash scripts/run_correctness.sh`。
7. 运行 `bash scripts/run_benchmark.sh`。

如果 linked original 不可用，`run_benchmark.sh` 可能记录 reference unavailable；
Kernel Engineer 替换 candidate 后可以用 `TARGET=candidate bash scripts/run_benchmark.sh`。
如果 correctness 或 candidate-only benchmark smoke 不能运行，不要修 benchmark、snapshot 或 shape list；
输出 `task_acceptance_review.md` 给 Framework Engineer。

## 允许修改

可以修改：

- `candidate_impl.py`
- `kernel_sources/`
- `docs/iteration_log.md`
- 自己生成的报告

不能修改：

- `snapshots/`
- `shape_list.json`
- `snapshot_runtime.py`
- `original_source/`
- `original_impl.py`
- `reference_impl.py`
- `correctness_test.py`
- `benchmark.py`
- tolerance
- timing rules

## 迭代规则

每轮只做一个明确方向：

- 实现或修改 candidate。
- 跑 correctness。
- correctness 过后跑 benchmark。
- 对 hot case 跑 NCU，若 ncu 可用。
- 记录假设、改动、结果、下一步。

## 收敛规则

允许停止的条件：

- 达到 `task.yaml` 的 performance target。
- 最近 3 个有效迭代提升小于 3%，且重新 profile 后没有新的可行方向。
- 至少尝试 3 类不同优化方向仍无收益。
- 证明瓶颈接近硬件上限或 launch/timing 下限。
- 需要 FrameworkChangeRequest 才能继续。

## 交付

最终交付：

- 修改后的 candidate 实现。
- `benchmark_report.md`
- `kernel_constraints.md`
- `kernel_delivery_package.md`
- 如需要，`framework_change_request.yaml`
