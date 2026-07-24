# Skill: Task Pack Optimization Protocol

此 skill 定义 Kernel Engineer 如何处理 Framework Engineer 交付的 Phase 1 task pack。

## 输入目录

```text
task_pack/
  README.md
  validate_task_pack.py
  task/
    task.yaml
    shape_list.json
    env_manifest.yaml
    snapshots/
    kernel_source_package/
    kernel_translate/
    kernel_engineer_ws/
    original_impl.py
    reference_impl.py
    candidate_impl.py
    correctness_test.py
    benchmark.py
    scripts/
```

## 第一动作

1. 从外层目录运行 `python validate_task_pack.py`。
2. 读 `task/task.yaml`，确认 ABI、目标和写权限合同。
3. 读 `task/shape_list.json` 与 `task/snapshots/manifest.json`。
4. 只读参考 `task/kernel_translate/` 和 `task/kernel_source_package/`。
5. 读 `task/env_manifest.yaml`。
6. 运行 `python task/scripts/run_correctness.py`。
7. 运行 `python task/scripts/run_benchmark.py`。

linked original 不可用时，可以执行：

```bash
python task/scripts/run_benchmark.py --target candidate
```

若 correctness 或 candidate-only benchmark 不能运行，不得修改 harness、snapshot 或
shape list；输出 task acceptance review 给 Framework Engineer。

## 写权限

Kernel Engineer 可以：

- 修改 `task/candidate_impl.py`；
- 在 `task/kernel_engineer_ws/` 内创建、修改、删除实现和构建产物；
- 在 `task/kernel_engineer_ws/iteration_log.md` 记录迭代。

Kernel Engineer 不得修改：

- `task/kernel_translate/`、`task/kernel_source_package/`；
- `task/snapshots/`、`task/shape_list.json`、`task/snapshot_runtime.py`；
- `task/original_impl.py`、`task/reference_impl.py`；
- `task/correctness_test.py`、`task/benchmark.py`、`task/scripts/`；
- `task/env_probe/`、`task/task.yaml`、根目录 README 和验证器；
- `task/docs/` 下的 Framework Engineer 交付证据；
- tolerance 或 timing rules。

## 迭代规则

每轮只做一个明确方向：

- 修改 candidate 或 `kernel_engineer_ws/` 中的实现。
- 跑 correctness。
- correctness 通过后跑 benchmark。
- 对 hot group 跑 NCU。
- 把假设、改动、结果和下一步追加到 iteration log。

## 收敛与交付

允许在达标、平台期、接近硬件上限或需要 FrameworkChangeRequest 时停止。最终交付：

- 修改后的 candidate 和 `kernel_engineer_ws/`；
- `benchmark_report.md`；
- `kernel_constraints.md`；
- `kernel_delivery_package.md`；
- 如需要，`framework_change_request.yaml`。
