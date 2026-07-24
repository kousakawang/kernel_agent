# Kernel Agent Phase 1.2: Config-driven Snapshot Task Pack

本文记录当前可落地版本的工程边界、使用方式和 task pack 合同。

## 1. 分角色设计

Phase 1.2 保留双角色边界：

- `framework_engineer/`：把真实框架 workload 中的待优化接口解耦成可 replay 的 task pack。
- `kernel_engineer/`：只消费 task pack，在不理解原框架对象的前提下优化 candidate。

Framework Engineer 不实现高性能 kernel。它负责真实输入、snapshot、UT、benchmark、环境合同和 task pack 验证。

Kernel Engineer 不构造框架输入。它只修改 `task/candidate_impl.py`、
`task/kernel_engineer_ws/` 和 iteration log，运行 task pack 内置的 Python correctness、
benchmark 和 profile 命令。

双方唯一交互物是：

```text
task_pack/
```

## 2. Active 工程结构

```text
framework_engineer/
  cli.py
  configs/phase1_targets.example.py
  prompts/
  skills/
  snapshot/
  templates/
  tests/

kernel_engineer/
  prompts/
  skills/
  templates/

kernel_agent/
  README.md
  backup/
```

`kernel_agent/backup/` 默认是历史设计、旧模板、Qwen 专用示例、Phase 2+ 设想和调研 repo，不参与 Phase 1.2 主流程。

## 3. Framework Engineer 使用模式

用户填写 Python 配置文件。

配置模板：

```text
framework_engineer/configs/phase1_targets.example.py
```

Framework Engineer 支持三种模式：

- CI / toy 测试：运行 `python -m unittest discover framework_engineer/tests`，不使用真实用户配置。
- Agent 模式：正式工作方式。先 `validate-config`，再按细粒度 CLI 分步执行并检查每一步产物。
- Batch 模式：使用 `run-phase1` 一键跑完整链路，适合稳定复跑、CI smoke 或用户明确要求自动跑完。

Batch 命令：

```bash
python -m framework_engineer.cli validate-config --config <config.py>
python -m framework_engineer.cli run-phase1 --config <config.py>
```

Agent 模式应参考：

```text
framework_engineer/prompts/start_phase1_validation.md
```

正式验证不要求只执行一个大命令。`run-phase1` 是批处理捷径，不是替代 agent 判断的唯一入口。

推荐多目标配置：

```python
task_group_id = "linear_attention_targets_h20"
output_root = "/tmp/linear_attention_targets"

service_cmd = "python -m sglang.launch_server ..."
workload_cmd = "python /path/to/run_workload.py ..."

forward_boundary_file = "/path/to/model_or_runner.py"
forward_boundary_line = 123

targets = [
    {"task_id": "target_1", "target_file": "/path/to/file.py", "target_line": 100},
    {"task_id": "target_2", "target_file": "/path/to/file.py", "target_line": 200},
]
```

单目标兼容配置：

```python
task_id = "single_target"
task_pack = "/tmp/single_target_task_pack"
target_file = "/path/to/file.py"
target_line = 100
```

## 4. Agent / Batch 执行流程

无论分步执行还是 batch，一条完整链路都包含：

1. 为每个 target 初始化独立 task pack。
2. group-level baseline，最多运行一次。
3. 解析每个 target 和共享 forward boundary。
4. 对每个 target 运行 `probe-target-calls`。
5. 对每个 target 运行 `capture-snapshots`。
6. 对每个 target 运行 `select-snapshots`。
7. 对每个 target 运行 `generate-harness`。
8. 可选对每个 target 运行 `probe-env`。
9. 对每个 target 运行 `validate-task-pack --run-correctness`。
10. 写出 group-level 报告。

输出：

```text
<output_root>/multi_target_report.json
<output_root>/multi_target_report.md
<output_root>/<target_task_id>/
```

本轮多目标只是 launcher：每个 target 生成独立 task pack。不做上层接口自动拆解、fusion planner、task_pack merge 或 cross-target benchmark。

## 5. Snapshot 策略

当前 snapshot capture 使用：

- forward boundary decorator 生成 `forward_id`。
- target decorator 读取当前 `forward_id`。
- `group_key` 只看 target、tree 结构、shape、dtype、stride/layout、primitive、None/presence。
- 每个 group 记录 `total_hit_count` / `forward_hit_count`。
- 每个 group 保存 bounded multi-sample，默认最多 8 个 sample，同一 forward window 内最多 3 个 sample。
- selector 按高频 group 选择少量 required groups。

Phase 1.2 不实现 kernel-aware semantic hash。Framework Engineer 不需要判断哪些 tensor value 会影响 kernel 控制流；Kernel Engineer 可以通过同一 group 的多个真实 sample 理解参数分布。

## 6. 自动 Mutable 输入检测

用户不需要填写 mutable 输入路径。

capture 会保存：

```text
pre_inputs.pt
post_inputs.pt
outputs.pt
```

recorder 会递归 diff `pre_inputs` 和 `post_inputs`：

- tensor：metadata + value hash。
- primitive：直接比较值。
- list/tuple/dict：递归比较。
- 结构变化或不可比较类型：写 warning。

自动检测结果写入：

```text
sample_meta["mutation"]["mutable_arg_paths"]
```

correctness 会比较 outputs，并按 sample meta 比较 candidate 运行后的 mutable post-state。

## 7. Task Pack 结构

每个 task pack 至少包含：

```text
README.md
validate_task_pack.py
task/
  task.yaml
  shape_list.json
  env_manifest.yaml
  snapshot_runtime.py
  snapshots/
    manifest.json
    selected/
  kernel_source_package/
  kernel_translate/
  kernel_engineer_ws/
  original_impl.py
  reference_impl.py
  candidate_impl.py
  correctness_test.py
  benchmark.py
  scripts/
    run_correctness.py
    run_benchmark.py
    run_ncu.py
  docs/
    task_pack_validation_report.json
```

关键含义：

- `snapshots/selected/` 是唯一 replay 来源。
- `shape_list.json` 只是摘要索引，不用于随机造输入。
- `snapshot_runtime.py` 是 task pack 内复制的最小 replay runtime。
- `kernel_source_package/` 和 `kernel_translate/` 是只读源码/翻译参考。
- `original_impl.py` 尝试 linked import 原框架接口，能跑则作为 benchmark reference。
- `reference_impl.py` 提供 linked reference 和 snapshot-golden fallback。
- `candidate_impl.py` 是 Kernel Engineer 的实现入口。
- `correctness_test.py` 固定 correctness 规则。
- `benchmark.py` 固定 timing/reset 规则。
- `env_manifest.yaml` 描述当前开发/分析环境，`probe-env` 可填充具体能力。

## 8. Kernel Engineer 使用方式

Kernel Engineer 接收到某个 task pack 后：

```bash
cd <task_pack>
python validate_task_pack.py
python task/scripts/run_correctness.py
python task/scripts/run_benchmark.py
```

如果 linked original 不可用：

```bash
python task/scripts/run_benchmark.py --target candidate
```

允许修改：

- `task/candidate_impl.py`
- `task/kernel_engineer_ws/`
- `task/kernel_engineer_ws/iteration_log.md`

禁止修改：

- `task/snapshots/`
- `task/snapshot_runtime.py`
- `task/shape_list.json`
- `task/kernel_translate/`
- `task/kernel_source_package/`
- `task/original_impl.py`
- `task/reference_impl.py`
- `task/correctness_test.py`
- `task/benchmark.py`
- correctness tolerance
- benchmark timing/reset rules

## 9. 完成标准

Framework Engineer 交付完成的标准是：

- `run-phase1` 中每个 target 状态为 `ok`。
- 每个 task pack 的 `validate-task-pack` 通过。
- `multi_target_report.json/md` 已生成。

`validate-task-pack` 会检查：

- 必需文件存在。
- selected snapshot 完整。
- workspace 目录和读写权限合同完整。
- correctness smoke 通过。
- env check 执行或明确 skipped。
- benchmark smoke 执行或明确 skipped。

## 10. 后续增强

待加强内容：

- 从上层接口自动拆解出 target 列表。
- 多 target fusion planner。
- task_pack merge。
- 接收 KernelDeliveryPackage / FrameworkChangeRequest。
- 接入优化后算子并做端到端性能/精度验收。
- 更多开发环境和硬件插件。
- 独立 audit 与 anti-cheat 复测。
