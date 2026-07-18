# Framework Engineer Phase 1 CLI/Skill Design

此文件记录 Phase 1 Framework Engineer 需要的可执行能力。旧编号 skill 已收敛为 CLI subcommands。

## 1. `run-baseline`

目的：

- 运行用户提供的服务启动命令和 workload。
- 提供 baseline 测试结果。

实现：

- 启动服务。
- 可选服务存活探测。
- 执行 workload。
- 记录 stdout/stderr、return code、elapsed time。
- 杀死服务进程。

输出：

- `docs/baseline_run_report.md`
- `docs/baseline_result.json`

## 2. `probe-target-calls`

目的：

- 以 non-cudagraph 服务/workload 验证目标接口确实被调用。

实现：

- 临时给目标 Python 函数加装饰器。
- 运行服务和 workload。
- 记录调用次数和调用日志。
- 自动恢复源码。

输出：

- `docs/target_call_probe_report.md`
- `docs/target_call_probe.jsonl`

## 3. `capture-snapshots`

目的：

- 收集待优化接口的 raw snapshots，用于生成 UT 和 benchmark。

实现：

- 临时给目标 Python 函数加 snapshot 装饰器。
- 捕获 `pre_inputs.pt`、`post_inputs.pt`、`outputs.pt`、`meta.json`。
- 支持 `mutable_arg_paths`，例如 `kwargs.ssm_states`。
- 自动恢复源码。

输出：

- `snapshots/raw/call_*/`

## 4. `select-snapshots`

目的：

- 对 raw snapshots 收敛，保留核心优化 case。

实现：

- 按 `semantic_hash` 分组。
- 按调用频率排序。
- 每组选择代表 case。
- 输出 selected snapshots 和 `shape_list.json` 摘要。

输出：

- `snapshots/manifest.json`
- `snapshots/selected/*`
- `shape_list.json`
- `docs/snapshot_selection_report.md`

## 5. `generate-harness`

目的：

- 从 selected snapshots 生成 task pack 的 UT 和 benchmark。

实现：

- 生成 `snapshot_runtime.py`。
- 生成 `reference_impl.py`、`candidate_impl.py`、`correctness_test.py`、`benchmark.py`。
- 生成 scripts。
- `candidate_impl.py` 初始调用 reference，确保初始 correctness pass。

## 6. `probe-env`

目的：

- 确认可用开发环境。

实现：

- 执行 Triton、CuTe DSL、CUDA extension、NCU probe。
- 输出标准 `env_manifest.yaml`。

## 7. `validate-task-pack`

目的：

- 验证 task pack 的有效性。

实现：

- 检查 required files。
- 检查 selected snapshots。
- 可选运行 correctness smoke。
- 可选运行 benchmark smoke。

输出：

- `docs/task_pack_validation_report.json`

