# Phase 1 Flow: Qwen3.5 Linear Attention Optimization

本文是 Phase 1 的主流程索引。目标是把“Qwen3.5 linear attention 性能优化”串成一条可执行协作链路：Framework Engineer 生成可信 snapshot task pack，Kernel Engineer 在 task pack 内优化，Framework Engineer 最后接回框架验收。

## 0. 输入与边界

用户提供：

- 可直接运行的 SGLang 启动命令。
- 可直接运行的 workload/test 命令。
- 明确优化目标：至少到 module forward，例如 `RadixLinearAttention.forward`；更推荐到具体 kernel/core 接口。

本阶段不做：

- 自动 top-K 热点发现。
- 多硬件插件化。
- Kernel Engineer 自行安装环境。
- Kernel Engineer 自行编造框架输入。

如果启动命令、workload 或优化目标不满足要求，Framework Engineer 应中断任务并要求用户修复。

## 1. Framework Engineer 启动任务与 Baseline Gate

使用文件：

- Prompt: `framework_engineer/prompts/framework_engineer.md`
- Skill: `framework_engineer/skills/qwen35_linear_core_task_pack.md`
- CLI: `python -m kernel_agent.framework_engineer.cli scaffold-task-pack`
- CLI: `python -m kernel_agent.framework_engineer.cli run-baseline`

工作：

- 创建 task pack 初始目录。
- 运行用户服务和 workload。
- 记录 baseline 和命令输出。
- 如果服务或 workload 不可用，任务终止。

输出：

- `task_pack/docs/baseline_run_report.md`
- `task_pack/docs/baseline_result.json`

## 2. 阅读代码并确认优化目标

使用文件：

- 主设计: `phase_1_qwen35_linear_core.md`
- Snapshot 设计: `phase_1_snapshot_design.md`
- Skill: `framework_engineer/skills/qwen35_linear_core_task_pack.md`
- Template: `framework_engineer/templates/qwen35_operator_breakdown.md`

工作：

- 如果用户目标是 module forward，拆解成一个或多个 tensor/scalar ABI 接口。
- 如果用户目标已是接口列表，确认其源码位置、调用签名和 mutable inputs。
- Qwen3.5/GDN 首个默认目标仍是 `candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc)`。

输出：

- `task_pack/docs/operator_breakdown.md`
- 目标接口列表和插桩参数。

## 3. 验证目标接口被调用

使用文件：

- CLI: `python -m kernel_agent.framework_engineer.cli probe-target-calls`

工作：

- 使用 non-cudagraph 服务/workload。
- 对目标接口临时加装饰器。
- 运行 workload，记录调用次数。
- 工具结束后恢复源码。

输出：

- `task_pack/docs/target_call_probe_report.md`
- `task_pack/docs/target_call_probe.jsonl`

如果目标接口没有被调用，该接口不能作为当前 workload 的有效优化目标。

## 4. Capture Raw Snapshots

使用文件：

- Snapshot 设计: `phase_1_snapshot_design.md`
- CLI: `python -m kernel_agent.framework_engineer.cli capture-snapshots`

工作：

- 对有效目标接口临时加 snapshot 装饰器。
- 捕获 `pre_inputs.pt`、`post_inputs.pt`、`outputs.pt`、`meta.json`。
- 对 mutable inputs，例如 `kwargs.ssm_states`，同时保存调用前和调用后状态。
- 工具结束后恢复源码。

输出：

```text
task_pack/snapshots/raw/
  call_000001/
    meta.json
    pre_inputs.pt
    post_inputs.pt
    outputs.pt
```

## 5. Select Snapshots

使用文件：

- CLI: `python -m kernel_agent.framework_engineer.cli select-snapshots`

工作：

- 使用 `shape_hash` 做粗分组。
- 使用 `semantic_hash` 区分 shape 相同但语义不同的 case。
- 使用 `value_hash` 做重复样本去重和完整性检查。
- 按调用频度选择 required cases。
- 派生 `shape_list.json` 摘要；它不是 replay 来源。

输出：

```text
task_pack/snapshots/manifest.json
task_pack/snapshots/selected/
task_pack/shape_list.json
task_pack/docs/snapshot_selection_report.md
```

## 6. Generate Snapshot Harness

使用文件：

- Skill: `framework_engineer/skills/ut_construction.md`
- CLI: `python -m kernel_agent.framework_engineer.cli generate-harness`

工作：

- 生成 task pack 内自包含 `snapshot_runtime.py`。
- 生成 `reference_impl.py`、`candidate_impl.py`、`correctness_test.py`、`benchmark.py`。
- `candidate_impl.py` 初始直接调用 reference，确保 task pack 初始 correctness pass。
- correctness 比较 outputs 和 mutable post-state。
- benchmark 每轮从 pre-state 恢复 mutable inputs，reset 不计入 timed region。

输出：

```text
task_pack/snapshot_runtime.py
task_pack/reference_impl.py
task_pack/candidate_impl.py
task_pack/correctness_test.py
task_pack/benchmark.py
task_pack/scripts/run_correctness.sh
task_pack/scripts/run_benchmark.sh
task_pack/scripts/run_ncu.sh
```

## 7. Probe Development Environment

使用文件：

- Template: `framework_engineer/templates/env_manifest.yaml`
- Probe templates: `framework_engineer/templates/probe_*.py`
- CLI: `python -m kernel_agent.framework_engineer.cli probe-env`

工作：

- 探测 Python、PyTorch、GPU、Triton、CuTe DSL、CUDA extension、NCU。
- 区分工具是否可用，以及是否可用于当前 task。

输出：

- `task_pack/env_manifest.yaml`
- `task_pack/docs/env_probe_result.json`

## 8. Validate And Deliver Task Pack

使用文件：

- CLI: `python -m kernel_agent.framework_engineer.cli validate-task-pack`

工作：

- 检查 task pack 文件完整性。
- 检查 selected snapshots 是否存在。
- 可选运行 correctness smoke 和 benchmark smoke。
- 若验证失败，不交给 Kernel Engineer。

输出：

- `task_pack/docs/task_pack_validation_report.json`
- 一个完整 `task_pack/`。

## 9. Kernel Engineer 接收与验收 Task Pack

使用文件：

- Prompt: `kernel_agent/prompts/kernel_engineer.md`
- Skill: `kernel_agent/skills/task_pack_optimization_protocol.md`
- Skill: `kernel_agent/skills/task_triage.md`
- Template: `kernel_agent/templates/task_acceptance_review.md`

工作：

- 读取 `task.yaml`、`snapshots/manifest.json`、`shape_list.json`、`env_manifest.yaml`。
- 运行 correctness 和 benchmark。
- 判断 task pack 是否可优化。
- 如果缺信息，返回 `task_acceptance_review.md`，不自行修改框架输入或测试逻辑。

输出：

- `task_acceptance_review.md`，结论为 `ACCEPT` 或 `NEEDS_MORE_INFO`。

## 10. Kernel Engineer 选择实现路径

使用文件：

- Skill: `kernel_agent/skills/triton_cuda_codegen.md`
- Skill: `kernel_agent/skills/nvidia_ncu_analysis.md`
- Reference: `ref/AKO4ALL/SKILL.md`
- Reference: `ref/KernelAgent/README.md`
- Reference: `ref/autokernel/program.md`
- Reference: `ref/kernel-design-agents/docs/agent-flow.md`

工作：

- 基于 `env_manifest.yaml` 选择实现路径。
- 首选 Triton 或 CuTe DSL。
- CUDA extension / CUTLASS 需要 benchmark/profile 证据触发。
- 先做最小正确实现，再做性能优化。

输出：

- 初始实现计划。
- `docs/iteration_log.md` 初始化。

## 11. Kernel Engineer 优化闭环

使用文件：

- Skill: `kernel_agent/skills/task_pack_optimization_protocol.md`
- Skill: `kernel_agent/skills/kernel_optimization_loop.md`
- Skill: `kernel_agent/skills/nvidia_ncu_analysis.md`
- Skill: `kernel_agent/skills/triton_cuda_codegen.md`
- Template: `kernel_agent/templates/iteration_log.md`
- Template: `kernel_agent/templates/benchmark_report.md`

工作循环：

1. 修改 `candidate_impl.py` 或 `kernel_sources/`。
2. 运行 `bash scripts/run_correctness.sh`。
3. correctness 通过后运行 `bash scripts/run_benchmark.sh`。
4. 对 hot case 运行 `bash scripts/run_ncu.sh <case_id>`。
5. 记录假设、改动、结果、是否保留。

输出：

- 优化后的 `candidate_impl.py` / `kernel_sources/`
- `docs/iteration_log.md`
- `benchmark_report.md`

## 12. Kernel Engineer 交付

使用文件：

- Template: `kernel_agent/templates/kernel_delivery_package.md`
- Template: `kernel_agent/templates/kernel_constraints.md`
- Template: `kernel_agent/templates/benchmark_report.md`
- Optional Skill: `kernel_agent/skills/framework_feedback.md`
- Optional Template: `kernel_agent/templates/framework_change_request.yaml`

输出：

- `kernel_delivery_package.md`
- `kernel_constraints.md`
- `benchmark_report.md`
- 可选 `framework_change_request.yaml`

## 13. Framework Engineer 接入与 E2E 验收

使用文件：

- Prompt: `framework_engineer/prompts/framework_engineer.md`
- Skill: `framework_engineer/skills/framework_integration.md`
- Skill: `framework_engineer/skills/e2e_accuracy_perf_validation.md`
- Template: `framework_engineer/templates/integration_plan.md`
- Template: `framework_engineer/templates/e2e_verification_report.md`

工作：

- 根据 delivery package 写接入计划。
- 接入 SGLang backend，保留 fallback。
- 运行用户原始 workload。
- 对比 baseline 与 candidate 的 e2e 性能和精度。
- 若 micro benchmark 快但 e2e 无收益，记录原因：热点误判、接入开销、调度/cache/同步瓶颈、shape 不匹配等。

输出：

- `integration_plan.md`
- `e2e_verification_report.md`

## Active Files

Phase 1 当前主链路使用：

```text
phase_1_details.md
phase_1_qwen35_linear_core.md
phase_1_flow.md
phase_1_usage_and_tooling.md
phase_1_snapshot_design.md
framework_engineer/prompts/framework_engineer.md
framework_engineer/skills/qwen35_linear_core_task_pack.md
framework_engineer/skills/ut_construction.md
framework_engineer/skills/framework_integration.md
framework_engineer/skills/e2e_accuracy_perf_validation.md
framework_engineer/snapshot/
framework_engineer/cli.py
framework_engineer/templates/*
kernel_agent/prompts/kernel_engineer.md
kernel_agent/skills/*
kernel_agent/templates/*
ref/
```

## Backup Files

Phase 1 主链路暂时不用的通用/后续阶段资产已移到：

```text
backup/
```

这些文件不是删除，只是暂时从 Phase 1 主路径中移开，后续做自动热点发现、多硬件插件化、独立审计或通用 KernelRequestSpec 时可以再恢复。

