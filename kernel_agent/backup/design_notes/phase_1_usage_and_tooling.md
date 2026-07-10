# Phase 1 Usage And Tooling Guide

本文说明用户如何实际使用 Phase 1 双角色 agent 流程，以及哪些步骤已经被工具化。

当前工程已经提供 Framework Engineer CLI：

```bash
python -m kernel_agent.framework_engineer.cli <subcommand>
```

md skill 仍然存在，但定位已经变成：告诉 agent 何时调用 CLI、如何解释 CLI 结果，而不是让模型手工完成确定性步骤。

## 1. 最小可用操作方式

推荐把 Phase 1 当成三个连续会话使用：

```text
Framework Engineer Agent
  -> 生成 snapshot task_pack/
Kernel Engineer Agent
  -> 在 task_pack/ 内优化 candidate
Framework Engineer Agent
  -> 接入 SGLang 并做 e2e 验收
```

Kernel Engineer 不需要完整 SGLang 背景，只需要 Framework Engineer 交付的 task pack。

## 2. Framework Engineer 示例请求

```text
请作为 Framework Engineer Agent 工作。

请先阅读：
- kernel_agent/framework_engineer/prompts/framework_engineer.md
- kernel_agent/phase_1_flow.md
- kernel_agent/phase_1_qwen35_linear_core.md
- kernel_agent/phase_1_snapshot_design.md
- kernel_agent/framework_engineer/skills/qwen35_linear_core_task_pack.md
- kernel_agent/framework_engineer/skills/ut_construction.md

目标：
- 为 Qwen3.5 linear attention 的 GDN extend core kernel 生成 Phase 1 snapshot task_pack。

输入：
- SGLang 启动命令：<粘贴启动命令>
- workload 命令：<粘贴 workload 命令>
- 目标接口源码文件：<目标 Python 文件>
- 目标函数名：<例如 extend>
- 目标接口名：<例如 sglang...TritonGDNKernel.extend>
- mutable arg paths：<例如 kwargs.ssm_states>

输出目录：
- kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack
```

Framework Engineer 应执行的 CLI 顺序：

```bash
python -m kernel_agent.framework_engineer.cli scaffold-task-pack \
  --task-id qwen35_gdn_extend_core_h20_YYYYMMDD \
  --out kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack

python -m kernel_agent.framework_engineer.cli run-baseline \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack \
  --service-cmd "<SGLang 启动命令>" \
  --workload-cmd "<workload 命令>"

python -m kernel_agent.framework_engineer.cli probe-target-calls \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack \
  --service-cmd "<non-cudagraph SGLang 启动命令>" \
  --workload-cmd "<workload 命令>" \
  --target-file <目标 Python 文件> \
  --function-name <函数名> \
  --target-name <完整目标接口名> \
  --drop-first-arg

python -m kernel_agent.framework_engineer.cli capture-snapshots \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack \
  --service-cmd "<non-cudagraph SGLang 启动命令>" \
  --workload-cmd "<workload 命令>" \
  --target-file <目标 Python 文件> \
  --function-name <函数名> \
  --target-name <完整目标接口名> \
  --mutable-arg-path kwargs.ssm_states \
  --drop-first-arg

python -m kernel_agent.framework_engineer.cli select-snapshots \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack

python -m kernel_agent.framework_engineer.cli generate-harness \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack

python -m kernel_agent.framework_engineer.cli probe-env \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack

python -m kernel_agent.framework_engineer.cli validate-task-pack \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack \
  --run-correctness \
  --run-benchmark
```

如果当前没有可用 GPU/SGLang 环境，可以先只做离线结构检查：

```bash
python -m kernel_agent.framework_engineer.cli validate-task-pack \
  --task-pack kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack \
  --skip-env-check
```

最终 task pack 结构：

```text
task_pack/
  README.md
  task.yaml
  shape_list.json
  env_manifest.yaml
  snapshot_runtime.py
  snapshots/
    manifest.json
    selected/
  reference_impl.py
  candidate_impl.py
  correctness_test.py
  benchmark.py
  scripts/run_correctness.sh
  scripts/run_benchmark.sh
  scripts/run_ncu.sh
  docs/baseline_run_report.md
  docs/target_call_probe_report.md
  docs/snapshot_selection_report.md
```

## 3. Kernel Engineer 示例请求

```text
请作为 Kernel Engineer Agent 工作。

请先阅读：
- kernel_agent/kernel_agent/prompts/kernel_engineer.md
- kernel_agent/kernel_agent/skills/task_pack_optimization_protocol.md
- kernel_agent/kernel_agent/skills/task_triage.md
- kernel_agent/kernel_agent/skills/kernel_optimization_loop.md
- kernel_agent/kernel_agent/skills/triton_cuda_codegen.md
- kernel_agent/kernel_agent/skills/nvidia_ncu_analysis.md

任务目录：
- kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack

约束：
- 只允许修改 candidate_impl.py 或 kernel_sources/。
- 不允许修改 snapshots/、snapshot_runtime.py、reference_impl.py、correctness_test.py、benchmark.py 的统计规则。
- 如果发现需要 layout、workspace、metadata 预计算等框架支持，输出 framework_change_request.yaml。
```

Kernel Engineer 的日常循环：

```bash
cd kernel_agent/tasks/qwen35_gdn_extend_core_h20_YYYYMMDD/task_pack
bash scripts/run_correctness.sh
bash scripts/run_benchmark.sh
bash scripts/run_ncu.sh <case_id>
```

## 4. 工具化边界

已经实现为 CLI 的部分：

- `scaffold-task-pack`
- `run-baseline`
- `probe-target-calls`
- `capture-snapshots`
- `select-snapshots`
- `generate-harness`
- `probe-env`
- `validate-task-pack`

CLI 和 snapshot 测试说明见：

- `kernel_agent/framework_engineer/tests/README.md`

仍然应该由 agent 判断的部分：

- 选择 prefill、decode 还是其他 kernel 边界。
- 判断目标接口是否足够接近最终 GPU kernel ABI。
- 判断 selected snapshots 是否覆盖真实 workload 的关键 case。
- 根据 KernelDeliveryPackage 做 SGLang 接入和 e2e 验收。
- 端到端无收益时分析原因。

## 5. 当前 md skill 的定位

- `prompts/*.md`：角色边界，启动 agent 时读。
- `skills/*.md`：操作规程，agent 决策时读。
- `framework_engineer/cli.py`：Framework Engineer 可调用命令入口。
- `framework_engineer/snapshot/`：snapshot 数据模型、hash、store、recorder、selector、harness builder。
- `templates/*`：task pack 初始模板。
- `phase_1_flow.md`：主流程索引。
- `phase_1_usage_and_tooling.md`：用户操作手册。
- `phase_1_snapshot_design.md`：snapshot 数据合同、hash、mutable input correctness、benchmark replay 标准。
