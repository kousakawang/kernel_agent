# Skill: 编写 KernelRequestSpec

`KernelRequestSpec` 是 Framework Engineer Agent 交给 Kernel Agent 的需求合同。它要避免“帮我优化这个算子”这类模糊描述，让算子侧可以只基于 spec 和 UT 开始工作。

## 必填内容

- `task_id`：稳定任务 ID，建议包含模型、算子、阶段和日期。
- `model`：模型名称、规模、关键 config、模型路径可选。
- `framework`：框架名称、commit/version、目标源码入口。
- `target_hardware`：GPU/NPU 型号、数量、driver/CUDA/toolchain。
- `operator_name`：明确到子路径，例如 `qwen35_linear_attention_prefill`。
- `operator_semantics`：输入到输出的数学语义和关键分支。
- `inputs` / `outputs`：每个 tensor 的 dtype、shape、layout、stride、contiguity、device。
- `shape_ranges`：真实 workload 的 shape 范围和优先级。
- `golden_impl`：PyTorch golden 或现有可靠实现。
- `unit_test_entry`：UT 文件和命令。
- `baseline_perf`：当前实现的 micro benchmark 和采集方式。
- `target_perf`：目标性能，最好包含最低可接受收益。
- `accuracy_tolerance`：绝对/相对误差、dtype、随机种子、容忍原因。
- `allowed_framework_changes`：允许/禁止 kernel 侧要求框架配合的范围。
- `e2e_context`：该算子在真实模型中的占比、调用频率、prefill/decode 场景。
- `known_constraints`：不支持或暂不覆盖的场景。

## 质量标准

一份好的 spec 应该回答：

- 这个算子在模型里为什么重要？
- Kernel Agent 是否能只看 spec + UT 就知道语义？
- shape/layout 是否覆盖真实热路径，而不是只覆盖玩具输入？
- 如果 kernel 更快，端到端理论收益上限是多少？
- 如果需要框架改造，哪些边界允许改变？

## 常见失败

- 只写模型层名字，没有写具体输入输出语义。
- 没有说明 prefill/decode 差异。
- shape 只给一个样例，没有给范围和优先级。
- 只给 latency 数字，没有说明 warmup、repeat、同步、统计口径。
- 精度容忍没有按 dtype 和累积误差解释。
- 允许框架改变范围不清，导致 kernel 侧提出不可接入的方案。
