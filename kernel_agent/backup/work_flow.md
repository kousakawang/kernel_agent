# Workflow: 优化 Qwen3.5 Linear Attention

本文用 “优化 Qwen3.5 linear attention” 作为 Phase 1 示例，说明 Framework Engineer Agent 和 Kernel Agent 如何协作。

第一阶段只支持：框架侧手工/半自动给出 `KernelRequestSpec + UnitTestHarness`，Kernel Agent 在该合同下完成算子实现和优化闭环。不要求自动 profile 全模型，不要求自动修改 SGLang，不要求多硬件。

## 1. 背景

目标模型是 SGLang 中的 Qwen3.5 / hybrid linear attention 场景。首期默认硬件是 NVIDIA H20，工具链是 Triton/CuTe DSL/CUDA + Nsight Compute/Nsight Systems。

参考入口：

- 模型侧 linear attention 模块：`sglang/python/sglang/srt/models/bailing_moe_linear.py`
- backend/kernel 侧：`sglang/python/sglang/srt/layers/attention/linear/`
- 相关服务启动和 workload：当前仓库里的 Qwen3.5 调研、server test、offline benchmark 文档和脚本。

初始目标不是让 agent 端到端自动改完整模型，而是选择一个最小可优化目标，例如 linear attention 的 prefill 子路径或 decode 子路径，先形成稳定的 kernel 任务合同。

## 2. Framework Engineer Agent 流程

### 2.1 读取模型和当前 backend

Framework Engineer Agent 先阅读模型入口和 backend：

- 确认 `BailingMoELinearAttention.forward` 中 qkv projection、q/k norm、rope、linear attention backend、gate/norm/out projection 的边界。
- 确认 `sglang/srt/layers/attention/linear/` 下当前 backend 的输入输出、metadata 和 prefill/decode 分支。
- 判断要优化的是完整 linear attention block，还是其中更小的 backend kernel。

第一阶段推荐选择更小的 backend kernel，不把 qkv projection、gate projection、out projection 一起塞给 Kernel Agent。

### 2.2 定位是否值得优化

Framework Engineer Agent 使用已有服务启动命令和 workload，手工或半自动确认 linear attention 是否是热点。

可用证据包括：

- SGLang 服务日志中的 prefill/decode token、batch、cuda graph、cache 状态。
- torch profiler 或 nsys 中 linear attention 相关 kernel 的 GPU time。
- ncu 中当前 backend 的带宽、occupancy、stall 指标。

如果 linear attention 不是热点，或者 e2e 瓶颈主要是调度/cache/通信，则不应生成 kernel 任务，而应在 `hotspot_report.md` 中标记为不适合。

### 2.3 选择最小可优化目标

示例目标：

```text
operator_name: qwen35_linear_attention_prefill
scope: q/k/v/metadata -> hidden
excluded: qkv projection, qk norm, rope, gate projection, output projection
```

如果 decode 和 prefill 的 shape、metadata、cache 行为差异明显，应拆成两个 `KernelRequestSpec`：

- `qwen35_linear_attention_prefill`
- `qwen35_linear_attention_decode`

### 2.4 编写 KernelRequestSpec

Framework Engineer Agent 复制并填写：

```text
kernel_agent/framework_engineer/templates/kernel_request_spec.yaml
```

必须写清：

- 输入 `q/k/v/metadata` 的 dtype、shape、layout、stride。
- 输出 `hidden` 的 dtype、shape、layout。
- prefill/decode/cache/metadata 的语义边界。
- hot shape，例如 token 数、batch、heads、head_dim。
- baseline 性能来源和命令。
- 目标性能，例如 hot shape 至少 1.10x。
- 精度容忍，例如 bf16 下 `atol/rtol`。
- 允许 kernel 侧提出哪些框架改造，例如 contiguous layout、metadata 预计算、workspace。

### 2.5 编写 UnitTestHarness

Framework Engineer Agent 复制并填写：

```text
kernel_agent/framework_engineer/templates/unit_test_harness.py
```

UT 需要提供：

- PyTorch golden：定义 linear attention 的真实语义。
- shape sweep：覆盖 prefill hot、decode hot、boundary。
- `candidate_kernel(inputs)` 占位接口。
- correctness check。
- micro benchmark。
- NCU command 占位。

如果 PyTorch golden 过慢，可以只用于 correctness；benchmark baseline 可以来自当前 SGLang backend，但必须在 spec 中写清来源。

### 2.6 交给 Kernel Agent

交接包最少包含：

```text
kernel_request_spec.yaml
unit_test_harness.py
baseline benchmark 摘要
可选 profiler 摘要
```

Framework Engineer Agent 到这里暂停，不要求它自己实现 kernel。

## 3. Kernel Agent 流程

### 3.1 接收并审查任务

Kernel Agent 读取 spec 和 UT，先判断是否可以开始：

- 语义是否清楚。
- 输入输出是否完整。
- shape 是否覆盖真实热路径。
- golden 是否独立。
- benchmark 是否可比较。
- 允许框架改造边界是否清楚。

如果缺信息，输出：

```text
kernel_agent/kernel_agent/templates/task_acceptance_review.md
```

结论为 `NEEDS_MORE_INFO`，并列出 Framework Engineer Agent 必须补充的内容。

### 3.2 选择实现路径

默认策略：

- 首选 Triton 或 CuTe DSL 原型，快速验证 correctness 和基本性能。
- 对 hot shape 做 Triton/CuTe DSL tuning。
- 如果 JIT 路径指标显示仍需要更重的低层控制或库集成，再考虑 CUDA extension / CUTLASS。

Kernel Agent 使用：

- `skills/kernel_optimization_loop.md`
- `skills/triton_cuda_codegen.md`
- `skills/nvidia_ncu_analysis.md`

### 3.3 执行优化闭环

每轮执行：

```text
实现 -> correctness -> benchmark -> NCU profile -> 分析 -> 修改
```

每轮记录：

- candidate ID。
- 修改点。
- correctness 结果。
- benchmark 结果。
- profiler 瓶颈。
- 保留或放弃原因。

如果发现需要框架侧支持，例如：

- q/k/v 需要 contiguous 或特定 stride。
- metadata 需要框架预计算。
- 需要 workspace 或 ping-pong buffer。
- prefill/decode 必须拆不同 backend。
- 需要 padding/alignment 才能稳定提速。

则输出：

```text
kernel_agent/kernel_agent/templates/framework_change_request.yaml
```

### 3.4 交付结果

达标后输出：

```text
kernel_agent/kernel_agent/templates/kernel_delivery_package.md
kernel_agent/kernel_agent/templates/benchmark_report.md
kernel_agent/kernel_agent/templates/kernel_constraints.md
```

交付中必须说明：

- 实现入口。
- 支持 dtype/shape/layout。
- correctness 结果。
- benchmark 结果。
- profiler 证据。
- fallback 条件。
- 是否需要框架改造。

如果无法达标，也要输出瓶颈解释，例如已经接近 memory bandwidth，或 e2e 上限不足。

## 4. Framework Engineer Agent 验收

Framework Engineer Agent 收到 `KernelDeliveryPackage` 后，不直接相信 micro benchmark。

验收步骤：

1. 评审支持范围和约束。
2. 如果有 `FrameworkChangeRequest`，先用 `framework_change_review.md` 做接受/拒绝决策。
3. 写 `integration_plan.md`，说明 backend flag、fallback、metadata/workspace、测试范围。
4. 接入 SGLang backend。
5. 用 Qwen3.5 真实 workload 做 e2e 性能测试。
6. 用任务样例做精度或输出一致性验证。
7. 生成 `e2e_verification_report.md`。

如果 micro benchmark 变快但 e2e 无收益，需要回写原因：

- 热点误判。
- kernel 占比太低。
- 调度/显存/cache/同步成为新瓶颈。
- 接入开销抵消收益。
- 只优化了非主要 shape。
- CUDA graph 或 cache 行为改变。

## 5. Phase 1 边界

Phase 1 只定义模板和交接协议：

- 要求人工/半自动提供 spec 和 UT。
- 不要求自动 profile 全模型。
- 不要求自动修改 SGLang。
- 不要求多硬件。
- 不要求真实 profiler、benchmark、kernel 脚本实现。

完成 Phase 1 后，一个新算子需求应该能被 `KernelRequestSpec + UnitTestHarness` 完整描述，Kernel Agent 不需要理解完整模型，也能知道要实现什么、怎么测、性能目标是什么。
