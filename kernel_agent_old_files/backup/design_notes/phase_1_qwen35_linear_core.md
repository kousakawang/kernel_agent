# Phase 1 Example: Qwen3.5 Linear Attention Core Kernel

本文把 Phase 1 具体化到一个任务：优化 Qwen3.5 linear attention 的核心 GPU kernel。这里不要求 Kernel Engineer 理解完整 SGLang serving，也不要求它构造框架内部输入；Framework Engineer 必须把真实框架路径里的复杂输入降低成可离线执行的 kernel-level task pack。

## 目标边界

首个建议目标是 Qwen3.5/GDN linear attention 的 prefill core kernel：

```text
TritonGDNKernel.extend(
  q, k, v, g, beta,
  ssm_states,
  cache_indices,
  query_start_loc,
) -> core_attn_out, last_recurrent_state, h
```

参考源码：

- Framework wrapper: `sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py`
- Core kernel wrapper: `sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`

为什么先选 prefill core：

- 用户给的 image workload 通常 prefill 比 decode 更重。
- `forward_extend` 会把 `ForwardBatch`、mamba cache、metadata 降低成 tensor 参数。
- core ABI 已经接近真实 Triton/CUDA/CuTe DSL kernel 参数。
- Kernel Engineer 可以在离线 harness 中迭代，不需要启动 SGLang 服务。

decode/packed decode 可以作为同一 task pack 的第二个 target，但 Phase 1 不建议一开始把 prefill、decode、conv1d、qkv split、state tracking 全部混成一个优化任务。

## 核心原则

Framework Engineer 对输入负责：

- shape 列表来自真实 workload trace 或明确的框架推导。
- `query_start_loc`、`cache_indices`、`ssm_states`、`g/beta` 等不是 Kernel Engineer 随机编造。
- 若无法从真实框架直接 replay，则必须说明 synthetic 构造规则和它与真实路径的对应关系。

Kernel Engineer 对实现负责：

- 只在 task pack 定义的 ABI、shape、tolerance、环境内优化。
- 不修改 golden、shape list、benchmark 统计逻辑。
- 如果需要 layout、metadata、workspace 变化，提交 `FrameworkChangeRequest`。

## Framework Engineer 工作流

### 1. 启动服务并确认 workload

输入是用户提供的启动命令和测试命令。Framework Engineer 必须先验证：

- 服务能启动。
- workload 能跑通。
- 原始 e2e 性能可记录。
- linear attention 路径确实被调用。

shape 收集时要关闭 CUDA graph，避免 replay 固定 shape 干扰真实动态调度。SGLang 可优先使用：

```text
--disable-cuda-graph
```

或分阶段关闭：

```text
--disable-prefill-cuda-graph
--disable-decode-cuda-graph
```

最终交付要同时记录：

- 原始用户命令。
- shape 收集时改写后的命令。
- 哪些参数被 Framework Engineer 自动追加或删除。

### 2. 插桩收集 kernel ABI trace

推荐插桩点不是模型顶层 `forward`，而是 core kernel wrapper 调用前：

```text
GDNAttnBackend.forward_extend
  -> self.kernel_dispatcher.extend(...)
  -> TritonGDNKernel.extend(...)
```

对 prefill core，至少记录：

- `mode`: `extend`
- `layer_id`
- `q/k/v` shape、stride、dtype、device、contiguous
- `g/beta` shape、stride、dtype
- `ssm_states` shape、stride、dtype
- `cache_indices` shape、dtype、min/max、是否有 -1
- `query_start_loc` shape、dtype、前几个值、diff 统计
- `extend_seq_lens`、`extend_prefix_lens`、`batch_size`
- 当前 backend：triton/cutedsl/flashinfer 等

原始 trace 写成 JSONL，每次 kernel call 一行。示例字段见：

```text
kernel_agent/framework_engineer/templates/qwen35_linear_shape_trace.schema.json
```

### 3. 形成最终 shape list

不要把 raw trace 全交给 Kernel Engineer。Framework Engineer 要做 shape canonicalization：

- 按 `mode + q_shape + query_start_loc.diff pattern + dtype + layout` 聚合。
- 统计每类 shape 的 `count`。
- 保留 top frequency。
- 保留最大 token、最大 batch、最小边界、非整齐边界。
- 对 `query_start_loc` 保留真实 diff 列表或可复现生成规则。
- 对 state pool 记录真实 `num_slots` 或最小可 replay slot 数。

最终输出：

```text
shape_list.json
```

模板见：

```text
kernel_agent/framework_engineer/templates/qwen35_linear_shape_list.json
```

### 4. 构造 correctness 和 benchmark

Framework Engineer 需要生成一个 task pack，而不是只给说明文档。推荐结构：

```text
task_pack/
  README.md
  task.yaml
  shape_list.json
  env_manifest.yaml
  reference_impl.py
  candidate_impl.py
  correctness_test.py
  benchmark.py
  scripts/
    run_correctness.sh
    run_benchmark.sh
    run_ncu.sh
```

其中：

- `reference_impl.py` 调用当前 SGLang 可靠实现或 PyTorch 等价实现。
- `candidate_impl.py` 只保留 Kernel Engineer 需要填的接口。
- `correctness_test.py` 同时比较 baseline/current 和 candidate。
- `benchmark.py` 对 baseline/current 与 candidate 用同一批输入计时。
- `shape_list.json` 是唯一 shape 来源。

对于 Qwen3.5 GDN prefill core，`candidate_impl.py` 的 ABI 应该接近：

```python
def candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc):
    ...
```

如果 reference 会更新 `ssm_states`，Framework Engineer 必须在测试中 clone 输入 state，避免 baseline 和 candidate 互相污染。

### 5. 调查开发环境

Framework Engineer 还要输出 `env_manifest.yaml`，证明 Kernel Engineer 能用哪些实现路径。

必须包含：

- Python、PyTorch、CUDA、Triton 版本。
- GPU 名称和 compute capability。
- `ncu`、`nvcc`、`ptxas` 是否可用。
- CuTe DSL、CUTLASS、cuBLAS、cuDNN、FlashInfer 等是否可 import 或找到头文件/库。
- 每个能力的验证命令和实际输出摘要。

Kernel Engineer 不能自己安装依赖；只能使用 `env_manifest.yaml` 中标记为 available 的路径。

## Kernel Engineer 工作流

Kernel Engineer 只读取 task pack，执行：

1. `scripts/run_correctness.sh`
2. `scripts/run_benchmark.sh`
3. 阅读 baseline performance 和 shape priority。
4. 选择实现路径：Triton 或 CuTe DSL 作为首选 JIT 路径；当 profiling 证明需要更重的低层控制或库集成时，再升级到 CUDA extension / CUTLASS。
5. 每轮只修改 `candidate_impl.py` 或 `kernel_sources/`。
6. 每轮记录 `iteration_log.md`。
7. correctness 通过后再 benchmark。
8. 对 hot shape 跑 `scripts/run_ncu.sh`。
9. 收敛后输出 `KernelDeliveryPackage`。

收敛标准：

- required shape 全部 correctness pass。
- hot shape 达到 task.yaml 中的 target，或证明当前瓶颈接近硬件上限。
- 最近 3 个有效迭代没有超过 3% 提升时，必须重新 profile 或换优化方向。
- 至少尝试 3 类不同方向后仍无收益，才允许停止。
- final 需要跑 full correctness、full benchmark、稳定性复测。

## 关键设计选择

### 为什么不让 Kernel Engineer 构造输入

linear attention 的输入虽然最终是 tensor/scalar，但 tensor 内容和 metadata 语义来自框架：

- `query_start_loc` 反映 serving 调度和 prefill packing。
- `cache_indices` 对应 mamba state slot。
- `ssm_states` 的 shape 和状态布局来自 `req_to_token_pool.mamba2_layer_cache(...)`。
- `g/beta` 来自 `fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)`。
- `q/k/v` 可能来自 conv1d + qkv split/fused split。

这些输入如果随机乱填，kernel benchmark 可能能跑，但不能代表真实模型路径。因此 Framework Engineer 必须负责构造或捕获输入。

### 为什么仍然在 core kernel level 做 UT/benchmark

完整 `ForwardBatch`/cache pool 过重，不适合 Kernel Engineer 快速迭代。core kernel level 的 ABI 已经足够接近最终 GPU kernel 参数，适合：

- correctness 快速验证。
- micro benchmark。
- NCU profile。
- autotune。
- 替换 Triton/CuTe DSL/CUDA/CUTLASS 实现。

Phase 1 的正确边界是：

```text
Framework Engineer 负责从真实框架降维到 core ABI。
Kernel Engineer 负责在 core ABI 下优化实现。
Framework Engineer 最后再把实现接回框架验收。
```

## Phase 1 交付清单

Framework Engineer 交给 Kernel Engineer：

- `task.yaml`
- `shape_list.json`
- `env_manifest.yaml`
- `reference_impl.py`
- `candidate_impl.py`
- `correctness_test.py`
- `benchmark.py`
- `scripts/run_correctness.sh`
- `scripts/run_benchmark.sh`
- `scripts/run_ncu.sh`
- 原始 e2e baseline 摘要
- shape trace 摘要

Kernel Engineer 交回 Framework Engineer：

- 优化后的 `candidate_impl.py` / `kernel_sources/`
- `benchmark_report.md`
- `kernel_constraints.md`
- `kernel_delivery_package.md`
- 如需要，`framework_change_request.yaml`
