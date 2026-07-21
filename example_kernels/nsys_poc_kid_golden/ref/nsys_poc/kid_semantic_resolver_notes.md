# KID Agent semantic notes: `nsys_poc`

本文件记录从 execution capture 到 semantic target 的判断过程，不属于固定 schema，也不被
`source_locate` 消费。对应 trace 由当前 `nsys_poc.py` 在 H20/SM90 上生成；high-level call
`1` 的 GPU kernel sum 为 48.097 us，12 个 kernel 全部完成归因。

## 解析方法

从拥有 CUDA launch 的最底层 capture 沿保存的 Python stack 向上分析，跳过通用 runtime
wrapper、PoC 调度函数和框架内部 redispatch frame，选择可作为后续输入输出 dump 与 kernel
优化单元的第一个稳定语义接口。`call_site` 取调用该语义接口的真实 stack edge；
`archetype` 取实际拥有 kernel 的最底层 capture 类别；`provider` 表示算子实现源码所在仓库，
而不是 runtime namespace 或编译器。

## 11 个目标的决议

| Rank | Semantic interface | Owner | Call site | Archetype | Provider | 关键证据 |
|---:|---|---:|---:|---|---|---|
| 1 | `torch.matmul` | 1 | `nsys_poc.py:818` | `pytorch_dispatch` | `pytorch` | `aten.mm.default` 的 stack 直接回到 `torch.matmul` 调用。 |
| 2 | `sgl_kernel.flash_attn.flash_attn_varlen_func` | 4 | `nsys_poc.py:853` | `pytorch_dispatch` | `sgl-attn` | `sgl_kernel.fwd.default` 一次调用拥有 preparation 与主 attention 两个 kernel。 |
| 3 | `flashinfer.sampling.min_p_sampling_from_probs` | 21 | `nsys_poc.py:910` | `tvm_ffi_call` | `flashinfer` | FFI export 的 stack 保留公开 sampling API。 |
| 4 | `sglang.jit_kernel.diffusion.triton.rmsnorm_onepass.triton_one_pass_rms_norm` | 6 | `nsys_poc.py:875` | `triton_launch` | `sglang` | 外层 PyTorch custom op 只提供上下文，真正 launch 在子 capture 6。 |
| 5 | `deep_gemm.bf16_gemm_nt` | 24 | `nsys_poc.py:952` | `python_binding` | `deepgemm` | 显式 export wrapper 保留稳定 public API。 |
| 6 | `sglang.srt.layers.mhc.hc_split_sinkhorn` | 35 | `nsys_poc.py:966` | `tilelang_launch` | `sglang` | `JITKernel.__call__` 的上游 stack 回溯到 SGLang semantic function。 |
| 7 | `flashinfer.triton.norm.rms_norm` | 7 | `nsys_poc.py:887` | `triton_launch` | `flashinfer` | Triton launcher stack 中存在公开 `rms_norm` wrapper。 |
| 8 | `sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift.fused_norm_scale_shift` | 23 | `nsys_poc.py:931` | `cute_dsl_launch` | `sglang` | 外层 PyTorch custom op 只提供上下文，子 capture 23 拥有 CuTe launch。 |
| 9 | `sgl_kernel.silu_and_mul` | 3 | `nsys_poc.py:839` | `pytorch_dispatch` | `sgl-kernel` | kernel 使用 FlashInfer activation 模板，但公开接口和集成源码由 sgl-kernel 提供。 |
| 10 | `sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties` | 36 | `nsys_poc.py:981` | `inductor_launch` | `sglang` | Inductor cache 文件不稳定，stack 中的 SGLang semantic function 是稳定接口。 |
| 11 | `sglang.jit_kernel.add_constant.add_constant` | 9 | `nsys_poc.py:898` | `tvm_ffi_call` | `sglang` | SGLang `load_jit` 返回的 FFI export 对应仓库内公开 wrapper。 |

## 嵌套与聚合判断

- capture 5 → 6 和 capture 22 → 23 均保留；kernel 只计入最底层 owner，外层不重复计时。
- DeepGEMM capture 24 内的 capture 25–27 是没有 GPU kernel 的 `aten.detach` 辅助调用，只保留
  在 raw JSONL 中。
- sgl-attn capture 4 关联两个 kernel，2.561 us 的 block preparation 与 6.784 us 的主
  attention kernel 合并为一个 9.345 us semantic target；代表 kernel 选择较热的主 kernel。
- 其余十个 semantic target 各拥有一个 GPU kernel。

所有 11 个决议的 confidence 均为 `high`：semantic call site 都能在当前源码及保存的 stack
edge 中直接验证，代表 kernel和耗时可通过 correlation id 回查同轮 SQLite。
