# Semantic Resolver Agent notes: `nsys_poc`

本文件记录 Agent 从 execution capture 到 semantic target 的判断过程，不属于
固定 schema，也不被 `source_locate` 消费。对应 trace 来自 H20/SM90；high-level
call `1` 的 GPU kernel sum 为 48.096 us，12 个 kernel 全部完成归因。

## 解析方法

Agent 从拥有 CUDA launch 的最底层 capture 沿保存的 Python stack 向上分析，
跳过通用 runtime wrapper、PoC 调度函数和框架内部 redispatch frame，选择可作为
后续输入输出 dump 与 kernel 优化单元的第一个稳定语义接口。`call_site` 取调用
该语义接口的 stack edge；`archetype` 取实际拥有 kernel 的最底层 capture 类别；
`provider` 表示算子实现源码所在仓库，而不是 runtime namespace 或编译器。

## 11 个目标的决议

| Rank | Semantic interface | Owner capture | Call site | Archetype | Provider | 关键证据 |
|---:|---|---:|---:|---|---|---|
| 1 | `torch.matmul` | 1 | `nsys_poc.py:810` | `pytorch_dispatch` | `pytorch` | `aten.mm.default` 的 stack 直接回到 PoC 中的 `torch.matmul` 调用。 |
| 2 | `sgl_kernel.flash_attn.flash_attn_varlen_func` | 4 | `nsys_poc.py:834` | `pytorch_dispatch` | `sgl-attn` | `sgl_kernel.fwd.default` 一次调用拥有 preparation 与主 attention 两个 kernel；源码实现落在 sgl-attn。 |
| 3 | `flashinfer.sampling.min_p_sampling_from_probs` | 21 | `nsys_poc.py:891` | `tvm_ffi_call` | `flashinfer` | FFI module export 的 stack 保留公开 sampling API；provider 由 FlashInfer 源码和 manifest 共同确认。 |
| 4 | `sglang.jit_kernel.diffusion.triton.rmsnorm_onepass.triton_one_pass_rms_norm` | 6 | `nsys_poc.py:856` | `triton_launch` | `sglang` | 外层 capture 5 是 PyTorch custom op，真正 launch 在子 capture 6；语义接口定义于 SGLang。 |
| 5 | `deep_gemm.bf16_gemm_nt` | 24 | `nsys_poc.py:933` | `python_binding` | `deepgemm` | pybind 没有更通用的 Python launch 入口；注册 export wrapper 直接保留稳定 public API。 |
| 6 | `sglang.srt.layers.mhc.hc_split_sinkhorn` | 35 | `nsys_poc.py:947` | `tilelang_launch` | `sglang` | `JITKernel.__call__` 的上游 stack 可回溯到 SGLang semantic function。 |
| 7 | `flashinfer.triton.norm.rms_norm` | 7 | `nsys_poc.py:868` | `triton_launch` | `flashinfer` | Triton launcher stack 中存在公开 `rms_norm` wrapper，kernel 定义在 FlashInfer。 |
| 8 | `sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift.fused_norm_scale_shift` | 23 | `nsys_poc.py:912` | `cute_dsl_launch` | `sglang` | 外层 capture 22 是 PyTorch custom op，子 capture 23 才拥有 CuTe launch；语义实现属于 SGLang。 |
| 9 | `sgl_kernel.silu_and_mul` | 3 | `nsys_poc.py:820` | `pytorch_dispatch` | `sgl-kernel` | runtime kernel 名含 FlashInfer activation 模板，但公开接口和集成源码由 sgl-kernel 提供。 |
| 10 | `sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties` | 36 | `nsys_poc.py:962` | `inductor_launch` | `sglang` | Inductor 临时 cache 文件不稳定；stack 中稳定的 SGLang semantic function 才是最终接口与源码入口。 |
| 11 | `sglang.jit_kernel.add_constant.add_constant` | 9 | `nsys_poc.py:879` | `tvm_ffi_call` | `sglang` | SGLang `load_jit` 返回的 FFI export 对应仓库内 public wrapper 和 `.cuh` 实现。 |

## 嵌套与聚合判断

- capture 5 → 6 和 capture 22 → 23 均保留；kernel 只计入最底层 owner，外层仅提供
  semantic context，因此不会重复计时。
- DeepGEMM capture 24 内的 capture 25–27 是 `aten.detach` 辅助调用，没有 GPU
  kernel；它们保留在 raw JSONL 中，但不拆成 semantic target。
- sgl-attn capture 4 关联两个 kernel，2.560 us 的 block preparation 与 6.848 us
  的主 attention kernel 合并为一个 9.408 us semantic target；最终代表 kernel
  选择较热的主 attention kernel。
- 其余十个 semantic target 在本次 trace 中各拥有一个 GPU kernel。

所有 11 个决议的 confidence 均为 `high`：语义 call site 都能在保存的 stack edge
中直接验证，代表 kernel 和耗时都能通过 correlation id 回查同轮 SQLite。
