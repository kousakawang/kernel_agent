# KID execution capture registry

本文定义 KID Runtime Capture CLI 的 execution capture 类别。代码侧唯一枚举位于
`capture_registry.py`；`archetype` 表示 execution 被哪一种 common interface 捕获，不表示源码
归属、构建方式或 provider。

## 1. Registry v2

registry 版本为 `kid-execution-capture/v2`，当前类别为：

| `archetype` | common interface | 已验证代表 |
| --- | --- | --- |
| `pytorch_dispatch` | `TorchDispatchMode.__torch_dispatch__` | PyTorch、sgl-kernel、sgl-attn |
| `triton_launch` | Triton `JITFunction/Autotuner/Heuristics` launcher | SGLang、FlashInfer Triton |
| `cute_dsl_launch` | `cutlass.cute.compile` 返回的 callable | SGLang CuTe DSL |
| `tilelang_launch` | `tilelang.JITKernel.__call__` | SGLang MHC TileLang |
| `tvm_ffi_call` | `tvm_ffi.module.Module` export | SGLang JIT、FlashInfer AOT/JIT |
| `inductor_launch` | Inductor `CachingAutotuner.run` | SGLang `torch.compile` 算子 |
| `python_binding` | 已登记的 Python extension export | DeepGEMM |

每个 capture event 保存 capture/high id、archetype、真实 execution interface、high→execution
Python frame/callsite、CPU launch range，以及可直接获得的 provider/implementation hint。

一个 execution 落入多个 hook 时全部保留，并用 `parent_capture_id` 建树。同一个 GPU kernel 只归因
给 CUDA launch API 所在的最内层 capture；外层只保存 inclusive 聚合。最终 semantic target 的
`archetype` 使用最底层有效 capture，上层 capture 和 frame 链用于语义定位。

## 2. Capture 边界

### `pytorch_dispatch`

只在 high-level invocation 内进入 `TorchDispatchMode`，覆盖 ATen、`torch.library` custom op 和
注册到 `torch.ops` 的 AOT/JIT extension。torch、sgl-kernel 和 third-party 可能具有相同
archetype；其源码差异由 provider 和后续源码分析表达。

不覆盖直接 pybind export、CUDA Graph replay 或 C++ 内部没有新 dispatcher 边界的子 kernel。

### `triton_launch`

在 workload import 前 patch `JITFunction`、`Autotuner`、`Heuristics.__getitem__` 返回的 launcher。
覆盖 SGLang 和 third-party Triton。若外层同时存在 custom op，保留外层 dispatcher capture，kernel
由内层 Triton capture 独占。

### `cute_dsl_launch`

patch `cutlass.cute.compile` 并包装返回 callable。必须在 compiled callable 首次创建前安装；provider
来自 kernel 定义源码，而不是 CUTLASS 编译器。例如 SGLang 自有 CuTe kernel 的 provider 是
`sglang`。

### `tilelang_launch`

patch TileLang 共享的 `JITKernel.__call__`。`@tilelang.jit` 的 factory 和 kernel 实现仍属于调用方
源码仓库；TileLang 只是编译/运行入口。当前在 H20 上用 SGLang `hc_split_sinkhorn` 验证。

### `tvm_ffi_call`

代理 runtime load factory 返回的 TVM-FFI module export。当前登记：

- `sglang.jit_kernel.utils.load_jit`；
- `flashinfer.jit.core.JitSpec.build_and_load`。

代理 export 而不是 semantic wrapper，因此 CUDA launch correlation 可以直接归入 FFI execution。
新增 TVM-FFI-producing factory 时需要显式登记。

### `inductor_launch`

patch PyTorch 2.11 Inductor `CachingAutotuner.run`，捕获 Inductor 生成的 Triton kernel。执行已编译
case 时临时弹出 discovery `TorchDispatchMode`，否则 mode 会改变 `torch.compile` 的执行路径并退回
eager；这一步不增加 semantic 标记。extern kernel 仍可能由 dispatcher 或 binding adapter 捕获。

### `python_binding`

Python 无法通用枚举所有 C-extension 调用，因此按 provider 登记接近 GPU execution 的稳定 export。
当前登记 DeepGEMM `bf16_gemm_nt`/`fp8_paged_mqa_logits`。binding 内多个 kernel 通过 correlation id
归入同一 capture。

## 3. Provider 口径

`provider` 表示包含 semantic operator/kernel 源码的仓库，不表示 compiler/runtime/package wheel。
例如：

- SGLang Triton、CuTe、TileLang、Inductor semantic function：`sglang`；
- sgl-kernel 自有算子：`sgl-kernel`；
- sgl-kernel FetchContent 的 FA3：`sgl-attn`；
- FlashInfer Triton/FFI：`flashinfer`；
- DeepGEMM binding：`deepgemm`。

provider 不能可靠确定时允许留空，由 Semantic Resolver Agent 填写。

## 4. 与旧 F0–F9 的关系

旧 F 类别描述源码/构建形态，与 runtime capture 不一一对应：

| 旧形态 | 常见 capture |
| --- | --- |
| PyTorch native | `pytorch_dispatch` 或 `inductor_launch` |
| SGLang/third-party Triton | `triton_launch` |
| sgl-kernel、third-party AOT | `pytorch_dispatch` 或 `python_binding` |
| SGLang/FlashInfer runtime JIT | `tvm_ffi_call` |
| CuTe DSL | `cute_dsl_launch` |
| TileLang | `tilelang_launch` |
| C++ 内部 NVRTC/downloaded cubin | 由 Python 暴露面决定，通常是 `python_binding` |

不能从 archetype 推断 source-locate 路径，也不能从 provider 反推 archetype。

## 5. PoC 验证

`nsys_poc.py` 只显式标记 high-level target。默认远端运行先生成环境 probe、逐 case smoke/prewarm，
再用 Nsight correlation 验证 11 个 H20 case；FA4 和 TokenSpeed MLA 是硬件条件 case。

```bash
python kernel_agent/framework_engineer/kernel_interface_decomposer/nsys_poc.py --probe-env
python kernel_agent/framework_engineer/kernel_interface_decomposer/nsys_poc.py
python3 kernel_agent/framework_engineer/kernel_interface_decomposer/nsys_poc.py --self-test
```

PoC 验证 warmup 不进入 high、NVTX pop 后的异步 kernel 仍能关联、nested capture 全量保留、kernel
最底层独占归因、多-kernel 聚合和未归因 coverage 报告。CUDA Graph discovery 仍不在本轮范围内。
