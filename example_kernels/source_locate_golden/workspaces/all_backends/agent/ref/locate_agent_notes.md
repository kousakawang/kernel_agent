# source_locate Agent notes

本文件由 Agent decisions 机械生成；正式下游契约只存在于 schema 的
四层结果中。

## torch_sdpa

- Interface: `torch.nn.functional.scaled_dot_product_attention`
- Summary: The semantic target is PyTorch scaled_dot_product_attention, but the matching PyTorch source checkout is not one of the allowed source roots.

### Layer evidence

- `interface_definition` — `missed`: The KID interface is known, but its Python definition cannot be verified without the PyTorch source repository.
- `kernel_impl` — `missed`: The dispatcher and native SDPA implementation are expected to exist in PyTorch, but no allowed root contains them.
- `py_cpp_binding` — `missed`: The Python-to-native dispatcher path cannot be verified without PyTorch sources.
- `kernel_header` — `missed`: Potential native declarations cannot be checked because the owning repository is absent.

### Gaps and follow-up

- Gap: PyTorch source is absent from the manifest, so none of the four layers can be evidenced within allowed roots.
- Manual follow-up: Add a matching PyTorch source checkout to the manifest and rerun locate and source_locate.

## fla_recompute_w_u_fwd

- Interface: `sglang.srt.layers.attention.fla.wy_fast.recompute_w_u_fwd`
- Summary: The SGLang wrapper directly launches a Triton kernel in the same module; there is no native bridge or independent header.

### Layer evidence

- `interface_definition` — `resolved`: The definition matches the KID call-site import and semantic target.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py:111` `recompute_w_u_fwd` — This is the imported Python wrapper used by the call site.
- `kernel_impl` — `resolved`: The wrapper launches this Triton JIT kernel directly.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py:23` `recompute_w_u_fwd_kernel` — This Triton kernel performs the target computation.
- `py_cpp_binding` — `not_applicable`: The Python wrapper launches Triton directly and does not cross a Python/C++ binding.
- `kernel_header` — `not_applicable`: The implementation is self-contained Triton code with no independent native declaration header.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## sgl_kernel_silu_and_mul

- Interface: `sgl_kernel.silu_and_mul`
- Summary: The Python torch.ops wrapper resolves through SGLang's Torch registration to a CUDA host launcher and FlashInfer device kernel.

### Layer evidence

- `interface_definition` — `resolved`: The wrapper is the KID-selected public interface and calls torch.ops.sgl_kernel.silu_and_mul.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/python/sgl_kernel/elementwise.py:258` `silu_and_mul` — This Python definition is imported and invoked by the SGLang call site.
- `kernel_impl` — `resolved`: The native launcher dispatches the templated FlashInfer device kernel in this order.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/csrc/elementwise/activation.cu:85` `silu_and_mul` — This is the registered CUDA host entry and launch site.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/include/flashinfer/activation.cuh:28` `act_and_mul_kernel` — This templated CUDA kernel performs the elementwise activation and multiplication.
- `py_cpp_binding` — `resolved`: Torch Library registration maps the Python torch.ops name to the native launcher.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/csrc/common_extension.cc:76` `m.def/m.impl(silu_and_mul)` — The schema and CUDA implementation registration close the Python-to-native boundary.
- `kernel_header` — `resolved`: The registered host function has a standalone public declaration.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/include/sgl_kernel_ops.h:139` `silu_and_mul declaration` — This declaration connects the registration unit to the CUDA definition.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## sgl_kernel_fa3_fwd

- Interface: `sgl_kernel.flash_attn.flash_attn_varlen_func`
- Summary: The Python FlashAttention wrapper crosses SGLang's Torch registration into the sgl-attn host API, launch template, and SM90 device implementation.

### Layer evidence

- `interface_definition` — `resolved`: This definition is the call-site-selected varlen FlashAttention interface.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/python/sgl_kernel/flash_attn.py:232` `flash_attn_varlen_func` — The function is the Python interface invoked by SGLang attention code.
- `kernel_impl` — `resolved`: The recorded order follows the native host entry, dispatch/launch template, then the SM90 kernel body.
  - `/Users/bytedance/Desktop/infra_agent/sgl-attn/hopper/flash_api.cpp:673` `mha_fwd` — This is the native host API reached by the registered operator.
  - `/Users/bytedance/Desktop/infra_agent/sgl-attn/hopper/flash_fwd_launch_template.h:31` `run_flash_fwd` — This template specializes and launches the forward attention implementation.
  - `/Users/bytedance/Desktop/infra_agent/sgl-attn/hopper/flash_fwd_kernel_sm90.h:179` `FlashAttnFwdSm90::operator()` — This device operator is the core SM90 forward kernel implementation.
- `py_cpp_binding` — `resolved`: The Torch Library schema/export anchors the Python operator to mha_fwd.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/csrc/flash_extension.cc:25` `m.def(fwd)` — This registration exposes the forward native entry used by the Python wrapper.
- `kernel_header` — `resolved`: The standalone header declares the native mha_fwd entry consumed by the extension.
  - `/Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel/include/sgl_flash_kernel_ops.h:45` `mha_fwd declaration` — This declaration connects the extension registration and sgl-attn definition.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## sglang_jit_add_constant

- Interface: `sglang.jit_kernel.add_constant.add_constant`
- Summary: The Python wrapper builds a TVM-FFI JIT module from a CUDA header, invokes its host wrapper, and reaches the add_constant CUDA kernel.

### Layer evidence

- `interface_definition` — `resolved`: This is the public JIT example function selected by KID.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/add_constant.py:24` `add_constant` — The function allocates the result and invokes module.add_constant.
- `kernel_impl` — `resolved`: The JIT-exported host wrapper launches the concrete CUDA kernel in this order.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/csrc/add_constant.cuh:59` `add_constant` — This templated TVM-FFI host function validates inputs and launches the kernel.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/csrc/add_constant.cuh:27` `add_constant_kernel` — This CUDA global function implements the scalar add.
- `py_cpp_binding` — `resolved`: The load_jit call declares the CUDA source and generated wrapper exported as add_constant.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/add_constant.py:16` `load_jit(add_constant)` — This loader configuration creates the Python-visible TVM-FFI module boundary.
- `kernel_header` — `not_applicable`: The .cuh file contains executable host/device implementation rather than an independent declaration-only layer.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## flashinfer_top_k_top_p_sampling

- Interface: `flashinfer.sampling.top_k_top_p_sampling_from_probs`
- Summary: The FlashInfer Python API obtains a generated sampling module, crosses its TVM-FFI binding, and reaches the CUDA launcher and rejection-sampling kernel.

### Layer evidence

- `interface_definition` — `resolved`: This is the KID-selected fused top-k/top-p sampling API.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/sampling.py:1579` `top_k_top_p_sampling_from_probs` — The public function performs the selected sampling operation.
- `kernel_impl` — `resolved`: The native host entry calls the templated dispatcher, which launches the CUDA kernel.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/csrc/sampling.cu:277` `top_k_top_p_sampling_from_probs` — This TVM-FFI host entry validates tensors and invokes the sampling implementation.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/include/flashinfer/sampling.cuh:1606` `TopKTopPSamplingFromProb` — This template dispatches launch geometry and the deterministic specialization.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/include/flashinfer/sampling.cuh:1192` `TopKTopPSamplingFromProbKernel` — This CUDA kernel performs the fused rejection sampling loop.
- `py_cpp_binding` — `resolved`: The generated module loader and its native declaration together establish the Python-to-CUDA boundary.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/sampling.py:68` `gen_sampling_module().build_and_load` — This builds and loads the Python-visible generated sampling module.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/csrc/flashinfer_sampling_binding.cu:54` `top_k_top_p_sampling_from_probs declaration` — This binding declaration is used to export the native function into the generated module.
- `kernel_header` — `not_applicable`: The .cuh file is implementation-bearing and is therefore part of kernel_impl, not a declaration-only header layer.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## flashinfer_triton_rms_norm

- Interface: `flashinfer.triton.norm.rms_norm`
- Summary: The FlashInfer Python function launches a Triton RMSNorm kernel that uses a Triton scale-and-clamp helper; no native bridge or separate header exists.

### Layer evidence

- `interface_definition` — `resolved`: The definition matches the KID interface and direct call-site import.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/triton/norm.py:9` `rms_norm` — This wrapper computes launch parameters and invokes rms_norm_kernel.
- `kernel_impl` — `resolved`: The main Triton kernel and its imported computation helper make up the implementation chain.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/triton/kernels/norm.py:7` `rms_norm_kernel` — This Triton JIT function is the directly launched RMSNorm kernel.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/triton/kernels/quant.py:5` `scale_and_clamp` — This Triton helper implements the optional output scaling and clamping used by the kernel.
- `py_cpp_binding` — `not_applicable`: The Python wrapper launches Triton without a C++ extension boundary.
- `kernel_header` — `not_applicable`: The implementation is Python/Triton and has no independent native declaration header.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## flashinfer_batch_prefill_paged

- Interface: `flashinfer.BatchPrefillWithPagedKVCacheWrapper.run`
- Summary: The FlashInfer wrapper loads a generated module, invokes paged_run through TVM-FFI, and reaches the paged prefill host entry, dispatcher, and CUDA kernel.

### Layer evidence

- `interface_definition` — `resolved`: The class method is the fully qualified KID interface used for paged KV prefill.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/prefill.py:2198` `BatchPrefillWithPagedKVCacheWrapper.run` — This overload anchor identifies the public class method selected by KID.
- `kernel_impl` — `resolved`: The order records native host entry, template dispatcher, and launched CUDA kernel.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/csrc/batch_prefill.cu:203` `BatchPrefillWithPagedKVCacheRun` — This TVM-FFI host function prepares parameters and dispatches paged prefill.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/include/flashinfer/attention/prefill.cuh:3605` `BatchPrefillWithPagedKVCacheDispatched` — This template selects launch traits and dispatches the paged implementation.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/include/flashinfer/attention/prefill.cuh:3409` `BatchPrefillWithPagedKVCacheKernel` — This CUDA global function is the core paged prefill kernel.
- `py_cpp_binding` — `resolved`: The Python generated-module loader and TVM-FFI paged_run export close the native boundary.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/flashinfer/prefill.py:447` `gen_batch_prefill_module().build_and_load` — This creates the generated module whose paged_run function the wrapper calls.
  - `/Users/bytedance/Desktop/infra_agent/flashinfer/csrc/batch_prefill_jit_binding.cu:38` `BatchPrefillWithPagedKVCacheRun export declaration` — This declaration is exported as paged_run by TVM-FFI.
- `kernel_header` — `not_applicable`: The .cuh layer contains launch and device implementation, so it belongs to kernel_impl rather than a declaration-only header.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## sglang_cutedsl_fused_norm_scale_shift

- Interface: `sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift.fused_norm_scale_shift`
- Summary: The SGLang custom op compiles a CuTe DSL callable with a TVM-FFI backend; that callable launches the CuTe device kernel.

### Layer evidence

- `interface_definition` — `resolved`: This custom-op definition is the KID-selected fused normalization interface.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py:279` `fused_norm_scale_shift` — The function validates tensors, compiles the DSL kernel, and executes it.
- `kernel_impl` — `resolved`: The CuTe callable is the host launch entry and its kernel method is the device implementation.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py:87` `ScaleResidualNormScaleShift.__call__` — This cute.jit callable prepares and launches self.kernel.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py:134` `ScaleResidualNormScaleShift.kernel` — This cute.kernel method contains the core device computation.
- `py_cpp_binding` — `resolved`: cute.compile with the TVM-FFI backend materializes the callable boundary used by the Python custom op.
  - `/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py:333` `cute.compile` — This compile anchor creates the TVM-FFI callable later invoked by Python.
- `kernel_header` — `not_applicable`: The CuTe DSL implementation is contained in Python and has no independent native declaration header.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.

## deepgemm_fp8_paged_mqa_logits

- Interface: `deep_gemm.fp8_paged_mqa_logits`
- Summary: DeepGEMM re-exports the binary extension function, whose pybind registration reaches a C++ host API, JIT code generator/launcher, and the SM90 device kernel template.

### Layer evidence

- `interface_definition` — `resolved`: The package-level re-export is the Python interface named by KID.
  - `/Users/bytedance/Desktop/infra_agent/DeepGEMM/deep_gemm/__init__.py:68` `fp8_paged_mqa_logits` — This import/re-export exposes the binary extension function to callers.
- `kernel_impl` — `resolved`: The order records the legacy host API, runtime code generation/launch, and generated SM90 device kernel body.
  - `/Users/bytedance/Desktop/infra_agent/DeepGEMM/csrc/apis/attention.hpp:401` `fp8_paged_mqa_logits` — This C++ host wrapper forwards the legacy API to the paged MQA implementation.
  - `/Users/bytedance/Desktop/infra_agent/DeepGEMM/csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp:281` `SMXXFP8PagedMQALogitsRuntime::generate/build/launch` — This is the JIT code generation, compilation, and launch anchor for paged MQA.
  - `/Users/bytedance/Desktop/infra_agent/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:30` `sm90_fp8_paged_mqa_logits` — This CUTLASS global function contains the SM90 paged MQA kernel implementation.
- `py_cpp_binding` — `resolved`: The pybind registration exposes the legacy fp8_paged_mqa_logits host wrapper.
  - `/Users/bytedance/Desktop/infra_agent/DeepGEMM/csrc/apis/attention.hpp:445` `m.def(fp8_paged_mqa_logits)` — This pybind anchor maps the Python name to the C++ host function.
- `kernel_header` — `not_applicable`: The headers in the chain contain executable templates and registrations, so they are implementation/binding hits rather than a separate declaration-only layer.

### Gaps and follow-up

- Gaps: none.
- Manual follow-up: none.
