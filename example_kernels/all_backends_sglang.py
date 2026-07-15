"""
all_backends_sglang.py
======================

SGLang 里算子后端(backend)分类总览 —— 每一类找一个"算子代表"，给出
**如何 import** 与 **如何调用** 的代码证据(附源码 file:line)。

调研目标(见 `调研sglang不同类别backend对应的算子.md`):

    pytorch_native        (F0)  torch/aten/cuBLAS API
    sglang_triton         (F1)  sglang 自带 triton
    sgl_kernel_builtin    (F2)  sgl-kernel 内实现 (AOT)
    sgl_kernel_thirdparty (F3)  sgl-kernel FetchContent 编入的三方 (sgl-attn)
    sglang_jit            (F4)  sglang-owned JIT from csrc
    thirdparty_aot        (F5)  三方 C++/cuda AOT (flashinfer)
    thirdparty_triton_dsl (F6)  三方 triton/cuteDSL (flashinfer)
    thirdparty_cpp_jit    (F7)  三方 C++ JIT  (flashinfer)
    sglang_jit            (F8)  sglang-owned JIT from cuteDSL
    thirdparty_deepgemm_jit (F9) 三方 DeepGEMM: AOT壳(_C 扩展)+C++内嵌 NVRTC JIT
                                 (补充类: 与 F7 同为三方运行时 C++/CUDA JIT, 但 JIT route 不同)

说明:
 - 每个 `demo_Fx()` 函数 = 该类别的 import + 调用，输入全部是**假的 dummy 张量**
   (小 shape)，只演示接口，不保证数值意义。
 - 每个 `demo_Fx()` 内还内嵌一个 `_golden_*` 子函数 = 该算子在**对应仓库测试目录**里
   找到的 golden/参考实现(CPU/torch 版 ground truth)。**仅供对照, 不调用、不比较结果**;
   子函数上方注释标注了 golden 的来源文件路径:行号。找不到的如实置空并说明原因(见 F6)。
 - 真正运行绝大多数需要 CUDA GPU + 对应包已安装; 这里的重点是"import 与调用的证据"。
 - 每段前的注释块给出**源码位置证据** (仓库相对路径 : 行号 + 关键代码片段)。
 - 仓库根:
       sglang     = /Users/bytedance/Desktop/infra_agent/sglang
       sgl-kernel = /Users/bytedance/Desktop/infra_agent/sglang/sgl-kernel
       sgl-attn   = /Users/bytedance/Desktop/infra_agent/sgl-attn      (F3 三方源)
       flashinfer = /Users/bytedance/Desktop/infra_agent/flashinfer    (F5/F6/F7 三方源)
       DeepGEMM   = /Users/bytedance/Desktop/infra_agent/DeepGEMM       (F9 三方源)
"""

import torch

CUDA = "cuda"  # 绝大多数 kernel 只能在 GPU 上跑


# =============================================================================
# F0  pytorch_native  —  torch / aten / cuBLAS API
# -----------------------------------------------------------------------------
# 代表算子: torch.nn.functional.scaled_dot_product_attention (SDPA)
#   —— 直接调 PyTorch 内建的 aten/flash/cuBLAS 实现，sglang 不自带 kernel。
#
# 代码证据 (sglang):
#   python/sglang/srt/layers/attention/torch_native_backend.py:6
#       from torch.nn.functional import scaled_dot_product_attention
#   python/sglang/srt/layers/attention/torch_native_backend.py:158-167
#       per_req_out_redudant = scaled_dot_product_attention(
#           per_req_query_redudant.unsqueeze(0), per_req_key.unsqueeze(0),
#           per_req_value.unsqueeze(0), attn_mask=attn_mask,
#           enable_gqa=enable_gqa, scale=scaling, is_causal=is_causal).squeeze(0)...
#
#   另一个纯 cuBLAS 代表: python/sglang/srt/layers/linear.py:1669
#       return torch.bmm(input, self.weight.transpose(-1, -2))
# =============================================================================
def demo_F0_pytorch_native():
    # ---- import (与 torch_native_backend.py:6 一致) ----
    from torch.nn.functional import scaled_dot_product_attention

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT: sglang/test/registered/attention/test_triton_attention_kernels.py
    #   :104-149 decode_attention_fwd_torch (朴素 softmax(QK^T*scale)@V, 作为 golden)
    #   :460     同测试也直接用 F.scaled_dot_product_attention 当 golden oracle
    def _golden_decode_attention(q, k, v, sm_scale):
        # q: [H_Q, D]; k/v: [L, H_Q, D]  (单 query decode 的朴素注意力; 摘自上面 L137-147)
        q_f32 = q.to(torch.float32)  # [H_Q, D]
        k_f32 = k.to(torch.float32)  # [L, H_Q, D]
        v_f32 = v.to(torch.float32)  # [L, H_Q, D]
        logits = torch.einsum("hd,lhd->hl", q_f32, k_f32) * float(sm_scale)  # [H_Q, L]
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.softmax(logits, dim=-1)  # [H_Q, L]
        return torch.einsum("hl,lhd->hd", p, v_f32)  # [H_Q, D]

    # ---- 假输入: (batch=1, num_heads=2, seq_len=4, head_dim=8) ----
    q = torch.randn(1, 2, 4, 8, device=CUDA)
    k = torch.randn(1, 2, 4, 8, device=CUDA)
    v = torch.randn(1, 2, 4, 8, device=CUDA)

    # ---- 调用 ----
    out = scaled_dot_product_attention(q, k, v, is_causal=True)  # -> (1, 2, 4, 8)
    return out


# =============================================================================
# F1  sglang_triton  —  sglang 自带的 triton kernel (仓库内 @triton.jit)
# -----------------------------------------------------------------------------
# 代表算子: recompute_w_u_fwd  (FLA / gated-delta-rule 前向)
#   —— @triton.jit kernel 定义在 sglang 仓库内，配一个 python launch wrapper。
#
# 代码证据 (sglang):
#   python/sglang/srt/layers/attention/fla/wy_fast.py:8-9   import triton / triton.language as tl
#   python/sglang/srt/layers/attention/fla/wy_fast.py:22    @triton.jit(do_not_specialize=["T"])
#   python/sglang/srt/layers/attention/fla/wy_fast.py:23        def recompute_w_u_fwd_kernel(...)
#   python/sglang/srt/layers/attention/fla/wy_fast.py:111   def recompute_w_u_fwd(...)   # wrapper
#   python/sglang/srt/layers/attention/fla/wy_fast.py:131       recompute_w_u_fwd_kernel[(NT, B*H)](...)  # kernel[grid](...)
#   调用方 import + 调用:
#   python/sglang/srt/layers/attention/fla/chunk_fwd.py:14  from ...fla.wy_fast import recompute_w_u_fwd
#   python/sglang/srt/layers/attention/fla/chunk_fwd.py:407     w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, ...)
# =============================================================================
def demo_F1_sglang_triton():
    # ---- import (与 chunk_fwd.py:14 一致) ----
    from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 说明: recompute_w_u_fwd 本身【没有】算子级 UT (test/ 下搜不到 recompute_w_u/wy_fast)。
    #   它被融进 chunk_gated_delta_rule 的整链前向, 因此只有【整链集成测试】。
    # 最接近的 UT: sglang/test/registered/attention/test_chunk_gated_delta_rule.py:26-53
    #   golden 不是 recompute_w_u 的闭式, 而是逐 token 递推的 fused_recurrent_gated_delta_rule,
    #   用它作为整个 chunked GDN 前向的 oracle (对照 torch.allclose, atol=2e-2)。
    def _golden_chunk_gdn_reference(pool_init, cache_indices, q, k, v, g, beta):
        # 摘自 test_chunk_gated_delta_rule.py:26-53 (_run_reference)
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )

        B = cache_indices.shape[0]
        T_per_seq = q.shape[1] // B
        pool = pool_init.clone()
        h_cur = pool[cache_indices].contiguous().clone()
        o_list = []
        for b in range(B):
            sl = slice(b * T_per_seq, (b + 1) * T_per_seq)
            o_b, h_b = fused_recurrent_gated_delta_rule(
                q=q[0, sl].unsqueeze(0),
                k=k[0, sl].unsqueeze(0),
                v=v[0, sl].unsqueeze(0),
                g=g[0, sl].unsqueeze(0),
                beta=beta[0, sl].unsqueeze(0),
                initial_state=h_cur[b : b + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o_list.append(o_b)
            h_cur[b] = h_b[0]
        pool[cache_indices] = h_cur
        return torch.cat(o_list, dim=1), pool

    # ---- 假输入 (非 varlen 路径, cu_seqlens=None) ----
    B, T, H, Hg, K, V, BT = 1, 16, 2, 2, 64, 64, 16
    k = torch.randn(B, T, Hg, K, device=CUDA, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device=CUDA, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=CUDA, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, device=CUDA, dtype=torch.float32)  # g_cumsum
    A = torch.randn(B, T, H, BT, device=CUDA, dtype=torch.bfloat16)  # BT == A.shape[-1]

    # ---- 调用 wrapper (内部 launch @triton.jit kernel) ----
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, g_cumsum=g, A=A, cu_seqlens=None)
    return w, u


# =============================================================================
# F2  sgl_kernel_builtin  —  sgl-kernel 内部实现的 CUDA 算子 (AOT 编进 .so)
# -----------------------------------------------------------------------------
# 代表算子: silu_and_mul
#   —— CUDA kernel 源码在 sgl-kernel/csrc/ 内，AOT 编入 sgl_kernel .so，
#      通过 torch.ops.sgl_kernel.* 暴露，python 侧一层薄 wrapper。
#
# 代码证据:
#   [python wrapper]  sgl-kernel/python/sgl_kernel/elementwise.py:258  def silu_and_mul(input, out=None)
#                     sgl-kernel/python/sgl_kernel/elementwise.py:269      torch.ops.sgl_kernel.silu_and_mul.default(out, input)
#   [in-tree CUDA源]  sgl-kernel/csrc/elementwise/activation.cu:85          void silu_and_mul(at::Tensor& out, at::Tensor& input) {...
#                     sgl-kernel/csrc/elementwise/activation.cu:95            flashinfer::activation::act_and_mul_kernel<c_type, silu><<<grid,block,0,stream>>>(...)
#   [torch.ops 注册]  sgl-kernel/csrc/common_extension.cc:76               m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
#                     sgl-kernel/csrc/common_extension.cc:77               m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
#   [编入扩展 CMake]  sgl-kernel/CMakeLists.txt:265                        "csrc/elementwise/activation.cu"   # 源码直接进源列表(非 FetchContent)
#   [sglang 使用]     python/sglang/srt/layers/sampler.py:32              from sgl_kernel import (top_k_renorm_prob, ...)   # 同样 from sgl_kernel import 方式
# =============================================================================
def demo_F2_sgl_kernel_builtin():
    # ---- import ----
    from sgl_kernel import silu_and_mul

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT: sglang/sgl-kernel/tests/test_activation.py:13-17 (test_fused_silu_mul)
    #   golden 是内联一行 (L15), 对照 torch.testing.assert_close(rtol/atol=1e-3)
    def _golden_silu_and_mul(x):
        # 摘自 test_activation.py:15 —— gate=前半, up=后半
        dim = x.shape[-1] // 2
        return x[..., dim:] * torch.nn.functional.silu(x[..., :dim])

    # ---- 假输入: 最后一维必须是偶数(切成 gate/up 两半)、fp16/bf16、16B 对齐 ----
    x = torch.randn(2, 8, device=CUDA, dtype=torch.float16)  # (num_tokens=2, 2*d=8)

    # ---- 调用 (等价于 torch.ops.sgl_kernel.silu_and_mul.default(out, x)) ----
    out = silu_and_mul(x)  # -> (2, 4)
    return out


# =============================================================================
# F3  sgl_kernel_thirdparty  —  sgl-kernel 通过 CMake FetchContent 编入的三方算子
# -----------------------------------------------------------------------------
# 代表算子: flash_attn_varlen_func / flash_attn_with_kvcache  (FlashAttention-3, Hopper)
#   —— kernel 源码 **不在** sgl-kernel/csrc，而是 FetchContent 从三方 repo
#      (sgl-project/sgl-attn) 拉取 hopper/*.cpp,*.cu 编成独立 flash_ops 模块，
#      通过 torch.ops.sgl_kernel.fwd 暴露。
#
# 代码证据:
#   [FetchContent 拉三方] sgl-kernel/CMakeLists.txt:82-87
#         FetchContent_Declare(repo-flash-attention
#             URL .../sgl-project/sgl-attn/archive/bcf72cc....tar.gz ...)
#         FetchContent_Populate(repo-flash-attention)
#   [三方源编入模块]     sgl-kernel/CMakeLists.txt:473-481
#         set(FLASH_SOURCES "csrc/flash_extension.cc"
#             "${repo-flash-attention_SOURCE_DIR}/hopper/flash_api.cpp" ...)   # 注意用的是三方 SOURCE_DIR
#         Python_add_library(flash_ops MODULE ... ${FLASH_SOURCES})
#   [三方 kernel 源/注册] sgl-attn/hopper/flash_api.cpp:1673   TORCH_LIBRARY(flash_attn_3, m) { m.def("fwd(" ...  # 只存在于 sgl-attn 仓库
#   [python wrapper]     sgl-kernel/python/sgl_kernel/flash_attn.py:7-12   from sgl_kernel import flash_ops  # FetchContent'd FA3 .so
#                        sgl-kernel/python/sgl_kernel/flash_attn.py:278        out,... = torch.ops.sgl_kernel.fwd.default(...)  (varlen 路径)
#   [sglang 使用]        python/sglang/srt/layers/attention/flashattention_backend.py:176
#                        python/sglang/srt/layers/attention/xpu_backend.py:24
#             from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
# =============================================================================
def demo_F3_sgl_kernel_thirdparty():
    # ---- import (与 xpu_backend.py:24 一致) ----
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT (两处, 均用同名 attention_ref 作 golden):
    #   sgl-kernel/tests/test_flash_attention.py:186-316  def attention_ref(...)  (完整版)
    #   sgl-attn/flash_attn/cute/testing.py:326-465        def attention_ref(...)  (FA3/FA4 cute 版)
    #   对照 torch.testing.assert_close。下面按其核心数学摘录朴素注意力 (省略 descale/dropout/sink 等分支)。
    def _golden_attention_ref(q, k, v, causal=False, softcap=0.0):
        # 摘自 attention_ref 的核心 scores/softmax 段 (test_flash_attention.py:246-289)
        import math

        from einops import repeat

        q, k, v = q.float(), k.float(), v.float()
        # GQA: 把 kv head 复制到 q head 数
        k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
        v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
        d = q.shape[-1]
        softmax_scale = 1.0 / math.sqrt(d)
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
        if softcap > 0:
            scores = torch.tanh(scores / softcap) * softcap
        if causal:
            seqlen_q, seqlen_k = q.shape[1], k.shape[1]
            row = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
            col = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
            causal_mask = col > row + (seqlen_k - seqlen_q)
            scores.masked_fill_(causal_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        return torch.einsum("bhts,bshd->bthd", attention, v)

    # ---- 假输入: (total_tokens, num_heads, head_dim), head_dim in {64,96,128} ----
    q = torch.randn(4, 2, 64, device=CUDA, dtype=torch.bfloat16)  # 4 query tokens
    k = torch.randn(4, 2, 64, device=CUDA, dtype=torch.bfloat16)
    v = torch.randn(4, 2, 64, device=CUDA, dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32, device=CUDA)  # 一条 len=4 的序列
    cu_seqlens_k = torch.tensor([0, 4], dtype=torch.int32, device=CUDA)

    # ---- 调用 (内部 -> torch.ops.sgl_kernel.fwd.default) ----
    out = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=4, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=4, cu_seqlens_k=cu_seqlens_k,
        causal=True,
    )
    return out


# =============================================================================
# F4  sglang_jit (from csrc)  —  sglang 自有的运行时 C++/CUDA JIT 编译
# -----------------------------------------------------------------------------
# 代表算子: sglang.jit_kernel.add_constant  (以及生产用的 norm.fused_add_rmsnorm)
#   —— sglang 自带 JIT 基建 (sglang/jit_kernel/)，运行时把自己 csrc/*.cu,*.cuh
#      用 load_jit()(底层 tvm_ffi.cpp.load/load_inline)现场编译成 .so。
#
# 代码证据 (sglang):
#   [JIT loader 机制]  python/sglang/jit_kernel/utils.py:227-228   cpp/cuda_files 从 KERNEL_PATH/"csrc" 解析 (sglang 自有源)
#                      python/sglang/jit_kernel/utils.py:247-257   load_inline(module_name, cpp_sources=..., cuda_sources=..., build_directory=...)  # 运行时编译
#   [import loader]    python/sglang/jit_kernel/add_constant.py:7  from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
#   [触发 JIT]         python/sglang/jit_kernel/add_constant.py:16     return load_jit("add_constant", *args, cuda_files=["add_constant.cuh"], ...)
#   [自有源]           python/sglang/jit_kernel/csrc/add_constant.cuh:26  __global__ void add_constant_kernel(...)  # #include <sgl_kernel/tensor.h> 等自有头
#   [调用]             python/sglang/jit_kernel/add_constant.py:24-28  def add_constant(src, constant): module=_jit_add_constant_module(constant); module.add_constant(dst, src)
#   [生产落地 rmsnorm] python/sglang/jit_kernel/norm.py:57            _jit_rmsnorm_module -> load_jit("rmsnorm", cuda_files=["elementwise/rmsnorm.cuh"], ...)
#                      python/sglang/srt/layers/layernorm.py:127      from sglang.jit_kernel.norm import fused_add_rmsnorm as _jit_fused_add_rmsnorm
#                      python/sglang/srt/layers/layernorm.py:309          _jit_fused_add_rmsnorm(...)
# =============================================================================
def demo_F4_sglang_jit_csrc():
    # ---- import (与 add_constant.py 对外接口一致) ----
    from sglang.jit_kernel.add_constant import add_constant

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT:
    #   add_constant: sglang/python/sglang/jit_kernel/tests/test_add_constant.py:13-37
    #       golden 内联为精确整数比较 assert torch.all(dst == src + constant)
    #   rmsnorm:      sglang/python/sglang/jit_kernel/tests/test_rmsnorm_hf.py:24-37 (hf_rmsnorm_reference)
    #   fused_add_rmsnorm: sglang/python/sglang/jit_kernel/tests/test_fused_add_rmsnorm.py:37-44
    def _golden_add_constant(src, constant):
        # 摘自 test_add_constant.py:16-18 (精确, 无容差)
        return src + constant

    def _golden_rmsnorm_hf(x, w, eps):
        # 摘自 test_rmsnorm_hf.py:24-30 (hf_rmsnorm_reference; HF LlamaRMSNorm 语义)
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return w * x_normed.to(x.dtype)

    def _golden_fused_add_rmsnorm(x, residual, w, eps):
        # 摘自 test_fused_add_rmsnorm.py:37-44 (forward_native_hf_reference)
        sum_fp32 = x.to(torch.float32) + residual.to(torch.float32)
        residual_out = sum_fp32.to(x.dtype)
        variance = sum_fp32.pow(2).mean(-1, keepdim=True)
        out = w * (sum_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)
        return out, residual_out

    # ---- 假输入 ----
    src = torch.arange(1024, dtype=torch.int32, device=CUDA)

    # ---- 调用: 首次调用会现场编译 csrc/add_constant.cuh (之后缓存) ----
    dst = add_constant(src, constant=5)  # dst == src + 5
    return dst


# =============================================================================
# F5  thirdparty_aot  —  三方 (flashinfer) 预编译 AOT 算子 (torch.ops.flashinfer.*)
# -----------------------------------------------------------------------------
# 代表算子: flashinfer.sampling.top_k_top_p_sampling_from_probs
#   —— flashinfer 所有 C++/CUDA op 都走 JitSpec；当预编译 .so 存在(AOT wheel)时
#      build_and_load() 直接 load 现成 .so，运行时不跑 nvcc，注册成 torch.ops.flashinfer.*。
#
# 代码证据:
#   [sglang import]  python/sglang/srt/layers/sampler.py:28
#         from flashinfer.sampling import (min_p_sampling_from_probs, top_k_top_p_sampling_from_probs)
#   [sglang 调用]    python/sglang/srt/layers/sampler.py:243
#         batch_next_token_ids = top_k_top_p_sampling_from_probs(probs.contiguous(), sampling_info.top_ks, sampling_info.top_ps, filter_apply_order="joint")
#   [AOT 机制]       flashinfer/flashinfer/jit/core.py:307  def build_and_load(self):
#                    flashinfer/flashinfer/jit/core.py:308      if self.is_aot: return self.load(self.aot_path)   # 预编译 .so 直接 load
#                    flashinfer/flashinfer/jit/core.py:260      is_aot = self.aot_path.exists()  (aot_path 指向 FLASHINFER_AOT_DIR 预编译缓存)
#                    flashinfer/flashinfer/sampling.py:67   module = gen_sampling_module().build_and_load(); @register_custom_op("flashinfer::top_k_top_p_sampling_from_probs", ...)
#   [另一 AOT 代表]  python/sglang/srt/layers/layernorm.py:56  import flashinfer.norm  ->  flashinfer.norm.layernorm(...)
# =============================================================================
def demo_F5_thirdparty_aot():
    # ---- import (与 sampler.py:28 一致) ----
    from flashinfer.sampling import top_k_top_p_sampling_from_probs

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT: flashinfer/tests/utils/test_sampling.py:320-359
    #   采样是随机的, 没有闭式 golden; 而是构造 top-p∧top-k 的【合法 token mask】,
    #   断言 kernel 采出的 token 一定落在 mask==1 上 (对照 assert mask[..., samples]==1)。
    def _golden_topk_topp_mask(normalized_prob, k, p):
        # 摘自 test_sampling.py:334-344 (inline golden mask)
        eps = 1e-4
        batch_size, vocab_size = normalized_prob.shape
        # top-p mask
        sorted_prob, indices = torch.sort(normalized_prob, descending=False)
        cdf = torch.cumsum(sorted_prob, dim=-1)
        mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=normalized_prob.device)
        mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
        # top-k mask
        sorted_prob, _ = torch.sort(normalized_prob, descending=True)
        pivot = sorted_prob[:, k - 1]
        mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
        # overall mask
        return torch.minimum(mask_top_p, mask_top_k)

    # ---- 假输入: (batch=4, vocab=32000) 的概率分布 ----
    probs = torch.softmax(torch.randn(4, 32000, device=CUDA), dim=-1)
    top_ks = torch.full((4,), 50, dtype=torch.int32, device=CUDA)
    top_ps = torch.full((4,), 0.9, device=CUDA)

    # ---- 调用 (底层 torch.ops.flashinfer.*, 预编译 AOT) ----
    ids = top_k_top_p_sampling_from_probs(probs, top_ks, top_ps, filter_apply_order="joint")
    return ids


# =============================================================================
# F6  thirdparty_triton_dsl  —  三方 (flashinfer) 的 triton / cuteDSL 纯 python DSL
# -----------------------------------------------------------------------------
# 代表算子: flashinfer.triton.norm.rms_norm  (@triton.jit rms_norm_kernel)
#   —— 纯 python DSL 路径，首次 launch 由 triton 编译器生成 PTX，
#      既不走 flashinfer 的 nvcc JitSpec，也不是预编译 torch op。
#
# 注意: sglang **没有** import flashinfer.triton (它用自己仓库内的 triton, 见 F1)。
#       这里演示的是 flashinfer 侧该机制的存在与调用方式。
#
# 代码证据:
#   [triton kernel]  flashinfer/flashinfer/triton/kernels/norm.py:7   import triton / @triton.jit def rms_norm_kernel(...)
#   [python launcher]flashinfer/flashinfer/triton/norm.py:9   def rms_norm(...)
#                    flashinfer/flashinfer/triton/norm.py:27      rms_norm_kernel[(b,)](...)   # triton grid launch
#   [cuteDSL 变体]   flashinfer/flashinfer/norm/__init__.py:63-70  from .kernels import rmsnorm_cute  (nvidia-cutlass-dsl 路径)
#   [sglang 侧]      无 (rg "flashinfer.triton" 在 sglang/python 下无命中; sglang 用自带 triton)
# =============================================================================
def demo_F6_thirdparty_triton_dsl():
    # ---- import (flashinfer 侧接口) ----
    from flashinfer.triton.norm import rms_norm

    # ---- golden / 参考实现: 【找不到 —— 置空】 ----
    # 原因: 目标算子 flashinfer.triton.norm.rms_norm (triton 版) 在 flashinfer 全仓
    #   【没有任何 UT】—— tests/ 下搜不到 flashinfer.triton / triton.norm.rms_norm 的调用点,
    #   该 triton 模块未被测试覆盖 (flashinfer 实际测的是 C++ 版 flashinfer.norm.rmsnorm)。
    # 参考: 同一 RMSNorm 数学在【兄弟 C++ 版】测试里有 golden, 可作等价对照 (非本算子的 UT):
    #   flashinfer/tests/utils/test_norm.py:27-34  def llama_rms_norm(x, w, eps):
    #       x = x.float(); variance = x.pow(2).mean(-1, keepdim=True)
    #       x = x * torch.rsqrt(variance + eps); x = x * w.float(); return x.to(orig_dtype)
    _golden = None  # 目标 triton 算子无 UT, 故置空

    # ---- 假输入 ----
    x = torch.randn(8, 4096, device=CUDA, dtype=torch.float16)
    w = torch.randn(4096, device=CUDA, dtype=torch.float16)
    out = torch.empty_like(x)

    # ---- 调用 (内部 launch @triton.jit rms_norm_kernel) ----
    rms_norm(x, w, out, eps=1e-6)
    return out


# =============================================================================
# F7  thirdparty_cpp_jit  —  三方 (flashinfer) 运行时 nvcc 编 C++/CUDA (JitSpec)
# -----------------------------------------------------------------------------
# 代表算子: flashinfer.BatchPrefillWithPagedKVCacheWrapper  (paged prefill attention)
#   —— .plan() 时按 (head_dim/mask/dtype...) 现场生成 .cu 模板并 gen_jit_spec，
#      build_and_load() 因无预编译 .so 而运行时跑 ninja+nvcc 编译 (对比 F5)。
#
# 代码证据:
#   [sglang import]  python/sglang/srt/layers/attention/flashinfer_backend.py:58
#         from flashinfer import (BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper, fast_decode_plan)
#   [sglang 调用]    python/sglang/srt/layers/attention/flashinfer_backend.py:309/736  构造 wrapper
#                    python/sglang/srt/layers/attention/flashinfer_backend.py:1236     begin_forward(=wrapper.plan) 触发 JIT
#                    python/sglang/srt/layers/attention/flashinfer_backend.py:827       o = prefill_wrapper_paged.forward(q.view(...), kv_buffer, causal=..., sm_scale=...)
#   [JIT 机制链]     flashinfer/flashinfer/prefill.py:447    module = gen_batch_prefill_module(backend, *args).build_and_load()
#                    flashinfer/flashinfer/jit/attention/modules.py:1620-1658  运行时渲染 batch_prefill_paged_kernel_mask_*.cu -> gen_jit_spec(...)
#                    flashinfer/flashinfer/jit/core.py:301-302  build_and_load 非 AOT 分支: write_ninja()/run_ninja() 现场编译
#                    flashinfer/flashinfer/jit/cpp_ext.py:305   command = $nvcc -shared $in ...   # 真·运行时 nvcc
# =============================================================================
def demo_F7_thirdparty_cpp_jit():
    # ---- import (与 flashinfer_backend.py:58 一致) ----
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT: flashinfer/tests/attention/test_batch_prefill_kernels.py:66-289
    #   注意: 该主测试的 golden 不是 torch, 而是另一个 flashinfer kernel
    #        single_prefill_with_kv_cache (kernel-vs-kernel, :280-289)。
    # 纯 torch-native 的 paged-attention golden 在 trtllm-gen 测试路径里:
    #   flashinfer/tests/attention/test_trtllm_gen_attention_decode.py:518-614  def sdpa_paged_reference(...)
    #   下面按其核心逻辑摘录 (gather 分页 KV -> GQA 扩展 -> 因果 mask -> SDPA)。
    def _golden_sdpa_paged(ref_q, ref_kv_cache, q_lens, seq_lens, page_table,
                           page_size, num_qo_heads, num_kv_heads, head_dim,
                           kv_layout, window_left):
        # 摘自 sdpa_paged_reference (test_trtllm_gen_attention_decode.py:518-614)
        sm_scale = 1.0 / (head_dim**0.5)
        batch_size = q_lens.shape[0]
        q_indptr = torch.cat([
            torch.zeros(1, dtype=q_lens.dtype, device=q_lens.device),
            torch.cumsum(q_lens, dim=0),
        ])
        outputs = []
        for b in range(batch_size):
            q_b = ref_q[q_indptr[b].item(): q_indptr[b + 1].item()]  # [q_len, Hq, D]
            s_len = seq_lens[b].item()
            num_pages = (s_len + page_size - 1) // page_size
            kv_pages = ref_kv_cache[page_table[b, :num_pages]]  # gather 分页 KV
            k_pages, v_pages = kv_pages[:, 0], kv_pages[:, 1]
            if kv_layout == "HND":
                k_flat = k_pages.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:s_len]
                v_flat = v_pages.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:s_len]
            else:  # NHD
                k_flat = k_pages.reshape(-1, num_kv_heads, head_dim)[:s_len]
                v_flat = v_pages.reshape(-1, num_kv_heads, head_dim)[:s_len]
            q_len = q_b.shape[0]
            head_grp = num_qo_heads // num_kv_heads
            k_exp = k_flat.unsqueeze(2).expand(-1, num_kv_heads, head_grp, -1).reshape(s_len, num_qo_heads, head_dim)
            v_exp = v_flat.unsqueeze(2).expand(-1, num_kv_heads, head_grp, -1).reshape(s_len, num_qo_heads, head_dim)
            q_t = q_b.transpose(0, 1).float()
            k_t = k_exp.transpose(0, 1).float()
            v_t = v_exp.transpose(0, 1).float()
            kv_offset = s_len - q_len
            q_pos = torch.arange(q_len, device=q_b.device).unsqueeze(1) + kv_offset
            k_pos = torch.arange(s_len, device=q_b.device).unsqueeze(0)
            causal_mask = k_pos <= q_pos
            if window_left >= 0:
                causal_mask = causal_mask & (q_pos - k_pos <= window_left)
            attn_mask = causal_mask.unsqueeze(0).expand(num_qo_heads, -1, -1)
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask, scale=sm_scale)
            outputs.append(out_b.transpose(0, 1).to(ref_q.dtype))
        return torch.cat(outputs, dim=0)

    # ---- 假输入 + plan (plan 触发 gen_batch_prefill_module().build_and_load() -> nvcc) ----
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=CUDA)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr=torch.tensor([0, 16], dtype=torch.int32, device=CUDA),
        paged_kv_indptr=torch.tensor([0, 4], dtype=torch.int32, device=CUDA),
        paged_kv_indices=torch.arange(4, dtype=torch.int32, device=CUDA),
        paged_kv_last_page_len=torch.tensor([16], dtype=torch.int32, device=CUDA),
        num_qo_heads=8, num_kv_heads=8, head_dim_qk=128, page_size=16,
    )

    # ---- 假 q / paged-kv 并调用 ----
    q = torch.randn(16, 8, 128, device=CUDA, dtype=torch.float16)
    kv = torch.randn(4, 2, 16, 8, 128, device=CUDA, dtype=torch.float16)  # paged kv cache
    o = wrapper.run(q, kv)
    return o


# =============================================================================
# F8  sglang_jit (from cuteDSL)  —  sglang 自有、用 CuTe DSL 写并运行时编译的 kernel
# -----------------------------------------------------------------------------
# 代表算子: sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift
#             .fused_norm_scale_shift
#   —— kernel 用 NVIDIA CuTe DSL (import cutlass / cutlass.cute / @cute.jit /
#      @cute.kernel) 在 sglang 仓库内编写，运行时 cute.compile(...) 现场编译。
#
# 代码证据 (sglang):
#   [cuteDSL import] python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py:3-5
#         import cuda.bindings.driver as cuda / import cutlass / import cutlass.cute as cute
#   [kernel 定义]    .../scale_residual_norm_scale_shift.py:87    @cute.jit def __call__(self, mY, mResOut, ...)
#                    .../scale_residual_norm_scale_shift.py:106   cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), ...)
#                    .../scale_residual_norm_scale_shift.py:134   @cute.kernel def kernel(self, ...)
#   [运行时 JIT]     .../scale_residual_norm_scale_shift.py:333   compiled_fn = cute.compile(kernel, *fake_sig_args, options="--enable-tvm-ffi")
#   [封装 torch op]  .../scale_residual_norm_scale_shift.py:278   @torch.library.custom_op("sglang::fused_norm_scale_shift", ...)
#   [调用]           python/sglang/multimodal_gen/runtime/layers/layernorm.py:489
#         from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import fused_scale_residual_norm_scale_shift
#         .../layernorm.py:503    return fused_scale_residual_norm_scale_shift(residual.contiguous(), x.contiguous(), ...)
#
#   另一代表(GDN 线性注意力 decode):
#     python/sglang/jit_kernel/cutedsl_gdn.py:41  @cute.kernel def gdn_kernel_small_batch(...)
#     python/sglang/jit_kernel/cutedsl_gdn.py:1338 compiled_kernel = cute.compile(kernel_func, ...)
#     python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py:15 import + :94 调用
# =============================================================================
def demo_F8_sglang_jit_cutedsl():
    # ---- import (与 layernorm.py:489 一致) ----
    from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
        fused_norm_scale_shift,
    )

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT: sglang/python/sglang/jit_kernel/tests/diffusion/test_fused_norm_scale_shift.py:56-120
    #   golden 用 torch.rms_norm/layer_norm + (1+scale)*x+shift, 对照 torch.testing.assert_close。
    def _golden_apply_scale_shift(y, scale, shift):
        # 摘自 test_fused_norm_scale_shift.py:56-66 (_apply_scale_shift; 仅取 2D/3D 分支)
        from einops import rearrange

        scale = rearrange(scale, "b d -> b 1 d") if scale.ndim == 2 else scale
        shift = rearrange(shift, "b d -> b 1 d") if shift.ndim == 2 else shift
        return y * (1 + scale) + shift

    def _golden_fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps):
        # 摘自 test_fused_norm_scale_shift.py:69-86 (fused_norm_scale_shift_ref)
        original_dtype = x.dtype
        x, weight, bias, scale, shift = (
            v.float() if v is not None else v for v in [x, weight, bias, scale, shift]
        )
        if norm_type == "layer":
            norm = torch.layer_norm(x, x.shape[-1:], eps=eps, weight=weight, bias=bias)
        else:
            norm = torch.rms_norm(x, x.shape[-1:], eps=eps, weight=weight)
        return _golden_apply_scale_shift(norm, scale, shift).to(original_dtype)

    # ---- 假输入: D 需为 256 的倍数, <= 8192 ----
    B, S, D = 2, 16, 2048
    x = torch.randn(B, S, D, dtype=torch.bfloat16, device=CUDA)
    weight = torch.randn(D, dtype=torch.bfloat16, device=CUDA)
    bias = torch.zeros(D, dtype=torch.bfloat16, device=CUDA)
    scale = torch.randn(B, 1, D, dtype=torch.bfloat16, device=CUDA)
    shift = torch.randn(B, 1, D, dtype=torch.bfloat16, device=CUDA)

    # ---- 调用: 首次调用 cute.compile 现场编译 CuTe DSL kernel, 之后缓存 ----
    y = fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type="rms", eps=1e-5)
    return y


# =============================================================================
# F9  thirdparty_cpp_jit (DeepGEMM)  —  三方运行时 C++/CUDA JIT, 但 route 与 flashinfer 不同
# -----------------------------------------------------------------------------
# 代表算子: deep_gemm.fp8_paged_mqa_logits  (DeepSeek indexer 的 fp8 paged MQA logits)
#   —— DeepGEMM 是 sgl 社区单独维护的三方库(独立 pip 包, 非 FetchContent 进 sgl-kernel)。
#      与 F7(flashinfer) 同为"三方运行时 C++/CUDA JIT", 但 JIT route 明显不同:
#        * flashinfer(F7): python 侧 JitSpec 渲染 .cu -> 外部 ninja+nvcc 编 .so -> import so
#            (flashinfer/jit/core.py build_and_load + cpp_ext.py:305 `$nvcc -shared`)
#        * DeepGEMM(F9):  先 AOT 编一个 _C 扩展(setup.py CUDAExtension, 依赖 cudart+nvrtc),
#            真正的 GPU kernel 在 _C 内部由 C++ JIT 引擎运行时生成 CUDA 源码字符串,
#            用 **NVRTC(默认) 或 NVCC** 编成 cubin/ptx 再 cuLaunchKernel, 全程在 C++ 层,
#            python 只 import 现成符号。=> "AOT 壳 + C++ 内嵌 NVRTC JIT"
#
# 代码证据:
#   [DeepGEMM 是独立包直接 import]  DeepGEMM/deep_gemm/__init__.py:68   from ._C import (... fp8_paged_mqa_logits ...)
#   [_C 是 AOT 编的扩展]            DeepGEMM/setup.py:106   CUDAExtension(name='deep_gemm._C', ...)
#                                   DeepGEMM/setup.py:43    build_libraries = ['cudart', 'nvrtc']   # 链接 nvrtc
#   [C++ 层 NVRTC JIT 引擎]         DeepGEMM/csrc/jit/compiler.hpp:8    #include <nvrtc.h>
#                                   DeepGEMM/csrc/jit/compiler.hpp:100  std::shared_ptr<KernelRuntime> build(name, code)  # 传入 CUDA 源码字符串, 运行时编译+cache
#   [kernel 源码运行时生成+编译]    DeepGEMM/csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp:86  code = ...::generate(args);
#                                   .../smxx_fp8_fp4_paged_mqa_logits.hpp:87  runtime = compiler->build("smxx_paged_mqa_logits_metadata", code);  # NVRTC 编译
#   [C++ 实现/注册]                 DeepGEMM/csrc/apis/attention.hpp:401  static torch::Tensor fp8_paged_mqa_logits(...)
#                                   DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:30  void sm90_fp8_paged_mqa_logits(...)  # 真 kernel
#   [sglang 侧封装/开关]            python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:26  import deep_gemm
#                                   python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:44  os.environ["DG_JIT_USE_NVRTC"] = ...  # 选 NVRTC/NVCC route
#   [sglang import + 调用]          python/sglang/srt/layers/attention/dsv4/indexer.py:529  from deep_gemm import fp8_paged_mqa_logits as fn
#                                   python/sglang/srt/layers/attention/dsv4/indexer.py:553      logits = fn(...)
# =============================================================================
def demo_F9_thirdparty_deepgemm_jit():
    # ---- import (与 dsv4/indexer.py:529 一致) ----
    from deep_gemm import fp8_paged_mqa_logits, get_paged_mqa_logits_metadata

    # ---- golden / 参考实现 (仅供对照, 不调用) ----
    # 找到的 UT: DeepGEMM/tests/test_attention.py:239-421 (test_paged_mqa_logits)
    #   golden = ref_paged_mqa_logits (:196-224), 对照 calc_diff < 1e-3 (fp8)。下面为逐行摘录。
    def _golden_ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                                     max_model_len, use_2d_context_lens):
        # 摘自 test_attention.py:196-224 (ref_paged_mqa_logits)
        batch_size, next_n, num_heads, dim = q.size()
        num_block, block_size, _, dim = kv_cache.size()
        logits = torch.full([batch_size * next_n, max_model_len], float('-inf'),
                            device=q.device, dtype=torch.float32)
        context_lens = context_lens.tolist()
        for i in range(batch_size):
            context_len = context_lens[i]
            q_offsets = torch.full((next_n,), context_len, device='cuda', dtype=torch.int32) \
                if use_2d_context_lens else torch.arange(context_len - next_n, context_len, device='cuda')
            weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
            num_blocks = (context_len + block_size - 1) // block_size
            block_idxs = block_tables[i][:num_blocks]
            kv_slice = kv_cache[block_idxs]                 # [num_blocks, block_size, kv_heads, dim]
            kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)  # [kv_heads, dim, total_tokens]
            qx = q[i].transpose(0, 1)                       # [num_heads, next_n, dim]
            s = torch.matmul(qx, kx).to(logits.dtype)       # [num_heads, next_n, total_tokens]
            total_len = num_blocks * block_size
            k_offsets = torch.arange(0, total_len, device=q.device)
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :], s, float('-inf'))
            s = torch.relu(s) * weight_slice[..., None]     # relu 后按 head 权重, 再对 head 求和
            s = s.sum(dim=0)                                # [next_n, total_tokens]
            logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(
                k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))
        return logits

    # ---- 假输入 (fp8 paged MQA logits; shape 仅示意, 不保证数值) ----
    num_tokens, next_n, heads, dim = 4, 1, 32, 128
    q = torch.randn(num_tokens, next_n, heads, dim, device=CUDA, dtype=torch.bfloat16)
    # kv_cache: fp8 分页缓存 (num_blocks, block_size, 1, dim) —— dummy
    kv_cache = torch.randn(8, 64, 1, dim, device=CUDA).to(torch.float8_e4m3fn)
    weights = torch.randn(num_tokens, heads, device=CUDA, dtype=torch.float32)
    context_lens = torch.tensor([64, 128, 64, 96], dtype=torch.int32, device=CUDA)
    block_tables = torch.zeros(num_tokens, 8, dtype=torch.int32, device=CUDA)

    # ---- 先算调度 metadata, 再调 kernel (内部 C++ NVRTC 现场编译, 之后 cache) ----
    schedule_meta = get_paged_mqa_logits_metadata(context_lens, 64, num_sms=132)
    logits = fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, schedule_meta, max_model_len=1024
    )
    return logits


# =============================================================================
# 汇总表
# =============================================================================
BACKENDS = {
    "F0_pytorch_native":     demo_F0_pytorch_native,      # torch/aten/cuBLAS: SDPA
    "F1_sglang_triton":      demo_F1_sglang_triton,       # 仓库内 @triton.jit: recompute_w_u_fwd
    "F2_sgl_kernel_builtin": demo_F2_sgl_kernel_builtin,  # sgl-kernel csrc AOT: silu_and_mul
    "F3_sgl_kernel_thirdparty": demo_F3_sgl_kernel_thirdparty,  # FetchContent sgl-attn: FA3 fwd
    "F4_sglang_jit_csrc":    demo_F4_sglang_jit_csrc,     # sglang JIT(csrc): add_constant / rmsnorm
    "F5_thirdparty_aot":     demo_F5_thirdparty_aot,      # flashinfer AOT: top_k_top_p_sampling
    "F6_thirdparty_triton_dsl": demo_F6_thirdparty_triton_dsl,  # flashinfer triton: rms_norm
    "F7_thirdparty_cpp_jit": demo_F7_thirdparty_cpp_jit,  # flashinfer JitSpec nvcc: BatchPrefill
    "F8_sglang_jit_cutedsl": demo_F8_sglang_jit_cutedsl,  # sglang CuTe DSL: fused_norm_scale_shift
    "F9_thirdparty_deepgemm_jit": demo_F9_thirdparty_deepgemm_jit,  # DeepGEMM AOT壳+C++内嵌NVRTC JIT: fp8_paged_mqa_logits
}


def all_backends_test():
    for name, fn in BACKENDS.items():
        try:
            out = fn()
            shape = tuple(out.shape) if hasattr(out, "shape") else type(out)
            print(f"[OK]   {name:26s} -> {shape}")
        except Exception as e:  # noqa: BLE001  (演示用途，容忍缺环境)
            print(f"[SKIP] {name:26s} -> {type(e).__name__}: {e}")
            

if __name__ == "__main__":
    # 仅演示"import + 调用"结构; 绝大多数需要 CUDA GPU + 对应包。
    # 这里逐个 try 运行，缺环境则打印跳过原因(不影响作为"证据清单"阅读)。
    all_backends_test()
