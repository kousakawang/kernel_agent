# Attention Layer Forward Path Decomposition (SM90) — SGLang

> 本文整理 sglang 仓库中 6 种注意力机制在 SM90 (Hopper) 硬件上整个 attention-layer forward 的主要调用路径。
> 按 kernel 粒度列出每一步的接口名及实现类别。

## 实现类别约定

| 标记 | 含义 |
|------|------|
| `pytorch.native(cuBLAS)` | PyTorch 原生接口，底层调用 cuBLAS GEMM |
| `triton` | Triton DSL 编写的 kernel |
| `cuteDSL` | CuTe DSL 编写的 kernel |
| `raw-cuda(sgl-kernel)` | sgl-kernel 仓库中的 CUDA kernel（含 CUTLASS 实现） |
| `raw-cuda(sgl-kernel/flash_attn)` | sgl-kernel 中集成的 Flash Attention 3 CUDA kernel |
| `third-party(flashinfer)` | FlashInfer 库 |
| `third-party(flash_attention)` | Flash Attention 库（非 sgl-kernel 内置版本） |
| `third-party(flashMLA)` | FlashMLA 库（已集成到 sgl-kernel） |
| `third-party(deep_gemm)` | DeepGemm FP8 GEMM 库 |
| `JIT-kernel` | sglang jit_kernel 编译的 CUDA kernel（rope.cuh 等） |

---

## 1. GQA — Grouped Query Attention（无 Sliding Window）

> 代表模型: Llama-3, Qwen2, Mistral (full attention layers)
> 模型文件: `sglang/python/sglang/srt/models/llama.py`
> 注意力层: `RadixAttention` → `AttentionBackend.forward()`

### 路径 1: FlashInfer 后端 (`--attention-backend flashinfer`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | 融合在 rope kernel 中 (`FusedRopeKernel::run_fused`) 或独立 `set_kv_buffer` | JIT-kernel / raw-cuda(sgl-kernel) |
| 4 | attention_decode | `BatchDecodeWithPagedKVCacheWrapper.forward(q, kv_cache)` | third-party(flashinfer) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `token_to_kv_pool.set_kv_buffer(layer_id, loc, k, v)` | raw-cuda(sgl-kernel) |
| 4 | attention_prefill | `BatchPrefillWithPagedKVCacheWrapper.forward(q, kv_cache)` 或 `BatchPrefillWithRaggedKVCacheWrapper.forward(q, k, v)` | third-party(flashinfer) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

---

### 路径 2: Flash Attention 3 后端 (`--attention-backend fa3`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` | raw-cuda(sgl-kernel) |
| 4 | attention_decode | `flash_attn_with_kvcache(q, k_cache, v_cache, page_table, cache_seqlens)` | raw-cuda(sgl-kernel/flash_attn) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` | raw-cuda(sgl-kernel) |
| 4 | attention_prefill | `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)` 或 `flash_attn_with_kvcache(q, k_cache, v_cache, page_table)` | raw-cuda(sgl-kernel/flash_attn) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

---

### 路径 3: Triton 后端 (`--attention-backend triton`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` | raw-cuda(sgl-kernel) |
| 4 | attention_decode | `decode_attention_fwd(q, kv_buffer, ...)` | triton |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` | raw-cuda(sgl-kernel) |
| 4 | attention_extend | `extend_attention_fwd(q, k, v, o, ...)` 或 `extend_attention_fwd_unified(...)` | triton |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

---

## 2. GQA — Grouped Query Attention（有 Sliding Window）

> 代表模型: Mistral (SWA layers), Qwen2 (alternating SWA)
> 与无 SWA 的 GQA 区别在于使用独立的 SWA KV Pool 和窗口参数

### 路径 1: FlashInfer 后端 (Sliding Window)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store_swa | `swa_kv_pool.set_kv_buffer(layer_id, swa_loc, k, v)` + `token_to_kv_pool.set_kv_buffer(layer_id, loc, k, v)` | raw-cuda(sgl-kernel) |
| 4 | attention_decode_swa | `BatchDecodeWithPagedKVCacheWrapper.forward(q, swa_kv_cache)` (wrapper[0], 使用 SWA pool) | third-party(flashinfer) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` (同时写 full pool 和 SWA pool) | raw-cuda(sgl-kernel) |
| 4a | attention_prefill_ragged | `BatchPrefillWithRaggedKVCacheWrapper.forward(q, k, v)` (ragged 部分) | third-party(flashinfer) |
| 4b | attention_prefill_paged | `BatchPrefillWithPagedKVCacheWrapper.forward(q, swa_kv_cache, window_left=W)` (paged 部分, 带窗口) | third-party(flashinfer) |
| 4c | merge_states | `_safe_merge_state(ragged_o, ragged_lse, paged_o, paged_lse)` | triton / third-party(flashinfer) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

---

### 路径 2: Flash Attention 3 后端 (Sliding Window)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` | raw-cuda(sgl-kernel) |
| 4 | attention_decode_swa | `flash_attn_with_kvcache(q, k_cache, v_cache, page_table, window_size=(W, 0))` | raw-cuda(sgl-kernel/flash_attn) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | qkv_proj | `QKVParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |
| 2 | rope | `apply_rope_with_cos_sin_cache_inplace` → `FusedRopeKernel::run` | JIT-kernel |
| 3 | kv_cache_store | `set_kv_buffer` | raw-cuda(sgl-kernel) |
| 4 | attention_prefill_swa | `flash_attn_with_kvcache(q, k_cache, v_cache, page_table, window_size=(W, 0))` | raw-cuda(sgl-kernel/flash_attn) |
| 5 | o_proj | `RowParallelLinear` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

---

## 3. MLA — Multi-Latent Attention (DeepSeek V2/V3)

> 代表模型: DeepSeek-V2, DeepSeek-V3
> 模型文件: `sglang/python/sglang/srt/models/deepseek_v2.py`
> 注意力层: `DeepseekV2AttentionMLA` + `DeepseekMLAForwardMixin`
> 
> MLA 特点: 使用低秩 KV latent (kv_lora_rank=512) + absorbed BMM，KV cache 仅存储 latent 维度

### 路径 1: FlashInfer-MLA 后端 (`--attention-backend flashinfer`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | fused_qkv_a_proj | `fused_qkv_a_proj_with_mqa` → `F.linear(x, weight)` (投影到 q_lora_rank + kv_lora_rank + qk_rope_head_dim) | pytorch.native(cuBLAS) |
| 2 | q_a_layernorm | `self.q_a_layernorm(q)` → RMSNorm | raw-cuda(sgl-kernel) |
| 3 | kv_a_layernorm | `self.kv_a_layernorm(k_nope)` → RMSNorm | raw-cuda(sgl-kernel) |
| 4 | q_b_proj | `self.q_b_proj(q)` → `F.linear` (解压 Q: q_lora_rank → num_heads × qk_head_dim) | pytorch.native(cuBLAS) |
| 5 | absorbed_bmm_kc (FP8) | `per_tensor_quant_mla_fp8` + `bmm_fp8(q_nope, w_kc, ...)` | raw-cuda(sgl-kernel) + third-party(deep_gemm) |
| 5' | absorbed_bmm_kc (DeepGemm) | `per_token_group_quant_mla_deep_gemm_masked_fp8` + `deep_gemm.grouped_gemm_nt_f8f8bf16_masked(q, w_kc)` | third-party(deep_gemm) |
| 5'' | absorbed_bmm_kc (BF16) | `torch.bmm(q_nope, w_kc)` | pytorch.native(cuBLAS) |
| 6 | rope | `self.rotary_emb(positions, q_pe, k_pe)` → `FusedRopeKernel::run` | JIT-kernel |
| 7 | kv_cache_store | `set_mla_kv_buffer(layer, loc, k_nope, k_pe)` (存储 kv_lora_rank + rope_dim) | raw-cuda(sgl-kernel) |
| 8 | attention_decode | `BatchMLAPagedAttentionWrapper.run(q_nope_out, q_rope, k_nope_cache, k_rope_cache)` | third-party(flashinfer) |
| 9 | absorbed_bmm_vc (FP8) | `bmm_fp8(attn_output, w_vc, ...)` | raw-cuda(sgl-kernel) + third-party(deep_gemm) |
| 9' | absorbed_bmm_vc (DeepGemm) | `deep_gemm.grouped_gemm_nt_f8f8bf16_masked(attn_output, w_vc)` | third-party(deep_gemm) |
| 9'' | absorbed_bmm_vc (BF16) | `torch.bmm(attn_output, w_vc)` | pytorch.native(cuBLAS) |
| 10 | o_proj | `self.o_proj(output)` → `F.linear(x, weight)` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

与 Decode 相同，仅 step 8 不同：

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 8 | attention_prefill | `BatchPrefillWithPagedKVCacheWrapper.run(q_nope_out, q_rope, k_nope_cache, k_rope_cache)` 或 ragged forward | third-party(flashinfer) |

---

### 路径 2: FlashMLA 后端 (`--attention-backend flashmla`)

> 继承 FlashInfer-MLA 后端，仅 decode kernel 替换为 FlashMLA

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-7 | (同路径1) | — | — |
| 8 | attention_decode | `sgl_kernel.flash_mla.flash_mla_with_kvcache(q, kv_cache, ...)` | raw-cuda(sgl-kernel/flashMLA) |
| 9-10 | (同路径1) | — | — |

#### Prefill/Extend 阶段

与路径1的 Prefill 完全相同（FlashMLA 仅优化 decode）。

---

### 路径 3: CUTLASS-MLA 后端 (`--attention-backend cutlass_mla`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-7 | (同路径1) | — | — |
| 8 | attention_decode | `sgl_kernel.cutlass_mla_decode(q, kv_cache, page_table, ...)` | raw-cuda(sgl-kernel, CUTLASS) |
| 9-10 | (同路径1) | — | — |

---

### 路径 4: FA3 后端 (`--attention-backend fa3`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-7 | (同路径1) | — | — |
| 8 | attention_decode | `flash_attn_with_kvcache(q, k_cache, v_cache, page_table, ...)` (k_cache/v_cache 为 latent 维度) | raw-cuda(sgl-kernel/flash_attn) |
| 9-10 | (同路径1) | — | — |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-7 | (同路径1) | — | — |
| 8 | attention_prefill | `flash_attn_with_kvcache(q, k_cache, v_cache, page_table, cu_seqlens_q, ...)` | raw-cuda(sgl-kernel/flash_attn) |
| 9-10 | (同路径1) | — | — |

---

## 4. GDN — Gated Delta Network (Qwen3.5 Linear Attention)

> 代表模型: Qwen3.5 (混合架构: 部分层 GQA + 部分层 GDN)
> 模型文件: `sglang/python/sglang/srt/models/qwen3_5.py`
> 注意力层: `Qwen3_5GatedDeltaNet` → `RadixLinearAttention` → `GDNAttnBackend`
> 
> GDN 特点: 无 RoPE, 使用 causal_conv1d + 循环状态更新，不存储 KV cache

### 路径 1: Triton 后端 (`--linear-attn-backend triton`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | in_proj_qkvz | `self.in_proj_qkvz(x)` → `F.linear` (融合 Q/K/V/Z 投影) | pytorch.native(cuBLAS) |
| 2 | in_proj_ba | `self.in_proj_ba(x)` → `F.linear` (融合 beta/alpha 门控投影) | pytorch.native(cuBLAS) |
| 3 | fused_qkvzba_split | `fused_qkvzba_split_reshape_cat_contiguous(...)` (拆分+reshape+concat) | triton |
| 4 | causal_conv1d_update | `causal_conv1d_update(mixed_qkv, conv_states, conv_weights)` | raw-cuda(sgl-kernel) |
| 5 | gdn_decode_fused | `fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv, a, b, states, ...)` (快速路径: 融合 split + gating + 循环状态更新 + output) | triton |
| 5' | gdn_decode_split | (慢速路径) `torch.split(mixed_qkv)` + `fused_sigmoid_gating_delta_rule_update(q, k, v, a, b, states)` | triton |
| 6 | gated_norm | `RMSNormGated(attn_out, z)` (门控 RMSNorm，使用 Z 作为 gate) | raw-cuda(sgl-kernel) |
| 7 | out_proj | `self.out_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | in_proj_qkvz | `self.in_proj_qkvz(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 2 | in_proj_ba | `self.in_proj_ba(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 3 | fused_qkvzba_split | `fused_qkvzba_split_reshape_cat_contiguous(...)` | triton |
| 4 | causal_conv1d_fwd | `causal_conv1d_fn(mixed_qkv, conv_weights)` (全序列因果卷积) | raw-cuda(sgl-kernel) |
| 5 | fused_qkv_split | `fused_qkv_split_gdn_prefill(mixed_qkv)` (融合 QKV 拆分) | triton |
| 6 | fused_gdn_gating | `fused_gdn_gating(A_log, a, b, dt_bias)` → 计算 g (log-decay) 和 beta (sigmoid) | triton |
| 7 | l2norm_q | `l2norm_fwd(q)` | triton |
| 8 | l2norm_k | `l2norm_fwd(k)` | triton |
| 9 | chunk_local_cumsum | `chunk_local_cumsum(g)` (chunk 内累积 gate) | triton |
| 10 | chunk_intra | `chunk_gated_delta_rule_fwd_intra(k, v, g, beta)` (chunk 内 KKT + 三角求解 → w, u) | triton |
| 11 | chunk_h | `chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)` (chunk 间循环状态更新 → h, v_new) | triton |
| 12 | chunk_o | `chunk_fwd_o(q, k, v_new, h, g, scale)` (计算最终输出) | triton |
| 13 | gated_norm | `RMSNormGated(attn_out, z)` | raw-cuda(sgl-kernel) |
| 14 | out_proj | `self.out_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

---

### 路径 2: FlashInfer 后端 (`--linear-attn-backend flashinfer`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | in_proj_qkvz | `self.in_proj_qkvz(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 2 | in_proj_ba | `self.in_proj_ba(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 3 | fused_qkvzba_split | `fused_qkvzba_split_reshape_cat_contiguous(...)` | triton |
| 4 | causal_conv1d_update | `causal_conv1d_update(mixed_qkv, conv_states, conv_weights)` | raw-cuda(sgl-kernel) |
| 5 | qkv_split + gdn_decode | `torch.split(mixed_qkv)` + `flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose(q, k, v, a, b, states, initial_state_indices)` | third-party(flashinfer) |
| 6 | gated_norm | `RMSNormGated(attn_out, z)` | raw-cuda(sgl-kernel) |
| 7 | out_proj | `self.out_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | in_proj_qkvz | `self.in_proj_qkvz(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 2 | in_proj_ba | `self.in_proj_ba(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 3 | fused_qkvzba_split | `fused_qkvzba_split_reshape_cat_contiguous(...)` | triton |
| 4 | causal_conv1d_fwd | `causal_conv1d_fn(mixed_qkv, conv_weights)` | raw-cuda(sgl-kernel) |
| 5 | fused_qkv_split | `fused_qkv_split_gdn_prefill(mixed_qkv)` | triton |
| 6 | fused_gdn_gating | `fused_gdn_gating(A_log, a, b, dt_bias)` | triton |
| 7 | l2norm_q | `l2norm_fwd(q)` | triton |
| 8 | l2norm_k | `l2norm_fwd(k)` | triton |
| 9 | chunk_gdn_prefill | `flashinfer.gdn_prefill.chunk_gated_delta_rule(q, k, v, g, beta, initial_state, ...)` | third-party(flashinfer) |
| 10 | gated_norm | `RMSNormGated(attn_out, z)` | raw-cuda(sgl-kernel) |
| 11 | out_proj | `self.out_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

---

### 路径 3: CuTe DSL 后端 (`--linear-attn-backend cutedsl`)

> 注: SM90 上 CuTe DSL 仅支持 decode，prefill 自动回退到 Triton

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | in_proj_qkvz | `self.in_proj_qkvz(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 2 | in_proj_ba | `self.in_proj_ba(x)` → `F.linear` | pytorch.native(cuBLAS) |
| 3 | fused_qkvzba_split | `fused_qkvzba_split_reshape_cat_contiguous(...)` | triton |
| 4 | causal_conv1d_update | `causal_conv1d_update(mixed_qkv, conv_states, conv_weights)` | raw-cuda(sgl-kernel) |
| 5 | qkv_split + gdn_decode | `torch.split(mixed_qkv)` + `cutedsl_fused_sigmoid_gating_delta_rule_update(q, k, v, a, b, states, initial_state_indices)` | cuteDSL |
| 6 | gated_norm | `RMSNormGated(attn_out, z)` | raw-cuda(sgl-kernel) |
| 7 | out_proj | `self.out_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

> SM90 上 CuTe DSL prefill 不可用，自动回退到 **路径 1 (Triton)** 的 prefill 流程。

---

## 5. DSA — DeepSeek Sparse Attention (DeepSeek V3.2)

> 代表模型: DeepSeek-V3.2 (DSA = MLA + Sparse TopK Indexing)
> 模型文件: `sglang/python/sglang/srt/models/deepseek_v2.py` (use_dsa=True)
> 注意力层: `DeepseekV2AttentionMLA` + `Indexer` + `DeepseekSparseAttnBackend`
> 
> DSA 特点: 在 MLA 基础上增加 per-layer TopK 稀疏索引，每个 query token 只 attend 一小部分 KV pages

### 路径 1: FlashMLA-Sparse 后端 (`--dsa-decode-impl flashmla_sparse`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | fused_qkv_a_proj | `fused_qkv_a_proj_with_mqa` → `F.linear` | pytorch.native(cuBLAS) |
| 2 | q_a_layernorm | `self.q_a_layernorm(q)` → RMSNorm | raw-cuda(sgl-kernel) |
| 3 | kv_a_layernorm | `self.kv_a_layernorm(k_nope)` → RMSNorm | raw-cuda(sgl-kernel) |
| 4 | q_b_proj | `self.q_b_proj(q)` → `F.linear` | pytorch.native(cuBLAS) |
| 5 | absorbed_bmm_kc | `bmm_fp8(q_nope, w_kc)` 或 `deep_gemm.grouped_gemm_nt_f8f8bf16_masked` | third-party(deep_gemm) / raw-cuda(sgl-kernel) |
| 6 | rope | `self.rotary_emb(positions, q_pe, k_pe)` | JIT-kernel |
| 7 | indexer_topk | `self.indexer(x, q_lora, positions, forward_batch)` → 内部使用 `deep_gemm.fp8_paged_mqa_logits` 计算稀疏打分 + topk 选择 | third-party(deep_gemm) + raw-cuda(sgl-kernel) |
| 8 | kv_cache_store | `set_mla_kv_buffer(layer, loc, k_nope, k_pe)` | raw-cuda(sgl-kernel) |
| 9 | concat_q | `concat_mla_absorb_q_general(q_nope_out, q_rope)` | raw-cuda(sgl-kernel) |
| 10 | transform_page_table | `transform_index_page_table_decode(page_table, topk_indices)` | raw-cuda(sgl-kernel) |
| 11 | attention_decode_sparse | `sgl_kernel.flash_mla.flash_mla_sparse_fwd(q, kv_cache, sparse_page_table)` | raw-cuda(sgl-kernel/flashMLA) |
| 12 | absorbed_bmm_vc | `bmm_fp8(attn_output, w_vc)` 或 `deep_gemm.grouped_gemm_nt_f8f8bf16_masked` | third-party(deep_gemm) / raw-cuda(sgl-kernel) |
| 13 | o_proj | `self.o_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-6 | (同 Decode 1-6) | — | — |
| 7 | indexer_topk | `self.indexer(...)` (prefill 时对全序列计算) | third-party(deep_gemm) + raw-cuda(sgl-kernel) |
| 8 | kv_cache_store | `set_mla_kv_buffer(layer, loc, k_nope, k_pe)` | raw-cuda(sgl-kernel) |
| 9 | attention_prefill_sparse | 根据 topk_indices 构建 sparse page_table，使用 `BatchPrefillWithPagedKVCacheWrapper` 或 FlashMLA sparse prefill | third-party(flashinfer) / raw-cuda(sgl-kernel/flashMLA) |
| 10 | absorbed_bmm_vc | (同 Decode step 12) | — |
| 11 | o_proj | `self.o_proj(output)` → `F.linear` | pytorch.native(cuBLAS) |

---

### 路径 2: FA3 后端 (`--dsa-decode-impl fa3`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-10 | (同路径1 step 1-10) | — | — |
| 11 | attention_decode_sparse | `flash_attn_with_kvcache(q=q_rope, k_cache=..., v_cache=..., qv=q_nope, page_table=sparse_page_table)` | raw-cuda(sgl-kernel/flash_attn) |
| 12-13 | (同路径1 step 12-13) | — | — |

---

### 路径 3: TiLeLang 后端 (`--dsa-decode-impl tilelang`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-10 | (同路径1 step 1-10) | — | — |
| 11 | attention_decode_sparse | TileLang MLA kernel (CUDA kernel via TileLang compiler) | raw-cuda(tilelang) |
| 12-13 | (同路径1 step 12-13) | — | — |

---

### 路径 4: TRT-LLM 后端 (`--dsa-decode-impl trtllm`)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-10 | (同路径1 step 1-10) | — | — |
| 11 | attention_decode_sparse | TRT-LLM ragged MLA attention kernel | raw-cuda(trt-llm) |
| 12-13 | (同路径1 step 12-13) | — | — |

---

## 6. C4A/C128A — Compressed Attention (DeepSeek V4)

> 代表模型: DeepSeek-V4
> 模型文件: `sglang/python/sglang/srt/models/deepseek_v4.py`
> 注意力层: `MQALayer` + `DeepseekV4AttnBackend` + `C4Indexer` + `Compressor`
> 
> V4 特点: 
> - MQA (num_kv_heads=1), head_dim=512 (nope=448 + rope=64)
> - 每层按 compress_ratio 分类: 0 (无压缩/SWA only), 4 (C4A), 128 (C128A)
> - 三级 KV cache: SWA pool (最近128 tokens) + C4 compressed pool + C128 compressed pool
> - Q/K 使用 Grouped Low-Rank 投影 + 分组 output (wo_a + wo_b)

### 路径 1: SM90 FlashMLA (compress_ratio=4, C4A 层)

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1 | wq_a (Q low-rank A) | `self.wq_a(x)` → `F.linear` (hidden → q_lora_rank=1024) | pytorch.native(cuBLAS) |
| 2 | q_norm | `self.q_norm(q_lora)` → RMSNorm | raw-cuda(sgl-kernel) |
| 3 | wq_b (Q low-rank B) | `self.wq_b(q_lora)` → `F.linear` (q_lora_rank → n_heads × head_dim=512) | pytorch.native(cuBLAS) |
| 4 | wkv (KV proj) | `self.wkv(x)` → `F.linear` (hidden → head_dim=512) | pytorch.native(cuBLAS) |
| 5 | fused_q_norm_rope | `dsv4_fused_q_norm_rope(q, cos_sin_cache, positions)` (Q 的 RMSNorm + 部分 RoPE) | raw-cuda(sgl-kernel) |
| 6 | fused_k_norm_rope_store | `dsv4_fused_k_norm_rope_flashmla(kv, cos_sin_cache, positions, swa_k_cache)` (K 的 RMSNorm + RoPE + FP8 量化 + SWA cache store) | raw-cuda(sgl-kernel) |
| 7 | c4_indexer | `self.indexer(x, q_lora, forward_batch, attn_backend)` 内部: | |
| 7a | — indexer_q_rope_hadamard | `dsv4_fused_q_indexer_rope_hadamard_quant(q_lora, ...)` (RoPE + Hadamard + FP8 quant) | raw-cuda(sgl-kernel) |
| 7b | — indexer_scoring | `deep_gemm.fp8_paged_mqa_logits(q_fp8, compressed_k_cache, page_table)` | third-party(deep_gemm) |
| 7c | — topk_select | `deepseek_v4_topk_512(logits)` → 选出 top-512 compressed pages | raw-cuda(sgl-kernel) |
| 8 | compressor (C4) | `attn_backend.forward_core_compressor(x, forward_batch, layer_id, compressor)` (每 4 tokens 压缩为 1 KV) 内部为线性层 + 存储 | pytorch.native(cuBLAS) + raw-cuda(sgl-kernel) |
| 9 | attention_decode | `sgl_kernel.flash_mla.flash_mla_with_kvcache(q, swa_k_cache, swa_page_indices, extra_k_cache=c4_cache, extra_indices=c4_sparse_indices, topk_length=...)` | raw-cuda(sgl-kernel/flashMLA) |
| 10 | fused_rope_inverse | `fused_rope_inplace(o[..., -rope_dim:], ..., inverse=True)` (对 output rope 部分做逆旋转) | raw-cuda(sgl-kernel) |
| 11 | wo_a (output low-rank A) | `self.wo_a(o)` → FP8 einsum `tgd,grd->tgr` 或 `deep_gemm.fp8_einsum` | third-party(deep_gemm) / pytorch.native(cuBLAS) |
| 12 | wo_b (output low-rank B) | `self.wo_b(o_a)` → `F.linear` (RowParallel) | pytorch.native(cuBLAS) |

#### Prefill/Extend 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-6 | (同 Decode 1-6) | — | — |
| 7 | c4_indexer | (同 Decode，对 prefill tokens 批量计算) | — |
| 8 | compressor | (同 Decode，对完整序列进行 C4 压缩) | — |
| 9 | attention_prefill | `sgl_kernel.flash_mla.flash_mla_with_kvcache(q, swa_k_cache, ..., extra_k_cache=c4_cache, ...)` (prefill 模式) | raw-cuda(sgl-kernel/flashMLA) |
| 10-12 | (同 Decode 10-12) | — | — |

---

### 路径 2: SM90 FlashMLA (compress_ratio=128, C128A 层)

> 与 C4A 的区别: 无 C4 Indexer (不做 topk 稀疏选择), 使用 C128 全量 compressed cache

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-6 | (同 C4A 路径 step 1-6) | — | — |
| 7 | compressor (C128) | 每 128 tokens 压缩为 1 KV → 写入 c128 cache | pytorch.native(cuBLAS) + raw-cuda(sgl-kernel) |
| 8 | attention_decode | `sgl_kernel.flash_mla.flash_mla_with_kvcache(q, swa_k_cache, swa_page_indices, extra_k_cache=c128_cache, extra_indices=c128_page_indices, ...)` (无 topk, 直接全量 attend c128 pages) | raw-cuda(sgl-kernel/flashMLA) |
| 9-11 | (同 C4A step 10-12) | — | — |

---

### 路径 3: SM90 FlashMLA (compress_ratio=0, SWA-only 层)

> 无压缩层: 仅使用 SWA 窗口内的 KV cache

#### Decode 阶段

| # | kernel_name | 接口 | 实现类别 |
|---|------------|------|---------|
| 1-6 | (同 C4A 路径 step 1-6) | — | — |
| 7 | attention_decode | `sgl_kernel.flash_mla.flash_mla_with_kvcache(q, swa_k_cache, swa_page_indices)` (仅 SWA, 无 extra cache) | raw-cuda(sgl-kernel/flashMLA) |
| 8-10 | (同 C4A step 10-12) | — | — |

---

## 附录 A: Kernel 实现来源汇总

| 实现类别 | 源码位置 | 说明 |
|---------|---------|------|
| pytorch.native(cuBLAS) | PyTorch `F.linear` | 所有 Linear 层的底层 GEMM |
| JIT-kernel (RoPE) | `sglang/python/sglang/jit_kernel/csrc/elementwise/rope.cuh` | JIT 编译的高性能 RoPE |
| raw-cuda(sgl-kernel) | `sglang/sgl-kernel/csrc/` | FlashMLA, concat_mla, causal_conv1d, RMSNorm, FP8 quant, topk 等 |
| raw-cuda(sgl-kernel/flash_attn) | `sglang/sgl-kernel/` 中集成的 FA3 | Flash Attention 3 (SM90 优化) |
| third-party(flashinfer) | `flashinfer` package | BatchDecode/Prefill Wrapper, MLA Wrapper, GDN kernels |
| third-party(deep_gemm) | `deep_gemm` package | FP8 grouped GEMM, fp8_paged_mqa_logits |
| triton | `sglang/python/sglang/srt/layers/attention/triton_ops/`, `fla/`, `jit_kernel/triton/` | decode_attention, extend_attention, GDN chunk, fused_gdn_gating 等 |
| cuteDSL | `sglang/python/sglang/jit_kernel/cutedsl_gdn.py` | GDN decode (SM90) |

---

## 附录 B: 后端选择参数

| 注意力类型 | 服务器参数 | SM90 可用选项 |
|-----------|-----------|--------------|
| GQA | `--attention-backend` | `flashinfer` (默认), `fa3`, `fa4`, `triton` |
| MLA | `--attention-backend` | `flashinfer` (默认), `flashmla`, `cutlass_mla`, `fa3`, `trtllm_mla` |
| GDN | `--linear-attn-backend` / `--linear-attn-decode-backend` / `--linear-attn-prefill-backend` | `triton` (默认), `flashinfer`, `cutedsl` (decode only on SM90) |
| DSA | `--attention-backend dsa` + `--dsa-decode-impl` | `flashmla_sparse` (默认), `flashmla_kv`, `fa3`, `tilelang`, `trtllm` |
| C4A/C128A | `--attention-backend dsv4` | SM90: `flash_mla_with_kvcache` (唯一路径) |
