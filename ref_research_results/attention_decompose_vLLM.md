# vLLM Attention Forward 调用路径分解 (SM90 / Hopper)

> 目标：对 vLLM 仓库中 5 类 attention 在 **SM90 (Hopper, compute capability 9.0)** 上**整个 attention-layer 的 forward**，逐 kernel 列出调用路径并标注实现类别。
> 仓库：`/Users/bytedance/Desktop/remote_dev_project/model_ana/vllm`

---

## 0. 说明与约定

### 0.1 统一 dispatch 机制
- **普通 attention 层** [`Attention`](vllm/model_executor/layers/attention/attention.py)：`forward()` 把 q/k/v 交给 `self.impl.forward()`；`impl` 由 [`selector.py`](vllm/v1/attention/selector.py) → `current_platform.get_attn_backend_cls()` 选出。KV cache 写入是**独立 op**（`forward_includes_kv_cache_update=False`），在 attention 前由 `unified_kv_cache_update` 发起。
- **MLA 层** [`MLAAttention`](vllm/model_executor/layers/attention/mla_attention.py)：`forward_impl()` 按 token 拆分为 `forward_mha`（prefill，compute-friendly MHA）与 `forward_mqa`（decode，data-movement friendly，权重吸收 MQA）。稀疏 MLA（DSA/V4）只走 `forward_mqa`。
- **GDN linear attention** [`GatedDeltaNetAttention`](vllm/model_executor/layers/mamba/gdn_linear_attn.py)：走独立的 Mamba/SSM 风格 backend，无标准 KV cache。

### 0.2 SM90 默认后端选择（`_get_backend_priorities`，[cuda.py L79-147](vllm/platforms/cuda.py#L79-L147)，major==9 走 else 分支）
| 类型 | SM90 默认后端（首选） |
|---|---|
| 非 MLA（GQA） | **FLASH_ATTN**（FA3）→ FlashInfer → Triton → Flex |
| MLA | **FLASH_ATTN_MLA**（FA3）→ FlashMLA → FlashInfer_MLA → Triton_MLA |
| Sparse MLA（DSA） | `FLASHMLA_SPARSE`（`FLASHINFER_MLA_SPARSE` 仅 SM100） |
| DeepSeek-V4（C4A/C128A） | **硬编码** `DeepseekV4FlashMLASparseBackend`（不走通用优先级，[L732](vllm/model_executor/layers/deepseek_v4_attention.py#L732)） |
| GDN | `GDN_ATTN`（prefill 默认 FlashInfer JIT，可切 Triton FLA） |

### 0.3 实现类别取值
- `pytorch-native`：PyTorch 原生接口（`F.linear`→cuBLAS、`torch.bmm`、`torch.cat`）
- `triton`：Triton kernel（含仓库内 FLA、量化、pack_seq 等）
- `cuteDSL`：CuTe DSL kernel（`import cutlass.cute` + `@cute.jit/@cute.kernel`）
- `raw-cuda`：源码在 vllm `_C`/`_C_cache_ops`/csrc 的手写 CUDA（含 cutlass）
- `third-party`：FlashAttention / FlashMLA / FlashInfer / DeepGEMM 等外部库
- `JIT-kernel`：运行时 JIT 编译且非上述 DSL 的 kernel

### 0.4 精度约定
- **BF16**：默认精度，权重与 kv-cache 均为 bf16。
- **FP8**：`--kv-cache-dtype fp8`（MLA/DSA/V4 另有 `fp8_ds_mla` 656B 格式）；DeepSeek 系模型权重多为 fp8 block-quant（128×128），投影 GEMM 走 DeepGEMM/cutlass/triton。
- **关于 RMSNorm/RoPE 归类**：默认 eager/非 inductor 走 vllm `_C` = `raw-cuda`；若启用 inductor 编译，RMSNorm 的 native 分解会被下降为 triton。RoPE、reshape_and_cache、scaled_fp8_quant 恒为 opaque `_C` custom op（raw-cuda）。

---

## 1. GQA (Grouped-Query Attention)

代表模型：Llama / Qwen3（[qwen3.py L145-162](vllm/model_executor/models/qwen3.py)）。核心 attention 后端 = FA3（`third-party`，来自 `vllm.vllm_flash_attn`）。

> **关键结论**：[`FlashAttentionImpl.forward`](vllm/v1/attention/backends/flash_attn.py) 的 prefill 与 decode 走**同一个** `flash_attn_varlen_func`（带 paged block_table），二者仅 attn_metadata 不同、逐 kernel 列表相同。

### 1.1 不含 sliding window

**路径1：Prefill / BF16**（**路径2：Decode / BF16** 逐 kernel 完全相同）
```
kernel_name: input_layernorm,   实现: fused_add_rms_norm → _C.fused_add_rms_norm (raw-cuda)   # 层外; 首层 residual=None 走 rms_norm
kernel_name: qkv_proj,          实现: QKVParallelLinear → F.linear (pytorch-native, cuBLAS)
kernel_name: q_norm/k_norm,     实现: RMSNorm.forward_cuda → _C.rms_norm (raw-cuda)            # Qwen3 特有, Llama 无
kernel_name: rope,              实现: RotaryEmbedding.forward_cuda → _C.rotary_embedding (raw-cuda)
kernel_name: reshape_and_cache, 实现: reshape_and_cache_flash → _C_cache_ops (raw-cuda)         # KV 写入, 独立 op
kernel_name: core_attn,         实现: flash_attn_varlen_func (third-party, FlashAttention FA3)
kernel_name: o_proj,            实现: RowParallelLinear → F.linear (pytorch-native, cuBLAS)
```

**路径3：Prefill / FP8（kv-cache=fp8）**（**路径4：Decode / FP8** 逐 kernel 完全相同）
```
kernel_name: input_layernorm,   实现: _C.fused_add_rms_norm (raw-cuda)
kernel_name: qkv_proj,          实现: F.linear (pytorch-native, cuBLAS)         # 权重仍 BF16, 仅 kv-cache 为 fp8
kernel_name: q_norm/k_norm,     实现: _C.rms_norm (raw-cuda)
kernel_name: rope,              实现: _C.rotary_embedding (raw-cuda)
kernel_name: query_quant,       实现: QuantFP8 → ops.scaled_fp8_quant → _C.static_scaled_fp8_quant (raw-cuda)   # 仅 FP8
kernel_name: reshape_and_cache, 实现: reshape_and_cache_flash → _C_cache_ops (raw-cuda; 内部量化写 fp8, 带 k/v_scale)
kernel_name: core_attn,         实现: flash_attn_varlen_func (third-party, FA3 fp8 scaled 分支; kv_cache.view(fp8)+q/k/v_descale)
kernel_name: o_proj,            实现: F.linear (pytorch-native, cuBLAS)
```

### 1.2 含 sliding window
**唯一差异点**：传给 `flash_attn_varlen_func` 的 `window_size` 参数不同——sliding 时 `(sliding_window-1, 0)`，非 sliding 时 `(-1, -1)`。其余 kernel 完全一致（副作用：sliding window 时 cascade attention 被禁用，但主路径本就是 varlen，不改变 kernel 列表）。

**证据**：模型顺序 [qwen3.py L145-162](vllm/model_executor/models/qwen3.py)；[`Attention.forward` L437-529](vllm/model_executor/layers/attention/attention.py#L437-L529)；FA impl [flash_attn.py L667-819](vllm/v1/attention/backends/flash_attn.py)；window_size 构造 L616-621、传参 L791-808；FA3 判定 [fa_utils.py L78-79](vllm/v1/attention/backends/fa_utils.py)。

---

## 2. MLA (Multi-head Latent Attention)

代表模型：DeepSeek-V2/V3（[deepseek_v2.py L865-1067](vllm/model_executor/models/deepseek_v2.py)，外层投影链在 [mla.py L119-181](vllm/model_executor/layers/mla.py)）。SM90 BF16 主后端 = **FLASH_ATTN_MLA**（FA3）；**kv-cache=fp8 时 FA_MLA 被排除**（[flashattn_mla.py L45-49,304-327](vllm/v1/attention/backends/mla/flashattn_mla.py)），退到 **FlashMLA**。prefill 段的 new-token / context attention 恒由 FA3 prefill 子后端承担（[prefill/selector.py L72-75](vllm/v1/attention/backends/mla/prefill/selector.py)）。

**路径1：Prefill / BF16**（compute-friendly，`forward_mha`）
```
kernel_name: q 投影链,          实现: fused_qkv_a_proj + q_a_layernorm + q_b_proj → F.linear/RMSNorm (pytorch-native + raw-cuda)
kernel_name: kv 投影链(down),   实现: kv_a_proj_with_mqa(融合) + kv_a_layernorm → F.linear + _C.rms_norm (pytorch-native + raw-cuda)
kernel_name: rope,              实现: ops.rotary_embedding → _C.rotary_embedding (raw-cuda)
kernel_name: concat_and_cache_mla, 实现: ops.concat_and_cache_mla → _C_cache_ops (raw-cuda)     # KV 写入
kernel_name: kv_b_proj 展开,    实现: self.kv_b_proj(kv_c) → F.linear (pytorch-native, cuBLAS)   # 展开 W_UK/W_UV
kernel_name: k 拼接,            实现: _concat_k_nope_k_pe → torch.empty+切片 (pytorch-native); DSV3 dims 走 flashinfer_concat_mla_k (third-party)
kernel_name: core_attn(new),    实现: run_prefill_new_tokens → flash_attn_varlen_func (third-party, FA3, causal, return_lse)
kernel_name: ctx_gather,        实现: ops.gather_and_maybe_dequant_cache → _C_cache_ops (raw-cuda)  # chunked context
kernel_name: core_attn(ctx),    实现: run_prefill_context_chunk → flash_attn_varlen_func (third-party, FA3, causal=False)
kernel_name: merge_attn_states, 实现: _C.merge_attn_states (raw-cuda; 回退 triton)                # lse 合并
kernel_name: o_proj,            实现: RowParallelLinear → F.linear (pytorch-native, cuBLAS)
```

**路径2：Decode / BF16**（data-movement friendly，`forward_mqa`，权重吸收 MQA）
```
kernel_name: q/kv 投影链+rope+concat_and_cache_mla,  实现: 同路径1 步骤 1-4
kernel_name: absorb_bmm#1,      实现: torch.bmm(q_nope, W_UK_T) (pytorch-native)                # q_nope 投到 latent
kernel_name: core_attn(MQA),    实现: FlashAttnMLAImpl.forward_mqa → flash_attn_varlen_func(q_v=q_nope, num_splits, FA3) (third-party)
kernel_name: absorb_bmm#2,      实现: _v_up_proj → torch.bmm(x, W_UV) (pytorch-native)
kernel_name: o_proj,            实现: F.linear (pytorch-native, cuBLAS)
```

**路径3：Prefill / FP8**（后端退到 FlashMLA；prefill 段仍 FA3；SM90 上 prefill query 不量化——`backend_supports_prefill_query_quantization` 需 cc100）
```
kernel_name: 投影链,            实现: 若权重 fp8 block-quant → DeepGEMM fp8_gemm_nt (third-party); 否则 F.linear (pytorch-native)
kernel_name: concat_and_cache_mla, 实现: _C_cache_ops (raw-cuda; fp8 写入带 scale)
kernel_name: ctx_gather(反量化), 实现: gather_and_maybe_dequant_cache (raw-cuda; fp8 cache → bf16 workspace)
kernel_name: kv_b_proj + k 拼接, 实现: F.linear + 切片 (pytorch-native, BF16 计算)
kernel_name: core_attn,         实现: flash_attn_varlen_func (third-party, FA3, BF16 计算)
kernel_name: merge_attn_states + o_proj, 实现: _C.merge_attn_states (raw-cuda) + F.linear (pytorch-native)
```

**路径4：Decode / FP8**（后端 FlashMLA）
```
kernel_name: 投影链+rope+concat_and_cache_mla,  实现: 同上 (fp8 GEMM: DeepGEMM / pytorch-native)
kernel_name: kv_cache view→fp8, 实现: kv_cache.view(fp8_dtype) (pytorch-native)
kernel_name: absorb_bmm#1,      实现: torch.bmm(q_nope, W_UK_T) (pytorch-native, BF16 权重)
kernel_name: decode_query_quant,实现: _DecodeConcatQuantFP8(QuantFP8) → cat+per-tensor 量化 (JIT-kernel / 底层 _C scaled_fp8_quant)
kernel_name: core_attn(MQA fp8),实现: FlashMLAImpl.forward_mqa → flash_mla_with_kvcache_fp8(descale_q/k) (third-party, FlashMLA)
kernel_name: absorb_bmm#2 + o_proj, 实现: torch.bmm(x, W_UV) (pytorch-native) + F.linear (pytorch-native)
```

> **FA_MLA vs FlashMLA（decode 差异）**：FA_MLA 调 `vllm.vllm_flash_attn.flash_attn_varlen_func`（用 `q_v` 单独喂 nope、FA3 varlen+num_splits、支持 VARLEN spec-decode/full-cudagraph，**不支持 fp8 kv-cache**）；FlashMLA 调 third-party `flash_mla_with_kvcache`(_fp8)（需 `get_mla_metadata` 生成 tile scheduler、block_size=64、fp8 变体原生吃 fp8 KV）。
> **weight-absorb 恒 BF16**：`kv_b_proj` 权重在 `process_weights_after_loading` 被反量化成 BF16 存 `W_UK_T/W_UV`，故两次 bmm 恒为 BF16 pytorch-native。

**证据**：[mla_attention.py](vllm/model_executor/layers/attention/mla_attention.py) `forward_mha` L2246-2316、`_compute_prefill_context` L2031-2136、decode bmm/量化 L705-794、`_v_up_proj` L972-994、`_DecodeConcatQuantFP8` L1137-1163；[flashattn_mla.py L309-356](vllm/v1/attention/backends/mla/flashattn_mla.py)；[flashmla.py L254-334](vllm/v1/attention/backends/mla/flashmla.py)；投影 fp8 GEMM [scaled_mm/deep_gemm.py](vllm/model_executor/layers/quantization/utils/scaled_mm)。

---

## 3. GDN (Gated DeltaNet, Qwen3.5 linear attention)

代表模型：Qwen3.5（[qwen3_5.py L136](vllm/model_executor/models/qwen3_5.py) → [gdn_linear_attn.py](vllm/model_executor/layers/mamba/gdn_linear_attn.py)）。三段式：`in_proj` → `gdn_attention_core`（prefill chunk / decode recurrent）→ `RMSNormGated` + `out_proj`。除 in/out 投影走 cuBLAS 外，几乎全为 **Triton (FLA)**；prefill chunk 核心默认走 **FlashInfer (JIT)**。

**路径1：Prefill / BF16**（chunk 并行，`num_prefills>0`）
```
kernel_name: in_proj_qkvz/ba,   实现: MergedColumnParallelLinear → F.linear (pytorch-native, cuBLAS)
kernel_name: causal_conv1d,     实现: causal_conv1d_fn → _causal_conv1d_fwd_kernel (triton)
kernel_name: post_conv_prep,    实现: fused_post_conv_prep → _fused_post_conv_kernel (triton; 含 l2norm+softplus gating+beta=sigmoid)
kernel_name: core_linear_attn,  实现【SM90 默认】: flashinfer.gdn_prefill.chunk_gated_delta_rule (third-party/JIT-kernel)
                                实现【Triton 回退】: fla_chunk_gated_delta_rule → cumsum/scaled_dot_kkt/solve_tril/recompute_w_u/fwd_h/fwd_o (triton 多 kernel 扫描链)
kernel_name: gated_rmsnorm,     实现: RMSNormGated → rmsnorm_fn → layer_norm_fwd_kernel (triton)
kernel_name: out_proj,          实现: RowParallelLinear → F.linear (pytorch-native, cuBLAS)
```

**路径2：Decode / BF16**（recurrent，`num_decodes>0`）
```
kernel_name: in_proj_qkvz/ba,   实现: F.linear (pytorch-native, cuBLAS)
kernel_name: causal_conv1d,     实现: causal_conv1d_update → _causal_conv1d_update_kernel (triton)
kernel_name: core_linear_attn,  实现【默认】: fused_sigmoid_gating_delta_rule_update_kernel (triton; 融合 gating+sigmoid-beta+kernel内l2norm+recurrent)
                                实现【env 打包路径】: fused_recurrent_gated_delta_rule_packed_decode_kernel (triton)
kernel_name: gated_rmsnorm,     实现: rmsnorm_fn → layer_norm_fwd_kernel (triton)
kernel_name: out_proj,          实现: F.linear (pytorch-native, cuBLAS)
```

**FP8 说明**：GDN 路径**无 fp8 linear-attn/conv kernel**。state（ssm_state/conv_state）与 chunk/recurrent 计算恒 bf16/fp32（kernel 内 `.to(tl.float32)` 累加）。FP8 仅可能影响 `in_proj_qkvz`/`out_proj` 的投影 GEMM（走量化 GEMM，类别变为 raw-cuda/third-party）；`in_proj_ba` 明确不支持 blockwise fp8，conv1d 权重非量化。

**证据**：[gdn_linear_attn.py](vllm/model_executor/layers/mamba/gdn_linear_attn.py) forward L723-783、`_forward_core` L1048-1311、SM90 FlashInfer 判定 L144-173、decode L1275-1294；FLA Triton 链 [fla/ops/chunk.py L37-82](vllm/model_executor/layers/fla/ops/chunk.py)；[causal_conv1d.py](vllm/model_executor/layers/mamba/ops/causal_conv1d.py)；`RMSNormGated` [layernorm.py L182-311](vllm/model_executor/layers/layernorm.py)。

---

## 4. DSA (DeepSeek Sparse Attention, DeepSeek-V3.2)

代表模型：DeepSeek-V3.2 = **MLA + Lightning Indexer**（[deepseek_v2.py L604-737](vllm/model_executor/models/deepseek_v2.py) `Indexer`；`is_v32=hasattr(config,"index_topk")`）。主体投影链与 MLA（第 2 节）相同。SM90 主 sparse 后端 = **FLASHMLA_SPARSE**；稀疏 impl **只走 `forward_mqa`**（prefill 与 decode 全部 token 都走同一 sparse kernel，[mla_attention.py L675-703](vllm/model_executor/layers/attention/mla_attention.py)）。

> **Indexer 始终 FP8**：无论主 cache 是否 fp8，Indexer 都用自身 FP8 naive cache（uint8 存 fp8 值 + fp32 scale）；DeepGEMM 是 Indexer 在 CUDA 上的硬依赖。

### Indexer 子链（所有路径共用，仅 logits/topk 的 prefill/decode 变体不同）
```
kernel_name: idx_wq_b,          实现: ReplicatedLinear → F.linear (pytorch-native; fp8 权重走 DeepGEMM)
kernel_name: idx_wk_weights,    实现: MergedColumnParallelLinear(融合 wk+weights) → F.linear (pytorch-native)
kernel_name: idx_k_norm,        实现: LayerNorm (raw-cuda)
kernel_name: idx_rope,          实现: indexer_rope_emb → _C.rotary_embedding (raw-cuda); torch.cat/split (pytorch-native)
kernel_name: idx_q_fp8_quant,   实现: per_token_group_quant_fp8 (triton)
kernel_name: idx_k_quant+cache, 实现: ops.indexer_k_quant_and_cache → _C (raw-cuda)               # K 写 indexer cache
kernel_name: idx_k_gather,      实现: ops.cp_gather_indexer_k_quant_cache → _C (raw-cuda)
kernel_name: idx_logits,        实现【prefill】: fp8_fp4_mqa_logits (third-party/JIT, DeepGEMM)
                                实现【decode】:  fp8_fp4_paged_mqa_logits (third-party/JIT, DeepGEMM; 含 pack_seq_triton padding = triton)
kernel_name: idx_topk,          实现【prefill】: _C.top_k_per_row_prefill (raw-cuda)
                                实现【decode】:  _C.persistent_topk / top_k_per_row_decode (raw-cuda; padding 时 unpack_seq_triton = triton)
```

### 主 sparse MLA attention

**路径1：Prefill / BF16 主 + Indexer FP8** & **路径2：Decode / BF16 主 + Indexer FP8**
（prefill/decode 主 attention 统一走同一 sparse kernel，仅 Indexer 的 logits/topk 变体不同）
```
kernel_name: 投影链+rope+concat_and_cache_mla, 实现: 同 MLA (pytorch-native + raw-cuda)
kernel_name: [Indexer 子链],    实现: 见上（prefill/decode 对应变体）
kernel_name: topk→global_index, 实现: triton_convert_req_index_to_global_index (triton)
kernel_name: absorb_bmm#1,      实现: torch.bmm(q_nope, W_UK_T) (pytorch-native); concat_mla_q → _C (raw-cuda)
kernel_name: core_sparse_attn,  实现: FlashMLASparseImpl.forward_mqa → _forward_bf16_kv → flash_mla_sparse_fwd (third-party, FlashMLA)
kernel_name: absorb_bmm#2(v_up),实现: torch.bmm(x, W_UV) (pytorch-native)
kernel_name: o_proj,            实现: F.linear (pytorch-native, cuBLAS)
```

**路径3/4：主 attention 也用 FP8 kv-cache（`fp8_ds_mla` 656B）**
Indexer 子链不变；差异仅在主 sparse attention（SM90 按 `num_heads` 分两种分发）：
```
# num_heads<32 (高 TP) —— mixed-batch: prefill+decode 全 token 单批走 FP8 decode kernel
kernel_name: core_sparse_attn,  实现: _forward_fp8_kv_mixed_batch → flash_mla_with_kvcache(is_fp8_kvcache=True) (third-party, FlashMLA FP8)

# num_heads>=32 —— separate:
kernel_name: core_sparse_attn(decode),  实现: flash_mla_with_kvcache FP8 decode (third-party, FlashMLA)
kernel_name: fp8→bf16 上转(prefill),     实现: ops.cp_gather_and_upconvert_fp8_kv_cache → _C (raw-cuda)
kernel_name: core_sparse_attn(prefill),  实现: _bf16_flash_mla_kernel → flash_mla_sparse_fwd (third-party, FlashMLA)
```

> **FLASHMLA_SPARSE vs FLASHINFER_MLA_SPARSE**：前者用 DeepSeek FlashMLA 的 `flash_mla_sparse_fwd`/`flash_mla_with_kvcache`（Hopper+Blackwell，支持 fp8_ds_mla 656B）；后者用 FlashInfer TRT-LLM `trtllm_batch_decode_with_kv_cache_mla`（**仅 SM100**），SM90 不命中。

**证据**：[sparse_attn_indexer.py](vllm/model_executor/layers/sparse_attn_indexer.py) `sparse_attn_indexer` L86、logits L225/L312、topk L251/L330/L351；[flashmla_sparse.py](vllm/v1/attention/backends/mla/flashmla_sparse.py) `forward_mqa` L1014、`_forward_bf16_kv` L777、fp8 分发 L800/L905；[indexer.py](vllm/v1/attention/backends/mla/indexer.py) metadata。

---

## 5. C4A / C128A (DeepSeek-V4)

代表模型：DeepSeek-V4（[deepseek_v4.py L928-1090](vllm/model_executor/models/deepseek_v4.py) `DeepseekV4Attention` → [deepseek_v4_attention.py L284](vllm/model_executor/layers/deepseek_v4_attention.py) `DeepseekV4MultiHeadLatentAttentionWrapper`）。`compress_ratio` 决定形态：**C4A（=4）带 `DeepseekV4Indexer` 稀疏**，**C128A（=128）为不带 indexer 的压缩 MLA**。

> **后端**：SM90（非 ROCm）**硬编码** `DeepseekV4FlashMLASparseBackend`（[L732](vllm/model_executor/layers/deepseek_v4_attention.py#L732)），C4A 与 C128A **都用它**（prefill=`flash_mla_sparse_fwd`，decode=`flash_mla_with_kvcache`）；差异仅在 **topk 来源**：C4A 由 Lightning Indexer 在线算，C128A 在 metadata build 时预算。
> **CuTe DSL 是本类型重点**：仅 **2 个** kernel 是真 CuTe DSL（`import cutlass.cute` + `@cute.jit/@cute.kernel`），均有 triton 回退。

V4 为 **fp8 模型**（`DeepseekV4FP8Config`）；BF16 参考路径（`rocm_inv_rope_einsum`）仅 ROCm，SM90 不触发。

### C128A — Prefill (fp8)
```
kernel_name: fused_wqa_wkv,     实现: MergedColumnParallelLinear fp8 → DeepGEMM fp8_gemm_nt (third-party; 回退 cutlass/triton)
kernel_name: compressor_kv_score, 实现: torch.mm(out_dtype=fp32) (pytorch-native, cuBLAS)
kernel_name: fused_q_kv_rmsnorm, 实现: _fused_q_kv_rmsnorm_kernel (triton)
kernel_name: wq_b,              实现: ColumnParallelLinear fp8 → DeepGEMM (third-party)
kernel_name: qnorm_rope_kv_rope_quant_insert, 实现: torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_insert (raw-cuda)   # q RMSNorm+GPT-J RoPE, kv RoPE+FP8量化+SWA cache 插入
kernel_name: compress_save_states, 实现: _save_partial_states_kernel (triton)
kernel_name: compress_norm_rope_insert, 实现: _fused_kv_compress_norm_rope_insert_sparse_attn (triton, head=512)
kernel_name: dequant_gather_k,  实现: DequantGatherKCacheKernel (【cuteDSL】; 无 cutedsl 时 triton 回退)          # FP8→BF16 gather+dequant
kernel_name: combine_topk_swa,  实现: _combine_topk_swa_indices_kernel (triton)
kernel_name: core_sparse_attn,  实现: flash_mla_sparse_fwd (third-party, FlashMLA)
kernel_name: o_inv_rope_quant,  实现: fused_inv_rope_fp8_quant → _fused_inv_rope_fp8_quant_per_head (triton)
kernel_name: v_up(einsum),      实现: deepseek_v4_fp8_einsum → DeepGEMM fp8_einsum (third-party)
kernel_name: wo_b,              实现: fp8 GEMM → DeepGEMM (third-party)
```

### C128A — Decode (fp8)
```
kernel_name: 投影+rmsnorm+rope_insert+compress, 实现: 同 Prefill (DeepGEMM + triton + raw-cuda)
kernel_name: core_sparse_attn,  实现: flash_mla_with_kvcache (third-party, FlashMLA; topk 由 metadata 预算)
kernel_name: o_inv_rope_quant → v_up → wo_b, 实现: fused_inv_rope_fp8_quant (triton) → fp8_einsum (DeepGEMM) → wo_b (DeepGEMM)
```

### C4A — Prefill (fp8 + indexer)
在 C128A Prefill 链基础上并行插入 **Indexer 子链**（`DeepseekV4Indexer`，复用 `SparseAttnIndexer`，K 由内部 compressor 产出、不单独插缓存）：
```
kernel_name: idx_wq_b,          实现: ReplicatedLinear fp8 → DeepGEMM (third-party)
kernel_name: idx_compressor,    实现: _save_partial_states_kernel + _fused_kv_compress_norm_rope_insert_indexer_attn (triton, head=128)
kernel_name: fused_indexer_q,   实现【FP8】: _fused_indexer_q_rope_quant_kernel (triton)
                                实现【MXFP4 + cutedsl】: IndexerQMxFp4Kernel (【cuteDSL】; 否则 triton)
kernel_name: idx_k_gather,      实现: cp_gather_indexer_k_quant_cache → _C (raw-cuda)
kernel_name: idx_logits,        实现: fp8_fp4_mqa_logits (third-party, DeepGEMM)
kernel_name: idx_topk,          实现: _C.top_k_per_row_prefill (raw-cuda)
kernel_name: [其余同 C128A Prefill], 实现: dequant_gather_k (cuteDSL) + combine_topk_swa (triton) + flash_mla_sparse_fwd (third-party) + O 投影 (triton+DeepGEMM)
```

### C4A — Decode (fp8 + indexer)
```
kernel_name: 投影+rmsnorm+rope_insert+compress, 实现: 同 C128A Decode
kernel_name: idx_wq_b + idx_compressor, 实现: DeepGEMM + triton
kernel_name: fused_indexer_q,   实现: FP8=triton / MXFP4=cuteDSL(IndexerQMxFp4Kernel)
kernel_name: idx_logits,        实现: fp8_fp4_paged_mqa_logits (third-party, DeepGEMM)
kernel_name: idx_topk,          实现: _C.persistent_topk / top_k_per_row_decode (raw-cuda)
kernel_name: global_topk_map,   实现: _compute_global_topk_indices_and_lens_kernel (triton)
kernel_name: core_sparse_attn,  实现: flash_mla_with_kvcache(extra_indices=topk) (third-party, FlashMLA)
kernel_name: O 投影,            实现: fused_inv_rope_fp8_quant (triton) → fp8_einsum (DeepGEMM) → wo_b (DeepGEMM)
```

> **CuTe DSL kernel 汇总（仅 2 个）**：
> 1. `DequantGatherKCacheKernel`（[dequant_gather_k_cutedsl.py L35](vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py)）— prefill FP8→BF16 gather+dequant K。
> 2. `IndexerQMxFp4Kernel`（[fused_indexer_q_cutedsl.py L68](vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py)）— 仅 C4A 且 MXFP4 indexer cache 时的 indexer-Q RoPE+MXFP4 量化。
> 其余 V4 专用算子（`fused_qk_rmsnorm`、`fused_inv_rope_fp8_quant`、`fused_indexer_q` FP8、`fused_compress_quant_cache`、`cache_utils` 的 gather/topk-map/combine）均为 **triton**。

**证据**：[deepseek_v4_attention.py](vllm/model_executor/layers/deepseek_v4_attention.py) `attention_impl` L413、后端 L725-732、O 投影 L890/L1017；[ops/deepseek_v4_ops/](vllm/v1/attention/ops/deepseek_v4_ops)（cutedsl vs triton 已逐文件核对）。

---

## 6. 汇总对比表

### 6.1 核心 attention / state kernel（各类型 × 阶段 × 精度）
| 类型 | Prefill 核心 | Decode 核心 | 实现类别 |
|---|---|---|---|
| GQA (BF16/FP8) | `flash_attn_varlen_func` | `flash_attn_varlen_func`（同一调用） | third-party (FA3) |
| MLA BF16 | FA3 new-token + chunked context + `merge_attn_states` | FA_MLA `forward_mqa`（q_v） | third-party + raw-cuda |
| MLA FP8 | FA3（gather 反量化后 BF16 计算） | FlashMLA `flash_mla_with_kvcache_fp8` | third-party |
| GDN | chunk：FlashInfer（默认）/ FLA Triton | recurrent：`fused_sigmoid_gating_delta_rule_update` | third-party/JIT + triton |
| DSA BF16 主 | `flash_mla_sparse_fwd`（forward_mqa 统一） | `flash_mla_sparse_fwd` | third-party (FlashMLA) |
| DSA FP8 主 | `flash_mla_sparse_fwd`（上转）/ mixed | `flash_mla_with_kvcache`(fp8) | third-party (FlashMLA) |
| C128A (fp8) | `flash_mla_sparse_fwd` | `flash_mla_with_kvcache` | third-party (FlashMLA) |
| C4A (fp8) | `flash_mla_sparse_fwd` + Indexer | `flash_mla_with_kvcache` + Indexer | third-party (FlashMLA) |

### 6.2 各实现类别在各类型中的分布
| 实现类别 | GQA | MLA | GDN | DSA | C4A/C128A |
|---|:-:|:-:|:-:|:-:|:-:|
| pytorch-native (cuBLAS/bmm) | 投影/o_proj | 投影/o_proj/absorb-bmm | in/out_proj | 投影/absorb-bmm | compressor_kv_score |
| triton | (inductor 时 norm) | merge 回退 | **conv1d/l2norm/gating/gated-norm/recurrent** | pack_seq/quant/idx_map | **fused_qk_rmsnorm/inv_rope/compress/indexer_q** |
| cuteDSL | — | — | — | — | **dequant_gather_k / indexer_q_mxfp4（2个）** |
| raw-cuda (`_C`) | rope/rms_norm/reshape_cache/fp8_quant | rope/rms_norm/concat_cache/gather/merge | — | rope/norm/k_quant/gather/topk | qnorm_rope_insert/idx_gather/topk |
| third-party | **FA3** | **FA_MLA/FlashMLA/DeepGEMM** | **FlashInfer(prefill)** | **FlashMLA + DeepGEMM(logits)** | **FlashMLA + DeepGEMM** |
| JIT-kernel | — | decode_query_quant | FlashInfer prefill | (DeepGEMM) | (DeepGEMM) |

### 6.3 关键差异速记
- **GQA**：最简单，prefill≡decode（同一 FA3 varlen）；sliding window 仅改 `window_size`。
- **MLA**：prefill（MHA 展开）与 decode（MQA 权重吸收）算法不同；BF16→FA_MLA，FP8→FlashMLA。
- **GDN**：非注意力，chunk（prefill）vs recurrent（decode）；主体 Triton + prefill FlashInfer。
- **DSA**：MLA + FP8 Indexer（DeepGEMM logits + topk）；主 attn 只走 forward_mqa。
- **C4A/C128A**：V4 fp8 专用 fused kernel 密集（triton 为主 + 2 个 cuteDSL）；后端硬编码 FlashMLA Sparse。
