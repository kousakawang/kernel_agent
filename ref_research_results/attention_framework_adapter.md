# Attention 算子库与推理框架的 ABI 适配调研报告

## 一、总论：算子库与推理框架的适配哲学

### 1.1 核心问题回答

**算子库在设计阶段就主动考虑了推理框架的资源形态，在 API 层面预先提供了 Paged KV Cache 的原生支持。** 具体而言：

- **FlashInfer** 从设计之初就定义了 `BatchDecodeWithPagedKVCacheWrapper` 和 `BatchPrefillWithPagedKVCacheWrapper` 两类 wrapper，原生接受 CSR 格式的分页索引（`kv_indptr`, `kv_indices`, `kv_last_page_len`），直接支持推理框架中常见的非连续 KV 缓存布局。
- **Flash-Attention (v3/v4)** 通过 `flash_attn_with_kvcache` / `flash_attn_varlen_func` 接口，原生支持 `block_table` 参数，允许按页表索引从预分配的 KV 缓存块中读取数据。

**但框架仍需承担大量适配工作**：算子库提供的是"通用分页接口"，而每个框架有自己的资源管理模型（slot 分配策略、request-to-token 映射表、page size 选择等）。框架需要将自身的资源管理抽象**翻译**为算子库期望的具体格式。

### 1.2 双层适配架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                             │
│  (Qwen3Attention / DeepseekV2MLA / DeepseekV4MQA ...)      │
│  职责: QKV projection, RoPE, 调用 attention                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ forward(q, k, v)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Framework Attention Layer                       │
│  (RadixAttention / Attention)                                │
│  职责: 路由到当前活跃的 AttentionBackend                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            Backend Adapter Layer  (核心适配层)                 │
│  (FlashInferAttnBackend / FlashAttentionBackend / ...)       │
│  职责:                                                       │
│    1. KV Cache Write: set_kv_buffer(loc, k, v)              │
│    2. Index Transform: req_to_token → 算子库格式             │
│    3. Forward Call: 调用算子库 API                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Operator Library API                             │
│  (flashinfer.decode/prefill, flash_attn_with_kvcache, ...)  │
│  职责: 纯计算——从分页 KV Cache 中按索引读取并计算 attention    │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 各方职责划分

| 层级 | 职责 | 代表组件 |
|------|------|---------|
| **算子库** | 定义分页 KV Cache 的 API 格式（CSR/block_table）；实现高效的 attention CUDA kernel | FlashInfer wrapper, flash_attn_varlen_func |
| **框架 Backend Adapter** | 将框架 KV Cache 管理（slot pool + req_to_token）翻译为算子库格式；管理 KV 写入时机；处理 sliding window 等特殊逻辑 | FlashInferAttnBackend, FlashAttentionBackend |
| **框架 Memory Pool** | 预分配物理存储；维护 slot 分配/释放；提供 req_to_token 映射 | MHATokenToKVPool, MLATokenToKVPool |
| **模型层** | 计算 Q/K/V projection 和 RoPE；调用统一的 attention 接口 | Qwen3Attention, DeepseekV2AttentionMLA |

---

## 二、场景一：常规 Attention (GQA/MHA + Sliding Window)

### 2.1 框架侧 KV Cache 物理布局

两个框架使用相似但不同的 KV Cache 存储策略：

**SGLang (`MHATokenToKVPool`)**:
- 物理存储: 每层两个 3D tensor `k_buffer[layer]` 和 `v_buffer[layer]`
- 形状: `[pool_size + page_size, num_kv_heads, head_dim]`
- 索引: 通过 `req_to_token[req_id, token_pos]` 获取每个 token 的物理 slot index
- 写入: `k_buffer[layer][loc] = new_k` (scatter write)

```python
# sglang/python/sglang/srt/mem_cache/memory_pool.py
class MHATokenToKVPool(KVCache):
    def _create_buffers(self):
        self.k_buffer = [
            torch.zeros((self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype, device=self.device)
            for _ in range(self.layer_num)
        ]
        # v_buffer 类似

    def get_kv_buffer(self, layer_id):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
        # loc: [num_new_tokens] int tensor, 指定写入的物理 slot 位置
        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v
```

**vLLM**:
- 物理存储: 5D tensor
  - Flash-Attention 布局: `[2, num_blocks, block_size, num_kv_heads, head_size]`
  - FlashInfer 布局: `[num_blocks, 2, block_size, num_kv_heads, head_size]`
- 索引: 通过 `block_table[req_id, block_idx]` 获取块号
- 写入: 通过 C++ op `reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping)`

### 2.2 FlashInfer 的 CSR Paging 适配

FlashInfer 期望的分页 KV 描述格式是 **CSR (Compressed Sparse Row)**:
- `kv_indptr`: `[bs+1]` — 每个 request 在 kv_indices 中的起始偏移
- `kv_indices`: `[total_kv_tokens]` — 所有 request 的 KV slot 索引展平
- `kv_last_page_len`: `[bs]` — 每个 request 最后一页的有效长度

**SGLang 的适配实现**:

框架通过一个 Triton kernel 将 `req_to_token` 二维表转换为 FlashInfer 需要的 CSR 格式：

```python
# sglang: flashinfer_backend.py — IndicesUpdater
# 步骤 1: 计算 kv_indptr (cumsum of seq_lens)
kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

# 步骤 2: Triton kernel 展平 req_to_token → kv_indices
create_flashinfer_kv_indices_triton[(bs,)](
    self.req_to_token,          # [max_reqs, max_ctx_len] 框架的映射表
    req_pool_indices,           # [bs] 当前 batch 的 req IDs
    paged_kernel_lens,          # [bs] 每个 req 的 KV 长度
    kv_indptr,                  # [bs+1] CSR 偏移
    kv_start_idx,               # [bs] 窗口起始 (SWA 用)
    kv_indices,                 # [total] 输出: 展平的 slot 索引
    req_to_token_stride,
)

# 步骤 3: plan (注册索引到 wrapper)
decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_len,
                             num_qo_heads, num_kv_heads, head_dim, ...)

# 步骤 4: forward (传入原始 KV buffer)
output = decode_wrapper.forward(
    q.view(-1, num_heads, head_dim),
    token_to_kv_pool.get_kv_buffer(layer_id),  # (k_buf, v_buf) 3D tensors
    sm_scale=scaling,
)
```

**vLLM 的适配实现**:

```python
# vllm: flashinfer.py — FlashInferMetadataBuilder.build()
# 步骤 1: 计算 CSR 三元组
paged_kv_indptr = cumsum(num_blocks_per_request)
paged_kv_indices = flatten(block_tables[:, :num_blocks])  # 展平块号
paged_kv_last_page_len = seq_lens % block_size

# 步骤 2: plan
prefill_wrapper.plan(
    qo_indptr=qo_indptr,
    paged_kv_indptr=paged_kv_indptr,
    paged_kv_indices=paged_kv_indices,
    paged_kv_last_page_len=paged_kv_last_page_len,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim_qk=head_dim,
    page_size=page_size,
)

# 步骤 3: forward (需要 permute KV cache 到 FlashInfer 布局)
kv_cache_permute = kv_cache.permute(*stride_order)  # NHD ↔ HND
output = prefill_wrapper.run(query, kv_cache_permute, ...)
```

### 2.3 Flash-Attention 的 Block Table 适配

Flash-Attention (v3/v4) 期望的分页格式更直接——一个 2D `block_table`:
- `k_cache` / `v_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
- `block_table`: `[batch_size, max_num_blocks]` — 每个 request 的块号序列
- `cache_seqlens`: `[batch_size]` — 每个 request 的实际 KV 长度

**SGLang 的适配实现**:

```python
# sglang: flashattention_backend.py
# 步骤 1: 直接从 req_to_token 表切片作为 page_table
metadata.page_table = self.req_to_token_pool.req_to_token[
    forward_batch.req_pool_indices, :metadata.max_seq_len_k
]

# 步骤 2: 如果 page_size > 1, 转换为块级索引
if self.page_size > 1:
    metadata.page_table = metadata.page_table[:, ::page_size] // page_size

# 步骤 3: reshape KV cache 为 4D 块格式
key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
key_cache = key_cache.view(-1, self.page_size, num_kv_heads, head_dim)
value_cache = value_cache.view(-1, self.page_size, num_kv_heads, head_dim)

# 步骤 4: 调用 flash_attn_with_kvcache
output = flash_attn_with_kvcache(
    q=q.view(-1, num_heads, head_dim),
    k_cache=key_cache,
    v_cache=value_cache,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    softmax_scale=scaling,
    window_size=window_size,
)
```

**vLLM 的适配实现**:

```python
# vllm: flash_attn.py — FlashAttentionImpl.forward()
key_cache, value_cache = kv_cache.unbind(0)  # 沿 dim=0 拆分 [2, blocks, ...]

flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key_cache,                    # 整个 key cache tensor
    v=value_cache,                  # 整个 value cache tensor
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,
    max_seqlen_k=max_seqlen_k,
    block_table=block_table,        # 从 CommonAttentionMetadata 获取
    window_size=sliding_window_size,
    scheduler_metadata=scheduler_metadata,
)
```

### 2.4 Sliding Window 的两种处理策略

| 策略 | FlashInfer (SGLang) | Flash-Attention (SGLang/vLLM) |
|------|--------------------|-----------------------------|
| **实现方式** | 在构建 `kv_indices` 时截断——通过 `kv_start_idx = seq_len - window_size` 只拷贝窗口内的索引 | 传 `window_size=(left, right)` 参数让 kernel 内部处理 |
| **Wrapper 数量** | 使用**两个独立 wrapper**：wrapper[0] 服务 SWA 层，wrapper[1] 服务 full attention 层 | 单一 kernel 调用，不同层传不同 `window_size` 参数 |
| **KV 存储** | 可选使用独立的 SWA KV pool（通过 `translate_loc_from_full_to_swa` 映射） | 同上——SWA 层可使用独立 pool |
| **优势** | 减少 kernel 内存访问（indices 中已无窗口外数据） | 实现简单，一个 kernel 参数解决 |

```python
# SGLang FlashInfer SWA 策略:
# IndicesUpdater.update_sliding_window()
if wrapper_id == 0:  # SWA wrapper
    paged_kernel_lens = torch.clamp(seq_lens, max=sliding_window_size + 1)
    kv_start_idx = seq_lens - paged_kernel_lens  # 窗口起始位置
else:  # Full attention wrapper
    paged_kernel_lens = seq_lens  # 全量
    kv_start_idx = None
```

### 2.5 适配模式总结

| 维度 | FlashInfer | Flash-Attention |
|------|-----------|----------------|
| **KV 索引格式** | CSR: `(kv_indptr, kv_indices)` 1D 展平 | 2D block_table `[bs, max_blocks]` |
| **KV buffer 传入** | 原始 3D tensor `[pool_size, H, D]` | 需 reshape 为 4D `[blocks, page_size, H, D]` |
| **两阶段 API** | `plan/begin_forward` → `forward/run` | 可选 `scheduler_metadata` 预计算 |
| **SWA 适配** | 截断 indices + 双 wrapper | `window_size` 参数 |
| **KV 写入** | 框架 `set_kv_buffer` (Python scatter) | 框架 `reshape_and_cache_flash` (C++ op) |

---

## 三、场景二：DeepSeek MLA (Multi-Head Latent Attention)

### 3.1 MLA 的核心特殊性

MLA 将多头 KV 投影到低秩潜在空间 (`kv_lora_rank`维)，KV Cache 只存储压缩后的 latent，而非完整的 per-head KV。这从根本上改变了 KV Cache 的形态：

- **标准 MHA**: 缓存 `[size, num_kv_heads, head_dim]` — 每个 head 独立存储
- **MLA**: 缓存 `[size, 1, kv_lora_rank + qk_rope_head_dim]` — 所有 head 共享一个 latent

### 3.2 框架侧 KV Cache 适配 (`MLATokenToKVPool`)

```python
# sglang: memory_pool.py — MLATokenToKVPool
class MLATokenToKVPool(KVCache):
    # kv_cache_dim = kv_lora_rank + qk_rope_head_dim (不使用 FP8 时)
    # 例如 DeepSeek-V2: kv_lora_rank=512, qk_rope_head_dim=64 → kv_cache_dim=576
    
    def _create_buffers(self):
        self.kv_buffer = [
            torch.zeros((self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype, device=self.device)
            for _ in range(self.layer_num)
        ]
    
    def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
        # 将 k_nope (kv_lora_rank 维) 和 k_rope (qk_rope_head_dim 维) 拼接存入
        # 通过 Triton kernel 实现 fused 写入
        kv_buffer[layer_id][loc, 0, :kv_lora_rank] = k_nope
        kv_buffer[layer_id][loc, 0, kv_lora_rank:] = k_rope
    
    def get_key_buffer(self, layer_id):
        return self.kv_buffer[layer_id]  # [size, 1, kv_cache_dim]
```

**vLLM 侧** 通过 `MLAAttentionSpec` 定义 MLA 的 KV Cache 规格:

```python
# vllm: kv_cache_interface.py
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str | None = None   # "fp8_ds_mla" 等
    alignment: int | None = None         # FlashMLA 对齐要求
    compress_ratio: int = 1
    model_version: str | None = None

    @property
    def real_page_size_bytes(self):
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                return self.storage_block_size * 584  # 448B NoPE FP8 + 128B RoPE BF16 + 8B scale
            return self.block_size * 656  # V3.2: 512B NoPE FP8 + 16B scale + 128B RoPE BF16
```

### 3.3 算子库适配：FlashInfer MLA Wrapper

FlashInfer 提供了专用的 `BatchMLAPagedAttentionWrapper`，其 API 要求将 Q 和 K 按 nope/rope 分别传入：

```python
# sglang: flashinfer_mla_backend.py — forward_decode
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    # 1. 写入 KV cache (latent format)
    self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
    
    # 2. 获取整层 KV buffer
    k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
    
    # 3. Q 拆分为 nope 和 rope 部分
    q_nope = q[..., :layer.v_head_dim]   # 用于 value output 的部分
    q_rope = q[..., layer.v_head_dim:]   # 用于 position scoring 的部分
    
    # 4. 调用 MLA wrapper (absorbed attention)
    output = decode_wrapper.run(
        q_nope,                                 # [bs, num_heads, v_head_dim]
        q_rope,                                 # [bs, num_heads, rope_dim]
        k_buffer[:, :, :layer.v_head_dim],      # k_nope 部分 (latent)
        k_buffer[:, :, layer.v_head_dim:],      # k_rope 部分
        out=output,
    )
```

**"Absorbed Attention" 的含义**：

传统 MLA 需要在 decode 时将 latent 还原为完整 KV (通过 `kv_b_proj` 矩阵)，计算开销大。Absorbed attention 的优化是：将 `kv_b_proj` 矩阵"吸收"到 Q 的投影中，这样 decode 时直接用 latent 做 attention scoring，无需还原。FlashInfer MLA wrapper 的 API 就是为这种 absorbed 模式设计的。

### 3.4 FlashMLA Kernel 的专用 ABI

对于非 FlashInfer 路径（如 FlashMLA 独立库），使用不同的接口：

```python
# flash_mla_with_kvcache 的核心 API:
flash_mla.flash_mla_with_kvcache(
    q=q,                    # [num_tokens, num_heads, head_dim]
    k_cache=kv_buffer,      # [num_pages, page_size, 1, kv_cache_dim] 
    head_dim_v=v_head_dim,  # 告诉 kernel latent 中 value 部分的维度
    softmax_scale=sm_scale,
    indices=page_indices,   # [bs, 1, num_pages_per_req] 页面索引
    topk_length=topk_lens,  # [bs] 每个 req 的实际 KV 长度
)
```

### 3.5 MLA 的 ABI 对齐总结

| 维度 | 标准 MHA | MLA |
|------|---------|-----|
| **KV 缓存形态** | `[size, num_kv_heads, head_dim]` × 2 (K/V 分离) | `[size, 1, kv_lora_rank + rope_dim]` (K/V 合一 latent) |
| **算子库接口** | `wrapper.forward(q, (k_buf, v_buf))` | `wrapper.run(q_nope, q_rope, k_nope_buf, k_rope_buf)` |
| **框架适配** | 直接传递 buffer | 需要按维度切分 k_buffer 为 nope/rope 两部分 |
| **算子设计策略** | 算子库通用 API | FlashInfer 提供专用 MLA wrapper；FlashMLA 是独立算子 |
| **是否算子库主动对齐** | 是 (paged API) | **是** — FlashInfer/FlashMLA 专门为 MLA 设计了 absorbed attention API |

---

## 四、场景三：Qwen3.X GDN Mamba 算子

### 4.1 默认实现确认

**GDN (Gated DeltaNet) 的默认实现使用 Triton 编写**，同时提供 FlashInfer 和 CuTe DSL 作为可选高性能后端：

| 后端 | 适用场景 | 来源 |
|------|---------|------|
| **Triton (FLA)** | 默认通用后端 | 两个框架内部实现 (`fla/ops/`) |
| **FlashInfer** | SM90+ (Hopper) | 外部库 `flashinfer.gdn_prefill` |
| **CuTe DSL** | SM90+ decode, SM100+ prefill | sglang 独有 (`cutedsl_gdn.py`) |

其中 FlashInfer GDN 是外部算子库提供的独立实现（`flashinfer.gdn_decode` / `flashinfer.gdn_prefill`），CuTe DSL 则是 sglang 基于 CUTLASS cute 编程模型编写的 JIT 编译内核。这两者虽不像 attention 那样涉及 paged KV cache 的复杂适配，但仍存在**状态池 (state pool) 访问模式**和**张量布局/dtype 转换**方面的 ABI 对齐需求。

### 4.2 GDN 的状态管理——与 KV Cache 的本质区别

GDN/Mamba 是**循环模型**，其"缓存"是固定大小的 state（不随序列长度增长），而非像 KV Cache 那样随 token 数增长。这带来根本性不同的资源管理模式：

| 维度 | Attention KV Cache | Mamba/GDN State |
|------|-------------------|-----------------|
| **大小** | 随序列长度线性增长 | 固定大小（与序列长度无关） |
| **形态** | Token-level: 每个 token 一个 slot | Request-level: 每个 request 一份完整 state |
| **更新模式** | Append-only (新 token 追加到末尾) | In-place update (每步覆写整个 state) |
| **分页需求** | 需要 paged allocation 应对变长 | 无需分页——固定分配 |

**SGLang 的状态管理 (`MambaPool`)**:

```python
# sglang: memory_pool.py — MambaPool
class MambaPool:
    # 每个 request 分配一个固定 slot
    # conv_state: [num_layers, pool_size, d_conv, d_inner]
    # temporal_state (SSM state): [num_layers, pool_size, d_state, d_inner]
    
    def __init__(self, ...):
        self.conv_state = [
            torch.zeros((num_layers, pool_size + 1, *conv_shape), device=device)
            for shape in conv_shapes
        ]
        self.temporal_state = torch.zeros(
            (num_layers, pool_size + 1, *temporal_shape), device=device
        )
```

**vLLM 的状态管理** 则将 Mamba state 视为特殊的 block-based KV Cache:

```python
# vllm: kv_cache_interface.py — MambaSpec
class MambaSpec(KVCacheSpec):
    # 复用 block allocator 管理 state，每个 "block" 存一份完整 state
    shapes: list  # state tensor shapes
    dtypes: list  # state tensor dtypes
```

### 4.3 GDN 算子的接口 (Triton FLA — 框架内定制)

**Decode 路径 — `fused_recurrent_gated_delta_rule`**:

```python
# vllm/sglang: fla/ops/fused_recurrent.py (Triton kernel)
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q, k, v, alpha, beta,    # 当前 token 的输入
    h,                        # SSM state [batch, num_heads, d_state, d_inner]
    o,                        # output
    ssm_state_indices,        # [batch] state pool 中的索引
    IS_CONTINUOUS_BATCHING: tl.constexpr,  # 非连续 batching 标志
    IS_SPEC_DECODING: tl.constexpr,
    ...
):
    # 从 state pool 中按 ssm_state_indices 读取对应 request 的 state
    # 执行 delta rule update: h = alpha * h + beta * outer(k, v)
    # 计算 output: o = q @ h
```

**Prefill 路径 — `chunk_gated_delta_rule`**:

```python
# Chunkwise parallel 计算: 将序列分块, 块内并行, 块间传递 state
def chunk_gated_delta_rule(q, k, v, alpha, beta, initial_state, ...):
    # 1. chunk_delta_h: 计算每个 chunk 的增量 state
    # 2. state_passing: 串行传递 state across chunks
    # 3. chunk_o: 计算每个 chunk 的 output
    return output, final_state
```

Triton 版本由框架完全控制接口设计，不存在外部 ABI 对齐问题。

### 4.4 FlashInfer GDN 的 ABI 对齐

FlashInfer 为 GDN 提供了三个独立函数：`gated_delta_rule_decode_pretranspose` (decode)、`chunk_gated_delta_rule` (prefill)、`gated_delta_rule_mtp` (MTP verify)。框架需要将自身的 state pool 和 tensor 布局转换为 FlashInfer 期望的格式。

#### 4.4.1 Decode 路径 — FlashInfer GDN

```python
# sglang: kernels/gdn_flashinfer.py — FlashInferGDNKernel.decode
# FlashInfer decode API 签名:
gated_delta_rule_decode_pretranspose(
    q=query_fi,           # [batch, 1, num_heads, head_k_dim], bf16
    k=key_fi,             # [batch, 1, num_heads, head_k_dim], bf16
    v=value_fi,           # [batch, 1, num_v_heads, head_v_dim], bf16
    state=state_batch,    # SM90: [batch, HV, V, K], float32 (手动 gather)
                          # SM100: None (由内核从 pool 索引)
    A_log=A_log,          # [HV], float32
    a=a_fi,               # [batch, 1, num_v_heads], bf16 (gating raw input)
    dt_bias=dt_bias,      # [HV], bf16
    b=b_fi,               # [batch, 1, num_v_heads], bf16 (beta raw input)
    use_qk_l2norm=True,
    initial_state=ssm_states,          # SM100: 整个 pool [pool_size, HV, V, K]
    initial_state_indices=cache_indices,# SM100: [batch] 索引
)
```

**SM90 (Hopper) vs SM100 (Blackwell) 的关键差异**:

```python
# SM90: 框架负责 gather/scatter
state_batch = ssm_states[cache_indices]      # gather from pool
output, new_state = decode_fn(... state=state_batch ...)
ssm_states[cache_indices] = new_state        # scatter back

# SM100: FlashInfer 内核原生支持 pool 索引
output, _ = decode_fn(
    ... state=None,
    initial_state=ssm_states,                # 传整个 pool
    initial_state_indices=cache_indices,      # 内核内部做间接寻址
)
```

#### 4.4.2 Prefill 路径 — FlashInfer GDN

```python
# FlashInfer prefill API:
chunk_gated_delta_rule(
    q=q_fi,                    # [total_seq_len, num_heads, head_k_dim], bf16 (已 L2norm)
    k=k_fi,                    # [total_seq_len, num_heads, head_k_dim], bf16 (已 L2norm)
    v=v_fi,                    # [total_seq_len, num_v_heads, head_v_dim], bf16
    g=torch.exp(g_fi),         # [total_seq_len, num_v_heads], float32 — 注意需要 exp!
    beta=beta_fi,              # [total_seq_len, num_v_heads], float32
    initial_state=state_fi,    # [batch, HV, V, K], float32
    output_final_state=True,
    cu_seqlens=query_start_loc, # cumulative seq lens for varlen
)
```

#### 4.4.3 框架侧需完成的格式转换

| 维度 | 框架原始格式 | FlashInfer 期望格式 | 转换操作 |
|------|-----------|-------------------|---------|
| **q/k/v (decode)** | `[1, batch, H, D]` | `[batch, 1, H, D]` | `.view(batch, 1, H, D)` |
| **q/k/v (prefill)** | `[1, total_seq, H, D]` (4D) | `[total_seq, H, D]` (3D) | `.squeeze(0).contiguous()` |
| **g (alpha gate)** | log-space, bf16 | `exp(g)`, float32 | `torch.exp(g.to(float32))` |
| **beta** | bf16 | float32 | `.to(torch.float32)` |
| **state (SM90 prefill)** | pool `[pool_size, HV, V, K]` | `[batch, HV, V, K]` float32 | `ssm_states[cache_indices].to(float32)` |
| **state (SM100 decode)** | pool `[pool_size, HV, V, K]` | 直接传 pool + indices | 无需转换 |
| **cu_seqlens** | int32 | SM90 需 int64，SM100 int32 | 条件 `.to(torch.int64)` |
| **L2 norm** | 外部 `l2norm_fwd(q)` | `use_qk_l2norm=True/False` | prefill 外部做传 False；decode 内核做传 True |

#### 4.4.4 MTP (Multi-Token Prediction) 路径

```python
# MTP verify: 每次处理多个 draft tokens
gated_delta_rule_mtp(
    q=query_mtp,               # [batch, draft_len, H, K], bf16
    k=key_mtp,                 # [batch, draft_len, H, K], bf16
    v=value_mtp,               # [batch, draft_len, HV, V], bf16
    initial_state=ssm_states,  # 整个 pool (SM100 原生支持)
    initial_state_indices=cache_indices,
    A_log=A_log, a=a_mtp, dt_bias=dt_bias, b=b_mtp,
    intermediate_states_buffer=buffer,  # 预分配的中间 state buffer
    disable_state_update=True,          # MTP 验证模式不写回 state
    use_qk_l2norm=True,
)
```

### 4.5 CuTe DSL GDN 的 ABI 对齐

CuTe DSL 是 sglang 基于 CUTLASS `cutlass.cute` API 编写的 JIT CUDA 内核，通过 `from_dlpack` 实现 PyTorch tensor 到 CUTE tensor 的零拷贝桥接。

#### 4.5.1 Decode 路径 — CuTe DSL (SM90+)

```python
# sglang: kernels/gdn_cutedsl.py — CuteDSLGDNKernel.decode
cutedsl_fused_sigmoid_gating_delta_rule_update(
    A_log=A_log,               # [HV], float32
    dt_bias=dt_bias,           # [HV], bf16
    q=q, k=k, v=v,            # [1, N, H, K/V], bf16 (原始 4D)
    a=a, b=b,                  # gating raw inputs, bf16
    initial_state_source=ssm_states,      # 整个 pool [pool_size, HV, K, V], float32
    initial_state_indices=cache_indices,   # [N], int32
    cu_seqlens=query_start_loc,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
)
```

**核心设计**: CuTe DSL decode 内核**直接接收整个 state pool 指针**，内核内部通过 `pool_idx = h0_indices[i_n]` 做间接寻址，原地读写 state，无需框架侧的 gather/scatter。

```python
# 内核内部 (cutedsl_gdn.py — gdn_kernel_large_batch_varlen):
pool_idx = h0_indices[i_n]                    # 间接寻址
gSrc_batch = h0_source[(pool_idx, i_hv, :, :)]  # 从 pool 读取 state tile
# ... delta rule 计算 ...
h0_source[(pool_idx, i_hv, k, v)] = h_new    # 原地写回 pool
```

#### 4.5.2 Prefill 路径 — CuTe DSL (SM100+ Blackwell only)

```python
# sglang: kernels/gdn_blackwell/__init__.py
chunk_gated_delta_rule_cutedsl(
    q=q_norm,              # [1, T, H, K], bf16, 已 L2-normed
    k=k_norm,              # [1, T, H, K], bf16, 已 L2-normed
    v=v,                   # [1, T, HV, V], bf16
    g=g,                   # [1, T, HV], float32, log-space (未 exp!)
    beta=beta,             # [1, T, HV], float32
    initial_state=initial_state,  # [N, HV, V, K], float32 — 注意: V,K 转置!
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,   # [num_chunks, 2], int32
    chunk_offsets=chunk_offsets,   # [N+1], int32
)
```

**Prefill 的关键区别**: 与 decode 不同，CuTe DSL prefill **不支持** pool 直接索引。框架需要手动 gather/scatter：

```python
# 框架侧 extend 方法:
# Gather
initial_state = ssm_states[ssm_cache_indices].contiguous()  # [N, HV, K, V]

# 计算 chunk 元数据
chunk_indices, chunk_offsets = prepare_chunk_metadata(cu_seqlens, chunk_size=64)

# 调用内核
output, final_state = extend_fn(q, k, v, g, beta, initial_state, ...)

# Scatter 回 pool
ssm_states.index_copy_(0, ssm_cache_indices, final_state.to(ssm_states.dtype))
```

#### 4.5.3 后端分发逻辑 (`GDNKernelDispatcher`)

```python
# sglang: gdn_backend.py — GDNKernelDispatcher
class GDNKernelDispatcher:
    def __init__(self, decode_backend, prefill_backend):
        triton_kernel = TritonGDNKernel()
        cutedsl_kernel = CuteDSLGDNKernel() if decode_backend.is_cutedsl() else None
        flashinfer_kernel = FlashInferGDNKernel() if decode_backend.is_flashinfer() else None
        
        # Decode 后端选择
        if decode_backend.is_cutedsl():
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            self.decode_kernel = flashinfer_kernel
        else:
            self.decode_kernel = triton_kernel
        
        # Prefill 后端选择 (CuTe DSL prefill 仅 SM100+)
        if prefill_backend.is_cutedsl() and cutedsl_kernel.supports_prefill:
            self.extend_kernel = cutedsl_kernel
        elif prefill_backend.is_flashinfer():
            self.extend_kernel = flashinfer_kernel
        else:
            self.extend_kernel = triton_kernel  # 默认回退
```

### 4.6 GDN 三种后端的 ABI 对比

| 维度 | Triton (FLA) | FlashInfer | CuTe DSL |
|------|-------------|-----------|-----------|
| **来源** | 框架内部 | 外部库 (`flashinfer`) | sglang JIT 编译 |
| **State 访问 (decode)** | `ssm_state_indices` 间接寻址 | SM90: gather/scatter；SM100: pool+indices | pool+indices 原地读写 |
| **State 访问 (prefill)** | gathered `[batch, HV, V, K]` | gathered `[batch, HV, V, K]` float32 | gathered `[N, HV, V, K]` float32 |
| **q/k 输入形状** | `[batch, H, K]` (decode) | `[batch, 1, H, K]` (decode)；`[seq, H, K]` (prefill) | `[1, N, H, K]` (decode)；`[1, T, H, K]` (prefill) |
| **g (alpha) 格式** | log-space, 内核内 softplus+exp | `exp(g)` float32 (prefill)；raw+softplus (decode) | raw (内核内 softplus+exp) |
| **L2 norm** | 内核内部 | prefill: 外部做；decode: 内核内 | prefill: 外部做；decode: 内核内 |
| **dtype 要求** | bf16 in/out, fp32 state | bf16 in/out, fp32 state (SM90) / bf16 state (SM100) | bf16 in/out, fp32 state |
| **额外 metadata** | `chunk_indices`, `chunk_offsets` | `cu_seqlens` | decode: `cu_seqlens`；prefill: `chunk_indices`+`chunk_offsets` |

### 4.7 GDN 的 ABI 对齐总结

| 维度 | 说明 |
|------|------|
| **核心适配点** | State pool 的访问模式（整池传入 vs gather/scatter）+ 张量 shape/dtype 转换 |
| **与 attention 的区别** | 无 paged KV cache，但 state pool 索引类似 page table 的作用 |
| **FlashInfer 的设计策略** | SM100 版本主动适配推理框架的 pool 模型（原生 `initial_state_indices`）；SM90 版本需要框架做 gather |
| **CuTe DSL 的设计策略** | Decode: 原生 pool 索引（最高效）；Prefill: 需框架 gather（Blackwell TMA 约束） |
| **算子库是否主动对齐** | **是** — FlashInfer SM100 和 CuTe DSL decode 都主动提供了 `initial_state_indices` 参数适配推理框架的 state pool 模型 |

---

## 五、场景四：DeepSeek V3.2 DSA (DeepSeek Sparse Attention)

### 5.1 DSA 的核心机制

DSA 在 MLA 基础上引入 **Indexer** 实现稀疏注意力：
1. 一个小型 Indexer (64 heads, head_dim=128) 对全序列计算 FP8 MQA logits
2. 从 logits 中选取 topk (默认 2048) 个最相关位置
3. 仅对这 topk 位置做完整 MLA attention

这引入了新的资源管理需求：**独立的 Indexer KV Cache**。

### 5.2 Indexer 的独立 FP8 KV 缓存

```python
# sglang: memory_pool.py — DSATokenToKVPool (继承 MLATokenToKVPool)
class DSATokenToKVPool(MLATokenToKVPool):
    def __init__(self, ...):
        super().__init__(...)  # MLA latent buffer
        
        # 额外: Indexer 专用的 FP8 K cache
        # 每 page 存储: page_size * (head_dim + head_dim/quant_block_size * 4) bytes
        # 即: 64 * (128 + 128/128 * 4) = 64 * 132 = 8448 bytes per page
        self.index_k_with_scale_buffer = [
            torch.zeros(
                (num_pages, page_size * (index_head_dim + scale_bytes)),
                dtype=torch.uint8, device=device,
            )
            for _ in range(layer_num)
        ]
    
    def get_index_k_with_scale_buffer(self, layer_id):
        return self.index_k_with_scale_buffer[layer_id]
```

### 5.3 Indexer 算子接口 — `deep_gemm.fp8_paged_mqa_logits`

Indexer 使用 DeepGEMM 库提供的 FP8 分页 MQA logits 计算：

```python
# sglang: dsa/dsa_indexer.py — Indexer._get_topk_paged
def _get_topk_paged(self, forward_batch, layer_id, q_fp8, weights, metadata):
    # 获取 indexer 的 FP8 KV cache
    kv_cache_fp8 = get_token_to_kv_pool().get_index_k_with_scale_buffer(layer_id)
    # reshape: [num_pages, page_size(64), 1, head_dim_with_scale(132)]
    kv_cache_fp8 = kv_cache_fp8.view(num_pages, block_kv, num_heads_kv, head_dim_with_sf)
    
    # 调用 DeepGEMM: FP8 paged MQA logits
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8.view(B, next_n, num_heads, head_dim),  # FP8 query
        kv_cache_fp8,                                 # FP8 paged K cache
        weights,                                      # 头门控权重
        seqlens_32_2d,                               # 序列长度
        block_tables,                                # 页表 (复用 req_to_token)
        schedule_metadata,                           # 调度
        max_seq_len,
    )
    
    # topk 选取
    topk_indices = metadata.topk_transform(logits, self.index_topk)
    return topk_indices
```

**DeepGEMM 的 ABI**:
- Q: FP8 tensor `[batch, next_n, num_heads, head_dim]`
- K cache: FP8 paged `[num_pages, page_size, num_heads_kv, head_dim + scale_bytes]`
- block_tables: 标准 2D 页表 `[batch, max_pages]` (复用框架的 req_to_token)
- 输出: logits `[batch, num_heads, max_seq_len]`

### 5.4 topk → Sparse Page Table 转换

Indexer 输出的 topk_indices 需要转换为 sparse kernel 可接受的页面索引：

```python
# sglang: dsa_backend.py — forward_decode
# topk_indices: [batch, topk] — 每个 request 选中的 token 位置
# 需要翻译为物理 page 地址

page_table_1 = transform_index_page_table_decode(
    page_table=metadata.page_table_1,     # 框架的 req_to_token 映射
    topk_indices=topk_indices,            # indexer 输出
    page_size=1,                          # MLA latent 的 page_size
)
# page_table_1: [batch, topk] — 选中 token 在 KV pool 中的物理 slot
```

### 5.5 Sparse Attention 算子接口 — `flash_mla_sparse_fwd`

```python
# sglang: dsa_backend.py — forward_decode
flash_mla_sparse_fwd(
    q=q_input,                              # [num_tokens, num_heads(padded), head_dim]
    kv=kv_cache,                            # MLATokenToKVPool 的整个 latent buffer
    indices=page_table_1.unsqueeze(1),      # [batch, 1, topk] 稀疏索引
    sm_scale=sm_scale,
    d_v=v_head_dim,                         # latent 中 value 部分的维度
)
```

**FlashMLA Sparse 的 ABI**：
- `kv`: 完整的 KV pool tensor `[pool_size, 1, kv_cache_dim]`
- `indices`: 3D 稀疏索引 `[batch, num_kv_heads=1, topk]` — 每个 request 要 attend 的 slot
- kernel 按 indices 从 kv pool 中 gather 对应 slot 的 latent

### 5.6 DSA 的完整调用流程

```
模型层 (DeepseekV2AttentionMLA)
    │
    ├── q_lora, k, k_rope = forward_prepare(hidden_states, positions)
    │
    ▼
DSA Backend.forward()
    │
    ├── 1. set_mla_kv_buffer(loc, k, k_rope)  → 写入 latent KV cache
    │
    ├── 2. Indexer.forward(q_lora, x, positions)
    │       ├── _get_q_k_bf16() → 计算 indexer Q/K
    │       ├── _store_index_k_cache() → FP8 量化 + 写入 indexer K cache
    │       └── _get_topk_paged()
    │             ├── deep_gemm.fp8_paged_mqa_logits() → 计算全序列 logits
    │             └── topk_transform() → 选取 top-2048 位置
    │
    ├── 3. transform_index_page_table_decode() → topk → 物理地址
    │
    └── 4. flash_mla_sparse_fwd(q, kv, indices) → 稀疏 attention
```

### 5.7 DSA 的 ABI 对齐总结

| 组件 | 算子来源 | ABI 对齐方式 |
|------|---------|------------|
| **Indexer logits** | DeepGEMM (外部库) | 框架将 indexer K cache reshape 为 DeepGEMM 期望的 FP8 paged 格式；复用标准 block_table |
| **Sparse attention** | FlashMLA (外部库) | 框架通过 `transform_index_page_table_decode` 将 topk → 物理地址；传入 unsqueeze 后的 3D indices |
| **Indexer K cache** | 框架管理 | `DSATokenToKVPool` 额外维护 FP8 buffer；写入时做 fused quantize + store |
| **主 KV cache** | 框架管理 | 继承 `MLATokenToKVPool` 的 latent buffer |

---

## 六、场景五：DeepSeek V4 C4A/C128A

### 6.1 V4 的多压缩率架构

DeepSeek V4 引入了**按层配置不同压缩率**的设计：
- **SWA-only 层** (compress_ratio=0): 仅使用滑动窗口 attention
- **C4A 层** (compress_ratio=4): 每 4 个 token 压缩为 1 个 compressed KV + SWA + Indexer 稀疏选取
- **C128A 层** (compress_ratio=128): 每 128 个 token 压缩为 1 个 + SWA (无 indexer——压缩比已足够大)

### 6.2 四 Pool 架构

```python
# sglang: deepseek_v4_memory_pool.py — DeepSeekV4TokenToKVPool
class DeepSeekV4TokenToKVPool(BaseSWAKVPool):
    def __init__(self, ...):
        # Pool 1: SWA 窗口 KV (所有层共享)
        self.swa_kv_pool = DeepSeekV4SingleKVPool(swa_size, swa_page_size, ...)
        
        # Pool 2: C4 压缩 KV (仅 C4A 层使用)
        self.c4_kv_pool = DeepSeekV4SingleKVPool(c4_size, c4_page_size, ...)
        
        # Pool 3: C128 压缩 KV (仅 C128A 层使用)
        self.c128_kv_pool = DeepSeekV4SingleKVPool(c128_size, c128_page_size, ...)
        
        # Pool 4: C4 Indexer KV (仅 C4A 层的 indexer 使用)
        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(indexer_size, ...)
```

**每 token 存储开销 (584 bytes)**:
- 448 bytes: NoPE FP8 data (kv_lora_rank=448 在 FP8 下)
- 128 bytes: RoPE BF16 data (qk_rope_head_dim=64 × 2 bytes)
- 8 bytes: FP8 scale

**层级映射 (将 layer_id 路由到正确的 pool)**:

```python
# 每层根据 compression_ratios 配置映射
class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int                  # 在对应 pool 内的局部层 ID
    compress_kv_pool: Optional[...]         # 指向 c4_kv_pool 或 c128_kv_pool

# API 路由:
def get_extra_key_buffer(self, layer_id):
    _, compress_layer_id, pool = self.layer_mapping[layer_id]
    return pool.get_key_buffer(compress_layer_id)
```

### 6.3 Compressor 的在线/离线模式

C4 和 C128 使用不同的压缩策略：

**C4 Compressor (overlap=True)**:
- 维护一个大小为 `2 * head_dim` 的 state buffer (双倍 state)
- 每积累 4 个 token 输出一个 compressed KV
- Decode 时: 每 token 更新 state，每 4 步输出一次

**C128 Compressor**:
- **在线模式**: 维护 `(max, sum, kv)` 单状态，每 128 步输出
- **离线模式 (prefill)**: 使用 128-slot ring buffer，一次性处理

```python
# sglang: dsv4/compressor.py — CompressorBackendMixin
def forward_compress(self, *, kv_score_buffer, kv_score_input, ape,
                     compress_ratio, head_dim, norm, freqs_cis_cache, ...):
    assert compress_ratio in (4, 128)
    metadata = self.get_paged_compress_metadata(compress_ratio)
    
    # JIT fused kernel: 输入 token KV → 压缩 → RoPE → 量化 → 写入 pool
    kv_compressed = compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        indices=metadata.indices,
        plan=metadata.plan,
        compress_ratio=compress_ratio,
    )
    # Norm + RoPE inplace
    compress_fused_norm_rope_inplace(kv_compressed, norm.weight, ...)
    return kv_compressed
```

### 6.4 `flash_mla_with_kvcache` 的统一 `extra_k_cache` 接口

V4 的核心设计亮点是：**同一个 FlashMLA kernel 通过 `extra_k_cache` 参数统一处理 C4/C128**，不需要为不同压缩率写不同 kernel：

```python
# sglang: deepseek_v4_backend.py — forward (decode 路径)
def forward(self, q, k, v, layer, forward_batch, compress_ratio, ...):
    # SWA cache (所有层都有)
    swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
    
    # Extra cache (按 compress_ratio 选择)
    if compress_ratio == 4:
        extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)  # C4 pool
        extra_indices = core_metadata.c4_sparse_page_indices  # indexer 选出的稀疏索引
        extra_topk_lengths = core_metadata.c4_sparse_topk_lengths
    elif compress_ratio == 128:
        extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)  # C128 pool
        extra_indices = core_metadata.c128_page_indices  # 全量索引 (无 indexer)
        extra_topk_lengths = core_metadata.c128_topk_lengths_clamp1
    
    # 统一 kernel 调用
    output = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=swa_k_cache,               # 主缓存: SWA 窗口
        head_dim_v=self.head_dim_v,
        softmax_scale=self.softmax_scale,
        indices=swa_page_indices,           # SWA 索引
        topk_length=swa_topk_lengths,       # SWA 长度
        extra_k_cache=extra_k_cache,        # 附加: C4/C128 压缩缓存
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_lengths,
    )
```

**FlashMLA `extra_k_cache` 的 ABI**:
- `k_cache` (必选): SWA 窗口缓存，索引由 `indices` 指定
- `extra_k_cache` (可选): 额外的压缩 KV 缓存（C4 或 C128）
- `extra_indices_in_kvcache`: 在 extra cache 中的索引
- `extra_topk_length`: 每个 request 在 extra cache 中的有效长度
- Kernel 内部将 SWA 结果和 extra 结果做 online softmax 合并

### 6.5 C4 Indexer 的接口

C4 Indexer 与 DSA Indexer 类似，但有关键区别：**它索引的是压缩后的 KV**（每 4 个原始 token = 1 个 compressed entry），而非原始 token。

```python
# sglang: dsv4/indexer.py — C4IndexerBackendMixin.forward_c4_indexer
def forward_c4_indexer(self, q_lora, x, positions, layer_id, forward_batch):
    # 1. 压缩新 token 的 indexer KV 并写入 indexer pool
    self.forward_indexer_compressor()
    
    # 2. 获取 indexer KV cache (已压缩 4x)
    c4_indexer_kv = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    
    # 3. DeepGEMM FP8 paged MQA logits
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8, c4_indexer_kv, weights, seqlens, block_tables, ...)
    
    # 4. topk (默认 512) → 翻译为 C4 pool 中的物理地址
    topk_indices = topk_transform_512(logits, ...)
    
    # 5. HiSparse: 将选中的远端 page 换入设备
    core_metadata.c4_sparse_page_indices = hisparse_coordinator.swap_in_selected_pages(...)
```

### 6.6 Metadata 路由与 Tile 调度

V4 的 metadata 按压缩率分组管理 FlashMLA 调度：

```python
# sglang: deepseek_v4_backend.py — DSV4AttnMetadata
@dataclass
class DSV4AttnMetadata:
    # SWA (所有层)
    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor
    
    # C4 (compress_ratio=4 的层)
    c4_sparse_page_indices: torch.Tensor   # indexer 选出的稀疏索引
    c4_sparse_topk_lengths: torch.Tensor
    
    # C128 (compress_ratio=128 的层)
    c128_page_indices: torch.Tensor        # 全量索引
    c128_topk_lengths_clamp1: torch.Tensor
    
    # FlashMLA Tile 调度 (三组独立调度)
    c1_flashmla_metadata: FlashMLASchedMeta   # SWA-only 层
    c4_flashmla_metadata: FlashMLASchedMeta   # C4A 层
    c128_flashmla_metadata: FlashMLASchedMeta # C128A 层
```

**vLLM 侧的对应实现**:

```python
# vllm: deepseek_v4_attention.py — _forward_decode
def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, ...):
    # 按 compress_ratio 选取 tile scheduler
    if self.compress_ratio <= 1:
        tile_metadata = swa_metadata.tile_sched_swaonly
    elif self.compress_ratio == 4:
        tile_metadata = swa_metadata.tile_sched_c4a
    elif self.compress_ratio == 128:
        tile_metadata = swa_metadata.tile_sched_c128a
    
    flash_mla_with_kvcache(
        q=q.unsqueeze(1),
        k_cache=swa_cache,
        extra_k_cache=kv_cache if not swa_only else None,
        extra_indices_in_kvcache=topk_indices,
        extra_topk_length=topk_lens,
        tile_scheduler_metadata=tile_metadata,
        out=output.unsqueeze(1),
    )
```

### 6.7 C4A/C128A 的 ABI 对齐总结

| 组件 | 算子/库 | ABI 格式 | 框架适配工作 |
|------|--------|---------|------------|
| **SWA Attention** | FlashMLA | `k_cache + indices + topk_length` | 维护 swa_kv_pool，构建 swa_page_indices |
| **C4/C128 Attention** | FlashMLA (同一 kernel) | `extra_k_cache + extra_indices + extra_topk_length` | 维护 c4/c128_kv_pool；C4 需 indexer 选稀疏索引 |
| **C4 Indexer** | DeepGEMM | `fp8_paged_mqa_logits(q, kv, weights, block_tables)` | 维护 indexer_kv_pool；压缩后写入；topk → 物理地址翻译 |
| **Compressor** | 框架内 Triton/JIT | `compress_forward(kv, indices, plan)` | 维护 compress state pool；输出写入 c4/c128 pool |
| **Tile 调度** | FlashMLA | `tile_scheduler_metadata` | 按 compress_ratio 分组预计算三组独立调度 |

---

## 七、总结

### 7.1 各场景对齐方式对比

| 场景 | 核心算子 | 算子来源 | KV 缓存形态 | 适配模式 |
|------|---------|---------|------------|---------|
| **常规 MHA/GQA** | FlashInfer / Flash-Attention | 第三方通用库 | `[size, H, D]` per-head token-level | 算子库**预设** paged API (CSR / block_table)；框架做格式转换 |
| **DeepSeek MLA** | FlashInfer MLA / FlashMLA | 第三方专用库 | `[size, 1, lora_rank + rope_dim]` latent | 算子库**专门设计** absorbed attention API；框架按 nope/rope 切分传入 |
| **Qwen3 GDN** | FLA Triton / FlashInfer GDN / CuTe DSL | 框架内定制 + 外部库 | 固定大小 SSM state (非 token-level) | State pool 索引传递 + shape/dtype 转换；FlashInfer SM100/CuTe DSL decode 原生支持 pool 索引 |
| **DeepSeek V3.2 DSA** | DeepGEMM + FlashMLA Sparse | 外部专用库 | MLA latent + 独立 FP8 indexer K cache | 框架管理双 pool；Indexer 复用标准 block_table；sparse kernel 用 topk indices |
| **DeepSeek V4 C4A/C128A** | FlashMLA + DeepGEMM | 外部专用库 | 四 Pool (SWA + C4 + C128 + Indexer) | FlashMLA 的 `extra_k_cache` 统一接口；框架做多 pool 路由 + compress state 管理 |

### 7.2 核心结论

1. **算子库确实在设计阶段就考虑了框架需求**：FlashInfer 和 Flash-Attention 从一开始就提供了 paged KV cache 的原生 API。对于新兴的注意力变体（MLA、sparse attention），算子库也快速跟进了专用接口（如 FlashInfer MLA wrapper、FlashMLA sparse）。

2. **但框架侧的适配工作量依然很大**：
   - 格式转换（req_to_token → CSR / block_table）
   - 多 Pool 管理（DSA/V4 需要 2-4 个独立 pool）
   - 写入时序控制（先写 KV 再读取做 attention）
   - Sliding window 策略（截断 indices vs 传参）
   - Metadata 预计算（FlashInfer plan / FA scheduler_metadata）

3. **GDN/Mamba 的外部算子（FlashInfer/CuTe DSL）也存在 ABI 对齐**：虽然不涉及 paged KV cache，但 state pool 的访问模式是核心适配点。较新的实现（FlashInfer SM100、CuTe DSL decode）已主动提供 `initial_state_indices` 参数，直接接受框架的 pool + 索引模型，避免了 gather/scatter 开销。这体现了算子库逐步向推理框架的资源管理模型靠拢的趋势。

4. **越新越复杂的架构，框架承担的适配工作越重**：
   - 常规 MHA: 仅需 1 个 pool + 1 次格式转换
   - MLA: 1 个 latent pool + nope/rope 切分
   - DSA: 2 个 pool + indexer 全流程
   - V4: 4 个 pool + compressor state + indexer + per-layer routing

### 7.3 架构演进趋势

```
MHA (单 Pool, 通用 API)
  │
  ├── MLA (压缩 Pool, 专用 absorbed API)
  │     │
  │     ├── DSA = MLA + Indexer (双 Pool, sparse API)
  │     │     │
  │     │     └── V4 C4A/C128A = DSA + Compressor + 多压缩率
  │     │           (四 Pool, 统一 extra_k_cache API, per-layer routing)
  │
  └── GDN/Mamba (State Pool, 框架内定制 Triton)
```

每一代演进都在框架侧增加了新的 Pool 类型和适配逻辑，但算子库侧通过 `extra_k_cache` 等扩展参数保持了 API 的向后兼容性，避免了每次都需要全新 kernel 的问题。
