# 通用算子库与推理框架的 ABI 对齐调研报告

> 调研对象：`sglang`（主）/ `vllm`（对照）当前仓库默认推理路径上的 attention 相关算子
> 输出：本文件 `attention_framework_adapter_3o.md`
> 范围：仅 attention / linear-attention（mamba）相关算子的资源（KV cache / sliding window / mamba state）与 ABI 对齐；不涉及 FFN / MOE。

---

## 1. 执行摘要与核心结论

### 1.1 核心问题的回答

> **问题**：FlashInfer / Flash-Attention 这类通用算子库，接入 sglang / vLLM 时，是"算子库开发阶段就预先对齐了框架的 KV 缓存形态"，还是"先有基础实现，接入时框架侧再改造去适配"？

**结论：既不是单方面预对齐，也不是接入时改算子库源码，而是"契约式分工，且 ABI 的主导权在算子库"。** 具体地：

1. **算子库主导 ABI**：通用算子库在开发阶段就把"分页 KV（paged KV）"当作一等公民，定义了自己的**原生分页接口约定**。它并不针对某个框架，而是定义了一套通用的"页表 + 序列长度 + 页大小"入参协议：
   - Flash-Attention（FA3）：`flash_attn_with_kvcache(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window_size, ...)`；
   - FlashInfer：`BatchPrefill/DecodeWithPagedKVCacheWrapper.plan(kv_indptr, kv_indices, kv_last_page_len, page_size, ...)`（CSR 三元组）；
   - FlashMLA：`flash_mla_with_kvcache(k_cache.view(-1, 64, 1, 576), block_table, cache_seqlens, head_dim_v=512, ...)`。

2. **框架负责三件事，且不改算子库源码**：
   - **资源分配对齐**：框架按算子库要求的 dtype / 维度顺序 / 页大小去**预分配** KV / State 资源池（例如 MLA 把 `latent(512)+rope(64)=576` 存成一块连续 buffer，正是为了喂给 FlashMLA/FlashInfer-MLA）；
   - **元数据翻译（marshalling）**：框架用轻量的、多为 **Triton** 写的"索引搬运"kernel，把自己内部的 `req_to_token` 页表实时翻译成算子库当步需要的 `page_table` / `kv_indices` / `cache_seqlens` 等元数据；
   - **调用收口**：所有后端实现同一个抽象基类 `AttentionBackend`，把"准备元数据 + 调算子"这套动作统一到 `init_forward_metadata` / `forward_extend` / `forward_decode` 几个方法里。

3. **越新的架构，框架侧新增的"专属资源池 + 定制 Triton 元数据/搬运 kernel"越多，但核心计算算子仍复用第三方库**。MLA → DSA → V4 一路演进，框架不断新增 `MLATokenToKVPool` → `DSATokenToKVPool`（多了 index-K cache）→ `DeepSeekV4TokenToKVPool` + `CompressStatePool`，以及配套的 topk / 压缩 / 元数据 Triton kernel；但真正做 attention 数值计算的核心 kernel（`flash_mla_with_kvcache` / `flash_mla_sparse_fwd` / `deep_gemm` logits）依然是第三方或 `sgl_kernel` 提供。

一句话：**算子库定义"我需要什么形状的 KV 和什么格式的页表"，框架负责"把资源摆成那个形状、把页表翻译成那个格式、然后调用"。这是一种编译期就约定好的 ABI 契约，不是运行时改源码。**

### 1.2 五场景一句话结论表

| # | 场景 | 默认核心算子 | 第三方 / 定制 | 一句话对齐方式 |
|---|------|-------------|--------------|---------------|
| 1 | 常规 GQA/MHA（含 SWA） | `flash_attn_with_kvcache`（FA3） | 第三方 | 框架把 `req_to_token` 翻成 `page_table`+`cache_seqlens`，滑窗用 `window_size` 元组 |
| 2 | DeepSeek MLA | `BatchMLAPagedAttentionWrapper.run` / `flash_mla_with_kvcache` | 第三方 | 框架把 latent+rope 存成 576 宽单 buffer，调用时切片喂给算子 |
| 3 | Qwen3.X GDN mamba | `chunk_gated_delta_rule` / `causal_conv1d` | **框架定制 Triton**（vendored `fla/`+`mamba/`） | 无第三方 ABI 问题；对齐点是 MambaPool 的 conv/temporal 状态 + `cache_indices` |
| 4 | DeepSeek V3.2 DSA | `flash_mla_sparse_fwd` + `deep_gemm` logits | 第三方核心 + 定制 Triton 辅助 | MLA 之上新增 index-K cache + topk 索引，稀疏页表经 `indices=` 传入 |
| 5 | DeepSeek V4 C4A/C128A | `flash_mla.flash_mla_with_kvcache` / `flash_mla_sparse_fwd` | 第三方核心 + 定制 Triton 压缩 | 复用 DSA indexer，新增双压缩比资源池 + 压缩/元数据 Triton kernel |

---

## 2. 总体架构：框架与算子库的职责边界

### 2.1 三层分工模型

```
┌─────────────────────────────────────────────────────────────┐
│  ①  资源池预分配（形状由算子库 ABI 决定，框架负责摆放）           │
│     mem_cache/memory_pool.py                                  │
│     MHATokenToKVPool / MLATokenToKVPool / DSATokenToKVPool /  │
│     MambaPool / HybridLinearKVPool                            │
└───────────────────────────┬─────────────────────────────────┘
                            │ get_key_buffer / get_kv_buffer / set_kv_buffer
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ②  每步元数据 marshalling（多为 Triton "索引搬运" kernel）      │
│     triton_ops/kv_indices.py                                  │
│     create_flashinfer_kv_indices_triton / *_flashmla_*        │
│     req_to_token 页表  ──►  page_table / kv_indices /          │
│                             cache_seqlens / kv_indptr          │
└───────────────────────────┬─────────────────────────────────┘
                            │ init_forward_metadata_{out,in}_graph
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ③  统一调用收口（AttentionBackend 抽象基类）                    │
│     layers/attention/base_attn_backend.py                     │
│     forward → forward_extend / forward_decode                 │
│              ──► 调用第三方算子（FA3 / FlashInfer / FlashMLA）    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 框架↔算子的统一契约：`AttentionBackend`

所有后端（无论包装第三方算子还是定制算子）都继承 [base_attn_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/base_attn_backend.py) 的 `AttentionBackend(ABC)`（L18）。契约方法分两类：

**A. 元数据准备（把框架资源翻译成算子入参）**

```python
# base_attn_backend.py L41-83
def init_forward_metadata(self, forward_batch):           # 每步 eager 入口
    self.init_forward_metadata_out_graph(forward_batch)   # host 侧动态逻辑（.item()/.cpu() 等）
    self.init_forward_metadata_in_graph(forward_batch)    # 可录进 CUDA graph 的静态 GPU op
```

`out_graph` 负责跑动态形状/主机侧逻辑（如页表翻译），`in_graph` 只放能被 CUDA graph 录制的静态 GPU op —— 这个拆分正是为了让"元数据翻译"这一步能安全地与 CUDA graph 共存。

**B. 计算分发**

```python
# base_attn_backend.py L155-223
def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    if forward_batch.forward_mode.is_decode():
        return self.forward_decode(...)     # L199
    else:
        return self.forward_extend(...)     # L212
```

**C. 稀疏扩展钩子**：`get_indexer_metadata(layer_id, forward_batch)`（L241），默认返回 `None`（不支持 indexer）；DSA/V4 覆盖它以返回 topk indexer 元数据。

### 2.3 资源适配器：`KVCache` 抽象 IO

[memory_pool.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/mem_cache/memory_pool.py) 的 `KVCache(ABC)`（L768）定义了四个抽象 IO —— 这是"资源适配器"的落地接口，把算子对 KV 的读写与具体存储布局解耦：

```python
# memory_pool.py L830-842（抽象方法）
def get_key_buffer(self, layer_id) -> torch.Tensor: ...
def get_value_buffer(self, layer_id) -> torch.Tensor: ...
def get_kv_buffer(self, layer_id) -> Tuple[torch.Tensor, torch.Tensor]: ...
def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...): ...
```

各实现的**存储布局差异，正是为了匹配不同算子的 ABI 期望**：

- `MHATokenToKVPool`（L864）：逐层独立 `k_buffer` / `v_buffer`，默认 NHD 布局 `(size+page_size, head_num, head_dim)`（L1054-1073）。
- `MLATokenToKVPool`（L1806）：**单一融合 buffer** `(size+page_size, 1, kv_lora_rank+qk_rope_head_dim)`（L1868-1875），其中 `kv_cache_dim = 512+64 = 576`（L1843-1847）—— 直接对应 FlashMLA/FlashInfer-MLA 的期望。
- `DSATokenToKVPool(MLATokenToKVPool)`（L2188）：在 MLA 之上**新增** `index_k_with_scale_buffer`（L2251）供 indexer 用。
- `MambaPool`（L216）：非 KVCache 子类，存 `conv` + `temporal` 状态。

### 2.4 默认后端选择

[server_args.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/server_args.py) `_get_default_attn_backend`（L2773）决定当前硬件/模型默认走哪个后端：

- **MHA 模型**：Hopper → `"fa3"`；SM100/Blackwell → `"trtllm_mha"`；HIP → `"aiter"`；否则 `"flashinfer"`（不可用则 `"triton"`）。
- **MLA 模型**：Hopper → `"fa3"`；SM100 → `"flashinfer"`；否则 `"triton"`。

注册表 [attention_registry.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/attention_registry.py) 的 `ATTENTION_BACKENDS` 字典（L21）+ `register_attention_backend` 装饰器（L24）把字符串 key（`fa3`/`flashinfer`/`flashmla`/`dsa`/`dsv4` 等）映射到工厂函数。`attn_backend_wrapper`（L246）在遇到 GDN/Mamba2/KDA 等混合模型时，用 `HybridLinearAttnBackend` 把 full-attn 后端和 mamba 后端包在一起。

**因此本报告后续以 Hopper 默认路径（fa3 / flashinfer 系）为准。**

---

## 3. 场景 1：常规 GQA/MHA（含 sliding window）

默认后端：[flashattention_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/flashattention_backend.py)（`fa3`），核心算子第三方 `flash_attn_with_kvcache`（FA3）。

### 3.1 算子调用与 paged 入参

```python
# flashattention_backend.py L1004-1024（forward_extend 主路径）
result = flash_attn_with_kvcache(
    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
    k_cache=key_cache,          # 框架预分配的分页 KV buffer
    v_cache=value_cache,
    page_table=page_table,      # ← 关键：分页块表（block_table // page_size）
    cache_seqlens=cache_seqlens,# ← 关键：每请求 KV 长度（int32）
    cu_seqlens_q=cu_seqlens_q,  # ← 关键：query 累计长度（varlen）
    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
    max_seqlen_q=max_seqlen_q,
    softmax_scale=layer.scaling,
    causal=... ,
    window_size=window_size,    # ← 关键：滑动窗口
    softcap=layer.logit_cap,
    ...
    ver=self.fa_impl_ver,       # fa3 / fa4
)
```

**对齐要点**：`page_table` 并非 FA 私有结构，而是框架把内部 `req_to_token_pool.req_to_token[req_pool_indices]` 这张"请求→token 全局槽位"表，按 `page_size` 折算成块索引后传入。也就是说 **FA 算子开发时就定义了"我接受一个 `page_table` + `cache_seqlens`"的分页协议，框架只负责把自己的页表填进去**。`cache_seqlens` 是 `metadata.cache_seqlens_int32`（每请求 KV 长度，int32）。

### 3.2 sliding window 的对齐

```python
# flashattention_backend.py L843-846
is_swa_layer = (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
```

FA 算子期望一个 `window_size=(left, right)` 元组；框架把 `RadixAttention` 层上配置的 `sliding_window_size` 直接翻译成 `(sliding_window_size, 0)`（右侧 0 表示因果 + 左窗）。对 SWA 层，框架还会额外维护一张 `swa_page_table`（滑窗专用 KV 池），在 decode 时换入。这说明**滑窗对齐 = 框架层的元数据翻译 + 算子层的 `window_size` 入参**，无需改算子。

### 3.3 FlashInfer 路径（CSR 三元组）

若默认走 `flashinfer`（如 SM100 或无 fa3 时），KV 布局改用 CSR 风格。框架在 [flashinfer_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py) 里这样翻译：

```python
# flashinfer_backend.py L1168-1187（call_begin_forward）
kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)   # CSR 行指针
...
create_flashinfer_kv_indices_triton[(bs,)](   # ← Triton "索引搬运" kernel
    self.req_to_token, req_pool_indices, paged_kernel_lens,
    kv_indptr, kv_start_idx, kv_indices, self.req_to_token.shape[1],
)
# L1217-1226：把 CSR 三元组喂给 FlashInfer wrapper 的 plan/begin_forward
wrapper.begin_forward(
    kv_indptr, kv_indices, self.kv_last_page_len[:bs],
    self.num_qo_heads, self.num_kv_heads, self.head_dim,
    1,                       # page_size
    data_type=..., q_data_type=..., ...
)
```

`create_flashinfer_kv_indices_triton` 定义在 [triton_ops/kv_indices.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/triton_ops/kv_indices.py)（由 [utils.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/utils.py) L18-20 re-export），是框架**自研的轻量 Triton kernel**，专职把 `req_to_token` 摊平成 FlashInfer 期望的 `kv_indices`。

### 3.4 小结

常规 attention 的对齐纯粹是"页表格式翻译"：算子库原生接受分页 KV（FA 用 `page_table`+`cache_seqlens`，FlashInfer 用 CSR `kv_indptr/kv_indices/kv_last_page_len`），框架用 Triton 索引 kernel 把内部页表翻译过去，滑窗经 `window_size` 入参。**算子库不为某个框架定制，框架也不改算子源码。**

---

## 4. 场景 2：DeepSeek MLA

默认后端：Hopper 上 `fa3`（MLA 变体）/ SM100 上 `flashinfer`；另有 `flashmla`（`flash_mla` 库）。核心算子均为第三方。

### 4.1 融合 latent 布局由算子决定

MLA 把 KV 压缩成一个低秩 latent（`kv_lora_rank=512`）+ 一段解耦 rope（`qk_rope_head_dim=64`）。框架据此把 KV 存成**一块 576 宽的连续 buffer**（见 §2.3，`MLATokenToKVPool` L1868-1875）。这个布局不是框架的自由选择，而是为了直接喂给 MLA 算子。

### 4.2 FlashInfer-MLA：调用时现场切片

```python
# flashinfer_mla_backend.py L636-642（forward_decode）
o = decode_wrapper.run(
    q_nope,                              # query 的 nope 部分
    q_rope,                              # query 的 rope 部分
    k_buffer[:, :, : layer.v_head_dim],  # ← latent(512)：从 576 buffer 切片
    k_buffer[:, :, layer.v_head_dim :],  # ← rope(64)：从 576 buffer 切片
    out=o,
)
```

`decode_wrapper` 是第三方 `BatchMLAPagedAttentionWrapper`。框架把单块 `k_buffer` 现场切成 latent + rope 两段传入 —— 算子的 `.plan()` 里用 `head_dim_ckv=512` / `head_dim_kpe=64` 声明这两段维度。**对齐 = 框架按算子约定存成 576 单缓冲 + 调用时切片。**

### 4.3 FlashMLA：单 buffer view 成分页

```python
# flashmla_backend.py L363-373（forward_decode，非 fp8 分支）
o, _ = flash_mla_with_kvcache(
    q=reshape_q,
    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),  # [blocks,64,1,576]
    block_table=self.forward_metadata.block_kv_indices[:bs],   # 分页块表
    cache_seqlens=forward_batch.seq_lens.to(torch.int32),
    head_dim_v=self.kv_lora_rank,                              # ← 512：value 维取自 latent
    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
    num_splits=self.forward_metadata.num_splits,
    softmax_scale=layer.scaling, causal=True,
)
```

`flash_mla_with_kvcache` 来自 `sgl_kernel.flash_mla`（第三方）。这里 `PAGE_SIZE=64`，框架把 576 宽 buffer `view` 成 `[-1, 64, 1, 576]` 的分页形态，`block_table` 由 `create_flashmla_kv_indices_triton`（框架 Triton kernel）从 `req_to_token` 生成，`head_dim_v=512` 告诉算子 value 维度取 latent 段。`tile_scheduler_metadata` / `num_splits` 来自 `get_mla_metadata(...)`（算子提供的调度元数据 API）。

### 4.4 小结

MLA 是"算子库预设压缩布局、框架按此分配单缓冲"的典型。框架的对齐工作有两点：(1) 把 KV 存成 `512+64=576` 的融合单 buffer；(2) 调用时按算子约定切片 / view / 传 `head_dim_v`。**算子库预先定义了压缩 KV 的 ABI，框架服从之。**

---

## 5. 场景 3：Qwen3.X GDN mamba（定制算子）

默认后端：[linear/gdn_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py) 的 `GDNAttnBackend`，由 `attn_backend_wrapper` 包进 `HybridLinearAttnBackend`。

### 5.1 默认实现确认：是框架定制 Triton，非第三方 pip 包

用户问"GDN 的 mamba 默认实现是不是 triton 写的" —— **是，且是 sglang 自带（vendored）的 Triton，不是安装的第三方 `fla` 包**。证据链：

```python
# gdn_backend.py L5-16（imports）
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn, causal_conv1d_update,
)
```

线性注意力核 `chunk_gated_delta_rule` 来自仓库内 [fla/chunk.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/fla/chunk.py)，文件头注释：

```python
# fla/chunk.py L1
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
```

即"改编自"flash-linear-attention，但代码已 vendored 进 sglang 自己的目录树，内部 kernel 均为 `@triton.jit`（如 `chunk_gated_delta_rule_fwd`）。conv 用 `causal_conv1d`（CUDA 版优先、Triton 版兜底，见 gdn_backend.py L34-39）。

> 备注：另有 `gdn_cutedsl.py`（Blackwell CuteDSL）、`gdn_flashinfer.py`（第三方 FlashInfer）作为可选 dispatch，但 Hopper 默认路径是上面的 Triton 实现。

### 5.2 两类路径：默认定制 Triton（可跳过）+ 可选第三方（有对齐工作）

需要区分两件事：

1. **默认路径（Hopper Triton）** 是框架自研 Triton kernel，按用户规则"框架内部自研 triton 算子可跳过匹配对齐问题"，**不存在与第三方算子库的 ABI 对齐问题**，它真正要对齐的只是框架预分配的 mamba 状态资源（见 §5.3）。
2. **可选路径 FlashInfer / CuteDSL** 是**真正的第三方 / 外部算子**（`flashinfer.gdn_decode`、`flashinfer.gdn_prefill` 等），一旦用户通过 `--linear-attn-backend flashinfer/cutedsl` 切换过去，框架就必须做与前面 attention 场景同类的 ABI 对齐工作。这部分是本次补充的重点（见 §5.4）。

因此 GDN 并非"完全没有外部算子对齐"，而是"默认走定制 Triton、但保留了可切换的第三方算子路径，且框架为这些路径实现了统一的适配层"。

### 5.3 资源对齐：MambaPool 的 conv/temporal 状态

```python
# memory_pool.py L282-310（MambaPool.__init__）
conv_state = [
    torch.zeros(size=(num_mamba_layers, size + 1) + conv_shape, dtype=conv_dtype, device=device)
    for conv_shape in conv_state_shape
]
temporal_state = torch.zeros(
    size=(num_mamba_layers, size + 1) + temporal_state_shape, dtype=ssm_dtype, device=device
)
```

`MambaPool.State` 就是 `{conv: List[Tensor], temporal: Tensor}`（L217-220）。GDN 后端在 forward 里这样读写：

```python
# gdn_backend.py L310-324（forward_decode）
layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
conv_states = layer_cache.conv[0]
ssm_states  = layer_cache.temporal
cache_indices = self.forward_metadata.mamba_cache_indices   # ← 关键：状态槽位索引
mixed_qkv = causal_conv1d_update(
    mixed_qkv, conv_states, layer.conv_weights, layer.bias, layer.activation,
    conv_state_indices=cache_indices,       # ← 把请求映射到状态槽
)
```

**对齐要点**：与 KV cache 用 `page_table` 定位不同，mamba 状态是**每请求一份的连续状态**（非分页），框架通过 `cache_indices`（=`mamba_cache_indices`）把 batch 内每个请求映射到 `conv_states` / `ssm_states` 的对应槽位。由于状态是连续内存、天然对齐，这里框架的活儿主要是"分配对形状 + 传对索引"，而 kernel 是自研 Triton，二者本就在同一套代码里协同设计。

### 5.4 补充：切换到第三方算子（FlashInfer / CuteDSL）时的 ABI 对齐

这里正面回答"GDN 也有 FlashInfer / CuteDSL 实现，框架应该也有与外部算子对齐的地方"。答案是**有，且 sglang 用一个"内层 ABI 契约 + 每 kernel 适配器"的结构统一收口**，与 §2 的 `AttentionBackend` 是同一套设计思想在 linear-attention 上的复刻。

**（1）内层统一契约 `LinearAttnKernelBase`**

不同于普通 attention 直接在 backend 里调算子，GDN 多了一层 kernel 抽象。[kernels/kernel_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/kernel_backend.py) 定义 `LinearAttnKernelBase(ABC)`（L6），规定所有 GDN kernel（Triton / FlashInfer / CuteDSL）都实现同一组方法和**同一份入参签名**：

```python
# kernel_backend.py L13-43（抽象方法，统一签名）
@abstractmethod
def decode(self, q, k, v, a, b, *, A_log, dt_bias,
           ssm_states, cache_indices, query_start_loc, **kwargs) -> torch.Tensor: ...
@abstractmethod
def extend(self, q, k, v, g, beta, *,
           ssm_states, cache_indices, query_start_loc, **kwargs) -> tuple: ...
def target_verify(self, ...) -> torch.Tensor: ...   # decode/prefill/MTP 三态
```

`GDNAttnBackend` 只面向这个统一签名调用（gdn_backend.py L357 `kernel_dispatcher.decode(...)` / L493 `extend(...)`），**具体是 Triton 还是第三方 FlashInfer 由 dispatcher 决定**。也就是说：外部算子的适配被收敛进各自的 `LinearAttnKernelBase` 子类里，backend 主体不感知差异。

**（2）dispatcher 按硬件 + 用户配置选算子**

后端由 `--linear-attn-backend / --linear-attn-decode-backend / --linear-attn-prefill-backend` 配置（[linear/utils.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/utils.py) L42-53，枚举 `triton/cutedsl/flashinfer/custom`）。`GDNKernelDispatcher`（gdn_backend.py L58）据此为 decode / extend / verify **分别**挑 kernel，还带硬件兜底：

```python
# gdn_backend.py L104-114（CuteDSL prefill 只在 SM100+，SM90 回落 Triton）
if cutedsl_kernel.supports_prefill:
    self.extend_kernel = cutedsl_kernel
else:
    rank0_log("CuTe DSL GDN prefill ... requires SM100+. Falling back to Triton for prefill.")
    self.extend_kernel = triton_kernel
```

这意味着一次推理里 decode 可能走第三方、prefill 回落 Triton —— 正因为有统一契约，混搭才可行。

**（3）FlashInfer 路径的具体对齐工作（核心）**

[kernels/gdn_flashinfer.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py) 里 `FlashInferGDNKernel` 包装的是**真正的第三方算子** `flashinfer.gdn_decode` / `flashinfer.gdn_prefill`（L43-50，要求 `flashinfer >= 0.6.7`）。框架为对齐它做了四类工作：

- **状态池布局对齐（资源侧 ABI）**：FlashInfer GDN 要求 SSM 状态是 **K-last 的 `[pool, HV, V, K]` 布局**（文件头 L3、类注释 L80）。这正是框架预分配 `MambaPool.temporal` 时约定的布局（Triton 侧 [gdn_triton.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py) L68 注释 `ssm_states: [num_slots, HV, V, K]` 与之一致），使得同一状态池能被 Triton 和 FlashInfer 两条路径共享 —— **框架按算子要求的布局分配资源**，与 MLA 的 576 单缓冲是同一逻辑。
- **状态寻址方式对齐（SM90 vs SM100 分叉）**：SM100 走"状态池内寻址"，直接把整池 `initial_state=ssm_states` + `initial_state_indices=cache_indices` 交给算子（L179-192）；SM90 算子尚不支持池内索引，框架就**手动 gather/scatter**：`state_batch = ssm_states[cache_indices]` → 调用 → `ssm_states[cache_indices] = new_state`（L196-210）。这是典型的"算子能力不同、框架补齐差异"的适配。
- **dtype / 归一化 / 元数据对齐**：`A_log` 转 `float()`、`g` 取 `exp` 成 float32 alpha、`cu_seqlens` 转 int32/int64（因 kernel 要求不同）、Q/K 的 `l2norm_fwd` 在 kernel 外做（L235-241、L262、L284）；padding 槽位 `-1` 被 `clamp(min=0)` / remap 到 sentinel slot 防止算子越界读（L247、L268-272）。
- **接口形状对齐**：把框架的 `[1, seq, H, D]` 张量 `view` 成算子要的 `[batch, 1, H, D]`（decode，L173-177）或 `[batch, draft_token_num, H, D]`（MTP verify，L339-341），输出再 `view` 回框架约定形状（L212、L296）。对 SM100 的 bf16 MTP kernel，还包了一个 `_mtp_bf16_adapted` 适配器（L116-146）把它伪装成 fp32 kernel 接口，让 backend 无需分支。

**（4）CuteDSL 路径**

[kernels/gdn_cutedsl.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py) 的 `CuteDSLGDNKernel` decode 调 `sglang.jit_kernel.cutedsl_gdn` 的算子（L15/L94），prefill 调 ported 的 Blackwell chunkwise 算子 `chunk_gated_delta_rule_cutedsl`（L69-74，SM100+ 且 `head_k_dim==128`）。对齐工作与 FlashInfer 同型：外部 kernel 前做 `l2norm` + gather（`ssm_states[ssm_cache_indices]`，L141-146）、算完 `index_copy_` 写回状态池（L164-168）、生成 chunk 元数据 `prepare_metadata_cutedsl`（L148-150），并保持与 Triton 相同的 `(output, None, None)` 返回三元组。

**小结（本节）**：GDN 的外部算子对齐 = **内层 `LinearAttnKernelBase` 统一契约** + **dispatcher 按硬件/配置选算子并回落** + **各 kernel 子类里做状态池布局对齐、SM 分叉的 gather/scatter、dtype/形状/元数据 marshalling**。与前面 attention 场景相比，差异在于 mamba 状态是连续状态池（对齐点是"池布局 + `cache_indices` 寻址"），而非分页 KV（对齐点是"页表翻译"）。

### 5.5 小结

GDN mamba **默认**实现是 sglang vendored 的 Triton kernel（改编自 flash-linear-attention），非第三方库，此路径**无第三方 ABI 对齐问题**；资源对齐点是框架用 `MambaPool` 分配 `conv`+`temporal` 状态（`(num_layers, size+1)+shape`），forward 时用 `cache_indices` 把请求映射到状态槽。但 GDN **保留了可切换的第三方算子路径**（FlashInfer `gdn_decode/gdn_prefill`、CuteDSL）：框架用内层 `LinearAttnKernelBase` 统一契约 + `GDNKernelDispatcher` 收口，并在各 kernel 子类里做**状态池 K-last 布局对齐、SM90/SM100 的 gather-scatter 分叉、dtype/形状/元数据 marshalling** —— 这与普通 attention 对齐第三方算子是同一套设计，只是对齐对象从"分页 KV 页表"变成"连续状态池 + `cache_indices`"。（可与仓库既有的 [Qwen3_5_mamba_research.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/Qwen3_5_mamba_research.md) 交叉印证。）

---

## 6. 场景 4：DeepSeek V3.2 DSA（DeepSeek Sparse Attention）+ indexer

默认后端：[dsa_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/dsa_backend.py)（`nsa_backend.py` 只是弃用转发壳）。核心稀疏注意力算子第三方，辅助逻辑含定制 Triton。

### 6.1 相对 MLA 新增的资源

DSA 在 MLA 基础上引入"先用 indexer 选出 topk 个最相关 KV，再只对这批 KV 做注意力"的稀疏机制。为此**新增两类资源**：

1. **index-K cache**（`DSATokenToKVPool.index_k_with_scale_buffer`）：

```python
# memory_pool.py L2251-2268
self.index_k_with_scale_buffer = [
    torch.zeros(
        # shape: (num_pages, page_size*head_dim + page_size*fp32_nbytes)
        ( (index_buf_size + page_size + 1) // self.page_size,
          self.page_size * (index_head_dim + index_head_dim // self.quant_block_size * 4) ),
        dtype=torch.uint8, device=device,
    )
    for _ in range(...)
]
```

这是专供 indexer 计算相关性打分用的、量化过的 index-K 缓存（fp8 数据 + fp32 scale 打包在 uint8 buffer 里）。

2. **每 query 的 topk 索引**：`DSAMetadata`（dsa_backend.py L159-176）新增 `topk_indices_offset` / `indexer_k_start_end` / `indexer_seq_lens` / `dsa_cache_seqlens_int32`（裁剪到 topk 后的序列长度）。

### 6.2 数据流与算子分工

- **indexer logits（打分）**：第三方 `deep_gemm.fp8_paged_mqa_logits` / `deep_gemm.fp8_mqa_logits`（dsa_indexer.py）。
- **topk 选择**：第三方 `sgl_kernel.fast_topk_v2` 或 `flashinfer.top_k*`（dsa_topk_backend.py）。
- **index-K 落盘**：框架**定制 Triton** `fused_store_index_k_cache`（来自 `sglang.jit_kernel.fused_store_index_cache`）。
- **稀疏核心注意力**：第三方 `sgl_kernel.flash_mla` 的 `flash_mla_sparse_fwd`：

```python
# dsa_backend.py L1816-1824（_forward_*_sparse）
indices_input = page_table_1.unsqueeze(1)   # ← topk 页表：shape (s_q, h_kv=1, topk)
o, _, _ = flash_mla_sparse_fwd(
    q=q_input,
    kv=kv_cache,
    indices=indices_input,   # ← 关键：稀疏页表经 indices= 传入算子
    sm_scale=sm_scale,
    d_v=v_head_dim,
)
```

### 6.3 对齐要点

DSA 的 ABI 对齐在常规 MLA 之上多了一层"稀疏索引"：**算子 `flash_mla_sparse_fwd` 定义了 `indices=(s_q, 1, topk)` 这个稀疏页表入参**，框架负责先跑 indexer（deep_gemm）+ topk（sgl_kernel/flashinfer）算出这批索引，再把它整形成 `(s_q, 1, topk)` 喂进去。新增的 index-K cache 由框架分配、由定制 Triton kernel 落盘。核心计算算子仍是第三方，`get_indexer_metadata`（§2.2 的钩子）在这里被覆盖以返回 indexer 元数据。

### 6.4 小结

DSA = MLA + 稀疏索引层。新增资源（index-K cache、topk 索引）由框架分配/生成；核心稀疏注意力用第三方 `flash_mla_sparse_fwd`（`indices=` 传 topk 页表）；打分用第三方 `deep_gemm`，topk 用第三方；仅 index-K 落盘/量化用定制 Triton。**第三方算子定义稀疏 ABI（`indices` 入参），框架负责喂出那批索引。**

---

## 7. 场景 5：DeepSeek V4 C4A/C128A

默认后端：[deepseek_v4_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py) 的 `DeepseekV4AttnBackend`。核心算子第三方 `sgl_kernel.flash_mla`，压缩/元数据用定制 Triton。

### 7.1 C4A/C128A 的含义

代码中 `compress_ratio ∈ {0, 4, 128}`，C4/C128 指**压缩比 4 / 128 的两路并行压缩流**（两个 compressor stream）。`DSV4Metadata`（L331-336）同时持有 `c4_compress_metadata` 和 `c128_compress_metadata`；`create_paged_compressor_data(compress_ratio=4/128)` 分别为两路构建分页压缩数据。

### 7.2 复用 DSA + 第三方核心算子

```python
# deepseek_v4_backend.py L1289-1298（forward_decode 分支）
import sgl_kernel.flash_mla as flash_mla
o = flash_mla.flash_mla_with_kvcache(
    q=q, k_cache=swa_k_cache, head_dim_v=self.head_dim_v,
    block_table=None, cache_seqlens=None,
    tile_scheduler_metadata=flashmla_metadata,
    softmax_scale=self.softmax_scale, is_fp8_kvcache=True,
    indices=swa_page_indices,          # ← 复用 DSA 的稀疏 indices 机制
    topk_length=swa_topk_lengths,
    ...
)[0]
```

prefill 稀疏路径同样调 `flash_mla_sparse_fwd`（L1403 附近）。V4 通过 `C4IndexerBackendMixin`（L41）**复用 DSA 的 indexer / 稀疏注意力概念**，核心 attention 仍是第三方 `sgl_kernel.flash_mla`。

### 7.3 新增资源与定制 Triton

- **新增资源池**：`DeepSeekV4TokenToKVPool`（L61 引入）+ `CompressStatePool` —— 承载压缩后的 KV 与压缩状态（`c4_out_loc` / `c128_out_loc` 写位置、`c4_sparse_page_indices` / `c128_page_indices` 等）。
- **定制 Triton**：压缩用 `sglang.jit_kernel.dsv4`（`triton_create_paged_compress_data`）、量化用 `dsa.triton_kernel.act_quant`、元数据初始化用 [dsv4/metadata_kernel.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py) 的 `init_compression_metadata`（L48-49 引入）。

### 7.4 小结

V4 是 DSA 的进一步演进：**复用** DSA 的 indexer + 第三方 `sgl_kernel.flash_mla`（稀疏 `indices=` ABI），**新增** 双压缩比（C4/C128）资源池 `DeepSeekV4TokenToKVPool` + `CompressStatePool`，以及压缩/元数据的定制 Triton kernel。再次印证"核心算子复用第三方、框架不断加专属资源池 + 定制 Triton 搬运/元数据 kernel"的演进规律。

---

## 8. vLLM 对照：同构的对齐范式

vLLM v1 引擎（[vllm/v1/attention/backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/vllm/vllm/v1/attention/backend.py)）采用与 sglang **完全同构**的三段式契约，但把"KV 布局由谁决定"这件事做得更显式：**每个后端自己声明 KV cache 形状**。

### 8.1 每后端声明 paged 布局（关键机制）

```python
# vllm/v1/attention/backends/flash_attn.py L145-149
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    if block_size % 16 != 0:
        raise ValueError("Block size must be a multiple of 16.")
    return (2, num_blocks, block_size, num_kv_heads, head_size)   # FA 布局
```

- FA：`(2, num_blocks, block_size, num_kv_heads, head_size)`；
- FlashInfer：`(num_blocks, 2, block_size, num_kv_heads, head_size)`（维度顺序不同 → 故有 `get_kv_cache_block_dim` 探测）；
- MLA：`(num_blocks, block_size, head_size)`（单 latent，`num_kv_heads=1`）。

框架 `gpu_model_runner.py` 分配 KV 时**先调 `attn_backend.get_kv_cache_shape(...)` 再按 `get_kv_cache_stride_order` 排布** —— 这就是"算子库决定形状、框架负责摆放"的最直白表达。

### 8.2 build 翻译元数据 + forward 调算子

- `AttentionMetadataBuilder.build(...)`（backend.py L582）：从 `common_attn_metadata.block_table_tensor` 等 marshalling 出 `block_table` / `cu_seqlens` / `seqused_k` / `scheduler_metadata`。
- `AttentionImpl.forward(...)`（backend.py L790）→ 直调第三方 op：

```python
# vllm/v1/attention/backends/flash_attn.py L796-818
flash_attn_varlen_func(
    q=query[:num_actual_tokens], k=key_cache, v=value_cache,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,                # 分页序列长度
    window_size=sliding_window_size,    # 滑窗元组
    block_table=block_table,            # 分页块表
    scheduler_metadata=scheduler_metadata,
    fa_version=self.vllm_flash_attn_version, ...
)
```

### 8.3 MLA / GDN / DSA 覆盖情况

- MLA：`vllm/v1/attention/backends/mla/` 下 `flashmla.py`（`flash_mla_with_kvcache` + `get_mla_metadata`）、`flashinfer_mla.py`、`triton_mla.py`、`cutlass_mla.py`。
- GDN：`gdn_attn.py` 用 FLA 库的 chunked gated-delta-rule Triton kernel。
- DSA：`mla/indexer.py`（indexer，Triton + `deep_gemm` topk logits）、`mla/flashmla_sparse.py`（`flash_mla_sparse_fwd`）。

### 8.4 小结

vLLM 与 sglang 同构：**框架按后端声明的 `get_kv_cache_shape` 分配 KV → `build()` 翻译元数据 → `forward()` 直调第三方 op**。`get_kv_cache_shape` + `get_kv_cache_stride_order` 的间接层，是让"每个算子库强加自己分页内存布局"的关键机制。这佐证了"框架适配算子库 ABI、算子库不为框架定制"是跨框架的通用范式。

---

## 9. 横向总结表

| 维度 | 场景1 GQA/MHA | 场景2 MLA | 场景3 GDN mamba | 场景4 DSA | 场景5 V4 C4A/C128A |
|------|--------------|-----------|----------------|-----------|-------------------|
| **默认后端** | `fa3`（`flashattention_backend`） | `fa3`/`flashinfer`/`flashmla` | `GDNAttnBackend`(+Hybrid) | `dsa_backend` | `DeepseekV4AttnBackend` |
| **核心计算算子** | `flash_attn_with_kvcache` | `BatchMLAPagedAttentionWrapper.run` / `flash_mla_with_kvcache` | `chunk_gated_delta_rule`+`causal_conv1d` | `flash_mla_sparse_fwd` | `flash_mla.flash_mla_with_kvcache` / `flash_mla_sparse_fwd` |
| **第三方 / 定制** | 第三方 | 第三方 | **框架定制 Triton**（vendored） | 第三方核心 + 定制辅助 | 第三方核心 + 定制压缩 |
| **KV/State 资源形态** | `MHATokenToKVPool` NHD `(N,head,dim)` | `MLATokenToKVPool` 单 buffer 576=512+64 | `MambaPool` conv+temporal `(L,N+1,..)` | `DSATokenToKVPool`（MLA+index-K uint8） | `DeepSeekV4TokenToKVPool`+`CompressStatePool` |
| **框架 marshalling 的元数据** | `page_table`+`cache_seqlens`+`cu_seqlens_q`（Triton 索引 kernel） | latent/rope 切片 + `block_table`+`head_dim_v` | `cache_indices`（请求→状态槽） | topk `indices`(s_q,1,topk) + index-K | 双压缩 `indices`+`topk_length`+压缩元数据 |
| **滑窗/稀疏特殊处理** | `window_size=(sw,0)` 元组 + swa_page_table | 无（全量 latent） | 无（连续状态） | indexer(deep_gemm)+topk 选择稀疏 KV | C4/C128 双压缩 + 复用 DSA 稀疏 |
| **对齐主导方** | 算子库 | 算子库 | 无（同一代码库） | 算子库（稀疏 ABI） | 算子库（稀疏 ABI） |

---

## 10. 附录：引用的文件与行号

**统一契约与资源池**
- [base_attn_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/base_attn_backend.py)：`AttentionBackend`(L18)、`init_forward_metadata`(L41-83)、`forward`(L155)、`forward_decode`(L199)、`forward_extend`(L212)、`get_indexer_metadata`(L241)
- [memory_pool.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/mem_cache/memory_pool.py)：`MambaPool`(L216)、`State`(L217-220)、`MambaPool` 分配(L282-310)、`KVCache`(L768)、抽象 IO(L830-842)、`MHATokenToKVPool`(L864)、NHD buffer(L1054-1073)、`HybridLinearKVPool`(L1578)、`MLATokenToKVPool`(L1806)、`kv_cache_dim=576`(L1843-1847)、单 buffer 分配(L1868-1875)、`DSATokenToKVPool`(L2188)、`index_k_with_scale_buffer`(L2251-2268)
- [attention_registry.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/attention_registry.py)：`ATTENTION_BACKENDS`(L21)、`register_attention_backend`(L24)、`attn_backend_wrapper`(L246)
- [server_args.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/server_args.py)：`_get_default_attn_backend`(L2773)

**场景 1 / 2**
- [flashattention_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/flashattention_backend.py)：`window_size`(L843-846)、`flash_attn_with_kvcache`(L1004-1024)
- [flashinfer_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py)：CSR/`create_flashinfer_kv_indices_triton`(L1168-1187)、`begin_forward`(L1217-1226)
- [flashinfer_mla_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py)：`decode_wrapper.run` 切片(L636-642)
- [flashmla_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/flashmla_backend.py)：`flash_mla_with_kvcache`(L363-373)
- [triton_ops/kv_indices.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/triton_ops/kv_indices.py) via [utils.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/utils.py)(L18-26)

**场景 3 / 4 / 5**
- [linear/gdn_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)：imports(L5-16)、CUDA conv 覆盖(L34-39)、`GDNKernelDispatcher`(L58)、CuteDSL prefill 回落(L104-114)、verify 选择(L134-139)、状态读写(L310-324)、dispatcher 调用(L357/L493)
- [linear/kernels/kernel_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/kernel_backend.py)：`LinearAttnKernelBase`(L6)、统一签名(L13-43)
- [linear/kernels/gdn_flashinfer.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py)：K-last 布局(L3/L80)、第三方 import(L43-50)、bf16 MTP 适配器(L116-146)、decode view(L173-177)、SM100 池内寻址(L179-192)、SM90 gather/scatter(L196-210)、extend l2norm/clamp(L235-293)、MTP view(L339-341)
- [linear/kernels/gdn_cutedsl.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py)：decode 算子(L15/L94)、Blackwell prefill import(L69-74)、gather+index_copy_(L141-168)
- [linear/kernels/gdn_triton.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py)：`ssm_states: [num_slots,HV,V,K]`(L68)
- [linear/utils.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/utils.py)：`LinearAttnKernelBackend` 枚举 + 配置(L15-53)
- [fla/chunk.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/fla/chunk.py)：vendored 头注释(L1)、`chunk_gated_delta_rule`(L133)
- [dsa_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/dsa_backend.py)：`DSAMetadata`(L159-176)、`flash_mla_sparse_fwd`(L1791-1824)
- [deepseek_v4_backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py)：`DSV4Metadata`(L331-336)、`DeepseekV4AttnBackend`(L419)、`flash_mla_with_kvcache`(L1289-1306)

**vLLM 对照**
- [vllm/v1/attention/backend.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/vllm/vllm/v1/attention/backend.py)：`get_kv_cache_shape`(L88)、`build`(L582)、`forward`(L790)
- [vllm/v1/attention/backends/flash_attn.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/vllm/vllm/v1/attention/backends/flash_attn.py)：`get_kv_cache_shape`(L145-149)、`flash_attn_varlen_func`(L796-818)
