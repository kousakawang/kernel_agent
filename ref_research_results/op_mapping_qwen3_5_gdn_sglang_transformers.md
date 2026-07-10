# Qwen3.5 Linear-Attention (GDN) 层 — sglang ↔ transformers 算子接口对应表

> 目标：把 **Qwen3.5 含 linear-attention (GDN) 的那一层 transformer-layer** 在 **sglang（推理框架）** 与 **transformers（参考实现）** 里调用的算子逐一对应，给 kernel-agent 提供"每个 sglang 热点算子在算什么"的 naive pytorch 参考，用于构造 UT。
>
> - sglang 侧：[qwen3_5.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py) `Qwen3_5LinearDecoderLayer` → `Qwen3_5GatedDeltaNet` → `RadixLinearAttention` → `GDNAttnBackend`
> - transformers 侧：[modeling_qwen3_5.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py) `Qwen3_5DecoderLayer` → `Qwen3_5GatedDeltaNet` mixer

---

## 0. 范围与约定

- **对象**：仅含 GDN 的 `linear_attention` 层（dense 变体 Qwen3.5-9B 为代表；MoE 变体仅在 MLP 处备注）。full-attention 层不展开。
- **映射记号**：`sglang 接口  ⇄  transformers 接口`。多对一 `{A,B}⇄C`、一对多 `A⇄{C,D}`、无对应 `—（框架特有）`。
- **精度**：不要求对齐，统一 bf16 代表；只保证"计算语义一致"。
- **参考实现**：每个数学算子给 **naive pytorch 伪代码 + 一句话计算描述**。core 算子直接复用 transformers 仓库已有的纯 torch 蓝本（`torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` / `torch_causal_conv1d_update` / `Qwen3_5RMSNormGated`），已标 file:/// 链接，kernel-agent 可直接拿来产参考输出。
- **SM90 后端**：主线 = sglang `--linear-attn-backend triton`；core 处并列给出 `flashinfer`、`cutedsl(仅 decode)` 的对应与差异（见第 4 节匹配度结论）。

### GDN 计算全景（一句话）
GDN 是 **Gated DeltaNet 线性注意力**：先对 hidden 做投影得到 q/k/v/z/b/a → 对 q/k/v 做**深度可分离因果卷积** → 由 a/b 算出**衰减门 g 与写入强度 beta** → 用 **delta-rule 递推**维护一个 `[num_heads, head_k_dim, head_v_dim]` 的状态矩阵 S（`S ← diag(exp(g))·S + k·(βv − βk·S)`，输出 `o = q·S`）→ **gated RMSNorm**（用 z 做门）→ 输出投影。无 RoPE、无 KV cache，取而代之的是 **conv_state + recurrent(ssm) state**。

---

## 1. 整层数据流对照总表（执行顺序）

| # | 计算步骤 | sglang 接口（实现类别） | transformers 接口（实现类别） | 映射 | 备注 |
|---|---|---|---|---|---|
| 0 | input_layernorm | `layer_communicator.prepare_attn`→`GemmaRMSNorm`（raw-cuda/融合） | `Qwen3_5RMSNorm`（pytorch） | 1⇄1 | 都是 RMSNorm；gemma 权重用 `(1+w)` |
| 1 | 输入投影 | `{in_proj_qkvz, in_proj_ba}`（2 融合 ColumnParallel, cuBLAS） | `{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a}`（4 独立 nn.Linear） | 多⇄多 | 同一组 QKVZBA 投影，仅打包/TP 切分不同 |
| 2 | qkvzba 拆分+reshape | `fused_qkvzba_split_reshape_cat_contiguous`（triton）→`mixed_qkv,z,b,a` | `fix_query_key_value_ordering`=`torch.split`+`reshape`（pytorch） | 1⇄多 | 纯 layout 重排，无算术 |
| 3 | causal conv1d | `causal_conv1d_update`(decode)/`causal_conv1d_fn`(prefill)（sgl-kernel/triton） | 同名 `causal_conv1d_*`（third-party）或回退 `torch_causal_conv1d_update`/`F.conv1d`+silu | 1⇄1 | depthwise 因果卷积+SiLU |
| 4 | qkv split（conv 后） | `fused_qkv_split_gdn_prefill`(triton, prefill) / kernel 内（decode） | `torch.split`+`reshape`+`repeat_interleave`（pytorch） | 1⇄多 | GQA：k/v 头广播到 v_heads |
| 5 | gating (g, beta) | `fused_gdn_gating`(fla, prefill) / 融进 core kernel(decode) | `beta=sigmoid(b)`；`g=-exp(A_log)·softplus(a+dt_bias)`（pytorch） | 1⇄多 | log-decay 门 + 写入强度 |
| 6 | **core linear-attn** | prefill `chunk_gated_delta_rule`(fla/triton)；decode `fused_recurrent_gated_delta_rule_packed_decode` 或 `fused_sigmoid_gating_delta_rule_update`(fla/triton) | prefill `chunk_gated_delta_rule`(fla)；decode `fused_recurrent_gated_delta_rule`(fla)；回退 `torch_chunk_*`/`torch_recurrent_*` | 1⇄1（同源） | **热点**；见 §2.6 |
| 7 | gated RMSNorm | `RMSNormGated`（fla layernorm_gated） | `FusedRMSNormGated`(fla) / `Qwen3_5RMSNormGated`(pytorch) | 1⇄1 | `RMSNorm(x)·SiLU(z)` |
| 8 | out_proj | `RowParallelLinear`（cuBLAS + all-reduce） | `nn.Linear`（cuBLAS） | 1⇄1 | all-reduce 框架特有 |
| 9 | post_attn_layernorm | `layer_communicator.prepare_mlp`→`GemmaRMSNorm` | `Qwen3_5RMSNorm` | 1⇄1 | 同 #0 |
| 10 | MLP | `Qwen2MoeMLP`(dense) / `Qwen2MoeSparseMoeBlock`(MoE) | `Qwen3_5MLP`(dense) | 1⇄1 | SwiGLU：`down(silu(gate)·up)` |
| — | 资源管理/调度 | `RadixLinearAttention` custom-op、state pool、metadata、通信、cudagraph | 无 | —（框架特有） | 见 §3 |

---

## 2. 逐算子对应详解

### 2.1 input_layernorm / post_attention_layernorm
- **计算**：RMSNorm，`y = x / sqrt(mean(x²)+eps) · w`。
- sglang：`GemmaRMSNorm`（[qwen3_5.py L614-617](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L614-L617)），由 `LayerCommunicator` 在 `prepare_attn`/`prepare_mlp` 里调用，常与 residual add 融合。
- transformers：`Qwen3_5RMSNorm`（层里显式 `input_layernorm`/`post_attention_layernorm`）。
- **差异**：Gemma 变体用 `(1.0 + weight)` 缩放（权重零初始化）；Qwen3_5RMSNorm 用 `weight` 直乘。计算同类。
```python
def rmsnorm(x, w, eps=1e-6, gemma=False):          # x:[T,H]
    v = x.float().pow(2).mean(-1, keepdim=True)
    y = x.float() * torch.rsqrt(v + eps)
    scale = (1.0 + w.float()) if gemma else w.float()
    return (y * scale).to(x.dtype)
```

### 2.2 输入投影（QKVZBA）
- **计算**：6 个线性投影 `q=Wq·x, k=Wk·x, v=Wv·x, z=Wz·x, b=Wb·x, a=Wa·x`。
- sglang：**2 个融合**权重——`in_proj_qkvz`(MergedColumnParallelLinear, 输出 `[k_dim,k_dim,v_dim,v_dim]`) + `in_proj_ba`(输出 `[num_v_heads,num_v_heads]`)（[L182-199](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L182-L199)）。
- transformers：**4 个独立** `nn.Linear`——`in_proj_qkv`(合 Q+K+V)、`in_proj_z`、`in_proj_b`、`in_proj_a`（[L433-436](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L433-L436)）。
- **映射**：多⇄多，本质是同一组投影的不同打包（sglang 把 z 并进 qkv 融合权重、b/a 融合）。TP 切分是框架特有。UT 里视作独立 GEMM 即可。
```python
# 数学上等价：先各自 matmul，再按 GDN 需要拆分/拼接
q = x @ Wq.T; k = x @ Wk.T; v = x @ Wv.T
z = x @ Wz.T; b = x @ Wb.T; a = x @ Wa.T
```

### 2.3 qkvzba 拆分 + reshape（投影后）
- **计算**：把融合投影输出切成 q/k/v/z/b/a 并 reshape 成 `[T, heads, head_dim]`，纯访存重排。
- sglang：`fused_qkvzba_split_reshape_cat_contiguous`（triton, [import L25-27](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L25-L27)；调用 [L495-502](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L495-L502)）——一个 triton kernel 完成 split+reshape+cat 成连续 `mixed_qkv`。
- transformers：mixer forward 内的裸 `torch.split` + `reshape`（[modeling_qwen3_5.py L503-524](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L503-L524)）；sglang 无 fused kernel 时的等价回退是 `fix_query_key_value_ordering`（[qwen3_5.py L425-445](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L425-L445)）。
- **映射**：1⇄多（一个 fused triton kernel ⇄ 多个 torch view op）。无算术，UT 只需校验 layout。
```python
q, k, v, z = mixed_qkvz.split([k_dim, k_dim, v_dim, v_dim], dim=-1)
b, a = mixed_ba.split([n_v, n_v], dim=-1)
q = q.view(T, n_k, head_k); k = k.view(T, n_k, head_k); v = v.view(T, n_v, head_v)
mixed_qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], -1)  # 供 conv 消费
```

### 2.4 causal conv1d（深度可分离因果卷积 + SiLU）
- **计算**：对 `mixed_qkv`（通道=conv_dim=2·k_dim+v_dim）沿时间做 **depthwise 因果卷积**（每通道独立、看左侧 `kernel_size-1` 个历史），接 SiLU。
- sglang：decode `causal_conv1d_update`（读写 `conv_states` pool，[gdn_backend.py :317](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)）；prefill `causal_conv1d_fn`（全序列，带 `query_start_loc`/`cache_indices`）。实现类别 sgl-kernel(CUDA)/triton。
- transformers：同名 `causal_conv1d_update`/`causal_conv1d_fn`（third-party causal-conv1d），回退纯 torch `torch_causal_conv1d_update`（[L223-238](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L223-L238)）/ `F.conv1d`+silu。
- **映射**：1⇄1，接口名都叫 `causal_conv1d_*`。差异仅：sglang 的 conv_state 来自 pool（`cache_indices` 间接寻址），transformers 来自本地 `cache_params`。
- **参考蓝本**（transformers 纯 torch，decode 单步）：
```python
# hidden_states:[B, conv_dim, seq], conv_state:[B, conv_dim, state_len]
def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None):
    _, C, L = hidden_states.shape
    x = torch.cat([conv_state, hidden_states], dim=-1)      # 拼历史
    conv_state.copy_(x[:, :, -conv_state.shape[-1]:])       # 更新 state
    out = F.conv1d(x, weight.unsqueeze(1), bias, groups=C)  # depthwise
    return F.silu(out[:, :, -L:])
```
> prefill 版 = 全序列 `F.conv1d(pad左侧, groups=C)` 再 SiLU；UT 里 conv_state 用零张量或随机张量即可。

### 2.5 gating：g（log-decay）与 beta（写入强度）
- **计算**：`beta = sigmoid(b)`；`g = -exp(A_log) · softplus(a + dt_bias)`（每 value-head 一个标量衰减，A_log/dt_bias 是可学习参数）。
- sglang：prefill 用 `fused_gdn_gating`(fla, [gdn_backend :492])；decode 融进 core recurrent kernel（不单独出现）。
- transformers：裸 torch（[L517-519](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L517-L519)）。
- **映射**：1⇄多（sglang 一个 fused kernel ⇄ transformers 几个 torch op）。
```python
beta = torch.sigmoid(b)                                  # [T, n_v]
g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias)   # [T, n_v]
```

### 2.6 core linear-attn（delta-rule）—— 热点，分 prefill / decode

**计算语义（两阶段等价）**：维护状态 `S:[n_heads, head_k, head_v]`。对每个 t：`S = exp(g_t)·S + k_t ⊗ (β_t·v_t − β_t·(k_tᵀS))`，输出 `o_t = q_t·S`（q/k 先做 L2 归一化，q 乘 `1/sqrt(head_k)`）。prefill 用 **chunk 并行**算法（块内矩阵化 + 块间递推），decode 用**逐步 recurrent**；二者数学等价。

#### (a) prefill —— chunk 并行
- sglang：`chunk_gated_delta_rule`（fla `chunk.py`，triton），内部链：`l2norm_fwd(q/k)` → `chunk_local_cumsum(g)` → `chunk_gated_delta_rule_fwd_intra`（块内 KKT + `solve_tril` → w,u）→ `chunk_gated_delta_rule_fwd_h`（块间状态递推，读写 ssm_state）→ `chunk_fwd_o`（输出）。
- transformers：**同一个** `chunk_gated_delta_rule`（fla），回退纯 torch `torch_chunk_gated_delta_rule`。
- **映射**：1⇄1（**同源 fla**）。sglang 多传 `initial_state`/`initial_state_indices`/`cu_seqlens`（state pool + varlen 边界）；transformers 传单一 `initial_state` 张量或 None。
- **参考蓝本**：[torch_chunk_gated_delta_rule L247-325](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L247-L325)（核心节选）：
```python
# q,k,v:[B,T,H,D], g,beta:[B,T,H]; chunk_size=64
q, k = l2norm(q), l2norm(k)                       # use_qk_l2norm_in_kernel
q = q * (1/ q.shape[-1]**0.5)
# 分块，块内累积 gate、构造 (I - tril(β k kᵀ·decay))^{-1} 解出 w,u
g = g.cumsum(-1)
decay = ((g[...,None]-g[...,None,:]).tril().exp()).tril()
attn = -((k*beta)[...] @ k.transpose(-1,-2) * decay).masked_fill(triu0, 0)
for i in 1..chunk: attn[...,i,:i] += (attn[...,i,:i,None]*attn[...,:i,:i]).sum(-2)
attn += I
value = attn @ (v*beta); k_cumdecay = attn @ (k*beta*g.exp())
S = initial_state or 0                              # [B,H,Dk,Dv]
for i in chunks:                                    # 块间递推
    o_intra = (q_i*exp(g_i)) @ S
    v_new  = v_i - k_cumdecay_i @ S
    o[i]   = o_intra + (q_i@k_iᵀ*decay_i) @ v_new
    S = S*exp(g_last) + (k_i*exp(g_last-g_i)).ᵀ @ v_new
return o, S
```

#### (b) decode —— 逐步 recurrent
- sglang：【packed 快路】`fused_recurrent_gated_delta_rule_packed_decode`（fla）或【split 路】`fused_sigmoid_gating_delta_rule_update`（fla）——把 split+sigmoid-beta+kernel 内 l2norm+recurrent 全融合；读写 `ssm_states` via `cache_indices`。
- transformers：`fused_recurrent_gated_delta_rule`（fla），回退纯 torch `torch_recurrent_gated_delta_rule`。
- **映射**：1⇄1（同源 fla）。sglang 的 packed 版把 gating 也吞进 kernel（对应 transformers 的 §2.5 + 本步），属**多⇄一**。
- **参考蓝本**：[torch_recurrent_gated_delta_rule L328-369](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L328-L369)：
```python
q, k = l2norm(q), l2norm(k); q = q * (1/q.shape[-1]**0.5)
S = initial_state or 0                               # [B,H,Dk,Dv]
for t in range(T):
    S = S * exp(g_t)[...,None,None]                  # 衰减
    kv = (S * k_t[...,None]).sum(-2)                 # kᵀS
    delta = (v_t - kv) * beta_t[...,None]            # delta-rule 修正
    S = S + k_t[...,None] * delta[...,None,:]        # 写入
    o_t = (S * q_t[...,None]).sum(-2)                # 读出
return o, S
```
> decode 时 T=1（单 token），S 从 pool 里按 req 取初值、算完写回。

#### (c) 三后端对照（SM90）
| 后端 | prefill core | decode core | 与 transformers 关系 |
|---|---|---|---|
| **triton**（默认） | `chunk_gated_delta_rule`(fla) | `fused_recurrent_*`/`fused_sigmoid_gating_*`(fla) | **完全同源**，逐子算子可对应 |
| **flashinfer** | `flashinfer.gdn_prefill.chunk_gated_delta_rule`（融合黑盒） | `gated_delta_rule_decode_pretranspose`（融合黑盒） | 仅语义等价，无法逐子算子拆 |
| **cutedsl** | 回退 triton | `cutedsl_fused_sigmoid_gating_delta_rule_update`（CuTe DSL） | 仅 decode；语义等价 |

### 2.7 gated RMSNorm
- **计算**：`y = RMSNorm(x) · SiLU(z)`（norm-before-gate）。
- sglang：`RMSNormGated`（fla `layernorm_gated`，[import L42](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L42)），`norm_before_gate=True`。
- transformers：`FusedRMSNormGated`(fla) 或回退 `Qwen3_5RMSNormGated`（[L188-203](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L188-L203)）。
- **映射**：1⇄1。
```python
def rmsnorm_gated(x, z, w, eps=1e-6):                # x,z:[*, head_v]
    v = x.float().pow(2).mean(-1, keepdim=True)
    y = x.float() * torch.rsqrt(v + eps)
    y = w * y.to(x.dtype)
    return (y * F.silu(z.float())).to(x.dtype)
```

### 2.8 out_proj
- **计算**：`out = Wo · core_attn_out`（value_dim → hidden_size）。
- sglang：`RowParallelLinear`（[L267-277](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/models/qwen3_5.py#L267-L277)）—— 行并行 + all-reduce。
- transformers：`nn.Linear`（[L419](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L419)）。
- **映射**：1⇄1；TP all-reduce 框架特有。`out = x @ Wo.T`。

### 2.9 MLP（层内最后一步）
- **计算**：SwiGLU：`down(SiLU(gate(x)) · up(x))`。
- sglang：dense=`Qwen2MoeMLP`（gate_up_proj 融合 + down_proj）；MoE 变体=`Qwen2MoeSparseMoeBlock`（router topk + fused experts + shared expert）。
- transformers：dense=`Qwen3_5MLP`（同 SwiGLU）。
- **映射**：dense 1⇄1；MoE 是 sglang 侧额外结构（若用 dense checkpoint 则无此分支）。
```python
def swiglu_mlp(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
```

---

## 3. 框架特有逻辑（sglang-only，无 transformers 对应）

这些是 serving 侧的资源管理/调度，**不是数学算子**；kernel-agent 构造 UT 时用普通连续张量 + 假 metadata 替代即可。

| 项 | 说明 | UT 替代 |
|---|---|---|
| `RadixLinearAttention` + `unified_linear_attention_with_output` custom-op | torch 自定义算子包装（cudagraph/piecewise 用），[radix_linear_attention.py L120-153](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/sglang/python/sglang/srt/layers/radix_linear_attention.py#L120-L153) | 直接调 backend forward |
| `ForwardBatch` / `forward_mode`(decode/extend/idle) | 批次调度上下文，决定走 decode 还是 prefill 分支 | 手动指定阶段 |
| **mamba state pool**：`conv_states`(=layer_cache.conv[0])、`ssm_states`(=layer_cache.temporal) | 跨请求的持久状态池 | 用本地零/随机张量 |
| **state 索引 metadata**：`mamba_cache_indices`/`cache_indices`/`initial_state_indices`/`ssm_state_indices`/`conv_state_indices` | 从 pool gather/scatter 每个 req 的 state（gather/scatter 寻址） | 用 `torch.arange(B)` |
| **varlen metadata**：`query_start_loc`/`cu_seqlens` | packed 变长序列边界 | 单序列 `[0, L]` |
| `LayerCommunicator`/`LayerScatterModes` | TP/DP all-reduce/reduce-scatter/all-gather 通信 | 单卡忽略 |
| `out_cache_loc`、`num_token_non_padded` | 输出写回位置 + padding 裁剪 | 忽略 |
| packed_decode 快路判定、DP-attn padding、`alt_stream` 双流、cuda graph capture | 性能/显存优化路径 | 忽略 |

> transformers 侧对应物只有一个极简 `cache_params`（本地 conv/recurrent 张量），无 pool、无索引、无通信——这正是"照 transformers 优化的 kernel 不能直接接 sglang"的根源：**数学一致，但访存契约（从 pool 按 index gather/scatter）不同**。

---

## 4. 后端匹配度结论（哪条 sglang 路径与 transformers 最贴合）

| sglang 后端(SM90) | core 是否与 transformers 同源 | 逐算子可对应度 | 适合"照 transformers 优化再回接"？ |
|---|---|---|---|
| **triton**（默认） | ✅ 都调 `fla`（`chunk_gated_delta_rule`/`fused_recurrent_*`） | **高**：conv/l2norm/gating/cumsum/chunk-intra/chunk-h/chunk-o 逐个可映射 | ✅ **最佳** |
| flashinfer | ❌ core 是 flashinfer 融合黑盒 | 低：只能整体语义对齐 | ⚠️ 仅参考语义 |
| cutedsl | ❌ decode 用 CuTe DSL；prefill 回退 triton | 中：decode 黑盒、prefill 同 triton | ⚠️ prefill 部分可参考 |

**结论**：要让"基于 transformers 参考优化出的算子"最低成本回接 sglang，应选 **`--linear-attn-backend triton`**——此时 sglang 与 transformers 的 core delta-rule 走**同一个 `fla` 库**，逐算子一一对应，UT 参考输出可直接用 transformers 的 `torch_chunk/torch_recurrent` 蓝本校验。flashinfer/cutedsl 把多步融成大 kernel，只能对齐整体数学语义，无法逐子算子对应。

---

## 5. 给 kernel-agent 的 UT 构造提示

**纯数学输入（UT 里正常随机张量即可）**：
- 投影权重 `Wq/Wk/Wv/Wz/Wb/Wa/Wo`、conv `weight`(depthwise)、`A_log`/`dt_bias`；
- 中间张量 `q/k/v/z/b/a`、`g/beta`、初始 `S`(recurrent_state)、`conv_state`。

**需用假 metadata 替代的 pool/调度参数**：
- `initial_state_indices`/`cache_indices` → `torch.arange(batch)`；
- `cu_seqlens`/`query_start_loc` → 单序列 `torch.tensor([0, L])`；
- `conv_states`/`ssm_states` → 预分配零/随机张量，UT 内自持（不走 pool）。

**参考输出生成**（直接调 transformers 纯 torch 蓝本，无需框架）：
- conv：[torch_causal_conv1d_update](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L223-L238)
- prefill core：[torch_chunk_gated_delta_rule](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L247-L325)
- decode core：[torch_recurrent_gated_delta_rule](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L328-L369)
- gated norm：[Qwen3_5RMSNormGated](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L188-L203)

**对应关系**：sglang triton kernel 的输出，应与"对应 transformers 蓝本 + 相同 g/beta/初始 state"的输出在 bf16 容差内一致；packed_decode 因把 gating 融进 kernel，参考时需先用 §2.5 算出 g/beta 再喂 `torch_recurrent_gated_delta_rule`。
