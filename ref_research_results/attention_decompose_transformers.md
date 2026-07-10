# Transformers Attention Forward 调用路径分解 (SM90 / Hopper)

> 目标：对 HuggingFace **Transformers** 仓库中 5 类 attention 在 **SM90 (Hopper, cc 9.0)** 上**整个 attention-layer 的 forward**，逐 kernel 列出调用路径并标注实现类别。
> 仓库：`/Users/bytedance/Desktop/remote_dev_project/model_ana/transformers`（`src/transformers/`）

---

## 0. 说明与约定

### 0.1 仓库性质（与 vLLM / sglang 报告的本质差异 —— 必读）
Transformers 是**纯 PyTorch 参考实现（reference implementation）**，不是 serving 引擎：

- **仓库内无任何自研 CUDA / Triton / CUTLASS 源码**：`find src -name "*.cu"/"*.cuh"/"*.cpp"` 为空，`grep "import triton" src/transformers/models` 为空。
- 因此用户在 vLLM/sglang 报告里用的类别 **`raw-cuda(sgl-kernel)`、`cuteDSL`、`JIT-kernel(DSL)` 在本仓库不存在**。
- 也**没有 MLA 权重吸收（weight-absorption）、没有 FlashMLA、没有 paged KV pool、没有 FP8 kernel**（DSA/V4 的 FP8 indexer 在这里被写成 bf16 等价实现）。
- 加速仅来自两处：① core-attention 交给 **torch 后端或第三方库**（SDPA/FlashAttention/FLEX）；② 可选的 **kernels-from-hub** 覆盖层（把 RMSNorm/RoPE/MLP 等换成 Liger 等 Triton kernel）。

### 0.2 统一 dispatch 机制：`ALL_ATTENTION_FUNCTIONS`
每个模型的 `XxxAttention.forward` 末尾都执行 `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`，再统一调用（Qwen3 [modeling_qwen3.py L273-287](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3/modeling_qwen3.py#L273-L287)）。注册表 [modeling_utils.py L5119-5130](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/modeling_utils.py#L5119-L5130)：

| `_attn_implementation` | 入口函数 | 底层实现 | 实现类别 |
|---|---|---|---|
| `eager` | 各 modeling 文件内 `eager_attention_forward` | `torch.matmul + softmax + matmul` | **pytorch-native** |
| `sdpa` | [sdpa_attention.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/sdpa_attention.py) `sdpa_attention_forward` | `F.scaled_dot_product_attention`（torch 运行时选 FlashAttn/cuDNN/mem-eff backend） | **torch-SDPA-backend** |
| `flash_attention_3` | [flash_attention.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/flash_attention.py) → `_flash_attention_forward` | `flash_attn_interface.flash_attn_(varlen_)func`（FA3） | **third-party (FlashAttention 3)** |
| `flash_attention_2` | 同上 | `flash_attn.flash_attn_(varlen_)func`（FA2） | **third-party (FlashAttention 2)** |
| `flex_attention` | [flex_attention.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/flex_attention.py) `flex_attention_forward` | `torch.compile(flex_attention)` | **torch.compile→triton** |

FA2/FA3/FA4 的选择在 [modeling_flash_attention_utils.py L157-221](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/modeling_flash_attention_utils.py#L157-L221) 的 `lazy_import_flash_attention` 里按可用性决定；SM90 上装了 FA3 即走 FA3。

### 0.3 core-attn 的 SM90 后端枚举口径
本报告对每个类型的 core-attention 一步，**并列列出全部可选后端**：

```
路径A = sdpa           (torch-SDPA-backend；SM90 实际落到 FlashAttention 或 cuDNN kernel)
路径B = flash_attention_3 (third-party, FA3)
路径C = flex_attention   (torch.compile→triton)
路径D = eager            (pytorch-native；数值参考/调试用)
```

投影 / norm / rope 等**非 core-attn** 步骤在四条后端路径下完全相同，故只在每节列一次。

### 0.4 实现类别取值（本仓库版）
- `pytorch-native`：`nn.Linear`(→cuBLAS) / `torch.matmul` / `torch.bmm` / `softmax` / `topk` / `scatter` / `F.pad` / `torch.cat` 等原生算子。
- `torch-SDPA-backend`：`F.scaled_dot_product_attention`，具体 kernel 由 torch 运行时选择（SM90 常为 FlashAttention 或 cuDNN flash backend）。
- `third-party`：外部库——`flash_attn` / `flash_attn_interface`（FA2/3）、`fla`（flash-linear-attention）、`causal_conv1d`。
- `torch.compile→triton`：`flex_attention`，经 `torch.compile` 生成 Triton kernel。
- `hub-kernel (triton, 可选)`：仅当 `use_kernels=True` 时由 kernels-from-hub 覆盖的 kernel（多为 Liger Triton），见 0.7。

### 0.5 prefill vs decode
- **GQA / MLA / DSA / C4A / C128A**：prefill 与 decode 走**同一个 forward、同一份 kernel 列表**，唯一差别是 `past_key_values`（Cache）里是否已有历史 K/V（决定 `key_states` 长度）。SDPA 分支的 `is_causal` 布尔在 `q_len==1` 时不同（[sdpa_attention.py L77](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/sdpa_attention.py#L77)），但仍是同一次 SDPA 调用。故这几类**不再对 prefill/decode 拆双份 kernel 列表**，只在文中标注差异。
- **GDN（Qwen3.5）是唯一真正按阶段分叉的**：prefill 走 chunk 并行 kernel，decode(`seq_len==1`) 走 recurrent 单步 kernel。故 GDN 明确拆「路径1 Prefill / 路径2 Decode」。

### 0.6 精度约定
统一用 **BF16** 作代表。特别地：DeepSeek-V3.2（DSA）与 V4（C4A/C128A）的 indexer/scoring 在真实推理里用 FP8 + Hadamard，但 HF 参考实现**显式跳过 FP8、用 bf16/fp32 等价计算**（[modeling_deepseek_v32.py L210-213](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L210-L213)），故本报告不列 FP8 量化 kernel。

### 0.7 可选 "kernels-from-hub" 覆盖层
装饰器 `@use_kernel_forward_from_hub("RMSNorm")` / `@use_kernel_func_from_hub("rotary_pos_emb")` / `@use_kernelized_func(...)`（[hub_kernels.py L83-99](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/hub_kernels.py#L83-L99)）在 `use_kernels=True` 且 `USE_HUB_KERNELS!=NO` 时，把默认 eager 模块替换为 hub 上的 kernel。映射表 `_build_kernel_mapping`（[hub_kernels.py L107+](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/hub_kernels.py#L107)）关键项：

| 被覆盖模块 | hub kernel | 类别 | SM90 是否命中 |
|---|---|---|---|
| `RMSNorm` | `kernels-community/liger-kernels::LigerRMSNorm` | triton | 是（cuda INFERENCE/TRAINING） |
| `rotary_pos_emb`(func) | hub rotary | triton | 是 |
| `SwiGLUMLP`/`GeGLUMLP` | `LigerSwiGLUMLP`/`LigerGEGLUMLP` | triton | 仅 `TORCH_COMPILE` mode |
| `Qwen3_5GatedDeltaNet` | `Atlas-Inference/gdn` | triton | **否**（仅 cc121 GB10） |

**默认关闭**：本报告主路径里 RMSNorm/RoPE/MLP 均按 `pytorch-native` 记，覆盖态在对应步骤后缀 `[hub 覆盖: triton]`。

---

## 1. GQA (Grouped-Query Attention)

代表模型：**Qwen3**（[modeling_qwen3.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3/modeling_qwen3.py)）/ Llama。`Qwen3Attention.forward`（[L252-291](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3/modeling_qwen3.py#L252-L291)）比 Llama 多 `q_norm`/`k_norm`（RMSNorm 作用在 head_dim 上）。

### 1.1 不含 sliding window（prefill ≡ decode，仅 Cache 状态不同）
```
kernel_name: input_layernorm,   实现: Qwen3RMSNorm.forward → x*rsqrt(mean(x^2)) (pytorch-native)   [hub 覆盖: LigerRMSNorm, triton]   # 层外, DecoderLayer
kernel_name: q_proj/k_proj/v_proj, 实现: nn.Linear (pytorch-native, cuBLAS)
kernel_name: q_norm/k_norm,     实现: Qwen3RMSNorm (pytorch-native)   [hub 覆盖: LigerRMSNorm, triton]   # Qwen3 特有, 作用在 head_dim
kernel_name: rope,              实现: apply_rotary_pos_emb → (q*cos)+(rotate_half(q)*sin) (pytorch-native)   [hub 覆盖: rotary_pos_emb, triton]
kernel_name: kv_cache_update,   实现: past_key_values.update(...) → torch.cat (pytorch-native)
kernel_name: core_attn 路径A,   实现: sdpa_attention_forward → F.scaled_dot_product_attention (torch-SDPA-backend; SM90→FlashAttention/cuDNN; 非GQA-SDPA时先 repeat_kv)
kernel_name: core_attn 路径B,   实现: flash_attention_forward → flash_attn_interface.flash_attn_func (third-party, FA3)
kernel_name: core_attn 路径C,   实现: flex_attention_forward → torch.compile(flex_attention) (torch.compile→triton)
kernel_name: core_attn 路径D,   实现: eager_attention_forward → matmul+softmax+matmul, 含 repeat_kv (pytorch-native)
kernel_name: o_proj,            实现: nn.Linear (pytorch-native, cuBLAS)
```
> SDPA 的 GQA 处理：torch≥2.5 且 attention_mask=None 时用 `enable_gqa=True`，否则先 `repeat_kv`（[sdpa_attention.py L57-62](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/integrations/sdpa_attention.py#L57-L62)）。

### 1.2 含 sliding window
**唯一差异**：`Qwen3Attention` 按 `layer_type=="sliding_attention"` 设 `self.sliding_window`（[L250](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3/modeling_qwen3.py#L250)），并作为 `sliding_window=` 传给 attention_interface（[L285](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3/modeling_qwen3.py#L285)）。kernel 列表与 1.1 完全一致，只是：
- 路径B(FA3)：`sliding_window` 转成 `window_size=(sliding_window-1,0)` 传入 FA。
- 路径A(sdpa)/路径D(eager)：滑窗体现在 `attention_mask`（由 `masking_utils` 预生成的 sliding causal mask），kernel 调用不变。
- 路径C(flex)：滑窗体现在 `block_mask`/`score_mod`。

prefill ≡ decode（见 0.5）。

---

## 2. MLA (Multi-head Latent Attention) — DeepSeek-V3

代表模型：**DeepSeek-V3**（[DeepseekV3Attention.forward L411-474](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L411-L474)）。

> **关键**：这是 **naive MLA**——通过 `kv_b_proj` 把 latent `kv_c` **完整展开**成 per-head 的 `k_pass`/`value` 再喂标准 attention，**没有** vLLM 里那种 `W_UK/W_UV` 权重吸收、`torch.bmm` latent 路径或 FlashMLA。prefill 与 decode 走同一份 kernel。

### 路径（prefill ≡ decode）
```
kernel_name: input_layernorm,   实现: DeepseekV3RMSNorm (pytorch-native)   [hub 覆盖: LigerRMSNorm, triton]   # 层外
kernel_name: q 投影链,          实现: q_a_proj → q_a_layernorm(RMSNorm) → q_b_proj, 均 nn.Linear/RMSNorm (pytorch-native)   # q_lora_rank 存在时; 否则单个 q_proj
kernel_name: q split,           实现: torch.split → q_pass(nope)/q_rot(rope) (pytorch-native)
kernel_name: kv down 投影,      实现: kv_a_proj_with_mqa → nn.Linear (pytorch-native)   # 输出 kv_lora_rank + qk_rope_head_dim
kernel_name: kv split,          实现: torch.split → k_pass(latent)/k_rot (pytorch-native)
kernel_name: kv up 投影(展开),  实现: kv_a_layernorm(RMSNorm) → kv_b_proj → nn.Linear (pytorch-native)   # 展开成 num_heads*(qk_nope+v_head_dim)
kernel_name: kv split2,         实现: torch.split → k_pass(nope)/value_states (pytorch-native)
kernel_name: rope,              实现: apply_rotary_pos_emb_interleave (rope_interleave 时) 或 apply_rotary_pos_emb (pytorch-native)   [hub 覆盖: rotary, triton]
kernel_name: k 拼接,            实现: torch.cat([q_pass,q_rot]) / torch.cat([k_pass,k_rot(expand)]) (pytorch-native)
kernel_name: kv_cache_update,   实现: past_key_values.update → torch.cat (pytorch-native)
kernel_name: value pad,         实现: F.pad(value, [0, qk_head_dim - v_head_dim]) (pytorch-native)   # 仅 flash_attention 后端且 qk_head_dim≠v_head_dim
kernel_name: core_attn 路径A,   实现: F.scaled_dot_product_attention (torch-SDPA-backend)
kernel_name: core_attn 路径B,   实现: flash_attn_interface.flash_attn_func (third-party, FA3)
kernel_name: core_attn 路径C,   实现: torch.compile(flex_attention) (torch.compile→triton)
kernel_name: core_attn 路径D,   实现: eager_attention_forward (pytorch-native)
kernel_name: attn_out 裁剪,     实现: attn_output[..., :v_head_dim] (pytorch-native)   # 仅 flash_attention 后端, 撤销上面的 pad
kernel_name: o_proj,            实现: nn.Linear (pytorch-native, cuBLAS)
```
> `apply_rotary_pos_emb` 在 DeepSeek-V3 里同样带 `@use_kernel_func_from_hub("rotary_pos_emb")`（[modeling_deepseek_v3.py L251](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L251)）。

---

## 3. GDN (Gated DeltaNet, Qwen3.5 linear attention)

代表模型：**Qwen3.5**（[Qwen3_5GatedDeltaNet.forward L438-560](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L438-L560)）。非标准注意力，无 KV cache，用 conv_state + recurrent_state。核心 kernel 来自 **third-party `fla`（flash-linear-attention）+ `causal_conv1d`**，两者都有**纯 torch 回退**（`is_fast_path_available` 决定，[L219](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L219)）。类装饰 `@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")` 的 hub 覆盖**仅 cc121（GB10），SM90 不命中**。

### 路径1：Prefill / BF16（多 token；chunk 并行）
```
kernel_name: in_proj_qkv/z/b/a, 实现: nn.Linear ×4 (pytorch-native, cuBLAS)
kernel_name: causal_conv1d,     实现【默认】: causal_conv1d_fn (third-party, causal-conv1d)
                                实现【回退】: F.conv1d + F.silu (pytorch-native)
kernel_name: split_qkv+reshape, 实现: torch.split / reshape / repeat_interleave (pytorch-native)   # GQA-style KV 广播
kernel_name: beta/g 计算,       实现: b.sigmoid(); -A_log.exp()*softplus(a+dt_bias) (pytorch-native)
kernel_name: core_linear_attn,  实现【默认】: chunk_gated_delta_rule (third-party, fla; kernel 内做 qk l2norm)
                                实现【回退】: torch_chunk_gated_delta_rule (pytorch-native; 分块扫描 [L247-325])
kernel_name: gated_rmsnorm,     实现【默认】: FusedRMSNormGated (third-party, fla)
                                实现【回退】: Qwen3_5RMSNormGated → norm*silu(gate) (pytorch-native)
kernel_name: out_proj,          实现: nn.Linear (pytorch-native, cuBLAS)
kernel_name: recurrent_state 写回, 实现: cache_params.update_recurrent_state (pytorch-native)
```

### 路径2：Decode / BF16（`use_precomputed_states and seq_len==1`；recurrent 单步）
```
kernel_name: in_proj_qkv/z/b/a, 实现: nn.Linear ×4 (pytorch-native)
kernel_name: causal_conv1d_update, 实现【默认】: causal_conv1d_update (third-party; 原地更新 conv_state)
                                   实现【回退】: torch_causal_conv1d_update (pytorch-native [L223-235])
kernel_name: split_qkv+reshape, 实现: torch.split / reshape / repeat_interleave (pytorch-native)
kernel_name: beta/g 计算,       实现: sigmoid / softplus / exp (pytorch-native)
kernel_name: core_linear_attn,  实现【默认】: fused_recurrent_gated_delta_rule (third-party, fla)
                                实现【回退】: torch_recurrent_gated_delta_rule (pytorch-native [L328-369])
kernel_name: gated_rmsnorm,     实现【默认】: FusedRMSNormGated (third-party, fla) / 回退 Qwen3_5RMSNormGated (pytorch-native)
kernel_name: out_proj,          实现: nn.Linear (pytorch-native)
```
> conv 与 chunk/recurrent 计算恒 bf16/fp32（kernel 内 `.float()` 累加）；无 FP8。`chunk_gated_delta_rule` 额外吃 `cu_seq_lens_q` 支持 packed 序列（[L546](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L546)）。

---

## 4. DSA (DeepSeek Sparse Attention, DeepSeek-V3.2)

代表模型：**DeepSeek-V3.2**（[DeepseekV32Attention.forward L403-475](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L403-L475)）= **naive MLA（同第 2 节）+ 纯 PyTorch Lightning Indexer**。`_supports_flash_attn=False`（[L652](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L652)）——flash-mla 尚未在 HF 接通。prefill 与 decode 同一份 kernel。

### 4.1 主 MLA 投影链（与第 2 节 MLA 相同）
```
kernel_name: q 投影链/split,    实现: q_a_proj→q_a_layernorm→q_b_proj + torch.split (pytorch-native)
kernel_name: kv down/up 投影,   实现: kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj → nn.Linear (pytorch-native)
kernel_name: rope,              实现: apply_rotary_pos_emb_interleave (pytorch-native)   # 主 MLA 用 interleave
kernel_name: k 拼接 + kv_cache_update, 实现: torch.cat + past_key_values.update (pytorch-native)
```

### 4.2 Indexer 子链（[DeepseekV32Indexer L197-262](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L197-L262)，全 pytorch-native）
```
kernel_name: idx_wq_b,          实现: nn.Linear(q_lora_rank → n_heads*head_dim) (pytorch-native)   # 复用主 q_resid
kernel_name: idx_wk,            实现: nn.Linear(hidden → head_dim) (pytorch-native)
kernel_name: idx_k_norm,        实现: nn.LayerNorm (pytorch-native)
kernel_name: idx_rope,          实现: apply_rotary_pos_emb (非 interleave; unsqueeze_dim=2) (pytorch-native)
kernel_name: idx_cache_update,  实现: past_key_values.update_indexer → torch.cat (pytorch-native)
kernel_name: idx_score,         实现: torch.matmul(q.float(), k.float()) * softmax_scale (pytorch-native)
kernel_name: idx_relu,          实现: F.relu (pytorch-native)
kernel_name: idx_weights,       实现: weights_proj(nn.Linear) * n_heads^-0.5, 再 matmul 加权求和 (pytorch-native)
kernel_name: idx_causal_mask,   实现: + attention_mask 或 masked_fill(causal, -inf) (pytorch-native)
kernel_name: idx_topk,          实现: index_scores.topk(index_topk).indices.to(int32) (pytorch-native)
```
> 注释明确此为 reference 的 **bf16 等价**：跳过了真实实现的 Hadamard 变换与 `fp8_index` FP8 scoring kernel（[L210-213](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py#L210-L213)）。

### 4.3 topk 注入 + core-attn（两条并列路径，取决于后端）
```
# 路径A/A' —— eager | sdpa（topk 折进 attention_mask）
kernel_name: topk→sparse_mask,  实现: new_ones.scatter(-1, topk_indices, False) → masked_fill(-inf) (pytorch-native)   # [B,1,S,T] additive mask
kernel_name: core_attn(sdpa),   实现: F.scaled_dot_product_attention(attn_mask=稀疏mask) (torch-SDPA-backend)
kernel_name: core_attn(eager),  实现: eager_attention_forward(attn_mask=稀疏mask) (pytorch-native)

# 路径B —— 其它后端（indices 透传, 主 attn 仍 dense；HF 暂无 flash-mla 稀疏 kernel）
kernel_name: core_attn,         实现: attention_interface(..., indices=topk_indices) (透传 sparse_indices; 当前后端无稀疏消费者)

kernel_name: o_proj,            实现: nn.Linear (pytorch-native, cuBLAS)   # 所有路径共用
```
> prefill ≡ decode：indexer 对全序列打分，decode 时 `q_len=1`、`T` 为累计 KV 长度，kernel 列表不变。

---

## 5. C4A / C128A (DeepSeek-V4)

代表模型：**DeepSeek-V4**（[DeepseekV4Attention.forward L801-873](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py#L801-L873)）。**Shared-KV MQA（单 KV head，K≡V）+ partial-RoPE + attention sink + sliding window + 分组低秩 O 投影**。后端仍走 `ALL_ATTENTION_FUNCTIONS`（4 后端可选），全部 compressor/indexer 均 **pytorch-native**。

### 5.0 layer_type → 形态映射（[configuration_deepseek_v4.py L21-31](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py#L21-L31)）
| 形态 | layer_type | compress_rate | compressor | Indexer |
|---|---|---|---|---|
| **C4A** | `compressed_sparse_attention` | 4 | `DeepseekV4CSACompressor` | **有**（Lightning Indexer + topk） |
| **C128A** | `heavily_compressed_attention` | 128 | `DeepseekV4HCACompressor` | 无 |
| sliding-only | `sliding_attention` | — | 无 | 无 |

主 forward 三者共用；差异仅在 `self.compressor` 是哪个类（`COMPRESSOR_CLASSES[layer_type]`，[L797-799](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py#L797-L799)）。

### 5.1 C128A（prefill ≡ decode；heavily_compressed_attention）
```
kernel_name: q 投影链,          实现: q_a_proj → q_a_norm(RMSNorm) → q_b_proj → q_b_norm(UnweightedRMSNorm) → nn.Linear/RMSNorm (pytorch-native)   [hub 覆盖: LigerRMSNorm, triton]
kernel_name: q partial-rope,    实现: apply_rotary_pos_emb (interleave, 作用在 trailing rope_head_dim) (pytorch-native)
kernel_name: kv 投影,           实现: kv_proj → kv_norm(RMSNorm) → nn.Linear (pytorch-native)   # 单 KV head, K≡V
kernel_name: kv partial-rope,   实现: apply_rotary_pos_emb (pytorch-native)
kernel_name: sliding_cache_update, 实现: past_key_values.update(kv,kv) (DynamicSlidingWindowLayer, pytorch-native)
kernel_name: hca_compressor,    实现: DeepseekV4HCACompressor.forward (pytorch-native): kv_proj/gate_proj(nn.Linear) → 窗口 reshape → softmax(dim=2,fp32) 聚合 → kv_norm → rope   # 每 128 token 压 1 entry [L394-428]
kernel_name: kv 拼接,           实现: torch.cat([kv, compressed_kv], dim=2) (pytorch-native)
kernel_name: mask 扩展,         实现: torch.cat([attention_mask, block_bias]) 或 F.pad (pytorch-native)
kernel_name: core_attn 路径A,   实现: F.scaled_dot_product_attention (torch-SDPA-backend)   # 注: sink(s_aux) 不被 SDPA 用
kernel_name: core_attn 路径B,   实现: flash_attn_interface.flash_attn_func (third-party, FA3; 支持 s_aux/sink + window)
kernel_name: core_attn 路径C,   实现: torch.compile(flex_attention) (torch.compile→triton; sink 在 flex 外单独处理 [L294-296])
kernel_name: core_attn 路径D,   实现: eager_attention_forward (pytorch-native)
kernel_name: out inv-rope,      实现: apply_rotary_pos_emb(attn_out, cos, -sin) (pytorch-native)   # 撤销 V 上的 rope (K≡V)
kernel_name: o_proj(grouped),   实现: DeepseekV4GroupedLinear(o_a_proj, torch.bmm 分块) → o_b_proj(nn.Linear) (pytorch-native)   # 低秩分组投影 [L303-332]
```
> `sinks`(attention sink, per-head 可学习) 作为 `s_aux=` 传入，仅 FA/eager 真正消费；SDPA 忽略、flex 在 kernel 外处理。

### 5.2 C4A（在 5.1 基础上：compressor 换 CSA + 插入 Lightning Indexer）
C4A 与 C128A 主 forward 完全相同，只把 `hca_compressor` 一步替换为 `csa_compressor`，后者内部多跑一个 Indexer 子链：
```
kernel_name: csa_compressor,    实现: DeepseekV4CSACompressor.forward (pytorch-native): kv_proj/gate_proj → Ca/Cb 双序列 overlap 布局 → softmax(fp32) 聚合 → kv_norm → rope (compress_rate=4) [L623-702]
  └─ kernel_name: idx_kv_proj/gate_proj, 实现: nn.Linear ×2 (pytorch-native)   # Indexer 自带缩小版 compressor
  └─ kernel_name: idx_compress,        实现: 窗口 reshape + softmax(fp32) 聚合 + kv_norm + rope (pytorch-native) [L528-561]
  └─ kernel_name: idx_q_b_proj + rope, 实现: nn.Linear + apply_rotary_pos_emb (pytorch-native)
  └─ kernel_name: idx_score,           实现: DeepseekV4IndexerScorer → Σ_h w·ReLU(q·K) (matmul/relu, pytorch-native) [L446-459]
  └─ kernel_name: idx_causal_mask,     实现: masked_fill(future_mask, -inf) (pytorch-native)
  └─ kernel_name: idx_topk,            实现: index_scores.topk(index_topk).indices + `-1` sentinel(torch.where) (pytorch-native) [L577-586]
kernel_name: block_bias,        实现: new_full(-inf).scatter_(-1, safe_indices, 0.0) (pytorch-native)   # 每 query 的稀疏 block 掩码 [L699-702]
kernel_name: [其余同 5.1],      实现: kv 拼接 / mask 扩展 / core_attn(4 后端) / out inv-rope / o_proj(grouped) (同 C128A)
```
> topk 越界（早期 query 可见块不足）用 `-1` sentinel 标记，再由 `block_bias` 的 `-inf` 剔除（[L571-584](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py#L571-L584)）。prefill ≡ decode：compressor 的窗口 buffer / overlap / entry_count 由 `DeepseekV4CSACache` 跨 forward 维护（[L255-302](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py#L255-L302)）。

---

## 6. 汇总对比表

### 6.1 各类型 core / state kernel（BF16 代表）
| 类型 | 投影/norm/rope | core-attn（4 后端并列） | 特有子链 | prefill vs decode |
|---|---|---|---|---|
| **GQA** | nn.Linear + RMSNorm + rope (pytorch-native) | sdpa / FA3 / flex / eager | — | **同**（仅 Cache 状态；sliding 仅改 window 参数） |
| **MLA** | naive 展开 kv_b_proj (pytorch-native) | sdpa / FA3 / flex / eager | value F.pad + 裁剪 | **同** |
| **GDN** | in/out_proj nn.Linear (pytorch-native) | 无标准 attn | conv(causal_conv1d) + chunk/recurrent(fla) | **分叉**：chunk(prefill) vs recurrent(decode) |
| **DSA** | naive MLA (pytorch-native) | sdpa/eager(稀疏mask) 或 indices透传 | Indexer 打分+topk (pytorch-native) | **同** |
| **C128A** | MQA + partial-rope + grouped o_proj (pytorch-native) | sdpa / FA3(sink) / flex / eager | HCA compressor (pytorch-native) | **同** |
| **C4A** | 同 C128A | 同上 | CSA compressor + Lightning Indexer + topk (pytorch-native) | **同** |

### 6.2 各实现类别在各类型中的分布
| 实现类别 | GQA | MLA | GDN | DSA | C4A/C128A |
|---|:-:|:-:|:-:|:-:|:-:|
| **pytorch-native** | 投影/norm/rope/o_proj | 投影链/展开/pad/o_proj | in/out_proj/beta-g/回退kernel | 主MLA + **Indexer全链** | **投影/compressor/indexer全链/grouped o_proj** |
| **torch-SDPA-backend** | 路径A | 路径A | — | 路径A(稀疏mask) | 路径A |
| **third-party** | FA2/FA3(路径B) | FA2/FA3(路径B) | **fla + causal_conv1d(默认)** | FA(路径B, 暂无稀疏消费) | FA3(路径B, 支持sink) |
| **torch.compile→triton** | flex(路径C) | flex(路径C) | — | flex(路径C) | flex(路径C) |
| **hub-kernel(triton,可选)** | RMSNorm/rope/MLP | RMSNorm/rope/MLP | GDN(仅cc121) | RMSNorm/rope | RMSNorm/rope/MLP |

### 6.3 关键差异速记（对比 vLLM / sglang 报告）
- **本仓库是 reference 实现**：无自研 CUDA/Triton/sgl-kernel、无 cuteDSL、无 JIT-DSL；「实现类别」以 pytorch-native 为主体。
- **core-attn 靠运行时 dispatch**：同一 forward 可落到 sdpa / FA2-3 / flex / eager 四选一，与 prefill/decode 正交。
- **MLA 是 naive 展开**：无 vLLM 的 W_UK/W_UV 权重吸收、无 FlashMLA、无 FP8 KV。
- **DSA/V4 的 indexer/compressor 全是 pytorch-native**：topk/scatter/softmax 明码写出，FP8 scoring 被简化为 bf16 等价；HF 尚未接 flash-mla 稀疏 kernel（`_supports_flash_attn=False`）。
- **只有 GDN 真正按 prefill/decode 分叉**（chunk vs recurrent），且核心 kernel 来自第三方 `fla`/`causal_conv1d`，附带纯 torch 回退。
- **加速全靠可选覆盖**：`use_kernels=True` 才把 RMSNorm/RoPE/MLP 换成 Liger Triton；GDN 的 hub kernel 仅 GB10(cc121) 命中，SM90 走本地实现。
