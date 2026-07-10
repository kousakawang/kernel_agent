"""Module-level math reference (PROBLEM SPEC) for the GDN chunk delta-rule core.

WHAT THIS IS
------------
A faithful, dependency-free PyTorch transcription of the *original problem
semantics* for the kernel under optimization. It is copied verbatim (only
imports adjusted) from upstream HuggingFace Transformers:

    repo:    huggingface/transformers
    commit:  2e68d89e50
    file:    src/transformers/models/qwen3_5/modeling_qwen3_5.py
    symbols: l2norm, torch_chunk_gated_delta_rule  (lines 241-325)

ANCHOR MAPPING (why this file is at THIS granularity)
-----------------------------------------------------
The hot kernel `chunk_gated_delta_rule_fwd_intra` produces chunked internal
intermediates (w, u, A) that DO NOT exist as standalone functions in the
论文级 dense implementation. So there is no transformers counterpart at the
kernel boundary. Instead we anchor one level up, where a clean dense
counterpart does exist:

    sglang  chunk_gated_delta_rule_fwd   (fla/chunk.py)
              ├─ chunk_local_cumsum
              ├─ chunk_gated_delta_rule_fwd_intra  → w,u,A   ← optimization target
              ├─ chunk_gated_delta_rule_fwd_h      → h,v_new
              └─ chunk_fwd_o                        → o
            ⇕  same math
    torch_chunk_gated_delta_rule (this file)         → core_attn_out, last_recurrent_state

IO correspondence (anchor level):
    sglang:  q,k,v,g,beta,scale,initial_state,...,cu_seqlens  -> o, h
    here:    query,key,value,g,beta,...,initial_state,chunk_size=64 -> core_attn_out, last_recurrent_state

The only real gap is layout: sglang packs variable-length sequences with
`cu_seqlens`; this reference is dense `[B, T, H, *]`. That gap lives OUTSIDE
the anchor (upstream paging/gather), so it is not reconciled here.

ROLE IN THE TASK PACK (READ ONLY — not a correctness gate)
----------------------------------------------------------
- Correctness gate stays `snapshot-golden` (snapshots/ + correctness_test.py).
  This file is NOT imported by correctness_test.py and MUST NOT relax the gate.
- Kernel Engineer reads this to understand the original math (e.g. whether the
  shipped sglang path took any numerical approximation) and to reason about
  fusion/algorithm choices.
- Optional approximation probe: unpack a packed snapshot by `cu_seqlens` into
  dense `[B, T, H, *]`, run `torch_chunk_gated_delta_rule`, and diff against the
  snapshot output. Large divergence => the shipped path is approximating.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
