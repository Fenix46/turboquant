"""
TurboQuant score module — attention computation over compressed + exact segments.

Handles the read path:
  - Compute attention scores over compressed historical KV (via Triton or PyTorch fallback)
  - Compute attention scores over exact recent buffer (via standard matmul / SDPA)
  - Merge logits and weighted values from both segments

Design rule: compressed path is only invoked when history is large enough
to justify it (>= 16 tokens).
"""

from __future__ import annotations

import math
import logging
import torch
import torch.nn.functional as F

from turboquant.store import FlatCache, CompressedKVStore
from turboquant.kv_cache import dequantize_values
from turboquant.quantizer import TurboQuantProd

try:
    from turboquant.triton_kernels import turboquant_fused_decode
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 for decode
        store: compressed KV store with historical tokens
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads (for GQA expansion)
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    # If we have only one segment, use specialized path
    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale,
            store.value_group_size
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale
        )

    # Both segments present — merge via online-softmax style aggregation
    # to avoid materializing the full history in FP32.
    return _attend_hybrid_fused(
        query, flat, store.quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale, store.value_group_size
    )


def _attend_compressed_only(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
    value_group_size: int = 32,
) -> torch.Tensor:
    """Attention over compressed history only. Prefers Triton if available."""
    T, Q, D = query.shape
    
    # Triton path — the paper's fast path
    if HAS_TRITON and T == 1 and gqa_ratio == 1:
        # Currently fused kernel is optimized for MHA (gqa_ratio=1)
        # and single-token decode (T=1).
        return turboquant_fused_decode(
            query=query,
            quantized_key=flat.prod_q,
            value_quantized=flat.value_q,
            Pi=quantizer.mse_quantizer.Pi,
            S=quantizer.S,
            centroids=quantizer.mse_quantizer.centroids,
            mse_bits=flat.prod_q.mse_bits,
            qjl_scale=quantizer.qjl_scale,
            sm_scale=scale,
            group_size=value_group_size,
        ).unsqueeze(0)

    # PyTorch fallback path
    k_dequant = quantizer.dequantize(flat.prod_q)  # (H_kv, N, D)
    v_dequant = dequantize_values(flat.value_q, value_group_size)

    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _attend_hybrid_fused(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    value_group_size: int = 32,
) -> torch.Tensor:
    """Merge history + recent segments without full history materialization."""
    T, Q, D = query.shape
    
    # Path A: Compressed History
    # We need the max logit (m) and sum of exponentials (l) to merge
    if HAS_TRITON and T == 1 and gqa_ratio == 1:
        # Use Triton kernels for history (unbiased score estimation)
        from turboquant.triton_kernels import turboquant_attention_score
        hist_logits = turboquant_attention_score(
            query=query,
            quantized_key=flat.prod_q,
            Pi=quantizer.mse_quantizer.Pi,
            S=quantizer.S,
            centroids=quantizer.mse_quantizer.centroids,
            mse_bits=flat.prod_q.mse_bits,
            qjl_scale=quantizer.qjl_scale,
        ) * scale # (BH, N_hist)
        
        hist_m = hist_logits.max(dim=-1, keepdim=True).values
        hist_p = torch.exp(hist_logits - hist_m)
        hist_l = hist_p.sum(dim=-1, keepdim=True)
        
        v_hist = dequantize_values(flat.value_q, value_group_size) # (H, N, D)
        # Weighted sum: (H, 1, N) @ (H, N, D) -> (H, 1, D)
        hist_out = torch.matmul(hist_p.unsqueeze(1), v_hist).squeeze(1)
    else:
        # Fallback for GQA/Prefill
        k_hist = quantizer.dequantize(flat.prod_q)
        v_hist = dequantize_values(flat.value_q, value_group_size)
        
        # q: (H, G, T, D), k: (H, 1, N, D) -> scores: (H, G, T, N)
        q_gqa = query.float().view(T, num_kv_heads, gqa_ratio, D).permute(1, 2, 0, 3)
        hist_logits = torch.einsum("hgtd,hgnd->hgtn", q_gqa, k_hist.unsqueeze(1)) * scale
        
        hist_m = hist_logits.max(dim=-1, keepdim=True).values
        hist_p = torch.exp(hist_logits - hist_m)
        hist_l = hist_p.sum(dim=-1, keepdim=True)
        hist_out = torch.einsum("hgtn,hgnd->hgtd", hist_p, v_hist.unsqueeze(1))

    # Path B: Recent Buffer (Exact)
    k_recent = recent_k.transpose(0, 1) # (H, N_rec, D)
    v_recent = recent_v.transpose(0, 1)
    
    if T == 1 and gqa_ratio == 1:
        # q: (BH, D), k: (H, N, D) -> (H, 1, N)
        q_flat = query.view(num_kv_heads, 1, head_dim)
        rec_logits = torch.matmul(q_flat, k_recent.transpose(-1, -2)) * scale
        rec_m = rec_logits.max(dim=-1, keepdim=True).values
        rec_p = torch.exp(rec_logits - rec_m)
        rec_l = rec_p.sum(dim=-1, keepdim=True)
        rec_out = torch.matmul(rec_p, v_recent)
    else:
        q_gqa = query.float().view(T, num_kv_heads, gqa_ratio, D).permute(1, 2, 0, 3)
        rec_logits = torch.einsum("hgtd,hgnd->hgtn", q_gqa, k_recent.unsqueeze(1)) * scale
        rec_m = rec_logits.max(dim=-1, keepdim=True).values
        rec_p = torch.exp(rec_logits - rec_m)
        rec_l = rec_p.sum(dim=-1, keepdim=True)
        rec_out = torch.einsum("hgtn,hgnd->hgtd", rec_p, v_recent.unsqueeze(1))

    # Merge contributions via online softmax logic
    # out = (hist_out * exp(m_hist - m_max) + rec_out * exp(m_rec - m_max)) / (l_hist * exp(m_hist - m_max) + l_rec * exp(m_rec - m_max))
    m_max = torch.maximum(hist_m, rec_m)
    alpha_hist = torch.exp(hist_m - m_max)
    alpha_rec = torch.exp(rec_m - m_max)
    
    combined_l = hist_l * alpha_hist + rec_l * alpha_rec
    combined_out = (hist_out * alpha_hist + rec_out * alpha_rec) / combined_l
    
    if T == 1 and gqa_ratio == 1:
        return combined_out.view(T, Q, D).to(query.dtype)
    return combined_out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(
            f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}"
        )

    # Avoid repeat_interleave(Q/H) on KV tensors to keep memory bounded at long context.
    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)   # (H_kv, 1, N, D) broadcast over G
    v = kv_values.float().unsqueeze(1) # (H_kv, 1, N, D) broadcast over G

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
