# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Pure FlashInfer attention from first principles.

FlashInfer provides efficient attention kernels optimized for inference.
Core principle: softmax(Q @ K^T / sqrt(d)) @ V with fused CUDA kernels.
"""
import torch
import flashinfer

__all__ = ['flash_attention']


def flash_attention(q, k, v, causal=False, window_size=(-1, -1)):
    """
    FlashInfer attention from first principles.

    Args:
        q: [B, L_q, H, D]
        k: [B, L_k, H, D]
        v: [B, L_k, H, D]
        causal: bool
        window_size: (left, right) - sliding window

    Returns:
        [B, L_q, H, D]
    """
    B, L_q, H, D = q.shape

    # FlashInfer processes each sample independently
    outputs = []
    for i in range(B):
        out_i = flashinfer.single_prefill_with_kv_cache(
            q=q[i],  # [L_q, H, D]
            k=k[i],  # [L_k, H, D]
            v=v[i],  # [L_k, H, D]
            causal=causal,
            kv_layout="NHD",
            pos_encoding_mode="NONE",  # RoPE applied externally
            window_left=window_size[0] if window_size != (-1, -1) else -1,
        )
        outputs.append(out_i)

    return torch.stack(outputs, dim=0)
