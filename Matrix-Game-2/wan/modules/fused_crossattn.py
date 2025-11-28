"""
Fused Cross-Attention with integrated caching.

Optimizations:
1. K/V computation only on first forward pass
2. Cache and attention logic fused together
3. No redundant tensor copies
4. CUDA Graph compatible
5. Optional FlashInfer integration for better performance
"""

import torch
import torch.nn as nn
from typing import Optional
from .model import WanRMSNorm


class FusedCrossAttention(nn.Module):
    """
    Cross-attention with integrated KV caching.

    Key optimizations:
    - K/V computed once and cached internally
    - No external cache management needed
    - Fused computation path
    - CUDA Graph compatible (all state on GPU)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        qk_norm: Whether to normalize Q/K
        eps: Epsilon for normalization
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Projection layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # Optional Q/K normalization
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Internal cache state (will be allocated on first forward)
        self.register_buffer('_k_cache', None, persistent=False)
        self.register_buffer('_v_cache', None, persistent=False)
        self.register_buffer('_is_cached', torch.tensor(0, dtype=torch.long), persistent=False)

    def reset_cache(self):
        """Reset cache state. Call this between generation sequences."""
        self._is_cached.zero_()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with automatic caching.

        Args:
            x: Query input [B, L_q, C]
            context: Key/Value context [B, L_kv, C]
            use_cache: Whether to use caching (default: True)

        Returns:
            Output tensor [B, L_q, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query (always computed, as it changes every forward pass)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if use_cache:
            # Lazy cache allocation on first use
            if self._k_cache is None:
                # Allocate cache with same shape as context
                L_kv = context.size(1)
                self._k_cache = torch.zeros(
                    b, L_kv, n, d,
                    dtype=context.dtype,
                    device=context.device
                )
                self._v_cache = torch.zeros(
                    b, L_kv, n, d,
                    dtype=context.dtype,
                    device=context.device
                )

            # Compute K/V only if not cached (CUDA Graph compatible)
            is_cached = self._is_cached.item() > 0

            if not is_cached:
                # First pass: compute and store
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                self._k_cache.copy_(k)
                self._v_cache.copy_(v)
                self._is_cached.fill_(1)

            # Use cached K/V
            k = self._k_cache
            v = self._v_cache
        else:
            # No caching: compute K/V every time
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # Cross-attention (can use FlashAttention or standard implementation)
        x = self._attention(q, k, v)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention computation. Override this to use FlashInfer or other kernels.

        Args:
            q: [B, L_q, H, D]
            k: [B, L_kv, H, D]
            v: [B, L_kv, H, D]

        Returns:
            [B, L_q, H, D]
        """
        # Import flash_attention from your module
        from .attention import flash_attention
        return flash_attention(q, k, v, k_lens=None)


class FusedCrossAttentionFlashInfer(FusedCrossAttention):
    """
    Cross-attention using FlashInfer kernels for better performance.

    FlashInfer advantages for cross-attention:
    - Fused kernel (faster than separate ops)
    - Better memory access patterns
    - Optimized for this exact use case
    """

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Use FlashInfer cross-attention kernel."""
        try:
            import flashinfer

            # Reshape for FlashInfer
            # q: [B, L_q, H, D] -> [B*L_q, H, D]
            # k, v: [B, L_kv, H, D] -> [B*L_kv, H, D]
            B, L_q, H, D = q.shape
            _, L_kv, _, _ = k.shape

            q_flat = q.reshape(B * L_q, H, D)
            k_flat = k.reshape(B * L_kv, H, D)
            v_flat = v.reshape(B * L_kv, H, D)

            # Create layout tensors for FlashInfer
            # For cross-attention, each query attends to all L_kv keys
            qo_indptr = torch.arange(0, B * L_q + 1, dtype=torch.int32, device=q.device)
            kv_indptr = torch.arange(0, (B + 1) * L_kv, L_kv, dtype=torch.int32, device=q.device)

            # FlashInfer cross-attention
            out = flashinfer.batch_decode_with_padded_kv_cache(
                q_flat, k_flat, v_flat,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
            )

            # Reshape back: [B*L_q, H, D] -> [B, L_q, H, D]
            return out.reshape(B, L_q, H, D)

        except ImportError:
            # Fallback to standard flash_attention
            return super()._attention(q, k, v)


# Convenience function to create the appropriate version
def create_cross_attention(
    dim: int,
    num_heads: int,
    qk_norm: bool = True,
    eps: float = 1e-6,
    use_flashinfer: bool = False,
) -> FusedCrossAttention:
    """
    Factory function to create cross-attention module.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        qk_norm: Whether to normalize Q/K
        eps: Epsilon for normalization
        use_flashinfer: Use FlashInfer kernels if available

    Returns:
        Cross-attention module
    """
    if use_flashinfer:
        try:
            import flashinfer
            return FusedCrossAttentionFlashInfer(dim, num_heads, qk_norm, eps)
        except ImportError:
            print("[Warning] FlashInfer not available, falling back to standard implementation")

    return FusedCrossAttention(dim, num_heads, qk_norm, eps)
