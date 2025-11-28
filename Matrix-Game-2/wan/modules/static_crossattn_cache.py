"""
Static Cross-Attention KV Cache for CUDA Graph compatibility.

Cross-attention cache has unique characteristics:
1. Fixed sequence length (e.g., 257 for CLIP features)
2. K/V computed once and reused across all denoising steps
3. No dynamic growth or eviction needed

This implementation eliminates conditional branches and dynamic assignments
to ensure full CUDA Graph compatibility.
"""

import torch
from typing import Tuple


class StaticCrossAttnCache:
    """
    CUDA Graph-compatible static KV cache for cross-attention.

    Unlike self-attention caches (which grow incrementally), cross-attention
    caches store fixed-length context (e.g., CLIP image features) that is
    computed once and reused.

    Key features:
    - Pre-allocated fixed-size tensors
    - In-place updates (no assignment during forward)
    - GPU-only control flow (torch.where instead of if/else)
    - Zero indexing overhead

    Args:
        batch_size: Batch size
        seq_len: Fixed context sequence length (e.g., 257)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to place tensors
        dtype: Data type for cache tensors
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate K/V cache [B, L, H, D]
        self.k_cache = torch.zeros(
            (batch_size, seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (batch_size, seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # GPU tensor flag for CUDA Graph compatibility
        # 0 = uninitialized, 1 = initialized
        self.is_initialized = torch.zeros(1, dtype=torch.long, device=device)

    def update_or_get(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache on first call, return cached values on subsequent calls.

        This method is fully CUDA Graph compatible:
        - Uses torch.where() instead of if/else
        - In-place operations only (no tensor assignment)
        - All control flow on GPU

        Args:
            k_new: New keys [B, L, H, D]
            v_new: New values [B, L, H, D]

        Returns:
            Tuple of (k, v) from cache
        """
        # Create update mask: True if uninitialized
        should_update = (self.is_initialized == 0)

        # Conditional in-place update (CUDA Graph safe)
        # If should_update: cache = k_new, else: cache = cache (no-op)
        torch.where(
            should_update.view(1, 1, 1, 1),
            k_new,
            self.k_cache,
            out=self.k_cache
        )
        torch.where(
            should_update.view(1, 1, 1, 1),
            v_new,
            self.v_cache,
            out=self.v_cache
        )

        # Mark as initialized (idempotent operation)
        self.is_initialized.fill_(1)

        return self.k_cache, self.v_cache

    def reset(self):
        """
        Reset cache to uninitialized state.

        Note: Should be called outside of CUDA Graph captured regions.
        """
        self.is_initialized.zero_()
        # Optional: zero out cache data (not required for correctness)
        # self.k_cache.zero_()
        # self.v_cache.zero_()

    def __repr__(self) -> str:
        return (
            f"StaticCrossAttnCache("
            f"batch_size={self.batch_size}, "
            f"seq_len={self.seq_len}, "
            f"heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"initialized={bool(self.is_initialized.item())})"
        )
