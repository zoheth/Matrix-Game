"""
Action KV Cache for mouse/keyboard attention.

This cache supports batch dimensions (e.g., spatial batching B*S) and provides
token-level sliding window management. Unlike PagedCache which is optimized
for FlashInfer with page-granular operations, ActionCache maintains the simple
dict-based behavior for action modules.
"""

import torch
from typing import Tuple


class ActionCache:
    """
    KV Cache for action modules (mouse/keyboard attention).

    Supports:
    - Arbitrary batch size (including spatial batching B*S)
    - Token-level sliding window eviction
    - Compatible with the old dict-based cache behavior

    Args:
        batch_size: Batch size (can be B*S for spatial batching)
        max_seq_len: Maximum sequence length to cache
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to place tensors
        dtype: Data type for cache tensors
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate cache tensors
        self.k_cache = torch.zeros(
            (batch_size, max_seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (batch_size, max_seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Index tracking (scalars on CPU for compatibility)
        self.global_end_index = 0
        self.local_end_index = 0

    def reset(self):
        """Reset cache to empty state."""
        self.global_end_index = 0
        self.local_end_index = 0
        # Note: No need to zero out tensors, indices ensure we don't read stale data

    def update_and_get_window(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        num_new_tokens: int,
        max_attention_size: int,
        sink_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Update cache with new KV pairs and return attention window.

        This implements the same logic as the old dict-based cache:
        - Token-level sliding window eviction
        - In-place updates to pre-allocated tensors
        - Returns sliced view (not copy) of the cache

        Args:
            k: New keys [BS, num_new_tokens, num_heads, head_dim]
            v: New values [BS, num_new_tokens, num_heads, head_dim]
            num_new_tokens: Number of new tokens to add
            max_attention_size: Maximum attention window size
            sink_tokens: Number of sink tokens to preserve (usually 0)

        Returns:
            k_window: Keys in attention window [BS, window_len, num_heads, head_dim]
            v_window: Values in attention window [BS, window_len, num_heads, head_dim]
            local_start_index: Start index in cache (Python int)
            local_end_index: End index in cache (Python int)
        """
        # Update global index
        current_end = self.global_end_index + num_new_tokens

        # Check if eviction is needed
        needs_eviction = (num_new_tokens + self.local_end_index) > self.max_seq_len

        if needs_eviction:
            # Calculate eviction parameters
            num_evicted_tokens = num_new_tokens + self.local_end_index - self.max_seq_len
            num_rolled_tokens = self.local_end_index - num_evicted_tokens - sink_tokens

            if num_rolled_tokens > 0:
                # Shift tokens: move tokens after sink region to make room
                src_start = sink_tokens + num_evicted_tokens
                src_end = src_start + num_rolled_tokens
                dst_start = sink_tokens
                dst_end = dst_start + num_rolled_tokens

                # Clone to avoid memory aliasing when regions overlap
                self.k_cache[:, dst_start:dst_end] = self.k_cache[:, src_start:src_end].clone()
                self.v_cache[:, dst_start:dst_end] = self.v_cache[:, src_start:src_end].clone()

            local_end_index = self.local_end_index + num_new_tokens - num_evicted_tokens
        else:
            # No eviction needed
            local_end_index = self.local_end_index + num_new_tokens

        local_start_index = local_end_index - num_new_tokens

        # Insert new keys/values (in-place)
        self.k_cache[:, local_start_index:local_end_index] = k
        self.v_cache[:, local_start_index:local_end_index] = v

        # Update indices
        self.global_end_index = current_end
        self.local_end_index = local_end_index

        # Return attention window (view, not copy)
        window_start = max(0, local_end_index - max_attention_size)
        k_window = self.k_cache[:, window_start:local_end_index]
        v_window = self.v_cache[:, window_start:local_end_index]

        return k_window, v_window, local_start_index, local_end_index


class ActionCacheManager:
    """
    Manages creation of ActionCache instances for different batch sizes.

    This is a factory class that handles the complexity of creating caches
    with potentially different batch sizes (e.g., B for keyboard, B*S for mouse).
    """

    @staticmethod
    def create_cache(
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ActionCache:
        """Create an ActionCache with specified parameters."""
        return ActionCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
