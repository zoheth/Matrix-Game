"""
Optimized KV cache management for action modules.

Now uses ActionCache which supports batch dimensions and token-level eviction,
maintaining compatibility with the old dict-based cache behavior.
"""

import torch
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wan.modules.action_cache import ActionCache


def update_kv_cache_optimized(
    kv_cache: "ActionCache",
    k: torch.Tensor,
    v: torch.Tensor,
    num_new_tokens: int,
    max_attention_size: int,
    sink_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Update ActionCache with new KV pairs and return attention window.

    This is a thin wrapper around ActionCache.update_and_get_window() for
    compatibility with the KVCacheManager interface.

    Args:
        kv_cache: ActionCache instance
        k: New keys [BS, num_new_tokens, num_heads, head_dim]
        v: New values [BS, num_new_tokens, num_heads, head_dim]
        num_new_tokens: Number of new tokens to add
        max_attention_size: Maximum attention window size
        sink_tokens: Number of sink tokens to preserve (unused for action cache)

    Returns:
        k_window: Keys in attention window [BS, window_len, num_heads, head_dim]
        v_window: Values in attention window [BS, window_len, num_heads, head_dim]
        local_start_index: Start index in cache (Python int)
        local_end_index: End index in cache (Python int)
    """
    return kv_cache.update_and_get_window(
        k=k,
        v=v,
        num_new_tokens=num_new_tokens,
        max_attention_size=max_attention_size,
        sink_tokens=sink_tokens,
    )
