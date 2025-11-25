"""
Optimized KV cache management for PagedCache.

Now uses PagedCache for efficient memory management with FlashInfer integration.
"""

import torch
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wan.modules.paged_cache import PagedCache


def update_kv_cache_optimized(
    kv_cache: "PagedCache",
    k: torch.Tensor,
    v: torch.Tensor,
    num_new_tokens: int,
    max_attention_size: int,
    sink_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Update PagedCache with new KV pairs and return attention window.

    Args:
        kv_cache: PagedCache instance
        k: New keys [BS, num_new_tokens, num_heads, head_dim]
        v: New values [BS, num_new_tokens, num_heads, head_dim]
        num_new_tokens: Number of new tokens to add
        max_attention_size: Maximum attention window size
        sink_tokens: Number of sink tokens to preserve (unused with PagedCache)

    Returns:
        k_window: Keys in attention window [BS, window_len, num_heads, head_dim]
        v_window: Values in attention window [BS, window_len, num_heads, head_dim]
        local_start_index: Start index in cache (Python int)
        local_end_index: End index in cache (Python int)
    """
    batch_size = k.size(0)

    # PagedCache only supports batch_size=1, so we take the first batch
    # For keyboard/mouse with spatial batching, we assume all spatial locations share the same cache
    if batch_size > 1:
        # Take first batch for caching (mean pooling would be better but more complex)
        k_for_cache = k[0:1]  # [1, num_new_tokens, num_heads, head_dim]
        v_for_cache = v[0:1]
    else:
        k_for_cache = k
        v_for_cache = v

    k_squeezed = k_for_cache.squeeze(0)  # [num_new_tokens, num_heads, head_dim]
    v_squeezed = v_for_cache.squeeze(0)  # [num_new_tokens, num_heads, head_dim]

    # Calculate current position
    current_start = kv_cache.global_end_index
    current_end = current_start + num_new_tokens

    # Evict old tokens BEFORE appending to make room (sliding window)
    # We need to respect both max_attention_size and cache capacity
    cache_capacity = kv_cache.max_pages * kv_cache.page_size
    page_size = kv_cache.page_size

    # If max_attention_size is -1 (global attention), use cache_capacity as limit
    if max_attention_size == -1:
        effective_max_size = cache_capacity
    else:
        effective_max_size = min(max_attention_size, cache_capacity)

    # Evict to make room for new tokens
    # PagedCache evicts at page granularity, so we need to ensure we evict enough pages
    current_seq_len = kv_cache.seq_len
    tokens_after_append = current_seq_len + num_new_tokens

    if tokens_after_append > effective_max_size:
        # Calculate how many tokens need to be removed
        tokens_to_remove = tokens_after_append - effective_max_size

        # Round up to full pages to ensure we remove enough
        # Use ceiling division: (tokens_to_remove + page_size - 1) // page_size
        pages_to_remove = (tokens_to_remove + page_size - 1) // page_size

        # Calculate target size after removing pages
        target_size = max(0, current_seq_len - pages_to_remove * page_size)
        kv_cache.evict(target_size)
    # else: no eviction needed, we're within limits

    # Update cache using PagedCache's update_or_append method
    kv_cache.update_or_append(k_squeezed, v_squeezed, current_start, current_end)

    # Get contiguous KV for attention
    k_contiguous, v_contiguous = kv_cache.get_kv_for_attention()

    # Expand to match original batch size
    k_window = k_contiguous.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [BS, seq_len, num_heads, head_dim]
    v_window = v_contiguous.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Return indices for compatibility
    local_end_index = kv_cache.seq_len
    local_start_index = max(0, local_end_index - num_new_tokens)

    return k_window, v_window, local_start_index, local_end_index


