"""
Paged KV Cache for FlashInfer integration.

This module provides PagedCache, a memory-efficient KV cache implementation
that uses page-based memory management for compatibility with FlashInfer's
paged attention kernels.

Key features:
1. Page-based memory allocation for efficient memory utilization
2. Sliding window eviction support
3. Sink token preservation during eviction
4. FlashInfer metadata generation for paged attention

Note: FlashInfer plan/run logic is handled by FlashInferPlanner in
flashinfer_attention.py. This module only provides cache storage.
"""

import math
from typing import Optional, Tuple
import torch


class PagedCache:
    """
    Paged KV Cache for efficient memory management with FlashInfer.

    This cache stores K/V tensors in fixed-size pages, allowing efficient
    memory allocation and eviction for long sequence generation.

    Memory layout:
        k_cache: [max_pages, page_size, num_heads, head_dim]
        v_cache: [max_pages, page_size, num_heads, head_dim]

    Attributes:
        page_size: Number of tokens per page
        sink_size: Number of sink tokens to preserve during eviction
        num_heads: Number of attention heads
        head_dim: Dimension per head
        max_pages: Maximum number of pages
        seq_len: Current sequence length in the cache
    """

    def __init__(
        self,
        max_total_tokens: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        sink_size: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda"
    ):
        """
        Initialize the paged cache.

        Args:
            max_total_tokens: Maximum number of tokens the cache can hold
            page_size: Number of tokens per page
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            sink_size: Number of sink tokens to keep at the beginning
            dtype: Data type for cache tensors
            device: Device to place cache tensors
        """
        self.page_size = page_size
        self.sink_size = sink_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Calculate number of pages needed
        self.max_pages = math.ceil(max_total_tokens / page_size)

        # Pre-allocate cache tensors
        # Shape: [max_pages, page_size, num_heads, head_dim]
        self.k_cache = torch.zeros(
            (self.max_pages, page_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (self.max_pages, page_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

        # Page management
        self.next_free_page_id = 0
        self.active_page_indices: list[int] = []
        self.free_page_pool: list[int] = []  # Recycled pages available for reuse

        # Position tracking within current page
        self.current_page_offset = 0
        self.seq_len = 0

        # Track global position (for RoPE compatibility and denoising overwrite detection)
        self.global_position = 0
        # Track the global end index (like dict cache's global_end_index)
        # This is used to detect if we're overwriting the same position during denoising
        self.global_end_index = 0

    def reset(self):
        """Reset the cache to initial state without reallocating memory."""
        self.next_free_page_id = 0
        self.active_page_indices = []
        self.free_page_pool = []
        self.current_page_offset = 0
        self.seq_len = 0
        self.global_position = 0
        self.global_end_index = 0
        # Note: We don't zero out k_cache/v_cache for efficiency
        # The indices will ensure we don't read stale data

    def _allocate_page(self) -> int:
        """
        Allocate a new page, either from free pool or new allocation.

        Returns:
            Page ID

        Raises:
            RuntimeError: If no pages available
        """
        # Try to recycle from free pool first
        if self.free_page_pool:
            return self.free_page_pool.pop()

        # Allocate new page
        if self.next_free_page_id >= self.max_pages:
            raise RuntimeError(
                f"KV Cache Out of Memory! "
                f"Tried to allocate page {self.next_free_page_id} but max is {self.max_pages}. "
                f"Current seq_len: {self.seq_len}, max_tokens: {self.max_pages * self.page_size}. "
                f"Call evict() before append() to free space."
            )

        page_id = self.next_free_page_id
        self.next_free_page_id += 1
        return page_id

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append new K/V pairs to the cache.

        Args:
            k: Key tensor [seq_len, num_heads, head_dim]
            v: Value tensor [seq_len, num_heads, head_dim]
        """
        incoming_len = k.shape[0]

        incoming_processed = 0
        while incoming_processed < incoming_len:
            # Allocate new page if needed
            if not self.active_page_indices or self.current_page_offset == self.page_size:
                new_page_id = self._allocate_page()
                self.active_page_indices.append(new_page_id)
                self.current_page_offset = 0

            # Calculate how much to write to current page
            current_page_id = self.active_page_indices[-1]
            space_left = self.page_size - self.current_page_offset
            to_write = min(space_left, incoming_len - incoming_processed)

            # Write to cache
            self.k_cache[
                current_page_id,
                self.current_page_offset:self.current_page_offset + to_write
            ] = k[incoming_processed:incoming_processed + to_write]

            self.v_cache[
                current_page_id,
                self.current_page_offset:self.current_page_offset + to_write
            ] = v[incoming_processed:incoming_processed + to_write]

            self.current_page_offset += to_write
            incoming_processed += to_write

        # Update seq_len after successful append
        self.seq_len += incoming_len
        self.global_position += incoming_len

    def update_or_append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        current_start: int,
        current_end: int
    ) -> None:
        """
        Update existing cache entries or append new ones.

        This method handles the denoising case where multiple forward passes
        happen for the same position. If current_end <= global_end_index,
        it overwrites the existing entries. Otherwise, it appends new entries.

        This matches the behavior of dict-based cache in CausalWanSelfAttention.

        Args:
            k: Key tensor [seq_len, num_heads, head_dim]
            v: Value tensor [seq_len, num_heads, head_dim]
            current_start: Start position in global sequence
            current_end: End position in global sequence (current_start + seq_len)
        """
        incoming_len = k.shape[0]

        if current_end <= self.global_end_index:
            # Overwrite mode: we're re-processing the same position (denoising)
            # Calculate the local position to overwrite
            # The cache stores tokens from some start position up to global_end_index
            # We need to find where current_start maps to in our paged structure

            # For simplicity, since we're overwriting the same range,
            # we just need to write to the last incoming_len positions
            # This works because during denoising:
            # - current_start and current_end are the same across steps
            # - We want to overwrite the same K/V positions

            # Calculate which pages and offsets to write to
            # The positions we want to overwrite are at the end of the valid cache
            write_start_in_cache = self.seq_len - incoming_len

            if write_start_in_cache < 0:
                # This shouldn't happen, but handle gracefully
                self.append(k, v)
                self.global_end_index = current_end
                return

            # Calculate page and offset for write_start_in_cache
            start_page_idx = write_start_in_cache // self.page_size
            start_offset = write_start_in_cache % self.page_size

            incoming_processed = 0
            current_page_idx = start_page_idx
            current_offset = start_offset

            while incoming_processed < incoming_len:
                if current_page_idx >= len(self.active_page_indices):
                    # Need more pages than we have - shouldn't happen in overwrite mode
                    break

                page_id = self.active_page_indices[current_page_idx]
                space_in_page = self.page_size - current_offset
                to_write = min(space_in_page, incoming_len - incoming_processed)

                # Overwrite cache
                self.k_cache[
                    page_id,
                    current_offset:current_offset + to_write
                ] = k[incoming_processed:incoming_processed + to_write]

                self.v_cache[
                    page_id,
                    current_offset:current_offset + to_write
                ] = v[incoming_processed:incoming_processed + to_write]

                incoming_processed += to_write
                current_page_idx += 1
                current_offset = 0

            # global_end_index stays the same (we're overwriting, not extending)
        else:
            # Append mode: new position beyond what we've seen
            self.append(k, v)
            self.global_end_index = current_end

    def evict(self, max_allowed_tokens: int) -> int:
        """
        Evict old pages to stay within token limit.

        Preserves sink tokens at the beginning of the sequence.
        Eviction happens at page boundaries for efficiency.
        Evicted pages are returned to the free pool for reuse.

        Args:
            max_allowed_tokens: Maximum tokens to keep in cache

        Returns:
            Number of tokens evicted
        """
        if self.seq_len <= max_allowed_tokens:
            return 0

        num_to_remove = self.seq_len - max_allowed_tokens

        # Calculate number of pages to drop (evict at page boundary)
        pages_to_drop = num_to_remove // self.page_size

        if pages_to_drop <= 0:
            return 0

        # Calculate sink pages to preserve
        sink_pages = math.ceil(self.sink_size / self.page_size) if self.sink_size > 0 else 0

        # Ensure we don't evict sink pages and keep at least one page
        max_evictable = len(self.active_page_indices) - sink_pages - 1
        if max_evictable <= 0:
            return 0

        pages_to_drop = min(pages_to_drop, max_evictable)

        if pages_to_drop > 0:
            # Get page IDs to evict (after sink region)
            evicted_page_ids = self.active_page_indices[sink_pages:sink_pages + pages_to_drop]

            # Return evicted pages to free pool for reuse
            self.free_page_pool.extend(evicted_page_ids)

            # Remove pages from active list
            del self.active_page_indices[sink_pages:sink_pages + pages_to_drop]

            evicted_tokens = pages_to_drop * self.page_size
            self.seq_len -= evicted_tokens
            return evicted_tokens

        return 0

    def get_kv_for_attention(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get contiguous K/V tensors for standard attention.

        This is a fallback path when FlashInfer is not available.

        Returns:
            Tuple of (k, v) tensors with shape [seq_len, num_heads, head_dim]
        """
        if not self.active_page_indices:
            return (
                torch.empty((0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device),
                torch.empty((0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device)
            )

        # Collect all K/V from active pages
        k_parts = []
        v_parts = []

        for i, page_id in enumerate(self.active_page_indices):
            if i == len(self.active_page_indices) - 1:
                # Last page: only up to current_page_offset
                k_parts.append(self.k_cache[page_id, :self.current_page_offset])
                v_parts.append(self.v_cache[page_id, :self.current_page_offset])
            else:
                # Full page
                k_parts.append(self.k_cache[page_id])
                v_parts.append(self.v_cache[page_id])

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def get_flashinfer_meta(
        self,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate FlashInfer paged attention metadata.

        Args:
            device: Device to place tensors on (defaults to cache device)

        Returns:
            Tuple of (page_indices, indptr, last_page_len):
                - page_indices: [num_pages] int32 tensor of page indices
                - indptr: [2] int32 tensor with [0, num_pages] (batch size = 1)
                - last_page_len: [1] int32 tensor with tokens in last page
        """
        if device is None:
            device = self.device

        # Page indices for active pages
        indices = torch.tensor(
            self.active_page_indices,
            dtype=torch.int32,
            device=device
        )

        # Indptr for single batch: [0, num_active_pages]
        indptr = torch.tensor(
            [0, len(self.active_page_indices)],
            dtype=torch.int32,
            device=device
        )

        # Number of valid tokens in last page
        last_page_len = torch.tensor(
            [self.current_page_offset if self.current_page_offset > 0 else self.page_size],
            dtype=torch.int32,
            device=device
        )

        return indices, indptr, last_page_len

    @property
    def total_tokens(self) -> int:
        """Total number of valid tokens in cache."""
        return self.seq_len

    def __repr__(self) -> str:
        return (
            f"PagedCache("
            f"seq_len={self.seq_len}, "
            f"pages={len(self.active_page_indices)}/{self.max_pages}, "
            f"page_size={self.page_size}, "
            f"heads={self.num_heads}, "
            f"head_dim={self.head_dim})"
        )


class PagedCacheManager:
    """
    Manager for multiple PagedCache instances (one per transformer layer).

    This class provides a unified interface for creating and managing
    layer-wise paged caches, compatible with the existing CacheManager API.

    Note: FlashInfer plan/run logic is handled by FlashInferPlanner in
    flashinfer_attention.py, not here. This class only manages cache storage.
    """

    def __init__(
        self,
        num_layers: int,
        max_total_tokens: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        sink_size: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda"
    ):
        """
        Initialize the paged cache manager.

        Args:
            num_layers: Number of transformer layers
            max_total_tokens: Maximum tokens per layer
            page_size: Tokens per page
            num_heads: Number of attention heads
            head_dim: Dimension per head
            sink_size: Sink tokens to preserve
            dtype: Cache data type
            device: Cache device
        """
        self.num_layers = num_layers
        self.max_total_tokens = max_total_tokens
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.dtype = dtype
        self.device = device

        # Create per-layer caches
        self.caches: list[PagedCache] = [
            PagedCache(
                max_total_tokens=max_total_tokens,
                page_size=page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                sink_size=sink_size,
                dtype=dtype,
                device=device
            )
            for _ in range(num_layers)
        ]

    def reset(self) -> None:
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def get_cache(self, layer_idx: int) -> PagedCache:
        """Get cache for a specific layer."""
        return self.caches[layer_idx]

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, idx: int) -> PagedCache:
        return self.caches[idx]
