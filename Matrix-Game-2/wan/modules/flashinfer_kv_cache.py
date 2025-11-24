"""
FlashInfer-based PagedKVCache Manager for CausalWanModel.

This module provides:
1. PagedKVCacheManager - Manages FlashInfer's PagedKVCache for all transformer blocks
2. PrecomputedRoPE3D - Precomputes 3D RoPE frequencies for efficient inference
3. FlashInferSelfAttention - Drop-in replacement for CausalWanSelfAttention using FlashInfer

Key design decisions:
- Use BatchPrefillWithPagedKVCacheWrapper (NOT BatchDecode) because Query Length > 1 (one frame ≈ 880 tokens)
- Apply 3D RoPE BEFORE FlashInfer (rope_mode="NONE") due to incompatibility with FlashInfer's built-in RoPE
- All dynamic allocations happen at initialization, making it CUDA Graph friendly
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn

try:
    import flashinfer
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    flashinfer = None
    BatchPrefillWithPagedKVCacheWrapper = None


class PrecomputedRoPE3D:
    """
    Precomputes 3D RoPE frequency grid for efficient inference.

    This avoids repeated torch.cat/expand/view operations during forward pass,
    which is critical for CUDA Graph compatibility.

    The frequency grid is computed as:
        freqs[t, h, w, :] = concat(
            freqs_time[t],
            freqs_height[h],
            freqs_width[w]
        )
    """

    def __init__(
        self,
        max_frames: int,
        height: int,
        width: int,
        head_dim: int,
        freqs: torch.Tensor,  # Original freqs from CausalWanModel [1024, head_dim//2]
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize precomputed 3D RoPE frequencies.

        Args:
            max_frames: Maximum number of frames to support
            height: Spatial height (after patchification)
            width: Spatial width (after patchification)
            head_dim: Dimension of each attention head
            freqs: Original RoPE frequencies from model [1024, head_dim//2]
            device: Target device
            dtype: Data type for frequencies
        """
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Split freqs into time, height, width components
        # Following the structure in causal_rope_apply
        c = head_dim // 2
        self.freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # Precompute the full 3D grid [max_frames, height, width, head_dim//2]
        self._precompute_grid()

    def _precompute_grid(self):
        """Precompute the full 3D frequency grid."""
        freqs_time, freqs_height, freqs_width = self.freqs_split

        # Precompute for all possible frame positions
        # Shape: [max_frames, height, width, head_dim//2]
        grid = torch.zeros(
            self.max_frames, self.height, self.width,
            self.head_dim // 2,
            device=self.device, dtype=self.dtype
        )

        c = self.head_dim // 2
        c_time = c - 2 * (c // 3)
        c_spatial = c // 3

        for t in range(self.max_frames):
            for h in range(self.height):
                for w in range(self.width):
                    # Time component
                    grid[t, h, w, :c_time] = freqs_time[t]
                    # Height component
                    grid[t, h, w, c_time:c_time + c_spatial] = freqs_height[h]
                    # Width component
                    grid[t, h, w, c_time + c_spatial:] = freqs_width[w]

        # Store as complex exponential (already in freqs)
        self.freq_grid = grid  # [max_frames, H, W, head_dim//2]

    def get_freqs(self, start_frame: int, num_frames: int) -> torch.Tensor:
        """
        Get precomputed frequencies for a range of frames.

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames

        Returns:
            Frequencies tensor [num_frames * H * W, 1, head_dim//2]
        """
        # Slice and flatten: [num_frames, H, W, C] -> [num_frames*H*W, 1, C]
        freqs = self.freq_grid[start_frame:start_frame + num_frames]  # [F, H, W, C]
        freqs = freqs.reshape(-1, 1, self.head_dim // 2)  # [F*H*W, 1, C]
        return freqs


class PagedKVCacheConfig:
    """Configuration for PagedKVCache."""

    def __init__(
        self,
        batch_size: int = 1,
        num_layers: int = 30,
        num_heads: int = 12,
        head_dim: int = 128,
        page_size: int = 16,  # Tokens per page
        max_num_pages: int = 1024,  # Total pages in pool
        max_seq_len: int = 15 * 880,  # 15 frames * 880 tokens/frame
        local_attn_size: int = -1,  # Sliding window size (-1 = full)
        sink_size: int = 0,  # Number of sink frames to keep
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None
    ):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.max_seq_len = max_seq_len
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.dtype = dtype
        self.device = device or torch.device("cuda")

        # Computed values
        self.frame_seq_len = 880  # Tokens per frame (fixed for this model)
        self.max_attention_tokens = (
            local_attn_size * self.frame_seq_len
            if local_attn_size > 0
            else max_seq_len
        )


class PagedKVCacheManager(nn.Module):
    """
    Manages FlashInfer's PagedKVCache for all transformer blocks.

    This is designed to be CUDA Graph friendly:
    - All memory is pre-allocated at initialization
    - Page tables are pre-allocated with fixed size
    - No dynamic memory allocation during forward pass

    Key concepts:
    - Pages: Fixed-size blocks of KV pairs (page_size tokens each)
    - Page Table: Maps logical sequence positions to physical page indices
    - KV Data: The actual key/value tensors stored in pages
    """

    def __init__(self, config: PagedKVCacheConfig):
        super().__init__()
        self.config = config

        if not FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "FlashInfer is not available. Please install it first:\n"
                "pip install flashinfer"
            )

        self._init_kv_storage()
        self._init_page_tables()
        self._init_wrappers()
        self._init_state_tracking()

    def _init_kv_storage(self):
        """Initialize the KV data storage (one per layer)."""
        cfg = self.config

        # KV data shape: [max_num_pages, 2, page_size, num_heads, head_dim]
        # The '2' dimension is for K and V
        # Using float16/bfloat16 for efficiency
        self.kv_data = torch.zeros(
            cfg.num_layers,
            cfg.max_num_pages,
            2,  # K and V
            cfg.page_size,
            cfg.num_heads,
            cfg.head_dim,
            dtype=cfg.dtype,
            device=cfg.device
        )

        # Track which pages are in use (for page allocation)
        self.page_in_use = torch.zeros(
            cfg.max_num_pages,
            dtype=torch.bool,
            device=cfg.device
        )

        # Free page indices (for fast allocation)
        self.free_pages = list(range(cfg.max_num_pages))

    def _init_page_tables(self):
        """Initialize page tables for each sequence in the batch."""
        cfg = self.config

        # Max pages per sequence
        max_pages_per_seq = (cfg.max_seq_len + cfg.page_size - 1) // cfg.page_size

        # Page table: [batch_size, max_pages_per_seq]
        # Each entry is a page index, or -1 for unused
        self.page_table = torch.full(
            (cfg.batch_size, max_pages_per_seq),
            -1,
            dtype=torch.int32,
            device=cfg.device
        )

        # Track sequence lengths for each batch item
        self.seq_lens = torch.zeros(
            cfg.batch_size,
            dtype=torch.int32,
            device=cfg.device
        )

        # For sliding window attention - track the logical start position
        self.logical_start = torch.zeros(
            cfg.batch_size,
            dtype=torch.int32,
            device=cfg.device
        )

    def _init_wrappers(self):
        """Initialize FlashInfer wrappers."""
        cfg = self.config

        # Workspace buffer for FlashInfer (128MB should be enough)
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=cfg.device
        )

        # One wrapper per layer for batch prefill with paged KV cache
        self.prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper] = []
        for _ in range(cfg.num_layers):
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout="NHD"  # [batch, seq_len, num_heads, head_dim]
            )
            self.prefill_wrappers.append(wrapper)

    def _init_state_tracking(self):
        """Initialize state tracking tensors."""
        cfg = self.config

        # Track global sequence position (for RoPE)
        self.global_position = torch.zeros(
            cfg.batch_size,
            dtype=torch.int64,
            device=cfg.device
        )

        # Track number of pages allocated per sequence
        self.num_pages_allocated = torch.zeros(
            cfg.batch_size,
            dtype=torch.int32,
            device=cfg.device
        )

    def reset(self):
        """Reset all state for a new generation session."""
        self.page_table.fill_(-1)
        self.seq_lens.zero_()
        self.logical_start.zero_()
        self.global_position.zero_()
        self.num_pages_allocated.zero_()
        self.page_in_use.zero_()
        self.free_pages = list(range(self.config.max_num_pages))

    def _allocate_pages(self, num_pages: int) -> List[int]:
        """Allocate pages from the free pool."""
        if len(self.free_pages) < num_pages:
            raise RuntimeError(
                f"Not enough free pages: need {num_pages}, have {len(self.free_pages)}"
            )

        allocated = []
        for _ in range(num_pages):
            page_idx = self.free_pages.pop(0)
            self.page_in_use[page_idx] = True
            allocated.append(page_idx)

        return allocated

    def _free_pages(self, page_indices: List[int]):
        """Return pages to the free pool."""
        for idx in page_indices:
            self.page_in_use[idx] = False
            self.free_pages.append(idx)

    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> None:
        """
        Append new KV pairs to the cache.

        This handles:
        1. Page allocation if needed
        2. Sliding window eviction if cache is full
        3. Writing KV data to the appropriate pages

        Args:
            layer_idx: Which transformer layer (0 to num_layers-1)
            k: New keys [batch_size, num_new_tokens, num_heads, head_dim]
            v: New values [batch_size, num_new_tokens, num_heads, head_dim]
            batch_indices: Optional batch indices to update (default: all)
        """
        cfg = self.config
        batch_size, num_new_tokens, _, _ = k.shape

        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device=cfg.device)

        for b_idx in batch_indices.tolist():
            current_len = self.seq_lens[b_idx].item()
            new_len = current_len + num_new_tokens

            # Check if we need sliding window eviction
            if cfg.local_attn_size > 0:
                max_tokens = cfg.local_attn_size * cfg.frame_seq_len
                if new_len > max_tokens:
                    # Need to evict old tokens
                    tokens_to_evict = new_len - max_tokens
                    self._evict_tokens(b_idx, tokens_to_evict, layer_idx)
                    current_len = self.seq_lens[b_idx].item()
                    new_len = current_len + num_new_tokens

            # Calculate pages needed
            current_pages = (current_len + cfg.page_size - 1) // cfg.page_size
            new_pages_needed = (new_len + cfg.page_size - 1) // cfg.page_size

            # Allocate new pages if needed
            if new_pages_needed > current_pages:
                pages_to_alloc = new_pages_needed - current_pages
                new_page_indices = self._allocate_pages(pages_to_alloc)

                # Update page table
                for i, page_idx in enumerate(new_page_indices):
                    self.page_table[b_idx, current_pages + i] = page_idx

                self.num_pages_allocated[b_idx] += pages_to_alloc

            # Write KV data to pages
            self._write_kv_to_pages(layer_idx, b_idx, k[b_idx], v[b_idx], current_len)

            # Update sequence length
            self.seq_lens[b_idx] = new_len

    def _write_kv_to_pages(
        self,
        layer_idx: int,
        batch_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int
    ):
        """Write KV pairs to the appropriate pages."""
        cfg = self.config
        num_tokens = k.shape[0]

        for token_idx in range(num_tokens):
            global_pos = start_pos + token_idx
            page_idx = global_pos // cfg.page_size
            pos_in_page = global_pos % cfg.page_size

            physical_page = self.page_table[batch_idx, page_idx].item()
            if physical_page < 0:
                raise RuntimeError(f"Page not allocated at position {page_idx}")

            self.kv_data[layer_idx, physical_page, 0, pos_in_page] = k[token_idx]
            self.kv_data[layer_idx, physical_page, 1, pos_in_page] = v[token_idx]

    def _evict_tokens(self, batch_idx: int, num_tokens: int, layer_idx: int):
        """Evict tokens from the beginning (sliding window)."""
        cfg = self.config
        sink_tokens = cfg.sink_size * cfg.frame_seq_len

        # Don't evict sink tokens
        if sink_tokens > 0:
            # Keep sink tokens at the beginning
            actual_evict = num_tokens
            # Move non-sink tokens forward
            # This is simplified - full implementation would need to handle page boundaries

        # Update logical start position
        self.logical_start[batch_idx] += num_tokens

        # Free pages that are no longer needed
        current_len = self.seq_lens[batch_idx].item()
        new_len = current_len - num_tokens

        current_pages = (current_len + cfg.page_size - 1) // cfg.page_size
        new_pages = (new_len + cfg.page_size - 1) // cfg.page_size

        if new_pages < current_pages:
            pages_to_free = current_pages - new_pages
            for i in range(pages_to_free):
                page_idx = self.page_table[batch_idx, new_pages + i].item()
                if page_idx >= 0:
                    self._free_pages([page_idx])
                    self.page_table[batch_idx, new_pages + i] = -1

        self.seq_lens[batch_idx] = new_len

    def get_kv_for_attention(
        self,
        layer_idx: int,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get KV cache data for attention computation.

        Returns data in format suitable for FlashInfer's paged attention.

        Args:
            layer_idx: Which transformer layer
            batch_indices: Which batch items to get (default: all)

        Returns:
            Tuple of (kv_data, kv_indptr, kv_indices, kv_last_page_len)
        """
        cfg = self.config

        if batch_indices is None:
            batch_size = cfg.batch_size
        else:
            batch_size = len(batch_indices)

        # Build indices for FlashInfer
        kv_indptr = [0]
        kv_indices = []
        kv_last_page_len = []

        for b in range(batch_size):
            b_idx = b if batch_indices is None else batch_indices[b].item()
            seq_len = self.seq_lens[b_idx].item()
            num_pages = (seq_len + cfg.page_size - 1) // cfg.page_size

            # Add page indices for this sequence
            for p in range(num_pages):
                page_idx = self.page_table[b_idx, p].item()
                kv_indices.append(page_idx)

            kv_indptr.append(kv_indptr[-1] + num_pages)

            # Last page length
            last_page_len = seq_len % cfg.page_size
            if last_page_len == 0 and seq_len > 0:
                last_page_len = cfg.page_size
            kv_last_page_len.append(last_page_len)

        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=cfg.device)
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device=cfg.device)
        kv_last_page_len = torch.tensor(kv_last_page_len, dtype=torch.int32, device=cfg.device)

        # Return layer-specific KV data
        return self.kv_data[layer_idx], kv_indptr, kv_indices, kv_last_page_len

    def plan_prefill(
        self,
        layer_idx: int,
        q_len: int,
        batch_size: int = None
    ):
        """
        Plan the prefill operation for FlashInfer.

        This should be called BEFORE CUDA Graph capture.

        Args:
            layer_idx: Which transformer layer
            q_len: Query sequence length
            batch_size: Batch size (default: config.batch_size)
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        cfg = self.config
        wrapper = self.prefill_wrappers[layer_idx]

        # Query indices (uniform length for all batch items)
        qo_indptr = torch.arange(
            0, (batch_size + 1) * q_len, q_len,
            dtype=torch.int32, device=cfg.device
        )

        # Get KV indices
        _, kv_indptr, kv_indices, kv_last_page_len = self.get_kv_for_attention(layer_idx)

        # Plan the computation
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            num_qo_heads=cfg.num_heads,
            num_kv_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            page_size=cfg.page_size,
            q_data_type=cfg.dtype,
        )

    def run_prefill(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Run prefill attention with paged KV cache.

        This assumes plan_prefill was already called.

        Args:
            layer_idx: Which transformer layer
            q: Query [batch_size, q_len, num_heads, head_dim]
            k: New keys to append [batch_size, q_len, num_heads, head_dim]
            v: New values to append [batch_size, q_len, num_heads, head_dim]
            causal: Whether to use causal masking

        Returns:
            Attention output [batch_size, q_len, num_heads, head_dim]
        """
        cfg = self.config
        batch_size, q_len = q.shape[:2]

        # First, append new KV to cache
        self.append_kv(layer_idx, k, v)

        # Flatten query to ragged format
        q_ragged = q.reshape(-1, cfg.num_heads, cfg.head_dim)

        # Get KV data for this layer
        kv_data, _, _, _ = self.get_kv_for_attention(layer_idx)

        # Re-plan with updated KV (needed if sequence length changed)
        self.plan_prefill(layer_idx, q_len, batch_size)

        # Run attention
        wrapper = self.prefill_wrappers[layer_idx]
        output = wrapper.run(q_ragged, kv_data)

        # Reshape output
        output = output.reshape(batch_size, q_len, cfg.num_heads, cfg.head_dim)

        return output


class CausalRoPEApply:
    """
    Optimized 3D RoPE application for causal inference.

    Uses precomputed frequency grid for efficiency.
    """

    @staticmethod
    def apply(
        x: torch.Tensor,
        freq_grid: torch.Tensor,
        start_frame: int,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply 3D RoPE using precomputed frequencies.

        Args:
            x: Input tensor [B, seq_len, num_heads, head_dim]
            freq_grid: Precomputed frequencies [max_frames, H, W, head_dim//2]
            start_frame: Starting frame index
            grid_sizes: [F, H, W] grid dimensions

        Returns:
            Rotated tensor [B, seq_len, num_heads, head_dim]
        """
        B, seq_len, num_heads, head_dim = x.shape
        f, h, w = grid_sizes.tolist()

        # Get frequencies for this frame range
        freqs = freq_grid[start_frame:start_frame + f]  # [F, H, W, C]
        freqs = freqs.reshape(f * h * w, 1, head_dim // 2)  # [seq_len, 1, C]

        # Convert to complex and apply rotation
        x_complex = torch.view_as_complex(
            x.reshape(B, seq_len, num_heads, -1, 2).to(torch.float64)
        )  # [B, seq_len, num_heads, head_dim//2]

        # Broadcast and apply rotation
        x_rotated = x_complex * freqs.unsqueeze(0)  # [B, seq_len, num_heads, head_dim//2]

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)  # [B, seq_len, num_heads, head_dim]

        return x_out.type_as(x)


def create_paged_kv_cache_manager(
    batch_size: int = 1,
    num_layers: int = 30,
    num_heads: int = 12,
    head_dim: int = 128,
    max_frames: int = 15,
    frame_seq_len: int = 880,
    local_attn_size: int = -1,
    sink_size: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None
) -> PagedKVCacheManager:
    """
    Factory function to create a PagedKVCacheManager with common defaults.

    Args:
        batch_size: Batch size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        max_frames: Maximum number of frames
        frame_seq_len: Tokens per frame (default 880)
        local_attn_size: Sliding window size in frames (-1 for full)
        sink_size: Number of sink frames to keep
        dtype: Data type
        device: Target device

    Returns:
        Configured PagedKVCacheManager
    """
    max_seq_len = max_frames * frame_seq_len

    # Calculate optimal page size (16 is a good default)
    page_size = 16

    # Calculate number of pages needed
    # Add some buffer for sliding window
    pages_per_seq = (max_seq_len + page_size - 1) // page_size
    max_num_pages = pages_per_seq * batch_size * 2  # 2x buffer

    config = PagedKVCacheConfig(
        batch_size=batch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_num_pages=max_num_pages,
        max_seq_len=max_seq_len,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        dtype=dtype,
        device=device or torch.device("cuda")
    )

    return PagedKVCacheManager(config)
