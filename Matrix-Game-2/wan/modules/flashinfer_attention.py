"""
FlashInfer-based Causal Self-Attention - Refactored from first principles.

This module provides a clean implementation of FlashInfer-based attention
with 3D RoPE support. All historical baggage has been removed.

Design principles (from first principles):
1. grid_sizes is a simple tuple (F, H, W), not a Tensor - no unnecessary GPU-CPU sync
2. Planning happens AFTER cache update - plan reads current cache state
3. Each class has a single, clear responsibility
4. All unused parameters removed (seq_lens, cache_start, max_frames, device)
5. Computation logic remains IDENTICAL to original implementation

Architecture:
- RoPE3DCache: On-demand 3D RoPE frequency computation (memory-efficient)
- apply_rope_3d: Apply 3D RoPE using precomputed frequencies
- FlashInferPlanner: Manages plan state (called once per generation step)
- CausalSelfAttention: Handles attention with conditional planning

Planning strategy:
- First layer: Updates cache → evicts pages → plans (reads fresh cache state) → runs
- Later layers: Skip planning (reuse plan from first layer) → run
- This ensures plan is based on current cache state and shared across all layers

Key features:
- FlashInfer's BatchPrefillWithPagedKVCacheWrapper for efficient paged attention
- 3D RoPE applied BEFORE attention for compatibility
- CUDA Graph friendly (all memory pre-allocated)
- Sliding window attention via page eviction
- Plan executed ONCE per generation step (in first layer) and shared across all layers
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    BatchPrefillWithPagedKVCacheWrapper = None

from wan.modules.attention import flash_attention as flash_attention_fallback

from .paged_cache import PagedCache


class RoPE3DCache:
    """
    On-demand 3D RoPE frequency computation for efficient inference.

    Instead of precomputing the full grid (which causes OOM for long videos),
    this class stores only the 1D frequency components and computes the 3D grid
    on-demand using vectorized operations.

    This is both memory-efficient and CUDA Graph friendly.
    """

    def __init__(
        self,
        freqs: torch.Tensor,  # [max_positions, head_dim//2]
        height: int,
        width: int,
    ):
        """
        Initialize RoPE frequency cache.

        Args:
            freqs: RoPE frequencies [max_positions, head_dim//2]
            height: Spatial height after patchification
            width: Spatial width after patchification
        """
        self.height = height
        self.width = width
        self.device = freqs.device
        self.dtype = freqs.dtype

        # freqs is already complex exponentials from rope_params
        # Split into time, height, width components
        head_dim_half = freqs.shape[1]
        c = head_dim_half
        self.c_time = c - 2 * (c // 3)
        self.c_height = c // 3
        self.c_width = c // 3
        self.head_dim_half = head_dim_half

        # Store only the 1D frequency components (memory efficient!)
        # These are small: [1024, c_time/c_height/c_width] each
        self.freqs_time = freqs[:, :self.c_time].to(self.device)       # [1024, c_time]
        self.freqs_height = freqs[:, self.c_time:self.c_time + self.c_height].to(self.device)  # [1024, c_height]
        self.freqs_width = freqs[:, self.c_time + self.c_height:].to(self.device)   # [1024, c_width]

    def get_freqs_for_frame_range(
        self,
        start_frame: int,
        num_frames: int
    ) -> torch.Tensor:
        """
        Compute frequencies for a range of frames on-demand.

        Uses vectorized operations instead of precomputing the entire grid.
        This is memory-efficient while still being fast.

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames

        Returns:
            Frequencies [num_frames * H * W, 1, head_dim_half]
        """
        end_frame = start_frame + num_frames

        # Validate bounds against freqs_time length (typically 1024)
        if end_frame > self.freqs_time.shape[0]:
            raise ValueError(
                f"Frame range [{start_frame}:{end_frame}] exceeds frequency table size {self.freqs_time.shape[0]}. "
                f"The model's RoPE frequencies don't support this many frames."
            )

        f, h, w = num_frames, self.height, self.width

        # Compute 3D grid using vectorized operations (same logic as causal_rope_apply)
        # Time component: [F, 1, 1, c_time] -> [F, H, W, c_time]
        time_freqs = self.freqs_time[start_frame:end_frame].view(f, 1, 1, -1).expand(f, h, w, -1)

        # Height component: [1, H, 1, c_height] -> [F, H, W, c_height]
        height_freqs = self.freqs_height[:h].view(1, h, 1, -1).expand(f, h, w, -1)

        # Width component: [1, 1, W, c_width] -> [F, H, W, c_width]
        width_freqs = self.freqs_width[:w].view(1, 1, w, -1).expand(f, h, w, -1)

        # Concatenate along last dimension: [F, H, W, head_dim_half]
        freqs = torch.cat([time_freqs, height_freqs, width_freqs], dim=-1)

        # Reshape to [F*H*W, 1, head_dim_half] for broadcasting with attention heads
        freqs = freqs.reshape(-1, 1, self.head_dim_half)

        return freqs


def apply_rope_3d(
    x: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
    freqs: torch.Tensor,  # [seq_len, 1, head_dim//2]
) -> torch.Tensor:
    """
    Apply 3D RoPE using precomputed frequencies.

    Args:
        x: Input tensor [B, seq_len, num_heads, head_dim]
        freqs: Precomputed frequencies [seq_len, 1, head_dim//2]

    Returns:
        Rotated tensor with same shape as x
    """
    B, seq_len, num_heads, head_dim = x.shape

    # Reshape x to complex: [B, seq_len, num_heads, head_dim//2, 2] -> complex
    x_complex = torch.view_as_complex(
        x.reshape(B, seq_len, num_heads, head_dim // 2, 2).to(torch.float64)
    )  # [B, seq_len, num_heads, head_dim//2]

    # Apply rotation (broadcast over batch and heads)
    # freqs: [seq_len, 1, head_dim//2]
    # x_complex: [B, seq_len, num_heads, head_dim//2]
    x_rotated = x_complex * freqs.unsqueeze(0)  # [B, seq_len, num_heads, head_dim//2]

    # Convert back to real
    x_out = torch.view_as_real(x_rotated).flatten(-2)  # [B, seq_len, num_heads, head_dim]

    return x_out.type_as(x)


class FlashInferPlanner:
    """
    Manages FlashInfer plan state for a generation step.

    This class is responsible for:
    1. Initializing FlashInfer workspace buffer and wrapper
    2. Executing plan() once per generation step
    3. Providing the planned wrapper to all attention layers

    Usage:
        planner = FlashInferPlanner(num_heads, head_dim, page_size)
        planner.init(device, dtype)

        # Once per generation step, after KV cache is updated:
        planner.plan(kv_cache, q_len)

        # All layers use the same planned wrapper:
        output = planner.run(q, kv_cache)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int,
        workspace_size: int = 128 * 1024 * 1024,  # 128MB
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.workspace_size = workspace_size

        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
        self._is_planned = False

    def init(self, device: torch.device) -> None:
        """Initialize FlashInfer workspace and wrapper."""
        if not FLASHINFER_AVAILABLE:
            return

        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                self.workspace_size,
                dtype=torch.uint8,
                device=device
            )

        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer,
                kv_layout="NHD"
            )

    def plan(
        self,
        kv_cache: PagedCache,
        q_len: int,
        device: torch.device,
        q_dtype: torch.dtype,
    ) -> None:
        """
        Execute plan for the current generation step.

        This should be called ONCE per generation step, before any attention layers.
        All layers will share this plan.

        Args:
            kv_cache: The PagedCache containing KV data (any layer's cache works,
                     as they all have the same structure)
            q_len: Query sequence length
            device: Device for tensors
            q_dtype: Query data type
        """
        if not FLASHINFER_AVAILABLE:
            self._is_planned = False
            return

        self.init(device)

        # Get FlashInfer metadata from cache
        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len = kv_cache.get_flashinfer_meta(device)

        # Build qo_indptr for single batch
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)

        # Plan the paged attention
        self._prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=False,  # Already handled by cache management
            q_data_type=q_dtype,
        )

        self._is_planned = True

    def run(
        self,
        q: torch.Tensor,  # [q_len, num_heads, head_dim]
        kv_cache: PagedCache,
    ) -> torch.Tensor:
        """
        Run paged attention using the pre-computed plan.

        Args:
            q: Query tensor [q_len, num_heads, head_dim]
            kv_cache: Layer-specific PagedCache

        Returns:
            Attention output [q_len, num_heads, head_dim]
        """
        if not self._is_planned:
            raise RuntimeError("FlashInferPlanner.plan() must be called before run()")

        return self._prefill_wrapper.run(
            q,
            (kv_cache.k_cache, kv_cache.v_cache),
        )

    @property
    def is_available(self) -> bool:
        return FLASHINFER_AVAILABLE

    @property
    def is_planned(self) -> bool:
        return self._is_planned

    def reset(self) -> None:
        """Reset plan state. Call at the start of each generation step if needed."""
        self._is_planned = False


class CausalSelfAttention(nn.Module):
    """
    FlashInfer-based causal self-attention with 3D RoPE support.

    Clean design from first principles:
    - grid_sizes is a simple tuple (F, H, W), not a Tensor
    - Planning is conditional: only first layer plans (after cache update)
    - All unused parameters removed (seq_lens, cache_start, max_frames, device)
    - Computation logic identical to original implementation

    Planning behavior:
    - First layer in each generation step:
      1. Updates KV cache with new tokens
      2. Evicts old pages (sliding window)
      3. Calls planner.plan() with fresh cache state
      4. Runs attention
    - Later layers: Skip planning, reuse plan from first layer

    This ensures:
    - Plan is based on current cache state (not stale state)
    - Plan is computed only once per generation step (not per layer)
    - All layers share the same plan (cache structure is identical)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        height: int = 22,
        width: int = 40,
        page_size: int = 16,
    ):
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        # Spatial dimensions
        self.height = height
        self.width = width
        self.frame_seq_len = height * width
        self.max_attention_size = (
            15 * self.frame_seq_len if local_attn_size == -1
            else local_attn_size * self.frame_seq_len
        )
        self.page_size = page_size

        # Linear layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization
        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # RoPE cache (initialized lazily on first forward)
        self.rope_cache: Optional[RoPE3DCache] = None

    def _init_rope_cache(self, freqs: torch.Tensor):
        """Initialize RoPE cache."""
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(
                freqs=freqs,
                height=self.height,
                width=self.width,
            )

    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, dim]
        grid_sizes: Tuple[int, int, int],  # (F, H, W)
        freqs: torch.Tensor,  # [1024, head_dim//2]
        block_mask,  # BlockMask (unused with FlashInfer)
        kv_cache: Optional[PagedCache] = None,
        current_start: int = 0,
        planner: Optional[FlashInferPlanner] = None,
    ) -> torch.Tensor:
        """
        Forward pass with FlashInfer attention.

        Planning is handled automatically:
        - First layer: Will call planner.plan() after updating cache
        - Later layers: Will reuse plan from first layer

        Args:
            x: Input hidden states [B, seq_len, dim]
            grid_sizes: (F, H, W) grid dimensions as tuple (not Tensor!)
            freqs: RoPE frequencies [1024, head_dim//2]
            block_mask: Attention mask (for full attention path only)
            kv_cache: PagedCache for incremental attention
            current_start: Current position in sequence
            planner: FlashInferPlanner (will be planned in first layer if needed)

        Returns:
            Output hidden states [B, seq_len, dim]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Compute Q, K, V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if kv_cache is None:
            # Full attention path (training or first frame)
            return self._forward_full_attention(q, k, v, grid_sizes, freqs, block_mask)
        else:
            # Incremental attention with PagedCache
            return self._forward_incremental_paged(
                q, k, v, grid_sizes, freqs, kv_cache, current_start, planner
            )

    def _forward_full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid_sizes: Tuple[int, int, int],
        freqs: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Full attention path (no KV cache)."""
        from wan.modules.model import rope_apply

        # Apply standard RoPE (rope_apply expects tensor, so convert tuple)
        grid_tensor = torch.tensor(grid_sizes, device=q.device)
        roped_q = rope_apply(q, grid_tensor, freqs).type_as(v)
        roped_k = rope_apply(k, grid_tensor, freqs).type_as(v)

        # Padding for FlexAttention
        padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
        if padded_length > 0:
            pad_shape = [q.shape[0], padded_length, q.shape[2], q.shape[3]]
            roped_q = torch.cat([roped_q, torch.zeros(pad_shape, device=q.device, dtype=v.dtype)], dim=1)
            roped_k = torch.cat([roped_k, torch.zeros(pad_shape, device=k.device, dtype=v.dtype)], dim=1)
            v_padded = torch.cat([v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)], dim=1)
        else:
            v_padded = v

        # Use FlexAttention with block mask
        from torch.nn.attention.flex_attention import flex_attention
        x = flex_attention(
            query=roped_q.transpose(2, 1),
            key=roped_k.transpose(2, 1),
            value=v_padded.transpose(2, 1),
            block_mask=block_mask
        )

        if padded_length > 0:
            x = x[:, :, :-padded_length]
        x = x.transpose(2, 1)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _forward_incremental_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid_sizes: Tuple[int, int, int],  # (F, H, W)
        freqs: torch.Tensor,
        kv_cache: PagedCache,
        current_start: int,
        planner: Optional[FlashInferPlanner],
    ) -> torch.Tensor:
        """
        Incremental attention with PagedCache using FlashInfer.

        Uses pre-computed plan from FlashInferPlanner.
        """
        B = q.shape[0]
        assert B == 1, "PagedCache currently only supports batch size 1"

        # Initialize RoPE cache if needed
        self._init_rope_cache(freqs)

        # Calculate frame position (grid_sizes is now a simple tuple!)
        num_frames, height, width = grid_sizes
        frame_seqlen = height * width
        current_start_frame = current_start // frame_seqlen

        # Get precomputed frequencies for this frame range
        precomputed_freqs = self.rope_cache.get_freqs_for_frame_range(
            current_start_frame, num_frames
        )

        # Apply 3D RoPE using precomputed frequencies
        roped_q = apply_rope_3d(q, precomputed_freqs)
        roped_k = apply_rope_3d(k, precomputed_freqs)

        # Squeeze batch dimension for PagedCache (batch=1)
        roped_k_squeezed = roped_k.squeeze(0)  # [seq_len, num_heads, head_dim]
        v_squeezed = v.squeeze(0)

        # Calculate current_end for update_or_append
        current_end = current_start + roped_k_squeezed.shape[0]

        # Update cache: handle denoising (overwrite) vs new frame (append)
        kv_cache.update_or_append(roped_k_squeezed, v_squeezed, current_start, current_end)

        # Evict old pages if needed (sliding window)
        kv_cache.evict(self.max_attention_size)

        # Compute attention
        if planner is not None and planner.is_available:
            # Plan if not already planned (first layer does this after cache update)
            # All layers have the same cache structure, so plan is reusable
            if not planner.is_planned:
                # IMPORTANT: This is called AFTER cache.update_or_append() and cache.evict()
                # so the cache state is up-to-date
                planner.plan(
                    kv_cache=kv_cache,
                    q_len=roped_q.shape[1],
                    device=roped_q.device,
                    q_dtype=roped_q.dtype,
                )

            # Use FlashInfer with pre-computed plan
            q_for_flash = roped_q.squeeze(0)  # [q_len, num_heads, head_dim]
            x = planner.run(q_for_flash, kv_cache)
            x = x.unsqueeze(0)  # [1, q_len, num_heads, head_dim]
        else:
            # Fallback: get contiguous K/V and use standard attention
            k_contiguous, v_contiguous = kv_cache.get_kv_for_attention()
            k_contiguous = k_contiguous.unsqueeze(0)
            v_contiguous = v_contiguous.unsqueeze(0)
            x = flash_attention_fallback(roped_q, k_contiguous, v_contiguous)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x
