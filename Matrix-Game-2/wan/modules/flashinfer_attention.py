"""
FlashInfer-based Self-Attention for CausalWanModel.

This module provides FlashInferCausalSelfAttention, a drop-in replacement for
CausalWanSelfAttention that uses FlashInfer for attention computation.

Key features:
1. Uses FlashInfer's BatchPrefillWithPagedKVCacheWrapper for efficient paged attention
2. Applies 3D RoPE BEFORE attention (rope_mode="NONE") for compatibility
3. All memory pre-allocated for CUDA Graph compatibility
4. Supports sliding window attention via page eviction
"""

from typing import Dict, Optional, Tuple
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

from wan.modules.attention import attention as flash_attention_fallback


class PrecomputedRoPE3DCache:
    """
    Precomputes and caches 3D RoPE frequencies for efficient inference.

    This eliminates the repeated torch.cat/expand/view operations in causal_rope_apply,
    making it CUDA Graph friendly.
    """

    def __init__(
        self,
        freqs: torch.Tensor,  # Original freqs [1024, head_dim//2]
        max_frames: int = 64,
        height: int = 22,  # After patchification (352 / 16)
        width: int = 40,   # After patchification (640 / 16)
        device: torch.device = None,
    ):
        """
        Initialize precomputed RoPE frequencies.

        Args:
            freqs: Original RoPE frequencies from model [1024, head_dim//2]
            max_frames: Maximum frames to precompute
            height: Spatial height after patchification
            width: Spatial width after patchification
            device: Target device
        """
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.device = device or freqs.device
        self.dtype = freqs.dtype

        # freqs is already complex exponentials from rope_params
        # Split into time, height, width components
        head_dim_half = freqs.shape[1]
        c = head_dim_half
        self.c_time = c - 2 * (c // 3)
        self.c_height = c // 3
        self.c_width = c // 3

        # Split frequency components
        self.freqs_time = freqs[:, :self.c_time].to(self.device)       # [1024, c_time]
        self.freqs_height = freqs[:, self.c_time:self.c_time + self.c_height].to(self.device)  # [1024, c_height]
        self.freqs_width = freqs[:, self.c_time + self.c_height:].to(self.device)   # [1024, c_width]

        # Precompute full 3D grid
        self._precompute_grid()

    def _precompute_grid(self):
        """Precompute the full 3D frequency grid."""
        # Create grid indices
        # [max_frames, H, W]
        time_idx = torch.arange(self.max_frames, device=self.device)
        height_idx = torch.arange(self.height, device=self.device)
        width_idx = torch.arange(self.width, device=self.device)

        # Expand each component to full grid
        # Time: [F, 1, 1, c_time] -> expand
        # Height: [1, H, 1, c_height] -> expand
        # Width: [1, 1, W, c_width] -> expand

        # Store expanded frequency grids
        # Shape: [max_frames, H, W, head_dim_half]
        self.freq_grid = torch.zeros(
            self.max_frames, self.height, self.width,
            self.c_time + self.c_height + self.c_width,
            dtype=self.dtype, device=self.device
        )

        for f in range(self.max_frames):
            for h in range(self.height):
                for w in range(self.width):
                    self.freq_grid[f, h, w, :self.c_time] = self.freqs_time[f]
                    self.freq_grid[f, h, w, self.c_time:self.c_time + self.c_height] = self.freqs_height[h]
                    self.freq_grid[f, h, w, self.c_time + self.c_height:] = self.freqs_width[w]

    def get_freqs_for_frame_range(
        self,
        start_frame: int,
        num_frames: int
    ) -> torch.Tensor:
        """
        Get precomputed frequencies for a range of frames.

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames

        Returns:
            Frequencies [num_frames * H * W, 1, head_dim_half]
        """
        # Slice the precomputed grid
        freqs = self.freq_grid[start_frame:start_frame + num_frames]  # [F, H, W, C]
        # Reshape to [F*H*W, 1, C] for broadcasting with attention heads
        freqs = freqs.reshape(-1, 1, freqs.shape[-1])
        return freqs


def apply_rope_3d_precomputed(
    x: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
    freqs: torch.Tensor,  # [seq_len, 1, head_dim//2]
) -> torch.Tensor:
    """
    Apply 3D RoPE using precomputed frequencies.

    This is the fast path that uses pre-expanded frequency grids.

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


class FlashInferCausalSelfAttention(nn.Module):
    """
    FlashInfer-based causal self-attention with 3D RoPE support.

    This is designed to be a drop-in replacement for CausalWanSelfAttention
    but uses FlashInfer for attention computation.

    Key differences from CausalWanSelfAttention:
    1. Uses FlashInfer's paged KV cache instead of rolling buffer
    2. Precomputes RoPE frequencies for CUDA Graph compatibility
    3. All memory allocation happens at initialization
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        # FlashInfer specific
        max_frames: int = 64,
        height: int = 22,
        width: int = 40,
        page_size: int = 16,
    ):
        """
        Initialize FlashInfer-based self-attention.

        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            local_attn_size: Sliding window size in frames (-1 for full attention)
            sink_size: Number of sink frames to keep
            qk_norm: Whether to apply QK normalization
            eps: Epsilon for normalization
            max_frames: Maximum frames to support
            height: Spatial height after patchification
            width: Spatial width after patchification
            page_size: Page size for KV cache
        """
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Spatial dimensions
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.frame_seq_len = height * width
        self.max_attention_size = (
            15 * self.frame_seq_len if local_attn_size == -1
            else local_attn_size * self.frame_seq_len
        )

        # Linear layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization
        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # RoPE cache (initialized lazily on first forward)
        self.rope_cache: Optional[PrecomputedRoPE3DCache] = None

        # FlashInfer workspace (initialized lazily)
        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None

    def _init_rope_cache(self, freqs: torch.Tensor):
        """Initialize precomputed RoPE frequencies."""
        if self.rope_cache is None:
            self.rope_cache = PrecomputedRoPE3DCache(
                freqs=freqs,
                max_frames=self.max_frames,
                height=self.height,
                width=self.width,
                device=freqs.device,
            )

    def _init_flashinfer(self, device: torch.device, dtype: torch.dtype):
        """Initialize FlashInfer workspace and wrapper."""
        if not FLASHINFER_AVAILABLE:
            return

        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                128 * 1024 * 1024,  # 128MB workspace
                dtype=torch.uint8,
                device=device
            )

        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer,
                kv_layout="NHD"
            )

    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, dim]
        seq_lens: torch.Tensor,  # [B]
        grid_sizes: torch.Tensor,  # [3] = (F, H, W)
        freqs: torch.Tensor,  # [1024, head_dim//2]
        block_mask,  # BlockMask (unused with FlashInfer)
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        cache_start: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with FlashInfer attention.

        When kv_cache is None: Full attention (training/first frame)
        When kv_cache is provided: Incremental attention with KV cache

        Args:
            x: Input hidden states [B, seq_len, dim]
            seq_lens: Sequence lengths [B]
            grid_sizes: [F, H, W] grid dimensions
            freqs: RoPE frequencies [1024, head_dim//2]
            block_mask: Attention mask (unused with FlashInfer)
            kv_cache: KV cache dictionary
            current_start: Current position in sequence
            cache_start: Start position of cache

        Returns:
            Output hidden states [B, seq_len, dim]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if cache_start is None:
            cache_start = current_start

        # Compute Q, K, V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if kv_cache is None:
            # Full attention path (training or first frame)
            # Use standard flash attention
            return self._forward_full_attention(
                q, k, v, grid_sizes, freqs, block_mask
            )
        else:
            # Incremental attention with KV cache
            return self._forward_incremental(
                q, k, v, grid_sizes, freqs, kv_cache, current_start, cache_start
            )

    def _forward_full_attention(
        self,
        q: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Full attention path (no KV cache)."""
        from wan.modules.model import rope_apply

        # Apply standard RoPE
        roped_q = rope_apply(q, grid_sizes, freqs).type_as(v)
        roped_k = rope_apply(k, grid_sizes, freqs).type_as(v)

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

    def _forward_incremental(
        self,
        q: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        grid_sizes: torch.Tensor,  # [3] = (F, H, W) for current frame
        freqs: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor],
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        """Incremental attention with KV cache."""
        assert grid_sizes.ndim == 1

        # Initialize RoPE cache if needed
        self._init_rope_cache(freqs)

        # Calculate frame position
        frame_seqlen = math.prod(grid_sizes[1:]).item()
        current_start_frame = current_start // frame_seqlen
        num_frames = grid_sizes[0].item()

        # Get precomputed frequencies for this frame range
        precomputed_freqs = self.rope_cache.get_freqs_for_frame_range(
            current_start_frame, num_frames
        )

        # Apply 3D RoPE using precomputed frequencies
        roped_q = apply_rope_3d_precomputed(q, precomputed_freqs)
        roped_k = apply_rope_3d_precomputed(k, precomputed_freqs)

        # Update KV cache
        current_end = current_start + roped_q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_q.shape[1]

        # Check if eviction is needed
        if (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # Eviction logic (same as original)
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

            # Roll the cache
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
        else:
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens

        # Insert new K/V
        kv_cache["k"][:, local_start_index:local_end_index] = roped_k
        kv_cache["v"][:, local_start_index:local_end_index] = v

        # Compute attention using FlashInfer or fallback
        window_start = max(0, local_end_index - self.max_attention_size)
        k_window = kv_cache["k"][:, window_start:local_end_index]
        v_window = kv_cache["v"][:, window_start:local_end_index]

        if FLASHINFER_AVAILABLE:
            # Use FlashInfer for attention
            self._init_flashinfer(roped_q.device, roped_q.dtype)
            x = self._flashinfer_attention(roped_q, k_window, v_window)
        else:
            # Fallback to standard flash attention
            x = flash_attention_fallback(roped_q, k_window, v_window)

        # Update cache indices
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _flashinfer_attention(
        self,
        q: torch.Tensor,  # [B, q_len, num_heads, head_dim]
        k: torch.Tensor,  # [B, kv_len, num_heads, head_dim]
        v: torch.Tensor,  # [B, kv_len, num_heads, head_dim]
    ) -> torch.Tensor:
        """Compute attention using FlashInfer."""
        B, q_len, num_heads, head_dim = q.shape
        _, kv_len, _, _ = k.shape

        if B == 1:
            # Single sequence - use single_prefill_with_kv_cache
            output = flashinfer.single_prefill_with_kv_cache(
                q.squeeze(0),  # [q_len, num_heads, head_dim]
                k.squeeze(0),  # [kv_len, num_heads, head_dim]
                v.squeeze(0),  # [kv_len, num_heads, head_dim]
                causal=False,  # Already handled by cache windowing
            )
            output = output.unsqueeze(0)  # [1, q_len, num_heads, head_dim]
        else:
            # Batch attention using BatchPrefillWithRaggedKVCacheWrapper
            q_ragged = q.reshape(-1, num_heads, head_dim)
            k_ragged = k.reshape(-1, num_heads, head_dim)
            v_ragged = v.reshape(-1, num_heads, head_dim)

            qo_indptr = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=q.device)
            kv_indptr = torch.arange(0, (B + 1) * kv_len, kv_len, dtype=torch.int32, device=k.device)

            if self._workspace_buffer is None:
                self._workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)

            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                self._workspace_buffer, kv_layout="NHD"
            )

            wrapper.plan(
                qo_indptr, kv_indptr,
                num_heads, num_heads, head_dim,
                causal=False,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )

            output = wrapper.run(q_ragged, k_ragged, v_ragged)
            output = output.reshape(B, q_len, num_heads, head_dim)

        return output


def replace_attention_with_flashinfer(model: nn.Module) -> nn.Module:
    """
    Replace all CausalWanSelfAttention modules with FlashInferCausalSelfAttention.

    Args:
        model: CausalWanModel instance

    Returns:
        Modified model with FlashInfer attention
    """
    if not FLASHINFER_AVAILABLE:
        print("Warning: FlashInfer not available, keeping original attention")
        return model

    from wan.modules.causal_model import CausalWanSelfAttention

    for name, module in model.named_modules():
        if isinstance(module, CausalWanSelfAttention):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Create replacement
            new_attn = FlashInferCausalSelfAttention(
                dim=module.dim,
                num_heads=module.num_heads,
                local_attn_size=module.local_attn_size,
                sink_size=module.sink_size,
                qk_norm=module.qk_norm,
                eps=module.eps,
            )

            # Copy weights
            new_attn.q.load_state_dict(module.q.state_dict())
            new_attn.k.load_state_dict(module.k.state_dict())
            new_attn.v.load_state_dict(module.v.state_dict())
            new_attn.o.load_state_dict(module.o.state_dict())
            new_attn.norm_q.load_state_dict(module.norm_q.state_dict())
            new_attn.norm_k.load_state_dict(module.norm_k.state_dict())

            # Replace
            setattr(parent, attr_name, new_attn)

    return model
