from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("Warning: flashinfer not available, falling back to flash_attn")

class IActionPreprocessor(nn.Module, ABC):
    """ (B, N_frames, C) -> (B, T_q_or_k, C_windowed)"""
    def __init__(self, vae_time_compression_ratio: int, windows_size: int):
        super().__init__()
        # self.vae_time_compression_ratio = vae_time_compression_ratio
        # self.windows_size = windows_size
        self.pat_t = vae_time_compression_ratio * windows_size
        
    @abstractmethod
    def forward(self, condition: torch.Tensor, N_feats: int, is_causal: bool, num_frame_per_block: int) -> torch.Tensor:
        pass
    
class IAttentionInjector(nn.Module, ABC):
    """"""
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass
    
class KVCacheManager(nn.Module):
    """Manages KV cache with sliding window and optional sink tokens"""
    def __init__(self, local_attn_size: int, sink_size: int = 0):
        super().__init__()
        self.max_attention_size = local_attn_size
        self.sink_tokens = sink_size

    def update_cache(
        self,
        kv_cache: Dict[str, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        num_new_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Update KV cache with new key-value pairs using sliding window strategy.

        Args:
            kv_cache: Dictionary containing 'k', 'v', 'global_end_index', 'local_end_index'
            k: New keys [BS, num_new_tokens, num_heads, head_dim]
            v: New values [BS, num_new_tokens, num_heads, head_dim]
            num_new_tokens: Number of new tokens to add

        Returns:
            k_window: Keys in attention window [BS, window_len, num_heads, head_dim]
            v_window: Values in attention window [BS, window_len, num_heads, head_dim]
            local_start_index: Start index in cache
            local_end_index: End index in cache
        """
        current_start = kv_cache["global_end_index"].item()
        current_end = current_start + num_new_tokens

        kv_cache_size = kv_cache["k"].shape[1]
        sink_tokens = self.sink_tokens

        # Check if we need to evict tokens
        if (current_end > kv_cache["global_end_index"].item()) and \
           (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # Calculate eviction
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

            # Roll the cache: move recent tokens to make space
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
        else:
            # No eviction needed
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens

        # Insert new keys/values
        kv_cache["k"][:, local_start_index:local_end_index] = k
        kv_cache["v"][:, local_start_index:local_end_index] = v

        # Update global indices
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # Extract attention window
        window_start = max(0, local_end_index - self.max_attention_size)
        k_window = kv_cache["k"][:, window_start:local_end_index]
        v_window = kv_cache["v"][:, window_start:local_end_index]

        return k_window, v_window, local_start_index, local_end_index


class IAttentionCore(nn.Module, ABC):
    """Abstract base class for attention computation"""

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        use_rope: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention output.

        Args:
            q: Query tensor [BS, seq_len_q, num_heads, head_dim]
            k: Key tensor [BS, seq_len_k, num_heads, head_dim]
            v: Value tensor [BS, seq_len_k, num_heads, head_dim]
            causal: Whether to apply causal masking
            use_rope: Whether to apply RoPE on-the-fly

        Returns:
            Attention output [BS, seq_len_q, num_heads, head_dim]
        """
        pass


class WanRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim] tensor
        Returns:
            Normalized tensor with same shape as input
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class FlashInferAttentionCore(IAttentionCore):
    """
    FlashInfer-based attention implementation.

    Note: RoPE should be applied externally before calling this module.
    This implementation focuses on efficient attention computation.
    """

    def __init__(self):
        super().__init__()
        # Wrapper for batch prefill (created lazily on first use)
        self._batch_wrapper = None
        self._workspace_buffer = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        use_rope: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention using FlashInfer.

        Args:
            q: [BS, seq_len_q, num_heads, head_dim] - Query (RoPE already applied)
            k: [BS, seq_len_k, num_heads, head_dim] - Key (RoPE already applied)
            v: [BS, seq_len_k, num_heads, head_dim] - Value
            causal: Whether to apply causal masking
            use_rope: Ignored (RoPE should be applied externally)

        Returns:
            Attention output [BS, seq_len_q, num_heads, head_dim]
        """
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("flashinfer is not available. Please install it first.")

        BS, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_kv, _, _ = k.shape

        if BS == 1:
            # Single sequence - use single_prefill_with_kv_cache
            # Expected shape: [seq_len, num_heads, head_dim]
            q_single = q.squeeze(0)  # [seq_len_q, num_heads, head_dim]
            k_single = k.squeeze(0)  # [seq_len_kv, num_heads, head_dim]
            v_single = v.squeeze(0)  # [seq_len_kv, num_heads, head_dim]

            output = flashinfer.single_prefill_with_kv_cache(
                q_single, k_single, v_single,
                causal=causal,
                use_fp16_qk_reduction=False,
            )
            # Output shape: [seq_len_q, num_heads, head_dim]
            # Add back batch dimension
            output = output.unsqueeze(0)  # [1, seq_len_q, num_heads, head_dim]
        else:
            # Multiple sequences - use BatchPrefillWithRaggedKVCacheWrapper
            # Flatten to ragged format: [BS, L, H, D] -> [BS*L, H, D]
            total_q_len = BS * seq_len_q
            total_kv_len = BS * seq_len_kv

            q_ragged = q.reshape(total_q_len, num_heads, head_dim)
            k_ragged = k.reshape(total_kv_len, num_heads, head_dim)
            v_ragged = v.reshape(total_kv_len, num_heads, head_dim)

            # Create ragged tensor indices: qo_indptr and kv_indptr
            # Both [BS+1], marking start/end positions of each sequence
            qo_indptr = torch.arange(
                0, (BS + 1) * seq_len_q, seq_len_q,
                dtype=torch.int32, device=q.device
            )
            kv_indptr = torch.arange(
                0, (BS + 1) * seq_len_kv, seq_len_kv,
                dtype=torch.int32, device=k.device
            )

            # Initialize wrapper if needed (allocate workspace buffer)
            if self._batch_wrapper is None or self._workspace_buffer is None:
                # Allocate 128MB workspace buffer (can be adjusted)
                self._workspace_buffer = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=q.device
                )
                self._batch_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    self._workspace_buffer, kv_layout="NHD"
                )

            # Plan the attention computation (creates auxiliary data structures)
            self._batch_wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_heads,
                num_heads,  # Assume num_kv_heads == num_qo_heads for now
                head_dim,
                causal=causal,
            )

            # Run the attention computation
            output = self._batch_wrapper.run(q_ragged, k_ragged, v_ragged)

            # Reshape back to [BS, seq_len_q, num_heads, head_dim]
            output = output.reshape(BS, seq_len_q, num_heads, head_dim)

        return output