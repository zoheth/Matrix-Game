"""
KV Cache management for the causal inference pipeline.

This module encapsulates all logic related to creating, initializing,
and resetting KV caches and cross-attention caches.

Visual cache uses PagedCache for efficient FlashInfer integration.
Action caches use ActionCache for token-level eviction with batch support.
"""

from typing import List, Dict, Optional
import torch

from pipeline.config import ModelConfig, CacheConfig
from wan.modules.paged_cache import PagedCache, PagedCacheManager
from wan.modules.action_cache import ActionCache


class CacheManager:
    """
    Manages KV caches for the transformer model.

    This class handles:
    - Visual self-attention cache (PagedCache for FlashInfer)
    - Mouse action conditioning cache (ActionCache - supports spatial batching)
    - Keyboard action conditioning cache (ActionCache - token-level eviction)
    - Cross-attention cache (simple tensor storage)

    Visual cache uses PagedCache for page-granular operations with FlashInfer.
    Action caches use ActionCache for token-level eviction with arbitrary batch sizes.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
        dtype: torch.dtype,
        page_size: int = 16,
    ):
        """
        Initialize the cache manager.

        Args:
            model_config: Model architecture configuration
            cache_config: Cache configuration
            device: Device to place caches on
            dtype: Data type for cache tensors
            page_size: Page size for PagedCache
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = dtype
        self.page_size = page_size

        # Cache storage
        self.visual_cache: Optional[List[PagedCache]] = None
        self.mouse_cache: Optional[List[ActionCache]] = None
        self.keyboard_cache: Optional[List[ActionCache]] = None
        self.crossattn_cache: Optional[List[Dict[str, torch.Tensor]]] = None

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        """
        Initialize all caches.

        Args:
            batch_size: Batch size for cache tensors
        """
        # Visual cache uses PagedCache (only supports batch_size=1)
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"
        self.visual_cache = self._create_paged_visual_cache(batch_size)

        # Action caches use ActionCache (supports arbitrary batch sizes)
        self.mouse_cache = self._create_action_mouse_cache(batch_size)
        self.keyboard_cache = self._create_action_keyboard_cache(batch_size)

        # Cross-attention cache (simple dict storage)
        self.crossattn_cache = self._create_crossattn_cache(batch_size)

    def _create_paged_visual_cache(self, batch_size: int) -> List[PagedCache]:
        """
        Create paged visual self-attention KV cache for FlashInfer.

        Args:
            batch_size: Batch size (currently only 1 is supported)

        Returns:
            List of PagedCache instances, one per transformer block
        """
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"

        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(PagedCache(
                max_total_tokens=cache_size,
                page_size=self.page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                sink_size=0,  # TODO: Get from config if needed
                dtype=self.dtype,
                device=self.device,
            ))

        return cache


    def _create_action_mouse_cache(self, batch_size: int) -> List[ActionCache]:
        """
        Create ActionCache for mouse action conditioning.

        Mouse self-attention uses spatial batching [B*S, T, H, D] where S is the
        number of spatial tokens (frame_seq_length). ActionCache pre-allocates
        memory for the full batch dimension.

        Args:
            batch_size: Base batch size (will be multiplied by spatial dimension)

        Returns:
            List of ActionCache instances, one per transformer block
        """
        # Mouse cache needs B * S where S is spatial dimension
        mouse_batch_size = batch_size * self.model_config.frame_seq_length

        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(ActionCache(
                batch_size=mouse_batch_size,
                max_seq_len=cache_size,
                num_heads=num_heads,
                head_dim=head_dim,
                device=self.device,
                dtype=self.dtype,
            ))

        return cache

    def _create_action_keyboard_cache(self, batch_size: int) -> List[ActionCache]:
        """
        Create ActionCache for keyboard action conditioning.

        Keyboard self-attention uses simple batching [B, T, H, D].
        ActionCache pre-allocates memory for the batch dimension.

        Args:
            batch_size: Batch size

        Returns:
            List of ActionCache instances, one per transformer block
        """
        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(ActionCache(
                batch_size=batch_size,
                max_seq_len=cache_size,
                num_heads=num_heads,
                head_dim=head_dim,
                device=self.device,
                dtype=self.dtype,
            ))

        return cache

    def _create_crossattn_cache(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create the cross-attention cache for visual context (CLIP features).

        Args:
            batch_size: Batch size

        Returns:
            List of cache dictionaries
        """
        seq_len = self.model_config.cross_attention_seq_length
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append({
                "k": torch.zeros(
                    [batch_size, seq_len, num_heads, head_dim],
                    dtype=self.dtype,
                    device=self.device
                ),
                "v": torch.zeros(
                    [batch_size, seq_len, num_heads, head_dim],
                    dtype=self.dtype,
                    device=self.device
                ),
                "is_init": False
            })

        return cache

    def reset_all_caches(self) -> None:
        """
        Reset all caches to their initial state.

        This is called at the start of each new generation sequence.
        Does NOT reallocate memory, just resets indices and flags.
        """
        if self.visual_cache is None:
            raise RuntimeError("Caches must be initialized before resetting")

        # Reset cross-attention cache
        for block_cache in self.crossattn_cache:
            block_cache["is_init"] = False

        # Reset all PagedCache instances
        for block_cache in self.visual_cache:
            block_cache.reset()

        for block_cache in self.mouse_cache:
            block_cache.reset()

        for block_cache in self.keyboard_cache:
            block_cache.reset()

    def get_caches(self) -> tuple[
        List[PagedCache],
        List[ActionCache],
        List[ActionCache],
        List[Dict[str, torch.Tensor]]
    ]:
        """
        Get all caches.

        Returns:
            Tuple of (visual_cache, mouse_cache, keyboard_cache, crossattn_cache)
            - visual_cache: PagedCache instances (page-granular eviction)
            - mouse_cache: ActionCache instances (token-level, spatial batching)
            - keyboard_cache: ActionCache instances (token-level, simple batching)
            - crossattn_cache: Simple dict storage
        """
        if (self.visual_cache is None or self.mouse_cache is None or
            self.keyboard_cache is None or self.crossattn_cache is None):
            raise RuntimeError("Caches must be initialized before access")

        return (
            self.visual_cache,
            self.mouse_cache,
            self.keyboard_cache,
            self.crossattn_cache
        )

    def is_initialized(self) -> bool:
        """Check if caches have been initialized."""
        return self.visual_cache is not None
