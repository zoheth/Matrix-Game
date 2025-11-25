"""
KV Cache management for the causal inference pipeline.

This module encapsulates all logic related to creating, initializing,
and resetting KV caches and cross-attention caches.

All caches now use PagedCache for efficient FlashInfer integration.
"""

from typing import List, Dict, Optional
import torch

from pipeline.config import ModelConfig, CacheConfig
from wan.modules.paged_cache import PagedCache, PagedCacheManager


class CacheManager:
    """
    Manages KV caches for the transformer model.

    This class handles:
    - Visual self-attention cache (PagedCache for FlashInfer)
    - Mouse action conditioning cache (PagedCache)
    - Keyboard action conditioning cache (PagedCache)
    - Cross-attention cache (simple tensor storage)

    All visual and action caches use PagedCache for efficient FlashInfer integration.
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

        # Cache storage - all use PagedCache except cross-attention
        self.visual_cache: Optional[List[PagedCache]] = None
        self.mouse_cache: Optional[List[PagedCache]] = None
        self.keyboard_cache: Optional[List[PagedCache]] = None
        self.crossattn_cache: Optional[List[Dict[str, torch.Tensor]]] = None

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        """
        Initialize all caches with PagedCache.

        Args:
            batch_size: Batch size for cache tensors (currently only 1 is supported)
        """
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"

        # Initialize all caches with PagedCache
        self.visual_cache = self._create_paged_visual_cache(batch_size)
        self.mouse_cache = self._create_paged_mouse_cache(batch_size)
        self.keyboard_cache = self._create_paged_keyboard_cache(batch_size)
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


    def _create_paged_mouse_cache(self, batch_size: int) -> List[PagedCache]:
        """
        Create paged mouse action conditioning cache.

        Args:
            batch_size: Batch size (currently only 1 is supported)

        Returns:
            List of PagedCache instances, one per transformer block
        """
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"

        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(PagedCache(
                max_total_tokens=cache_size,
                page_size=self.page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                sink_size=0,
                dtype=self.dtype,
                device=self.device,
            ))

        return cache

    def _create_paged_keyboard_cache(self, batch_size: int) -> List[PagedCache]:
        """
        Create paged keyboard action conditioning cache.

        Args:
            batch_size: Batch size (currently only 1 is supported)

        Returns:
            List of PagedCache instances, one per transformer block
        """
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"

        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(PagedCache(
                max_total_tokens=cache_size,
                page_size=self.page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                sink_size=0,
                dtype=self.dtype,
                device=self.device,
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
        List[PagedCache],
        List[PagedCache],
        List[Dict[str, torch.Tensor]]
    ]:
        """
        Get all caches.

        Returns:
            Tuple of (visual_cache, mouse_cache, keyboard_cache, crossattn_cache)
            All caches except cross-attention use PagedCache.
        """
        if self.visual_cache is None:
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
