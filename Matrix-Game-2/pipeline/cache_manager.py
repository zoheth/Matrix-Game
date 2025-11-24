"""
KV Cache management for the causal inference pipeline.

This module encapsulates all logic related to creating, initializing,
and resetting KV caches and cross-attention caches.

Supports two cache modes:
1. Dict-based cache (legacy): Uses torch tensors with rolling buffer eviction
2. PagedCache (FlashInfer): Uses paged memory management for efficient FlashInfer integration
"""

from typing import List, Dict, Optional, Union
import torch

from pipeline.config import ModelConfig, CacheConfig

# Try to import PagedCache for FlashInfer support
try:
    from wan.modules.paged_cache import PagedCache, PagedCacheManager
    PAGED_CACHE_AVAILABLE = True
except ImportError:
    PAGED_CACHE_AVAILABLE = False
    PagedCache = None
    PagedCacheManager = None


class CacheManager:
    """
    Manages KV caches for the transformer model.

    This class handles:
    - Visual self-attention cache
    - Mouse action conditioning cache
    - Keyboard action conditioning cache
    - Cross-attention cache

    Supports two modes:
    - Dict-based cache (legacy): Standard torch tensor caches
    - PagedCache (FlashInfer): Paged memory management for FlashInfer

    The caches are structured as lists of dictionaries/PagedCache, one per transformer block.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
        dtype: torch.dtype,
        use_paged_cache: bool = False,
        page_size: int = 16,
    ):
        """
        Initialize the cache manager.

        Args:
            model_config: Model architecture configuration
            cache_config: Cache configuration
            device: Device to place caches on
            dtype: Data type for cache tensors
            use_paged_cache: Whether to use PagedCache for FlashInfer
            page_size: Page size for PagedCache (only used if use_paged_cache=True)
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = dtype
        self.use_paged_cache = use_paged_cache and PAGED_CACHE_AVAILABLE
        self.page_size = page_size

        if use_paged_cache and not PAGED_CACHE_AVAILABLE:
            print("[WARNING] PagedCache requested but not available, falling back to dict-based cache")

        # Cache storage (type depends on use_paged_cache)
        self.visual_cache: Optional[Union[List[Dict[str, torch.Tensor]], List[PagedCache]]] = None
        self.mouse_cache: Optional[List[Dict[str, torch.Tensor]]] = None
        self.keyboard_cache: Optional[List[Dict[str, torch.Tensor]]] = None
        self.crossattn_cache: Optional[List[Dict[str, torch.Tensor]]] = None

        # PagedCacheManager for FlashInfer (if enabled)
        self._paged_cache_manager: Optional[PagedCacheManager] = None

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        """
        Initialize all caches with zeros.

        Args:
            batch_size: Batch size for cache tensors
        """
        if self.use_paged_cache:
            self.visual_cache = self._create_paged_visual_cache(batch_size)
        else:
            self.visual_cache = self._create_visual_cache(batch_size)

        # Action and cross-attention caches always use dict-based approach
        self.mouse_cache = self._create_mouse_cache(batch_size)
        self.keyboard_cache = self._create_keyboard_cache(batch_size)
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

    def _create_visual_cache(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create the visual self-attention KV cache.

        Args:
            batch_size: Batch size

        Returns:
            List of cache dictionaries, one per transformer block
        """
        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append({
                "k": torch.zeros(
                    [batch_size, cache_size, num_heads, head_dim],
                    dtype=self.dtype,
                    device=self.device
                ),
                "v": torch.zeros(
                    [batch_size, cache_size, num_heads, head_dim],
                    dtype=self.dtype,
                    device=self.device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device)
            })

        return cache

    def _create_action_cache_base(
        self,
        batch_size: int,
        batch_multiplier: int = 1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create a base action conditioning cache (shared logic for mouse/keyboard).

        Args:
            batch_size: Base batch size
            batch_multiplier: Multiplier for batch dimension (1 for keyboard, frame_seq_length for mouse)

        Returns:
            List of cache dictionaries
        """
        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim
        effective_batch = batch_size * batch_multiplier

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append({
                "k": torch.zeros(
                    [effective_batch, cache_size, num_heads, head_dim],
                    dtype=self.dtype,
                    device=self.device
                ),
                "v": torch.zeros(
                    [effective_batch, cache_size, num_heads, head_dim],
                    dtype=self.dtype,
                    device=self.device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device)
            })

        return cache

    def _create_mouse_cache(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create the mouse action conditioning cache.

        The mouse cache has a special structure where the batch dimension
        is multiplied by frame_seq_length.

        Args:
            batch_size: Batch size

        Returns:
            List of cache dictionaries
        """
        return self._create_action_cache_base(
            batch_size,
            batch_multiplier=self.model_config.frame_seq_length
        )

    def _create_keyboard_cache(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create the keyboard action conditioning cache.

        Args:
            batch_size: Batch size

        Returns:
            List of cache dictionaries
        """
        return self._create_action_cache_base(batch_size, batch_multiplier=1)

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

        # Reset visual self-attention cache
        if self.use_paged_cache:
            # PagedCache has its own reset method
            for block_cache in self.visual_cache:
                block_cache.reset()
        else:
            # Dict-based cache
            for block_cache in self.visual_cache:
                block_cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=self.device)
                block_cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=self.device)

        # Reset mouse cache (always dict-based)
        for block_cache in self.mouse_cache:
            block_cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=self.device)
            block_cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=self.device)

        # Reset keyboard cache (always dict-based)
        for block_cache in self.keyboard_cache:
            block_cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=self.device)
            block_cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=self.device)

    def get_caches(self) -> tuple[
        Union[List[Dict[str, torch.Tensor]], List[PagedCache]],
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]]
    ]:
        """
        Get all caches.

        Returns:
            Tuple of (visual_cache, mouse_cache, keyboard_cache, crossattn_cache)
            visual_cache can be either List[Dict] or List[PagedCache] depending on use_paged_cache
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
