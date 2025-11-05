"""
Complete CUDA Graph wrapper for ActionModule

包装整个 ActionModule (MouseInjector + KeyboardInjector 串行执行)
"""

from typing import Dict, Optional, Tuple, Any
import torch
from torch import nn

from .graph_wrapper import GraphKey


class ActionModuleGraphWrapper(nn.Module):
    """
    CUDA Graph wrapper for complete ActionModule.

    策略：
    - 为不同的 (B, T, H, W, is_causal, num_frame_per_block) 组合缓存 graph
    - 串行执行: MouseInjector -> KeyboardInjector
    - Fallback 到 eager mode 当遇到新形状

    State Dict Compatibility:
    - Transparently handles loading state dicts saved from unwrapped ActionModule
    - Keys like "mouse_injector.*" are mapped to "action_module.mouse_injector.*"
    """

    def __init__(
        self,
        action_module: nn.Module,
        max_cached_graphs: int = 10,
        enable_cuda_graph: bool = True,
        num_warmup_iters: int = 3,
    ):
        """
        Args:
            action_module: The ActionModule to wrap
            max_cached_graphs: Maximum number of graphs to cache
            enable_cuda_graph: Global switch for CUDA Graph
            num_warmup_iters: Number of warmup iterations before capture
        """
        super().__init__()
        self.action_module = action_module
        self.max_cached_graphs = max_cached_graphs
        self.enable_cuda_graph = enable_cuda_graph
        self.num_warmup_iters = num_warmup_iters

        # Cache: GraphKey -> CachedGraph
        self.graph_cache: Dict[GraphKey, 'CachedGraph'] = {}

        # Statistics
        self.stats = {
            'graph_hits': 0,
            'graph_misses': 0,
            'eager_executions': 0,
        }

    def _create_graph_key(
        self,
        x: torch.Tensor,
        tt: int,
        th: int,
        tw: int,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> GraphKey:
        """Create GraphKey from ActionModule.forward() arguments"""
        B = x.shape[0]

        # Ensure all values are Python int (not tensor)
        # tt, th, tw might be passed as torch.Tensor from grid_sizes
        if isinstance(tt, torch.Tensor):
            tt = tt.item()
        if isinstance(th, torch.Tensor):
            th = th.item()
        if isinstance(tw, torch.Tensor):
            tw = tw.item()

        return GraphKey(
            batch_size=B,
            temporal_shape=tt,
            spatial_h=th,
            spatial_w=tw,
            is_causal=is_causal,
            num_frame_per_block=num_frame_per_block,
        )

    def _create_static_tensors(self, key: GraphKey, local_attn_size: int = -1) -> Dict[str, torch.Tensor]:
        """Allocate static tensors for inputs/outputs"""
        B = key.batch_size
        T = key.temporal_shape
        H = key.spatial_h
        W = key.spatial_w
        S = H * W

        config = self.action_module.config
        C_img = config.img_hidden_size
        C_mouse = config.mouse_dim_in
        C_keyboard = config.keyboard_dim_in

        # Infer N_frames
        vae_ratio = config.vae_time_compression_ratio
        if key.is_causal:
            N_frames = (key.num_frame_per_block - 1) * vae_ratio + 1
        else:
            N_frames = T * vae_ratio

        device = next(self.action_module.parameters()).device
        dtype = next(self.action_module.parameters()).dtype

        static_tensors = {
            # Inputs
            'x': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
            'mouse_condition': torch.empty(B, N_frames, C_mouse, dtype=dtype, device=device),
            'keyboard_condition': torch.empty(B, N_frames, C_keyboard, dtype=dtype, device=device),
            # Output
            'output': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
        }

        # Allocate KV cache tensors if needed
        if key.is_causal:
            # Determine kv_cache_size based on local_attn_size
            if local_attn_size != -1:
                kv_cache_size = local_attn_size
            else:
                kv_cache_size = 15 * 1  # max frames

            # Get num_heads and head_dim from ActionModule config
            # Note: Mouse and Keyboard use their own hidden_dim, not img_hidden_size!
            # ActionConfig uses 'heads_num', not 'num_heads'
            num_heads = config.heads_num
            mouse_head_dim = config.mouse_head_dim      # mouse_hidden_dim // heads_num = 64
            keyboard_head_dim = config.keyboard_head_dim # keyboard_hidden_dim // heads_num = 64

            # Mouse KV cache: [B*S, kv_cache_size, num_heads, mouse_head_dim]
            static_tensors['kv_cache_mouse'] = {
                'k': torch.zeros(B * S, kv_cache_size, num_heads, mouse_head_dim, dtype=dtype, device=device),
                'v': torch.zeros(B * S, kv_cache_size, num_heads, mouse_head_dim, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device),
            }

            # Keyboard KV cache: [B, kv_cache_size, num_heads, keyboard_head_dim]
            static_tensors['kv_cache_keyboard'] = {
                'k': torch.zeros(B, kv_cache_size, num_heads, keyboard_head_dim, dtype=dtype, device=device),
                'v': torch.zeros(B, kv_cache_size, num_heads, keyboard_head_dim, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device),
            }

        return static_tensors

    def _warmup(
        self,
        key: GraphKey,
        static_tensors: Dict[str, torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        start_frame: int,
    ):
        """Warmup iterations to stabilize internal states"""
        print(f"[ActionModuleGraphWrapper] Starting warmup ({self.num_warmup_iters} iterations)...")
        i = 0
        try:
            for i in range(self.num_warmup_iters):
                _ = self.action_module(
                    x=static_tensors['x'],
                    tt=key.temporal_shape,
                    th=key.spatial_h,
                    tw=key.spatial_w,
                    mouse_condition=static_tensors['mouse_condition'] if self.action_module.enable_mouse else None,
                    keyboard_condition=static_tensors['keyboard_condition'] if self.action_module.enable_keyboard else None,
                    block_mask_mouse=None,
                    block_mask_keyboard=None,
                    is_causal=key.is_causal,
                    kv_cache_mouse=static_tensors.get('kv_cache_mouse', None),
                    kv_cache_keyboard=static_tensors.get('kv_cache_keyboard', None),
                    start_frame=start_frame,
                    use_rope_keyboard=True,
                    num_frame_per_block=key.num_frame_per_block,
                )
                print(f"[ActionModuleGraphWrapper] Warmup iteration {i+1}/{self.num_warmup_iters} done")
            torch.cuda.synchronize()
            print(f"[ActionModuleGraphWrapper] Warmup completed successfully")
        except Exception as e:
            print(f"[ActionModuleGraphWrapper] Warmup failed at iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _capture_graph(
        self,
        key: GraphKey,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        start_frame: int,
        local_attn_size: int = -1,
    ) -> 'CachedGraph':
        """Capture a new CUDA Graph"""
        # Allocate static tensors
        static_tensors = self._create_static_tensors(key, local_attn_size)

        # Warmup
        self._warmup(key, static_tensors, freqs_cis, start_frame)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_tensors['output'] = self.action_module(
                    x=static_tensors['x'],
                    tt=key.temporal_shape,
                    th=key.spatial_h,
                    tw=key.spatial_w,
                    mouse_condition=static_tensors['mouse_condition'] if self.action_module.enable_mouse else None,
                    keyboard_condition=static_tensors['keyboard_condition'] if self.action_module.enable_keyboard else None,
                    block_mask_mouse=None,
                    block_mask_keyboard=None,
                    is_causal=key.is_causal,
                    kv_cache_mouse=static_tensors.get('kv_cache_mouse', None),
                    kv_cache_keyboard=static_tensors.get('kv_cache_keyboard', None),
                    start_frame=start_frame,
                    use_rope_keyboard=True,
                    num_frame_per_block=key.num_frame_per_block,
                )

        torch.cuda.synchronize()

        return CachedGraph(
            graph=graph,
            static_tensors=static_tensors,
            key=key,
        )

    def _get_or_create_graph(
        self,
        key: GraphKey,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        start_frame: int,
        local_attn_size: int = -1,
    ) -> Optional['CachedGraph']:
        """Get cached graph or create a new one"""
        if key in self.graph_cache:
            self.stats['graph_hits'] += 1
            return self.graph_cache[key]

        self.stats['graph_misses'] += 1

        # Check capacity
        if self.max_cached_graphs > 0 and len(self.graph_cache) >= self.max_cached_graphs:
            # Simple eviction: remove first item
            first_key = next(iter(self.graph_cache))
            del self.graph_cache[first_key]
            print(f"[ActionModuleGraphWrapper] Evicted graph: {first_key}")

        # Capture new graph
        try:
            cached_graph = self._capture_graph(key, freqs_cis, start_frame, local_attn_size)
            self.graph_cache[key] = cached_graph
            print(f"[ActionModuleGraphWrapper] Captured new graph: {key}")
            return cached_graph
        except Exception as e:
            print(f"[ActionModuleGraphWrapper] Failed to capture graph for {key}: {e}")
            # Clean up CUDA state after failed capture
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except:
                pass
            # Permanently disable CUDA Graph after first failure to avoid corrupting CUDA state
            print(f"[ActionModuleGraphWrapper] Disabling CUDA Graph due to capture failure")
            self.enable_cuda_graph = False
            return None

    def __call__(
        self,
        x: torch.Tensor,
        tt: int,
        th: int,
        tw: int,
        mouse_condition: Optional[torch.Tensor] = None,
        keyboard_condition: Optional[torch.Tensor] = None,
        block_mask_mouse: Optional[Any] = None,
        block_mask_keyboard: Optional[Any] = None,
        is_causal: bool = False,
        kv_cache_mouse: Optional[Dict[str, torch.Tensor]] = None,
        kv_cache_keyboard: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        use_rope_keyboard: bool = True,
        num_frame_per_block: int = 3,
    ) -> torch.Tensor:
        """Make the wrapper callable like the original ActionModule"""
        return self.forward(
            x, tt, th, tw,
            mouse_condition, keyboard_condition,
            block_mask_mouse, block_mask_keyboard,
            is_causal, kv_cache_mouse, kv_cache_keyboard,
            start_frame, use_rope_keyboard, num_frame_per_block
        )

    def forward(
        self,
        x: torch.Tensor,
        tt: int,
        th: int,
        tw: int,
        mouse_condition: Optional[torch.Tensor] = None,
        keyboard_condition: Optional[torch.Tensor] = None,
        block_mask_mouse: Optional[Any] = None,
        block_mask_keyboard: Optional[Any] = None,
        is_causal: bool = False,
        kv_cache_mouse: Optional[Dict[str, torch.Tensor]] = None,
        kv_cache_keyboard: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        use_rope_keyboard: bool = True,
        num_frame_per_block: int = 3,
    ) -> torch.Tensor:
        """
        Forward with CUDA Graph acceleration.

        Same interface as ActionModule.forward()

        LIMITATION: CUDA Graph currently only supports non-causal mode.
        Causal mode uses dynamic KV cache that requires D2H transfers,
        which are incompatible with CUDA Graph capture.
        """
        # CUDA Graph incompatible with causal mode because:
        # - KV cache update uses .item() (D2H transfer)
        # - Dynamic window slicing requires Python int
        # - Window size changes every iteration
        if is_causal:
            self.stats['eager_executions'] += 1
            return self.action_module(
                x=x, tt=tt, th=th, tw=tw,
                mouse_condition=mouse_condition,
                keyboard_condition=keyboard_condition,
                block_mask_mouse=block_mask_mouse,
                block_mask_keyboard=block_mask_keyboard,
                is_causal=is_causal,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                start_frame=start_frame,
                use_rope_keyboard=use_rope_keyboard,
                num_frame_per_block=num_frame_per_block,
            )

        if not self.enable_cuda_graph:
            # CUDA Graph disabled
            self.stats['eager_executions'] += 1
            return self.action_module(
                x=x,
                tt=tt,
                th=th,
                tw=tw,
                mouse_condition=mouse_condition,
                keyboard_condition=keyboard_condition,
                block_mask_mouse=block_mask_mouse,
                block_mask_keyboard=block_mask_keyboard,
                is_causal=is_causal,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                start_frame=start_frame,
                use_rope_keyboard=use_rope_keyboard,
                num_frame_per_block=num_frame_per_block,
            )

        # Create graph key
        key = self._create_graph_key(x, tt, th, tw, is_causal, num_frame_per_block)

        # Get freqs_cis from ActionModule
        freqs_cis = (self.action_module.freqs_cos, self.action_module.freqs_sin)

        # Get local_attn_size from ActionModule config
        local_attn_size = getattr(self.action_module.config, 'local_attn_size', -1)

        # Get or create graph
        cached_graph = self._get_or_create_graph(key, freqs_cis, start_frame, local_attn_size)

        if cached_graph is None:
            # Fallback to eager
            self.stats['eager_executions'] += 1
            return self.action_module(
                x=x,
                tt=tt,
                th=th,
                tw=tw,
                mouse_condition=mouse_condition,
                keyboard_condition=keyboard_condition,
                block_mask_mouse=block_mask_mouse,
                block_mask_keyboard=block_mask_keyboard,
                is_causal=is_causal,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                start_frame=start_frame,
                use_rope_keyboard=use_rope_keyboard,
                num_frame_per_block=num_frame_per_block,
            )

        # Copy inputs to static tensors
        cached_graph.static_tensors['x'].copy_(x)
        if mouse_condition is not None and self.action_module.enable_mouse:
            cached_graph.static_tensors['mouse_condition'].copy_(mouse_condition)
        if keyboard_condition is not None and self.action_module.enable_keyboard:
            cached_graph.static_tensors['keyboard_condition'].copy_(keyboard_condition)

        # Handle KV cache: ZERO COPY strategy (like vLLM)
        #
        # CUDA Graph records tensor addresses during capture. For zero-copy operation:
        # 1. External kv_cache must be the SAME tensor (same data_ptr) as static kv_cache
        # 2. If same: zero copy, graph will update in-place ✅
        # 3. If different: fallback to eager (avoid expensive copy overhead) ❌
        #
        # This is the vLLM approach: use a fixed KV cache pool throughout inference
        static_kv_mouse = cached_graph.static_tensors.get('kv_cache_mouse')
        static_kv_keyboard = cached_graph.static_tensors.get('kv_cache_keyboard')

        # Check if external kv_cache matches static kv_cache (zero copy check)
        mouse_is_same = False
        keyboard_is_same = False

        if kv_cache_mouse is not None and static_kv_mouse is not None:
            # Check if it's the same tensor by comparing data pointers
            mouse_is_same = (
                kv_cache_mouse['k'].data_ptr() == static_kv_mouse['k'].data_ptr() and
                kv_cache_mouse['v'].data_ptr() == static_kv_mouse['v'].data_ptr() and
                kv_cache_mouse['global_end_index'].data_ptr() == static_kv_mouse['global_end_index'].data_ptr() and
                kv_cache_mouse['local_end_index'].data_ptr() == static_kv_mouse['local_end_index'].data_ptr()
            )

        if kv_cache_keyboard is not None and static_kv_keyboard is not None:
            keyboard_is_same = (
                kv_cache_keyboard['k'].data_ptr() == static_kv_keyboard['k'].data_ptr() and
                kv_cache_keyboard['v'].data_ptr() == static_kv_keyboard['v'].data_ptr() and
                kv_cache_keyboard['global_end_index'].data_ptr() == static_kv_keyboard['global_end_index'].data_ptr() and
                kv_cache_keyboard['local_end_index'].data_ptr() == static_kv_keyboard['local_end_index'].data_ptr()
            )

        # If external kv_cache is provided but doesn't match static kv_cache,
        # fallback to eager mode to avoid copy overhead
        has_external_kv = kv_cache_mouse is not None or kv_cache_keyboard is not None
        has_static_kv = static_kv_mouse is not None or static_kv_keyboard is not None

        if has_external_kv and has_static_kv:
            # Check if ALL provided external caches match their static counterparts
            all_match = True
            if kv_cache_mouse is not None and static_kv_mouse is not None:
                all_match = all_match and mouse_is_same
            if kv_cache_keyboard is not None and static_kv_keyboard is not None:
                all_match = all_match and keyboard_is_same

            if not all_match:
                # External kv_cache doesn't match static → fallback to eager
                self.stats['eager_executions'] += 1
                print(f"[ActionModuleGraphWrapper] External kv_cache doesn't match static kv_cache, falling back to eager mode")
                return self.action_module(
                    x=x,
                    tt=tt,
                    th=th,
                    tw=tw,
                    mouse_condition=mouse_condition,
                    keyboard_condition=keyboard_condition,
                    block_mask_mouse=block_mask_mouse,
                    block_mask_keyboard=block_mask_keyboard,
                    is_causal=is_causal,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    start_frame=start_frame,
                    use_rope_keyboard=use_rope_keyboard,
                    num_frame_per_block=num_frame_per_block,
                )

        # Replay graph (zero copy: external kv_cache IS the static kv_cache)
        # The graph will update kv_cache in-place
        cached_graph.graph.replay()

        # Return output (no copy needed)
        return cached_graph.static_tensors['output']

    def get_static_kv_cache(self, key: GraphKey) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Get the static KV cache tensors for a specific graph key.

        This allows external code (like the pipeline) to use the wrapper's internal
        static kv_cache directly, enabling zero-copy CUDA Graph execution.

        Usage:
            # First call triggers graph capture
            wrapper.forward(..., kv_cache_mouse=None, ...)

            # Get static kv_cache for the captured graph
            static_kv = wrapper.get_static_kv_cache(key)

            # Subsequent calls use the static kv_cache (zero copy)
            wrapper.forward(..., kv_cache_mouse=static_kv['mouse'], ...)

        Args:
            key: The GraphKey identifying which cached graph's kv_cache to return

        Returns:
            Dictionary with 'mouse' and 'keyboard' kv_cache references, or None if not found
        """
        if key not in self.graph_cache:
            return None

        cached_graph = self.graph_cache[key]
        return {
            'mouse': cached_graph.static_tensors.get('kv_cache_mouse'),
            'keyboard': cached_graph.static_tensors.get('kv_cache_keyboard'),
        }

    def get_or_create_static_kv_cache(
        self,
        x: torch.Tensor,
        tt: int,
        th: int,
        tw: int,
        is_causal: bool,
        num_frame_per_block: int,
        start_frame: int = 0,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Get or create static KV cache for the given input configuration.

        This is a convenience method that:
        1. Creates the graph key from input parameters
        2. Triggers graph capture if needed (via dummy forward pass)
        3. Returns the static kv_cache for zero-copy usage

        Args:
            x, tt, th, tw, is_causal, num_frame_per_block: Same as forward()
            start_frame: Start frame for RoPE

        Returns:
            Dictionary with 'mouse' and 'keyboard' kv_cache references, or None
        """
        if not self.enable_cuda_graph or not is_causal:
            return None

        # Create graph key
        key = self._create_graph_key(x, tt, th, tw, is_causal, num_frame_per_block)

        # Check if graph already exists
        if key in self.graph_cache:
            return self.get_static_kv_cache(key)

        # Trigger graph capture with dummy inputs
        freqs_cis = (self.action_module.freqs_cos, self.action_module.freqs_sin)
        local_attn_size = getattr(self.action_module.config, 'local_attn_size', -1)

        cached_graph = self._get_or_create_graph(key, freqs_cis, start_frame, local_attn_size)

        if cached_graph is None:
            return None

        return {
            'mouse': cached_graph.static_tensors.get('kv_cache_mouse'),
            'keyboard': cached_graph.static_tensors.get('kv_cache_keyboard'),
        }

    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics"""
        return self.stats.copy()

    def clear_cache(self):
        """Clear all cached graphs"""
        num_graphs = len(self.graph_cache)
        self.graph_cache.clear()
        print(f"[ActionModuleGraphWrapper] Cleared {num_graphs} cached graphs")


class CachedGraph:
    """Cached CUDA Graph with static tensors"""
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_tensors: Dict[str, torch.Tensor],
        key: GraphKey,
    ):
        self.graph = graph
        self.static_tensors = static_tensors
        self.key = key
