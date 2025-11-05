"""
CUDA Graph wrappers for MouseInjector and KeyboardInjector
"""

from typing import Dict, Optional, Tuple, Any
import torch
from torch import nn

from .graph_wrapper import CUDAGraphWrapper, GraphKey


class MouseInjectorGraphWrapper(CUDAGraphWrapper):
    """
    CUDA Graph wrapper for MouseInjector.

    Handles:
    - Mouse preprocessing (Triton kernel)
    - MLP forward
    - QKV projection
    - RoPE + Attention (FlashInfer)
    - Output projection
    """

    def __init__(
        self,
        mouse_injector: nn.Module,
        max_cached_graphs: int = 10,
        enable_cuda_graph: bool = True,
        num_warmup_iters: int = 3,
    ):
        """
        Args:
            mouse_injector: The MouseInjector module to wrap
            max_cached_graphs: Maximum number of graphs to cache
            enable_cuda_graph: Global switch for CUDA Graph
            num_warmup_iters: Number of warmup iterations before capture
        """
        super().__init__(max_cached_graphs, enable_cuda_graph, num_warmup_iters)
        self.injector = mouse_injector

    def _create_graph_key(self, **kwargs) -> GraphKey:
        """Create GraphKey from MouseInjector.forward() arguments"""
        x = kwargs['x']
        condition = kwargs['condition']
        temporal_shape = kwargs['temporal_shape']
        spatial_shape = kwargs['spatial_shape']
        is_causal = kwargs.get('is_causal', False)
        num_frame_per_block = kwargs.get('num_frame_per_block', 1)

        B = x.shape[0]
        H, W = spatial_shape

        return GraphKey(
            batch_size=B,
            temporal_shape=temporal_shape,
            spatial_h=H,
            spatial_w=W,
            is_causal=is_causal,
            num_frame_per_block=num_frame_per_block,
        )

    def _create_static_inputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static input tensors"""
        B = key.batch_size
        T = key.temporal_shape
        H = key.spatial_h
        W = key.spatial_w
        S = H * W

        # Get dimensions from injector config
        C_img = self.injector.action_config.img_hidden_size
        C_mouse = self.injector.action_config.mouse_dim_in

        # Infer N_frames from typical usage (can be adjusted)
        # For now, assume N_frames is proportional to T
        vae_ratio = self.injector.action_config.vae_time_compression_ratio
        if key.is_causal:
            N_frames = (key.num_frame_per_block - 1) * vae_ratio + 1
        else:
            N_frames = T * vae_ratio

        device = next(self.injector.parameters()).device
        dtype = next(self.injector.parameters()).dtype

        static_inputs = {
            'x': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
            'condition': torch.empty(B, N_frames, C_mouse, dtype=dtype, device=device),
            'start_frame': 0,  # scalar, will be stored separately
        }

        return static_inputs

    def _create_static_outputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static output tensors"""
        B = key.batch_size
        T = key.temporal_shape
        H = key.spatial_h
        W = key.spatial_w
        S = H * W
        C_img = self.injector.action_config.img_hidden_size

        device = next(self.injector.parameters()).device
        dtype = next(self.injector.parameters()).dtype

        static_outputs = {
            'output': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
        }

        return static_outputs

    def _copy_inputs_to_static(
        self,
        static_inputs: Dict[str, torch.Tensor],
        dynamic_inputs: Dict[str, Any],
    ):
        """Copy dynamic inputs to static tensors"""
        static_inputs['x'].copy_(dynamic_inputs['x'])
        static_inputs['condition'].copy_(dynamic_inputs['condition'])
        static_inputs['start_frame'] = dynamic_inputs.get('start_frame', 0)

    def _copy_outputs_from_static(
        self,
        static_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return output (no copy needed, static tensor is safe to use)"""
        # Note: we return the static output directly since it will be
        # consumed immediately. If needed, can clone here.
        return static_outputs['output']

    def _execute_graph_body(
        self,
        static_inputs: Dict[str, torch.Tensor],
        static_outputs: Dict[str, torch.Tensor],
    ):
        """Execute MouseInjector forward (the core computation)"""
        # Get static inputs
        x = static_inputs['x']
        condition = static_inputs['condition']
        start_frame = static_inputs['start_frame']

        # Reconstruct spatial shape from x
        B, T_S, C_img = x.shape
        # Need to get T, H, W from somewhere - we can store them in the key
        # For now, we'll need to pass them through. Let's add them to static_inputs.
        # Actually, we need to refactor this slightly.

        # For now, let's use a simpler approach: store all necessary scalar params
        # in static_inputs during copy phase
        T = static_inputs['temporal_shape']
        H = static_inputs['spatial_h']
        W = static_inputs['spatial_w']
        is_causal = static_inputs['is_causal']
        num_frame_per_block = static_inputs['num_frame_per_block']

        # Get freqs_cis from injector (assume it's pre-computed and cached)
        freqs_cis = static_inputs['freqs_cis']

        # Execute injector forward
        output = self.injector.forward(
            x=x,
            condition=condition,
            freqs_cis=freqs_cis,
            spatial_shape=(H, W),
            temporal_shape=T,
            is_causal=is_causal,
            kv_cache=static_inputs.get('kv_cache'),
            start_frame=start_frame,
            num_frame_per_block=num_frame_per_block,
            block_mask=None,
        )

        # Write to static output
        static_outputs['output'].copy_(output)

    def _execute_eager(self, **kwargs) -> torch.Tensor:
        """Execute in eager mode (direct call to injector)"""
        return self.injector.forward(**kwargs)

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
        """
        Forward with CUDA Graph acceleration.

        Same interface as MouseInjector.forward()
        """
        if condition is None:
            return x

        # Prepare kwargs for graph execution
        kwargs = {
            'x': x,
            'condition': condition,
            'freqs_cis': freqs_cis,
            'spatial_shape': spatial_shape,
            'temporal_shape': temporal_shape,
            'is_causal': is_causal,
            'kv_cache': kv_cache,
            'start_frame': start_frame,
            'num_frame_per_block': num_frame_per_block,
            'block_mask': block_mask,
        }

        # Use base class method
        return self.forward_with_graph(**kwargs)

    def _create_static_inputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static input tensors (revised to include all needed data)"""
        B = key.batch_size
        T = key.temporal_shape
        H = key.spatial_h
        W = key.spatial_w
        S = H * W

        # Get dimensions from injector config
        C_img = self.injector.action_config.img_hidden_size
        C_mouse = self.injector.action_config.mouse_dim_in

        # Infer N_frames
        vae_ratio = self.injector.action_config.vae_time_compression_ratio
        if key.is_causal:
            N_frames = (key.num_frame_per_block - 1) * vae_ratio + 1
        else:
            N_frames = T * vae_ratio

        device = next(self.injector.parameters()).device
        dtype = next(self.injector.parameters()).dtype

        static_inputs = {
            'x': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
            'condition': torch.empty(B, N_frames, C_mouse, dtype=dtype, device=device),
            'freqs_cis': (None, None),  # Will be set during copy
            'start_frame': 0,
            'temporal_shape': T,
            'spatial_h': H,
            'spatial_w': W,
            'is_causal': key.is_causal,
            'num_frame_per_block': key.num_frame_per_block,
            'kv_cache': None,  # Will be set during copy if provided
        }

        return static_inputs

    def _copy_inputs_to_static(
        self,
        static_inputs: Dict[str, torch.Tensor],
        dynamic_inputs: Dict[str, Any],
    ):
        """Copy dynamic inputs to static tensors (revised)"""
        static_inputs['x'].copy_(dynamic_inputs['x'])
        static_inputs['condition'].copy_(dynamic_inputs['condition'])
        static_inputs['freqs_cis'] = dynamic_inputs['freqs_cis']
        static_inputs['start_frame'] = dynamic_inputs.get('start_frame', 0)
        static_inputs['kv_cache'] = dynamic_inputs.get('kv_cache')
        # Scalars are already set in _create_static_inputs


class KeyboardInjectorGraphWrapper(CUDAGraphWrapper):
    """
    CUDA Graph wrapper for KeyboardInjector.

    Similar structure to MouseInjectorGraphWrapper.
    """

    def __init__(
        self,
        keyboard_injector: nn.Module,
        max_cached_graphs: int = 10,
        enable_cuda_graph: bool = True,
        num_warmup_iters: int = 3,
    ):
        super().__init__(max_cached_graphs, enable_cuda_graph, num_warmup_iters)
        self.injector = keyboard_injector

    def _create_graph_key(self, **kwargs) -> GraphKey:
        """Create GraphKey from KeyboardInjector.forward() arguments"""
        x = kwargs['x']
        temporal_shape = kwargs['temporal_shape']
        spatial_shape = kwargs['spatial_shape']
        is_causal = kwargs.get('is_causal', False)
        num_frame_per_block = kwargs.get('num_frame_per_block', 1)

        B = x.shape[0]
        H, W = spatial_shape

        return GraphKey(
            batch_size=B,
            temporal_shape=temporal_shape,
            spatial_h=H,
            spatial_w=W,
            is_causal=is_causal,
            num_frame_per_block=num_frame_per_block,
        )

    def _create_static_inputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static input tensors"""
        B = key.batch_size
        T = key.temporal_shape
        H = key.spatial_h
        W = key.spatial_w
        S = H * W

        C_img = self.injector.action_config.img_hidden_size
        C_keyboard = self.injector.action_config.keyboard_dim_in

        vae_ratio = self.injector.action_config.vae_time_compression_ratio
        if key.is_causal:
            N_frames = (key.num_frame_per_block - 1) * vae_ratio + 1
        else:
            N_frames = T * vae_ratio

        device = next(self.injector.parameters()).device
        dtype = next(self.injector.parameters()).dtype

        static_inputs = {
            'x': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
            'condition': torch.empty(B, N_frames, C_keyboard, dtype=dtype, device=device),
            'freqs_cis': (None, None),
            'start_frame': 0,
            'temporal_shape': T,
            'spatial_h': H,
            'spatial_w': W,
            'is_causal': key.is_causal,
            'num_frame_per_block': key.num_frame_per_block,
            'kv_cache': None,
        }

        return static_inputs

    def _create_static_outputs(self, key: GraphKey) -> Dict[str, torch.Tensor]:
        """Allocate static output tensors"""
        B = key.batch_size
        T = key.temporal_shape
        H = key.spatial_h
        W = key.spatial_w
        S = H * W
        C_img = self.injector.action_config.img_hidden_size

        device = next(self.injector.parameters()).device
        dtype = next(self.injector.parameters()).dtype

        static_outputs = {
            'output': torch.empty(B, T * S, C_img, dtype=dtype, device=device),
        }

        return static_outputs

    def _copy_inputs_to_static(
        self,
        static_inputs: Dict[str, torch.Tensor],
        dynamic_inputs: Dict[str, Any],
    ):
        """Copy dynamic inputs to static tensors"""
        static_inputs['x'].copy_(dynamic_inputs['x'])
        static_inputs['condition'].copy_(dynamic_inputs['condition'])
        static_inputs['freqs_cis'] = dynamic_inputs['freqs_cis']
        static_inputs['start_frame'] = dynamic_inputs.get('start_frame', 0)
        static_inputs['kv_cache'] = dynamic_inputs.get('kv_cache')

    def _copy_outputs_from_static(
        self,
        static_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return output"""
        return static_outputs['output']

    def _execute_graph_body(
        self,
        static_inputs: Dict[str, torch.Tensor],
        static_outputs: Dict[str, torch.Tensor],
    ):
        """Execute KeyboardInjector forward"""
        x = static_inputs['x']
        condition = static_inputs['condition']
        freqs_cis = static_inputs['freqs_cis']
        start_frame = static_inputs['start_frame']
        T = static_inputs['temporal_shape']
        H = static_inputs['spatial_h']
        W = static_inputs['spatial_w']
        is_causal = static_inputs['is_causal']
        num_frame_per_block = static_inputs['num_frame_per_block']
        kv_cache = static_inputs.get('kv_cache')

        output = self.injector.forward(
            x=x,
            condition=condition,
            freqs_cis=freqs_cis,
            spatial_shape=(H, W),
            temporal_shape=T,
            is_causal=is_causal,
            kv_cache=kv_cache,
            start_frame=start_frame,
            num_frame_per_block=num_frame_per_block,
            block_mask=None,
        )

        static_outputs['output'].copy_(output)

    def _execute_eager(self, **kwargs) -> torch.Tensor:
        """Execute in eager mode"""
        return self.injector.forward(**kwargs)

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
        """Forward with CUDA Graph acceleration"""
        if condition is None:
            return x

        kwargs = {
            'x': x,
            'condition': condition,
            'freqs_cis': freqs_cis,
            'spatial_shape': spatial_shape,
            'temporal_shape': temporal_shape,
            'is_causal': is_causal,
            'kv_cache': kv_cache,
            'start_frame': start_frame,
            'num_frame_per_block': num_frame_per_block,
            'block_mask': block_mask,
        }

        return self.forward_with_graph(**kwargs)
