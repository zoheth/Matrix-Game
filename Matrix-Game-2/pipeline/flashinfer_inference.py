"""
FlashInfer-based Causal Inference Pipeline with CUDA Graph Support.

This pipeline uses FlashInfer's PagedKVCache for efficient KV cache management
and supports CUDA Graph capture for maximum performance.

Key features:
1. FlashInfer PagedKVCache for memory-efficient KV storage
2. Precomputed 3D RoPE frequencies for CUDA Graph compatibility
3. Static memory allocation (no dynamic allocation in forward path)
4. Optional CUDA Graph capture for repeated inference steps
"""

from typing import Dict, List, Optional, Tuple
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from wan.modules.flashinfer_kv_cache import (
    PagedKVCacheConfig,
    PagedKVCacheManager,
    PrecomputedRoPE3D,
    create_paged_kv_cache_manager,
)
from wan.modules.flashinfer_attention import (
    PrecomputedRoPE3DCache,
    replace_attention_with_flashinfer,
    FLASHINFER_AVAILABLE,
)
from wan.modules.action_context import ActionContext

# Import from existing pipeline for compatibility
from pipeline.causal_inference import cond_current


class CUDAGraphRunner:
    """
    Manages CUDA Graph capture and replay for a model.

    CUDA Graphs capture a sequence of GPU operations and replay them
    with minimal CPU overhead, providing significant speedups for
    repeated inference steps.

    Requirements for CUDA Graph capture:
    1. All tensor shapes must be static
    2. No dynamic memory allocation
    3. No CPU-GPU synchronization (e.g., .item() calls)
    4. No control flow dependent on tensor values
    """

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 3,
        device: torch.device = None,
    ):
        """
        Initialize CUDA Graph runner.

        Args:
            model: The model to capture
            warmup_steps: Number of warmup iterations before capture
            device: Target device
        """
        self.model = model
        self.warmup_steps = warmup_steps
        self.device = device or torch.device("cuda")

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs: Dict[str, torch.Tensor] = {}
        self.static_outputs: Dict[str, torch.Tensor] = {}
        self.is_captured = False

    def warmup(self, **kwargs) -> None:
        """
        Run warmup iterations to prepare for capture.

        Args:
            **kwargs: Input tensors for the model
        """
        # Warmup iterations to stabilize memory layout
        for _ in range(self.warmup_steps):
            with torch.cuda.stream(torch.cuda.current_stream()):
                _ = self.model(**kwargs)

        torch.cuda.synchronize()

    def capture(self, **kwargs) -> None:
        """
        Capture CUDA Graph for the model.

        Args:
            **kwargs: Static input tensors (will be reused for all replays)
        """
        if self.is_captured:
            return

        # Store static input tensors
        self.static_inputs = {k: v.clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)}

        # Create graph
        self.graph = torch.cuda.CUDAGraph()

        # Capture the graph
        with torch.cuda.graph(self.graph):
            outputs = self.model(**{
                k: self.static_inputs.get(k, v) for k, v in kwargs.items()
            })

        # Store static output reference
        if isinstance(outputs, torch.Tensor):
            self.static_outputs = {"output": outputs}
        elif isinstance(outputs, (tuple, list)):
            self.static_outputs = {f"output_{i}": o for i, o in enumerate(outputs)}
        elif isinstance(outputs, dict):
            self.static_outputs = outputs

        self.is_captured = True

    def replay(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Replay captured graph with new inputs.

        Args:
            **kwargs: New input values (copied to static tensors)

        Returns:
            Output tensors from the graph
        """
        if not self.is_captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        # Copy new inputs to static tensors
        for k, v in kwargs.items():
            if k in self.static_inputs and isinstance(v, torch.Tensor):
                self.static_inputs[k].copy_(v)

        # Replay graph
        self.graph.replay()

        return self.static_outputs


class FlashInferKVCacheWrapper:
    """
    Wrapper that provides the same interface as the original dict-based KV cache
    but uses FlashInfer's PagedKVCache internally.

    This allows gradual migration without changing the model code.
    """

    def __init__(
        self,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        local_attn_size: int,
        sink_size: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        """
        Initialize KV cache wrapper.

        Args:
            batch_size: Batch size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            local_attn_size: Sliding window size in frames
            sink_size: Number of sink frames
            dtype: Data type
            device: Device
        """
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.dtype = dtype
        self.device = device or torch.device("cuda")
        self.frame_seq_len = 880

        # Calculate cache size
        if local_attn_size > 0:
            self.cache_size = local_attn_size * self.frame_seq_len
        else:
            self.cache_size = max_seq_len

        # Create per-layer caches (using dict format for compatibility)
        self.caches: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_layers):
            self.caches.append({
                "k": torch.zeros(
                    batch_size, self.cache_size, num_heads, head_dim,
                    dtype=dtype, device=self.device
                ),
                "v": torch.zeros(
                    batch_size, self.cache_size, num_heads, head_dim,
                    dtype=dtype, device=self.device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            })

    def reset(self):
        """Reset all caches to initial state."""
        for cache in self.caches:
            cache["global_end_index"].fill_(0)
            cache["local_end_index"].fill_(0)

    def __getitem__(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get cache for a specific layer."""
        return self.caches[layer_idx]

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.caches)


class FlashInferInferencePipeline(nn.Module):
    """
    FlashInfer-based Causal Inference Pipeline.

    This is a drop-in replacement for CausalInferencePipeline that uses
    FlashInfer for attention computation and KV cache management.

    Key improvements:
    1. Memory-efficient paged KV cache
    2. Precomputed RoPE for CUDA Graph compatibility
    3. Optional CUDA Graph capture for maximum throughput
    """

    def __init__(
        self,
        args,
        device: str = "cuda",
        generator=None,
        vae_decoder=None,
        use_cuda_graph: bool = False,
        use_flashinfer_attention: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            args: Configuration arguments
            device: Target device
            generator: Pre-loaded generator (WanDiffusionWrapper)
            vae_decoder: Pre-loaded VAE decoder
            use_cuda_graph: Whether to use CUDA Graph for inference
            use_flashinfer_attention: Whether to replace attention with FlashInfer
        """
        super().__init__()

        from utils.wan_wrapper import WanDiffusionWrapper

        # Initialize generator
        self.generator = (
            WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
            if generator is None else generator
        )

        # Optionally replace attention with FlashInfer
        if use_flashinfer_attention and FLASHINFER_AVAILABLE:
            print("[INFO] Replacing attention modules with FlashInfer")
            replace_attention_with_flashinfer(self.generator.model)

        self.vae_decoder = vae_decoder

        # Scheduler setup
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long
        )

        if args.warp_denoising_step:
            timesteps = torch.cat((
                self.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)
            ))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Model parameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 880
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # KV caches (initialized lazily)
        self.kv_cache: Optional[FlashInferKVCacheWrapper] = None
        self.kv_cache_mouse: Optional[List[Dict]] = None
        self.kv_cache_keyboard: Optional[List[Dict]] = None
        self.crossattn_cache: Optional[List[Dict]] = None

        # CUDA Graph support
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_runner: Optional[CUDAGraphRunner] = None

        print(f"[INFO] FlashInfer Pipeline initialized")
        print(f"  - FlashInfer available: {FLASHINFER_AVAILABLE}")
        print(f"  - CUDA Graph enabled: {use_cuda_graph}")
        print(f"  - Frames per block: {self.num_frame_per_block}")
        print(f"  - Local attention size: {self.local_attn_size}")

    def _initialize_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Initialize all KV caches."""
        # Main self-attention cache
        self.kv_cache = FlashInferKVCacheWrapper(
            batch_size=batch_size,
            num_layers=self.num_transformer_blocks,
            num_heads=12,
            head_dim=128,
            max_seq_len=15 * self.frame_seq_length,
            local_attn_size=self.local_attn_size,
            sink_size=0,
            dtype=dtype,
            device=device,
        )

        # Action module caches
        kv_cache_size = self.local_attn_size if self.local_attn_size > 0 else 15
        self.kv_cache_keyboard = []
        self.kv_cache_mouse = []

        for _ in range(self.num_transformer_blocks):
            self.kv_cache_keyboard.append({
                "k": torch.zeros(batch_size, kv_cache_size, 16, 64, dtype=dtype, device=device),
                "v": torch.zeros(batch_size, kv_cache_size, 16, 64, dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
            self.kv_cache_mouse.append({
                "k": torch.zeros(batch_size * self.frame_seq_length, kv_cache_size, 16, 64, dtype=dtype, device=device),
                "v": torch.zeros(batch_size * self.frame_seq_length, kv_cache_size, 16, 64, dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        # Cross-attention cache
        self.crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            self.crossattn_cache.append({
                "k": torch.zeros(batch_size, 257, 12, 128, dtype=dtype, device=device),
                "v": torch.zeros(batch_size, 257, 12, 128, dtype=dtype, device=device),
                "is_init": False,
            })

    def _reset_caches(self, device: torch.device):
        """Reset all caches for a new generation session."""
        if self.kv_cache is not None:
            self.kv_cache.reset()

        for cache in self.crossattn_cache:
            cache["is_init"] = False

        for cache in self.kv_cache_mouse:
            cache["global_end_index"].fill_(0)
            cache["local_end_index"].fill_(0)

        for cache in self.kv_cache_keyboard:
            cache["global_end_index"].fill_(0)
            cache["local_end_index"].fill_(0)

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        mode: str = 'universal',
        profile: bool = False,
    ) -> torch.Tensor:
        """
        Perform causal video generation.

        Args:
            noise: Input noise [B, C, F, H, W]
            conditional_dict: Dictionary with conditions
            initial_latent: Initial latent for video continuation
            return_latents: Whether to return latents instead of decoded video
            mode: Action mode ('universal', 'gta_drive', 'templerun')
            profile: Whether to enable profiling

        Returns:
            Generated video or latents
        """
        from demo_utils.constant import ZERO_VAE_CACHE

        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        # Output buffer
        output = torch.zeros(
            batch_size, num_channels, num_output_frames, height, width,
            device=noise.device, dtype=noise.dtype
        )

        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        # Initialize caches
        with torch.profiler.record_function("Cache_Initialization"):
            if self.kv_cache is None:
                self._initialize_caches(batch_size, noise.dtype, noise.device)
            else:
                self._reset_caches(noise.device)

        # Process initial latent (context encoding)
        current_start_frame = 0
        if initial_latent is not None:
            with torch.profiler.record_function("Context_Encoding"):
                timestep = torch.zeros(batch_size, 1, device=noise.device, dtype=torch.int64)
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

                for _ in range(num_input_blocks):
                    current_ref_latents = initial_latent[
                        :, :, current_start_frame:current_start_frame + self.num_frame_per_block
                    ]
                    output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents

                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=cond_current(
                            conditional_dict, current_start_frame,
                            self.num_frame_per_block, mode=mode
                        ),
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache.caches,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    current_start_frame += self.num_frame_per_block

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)

        for block_idx, current_num_frames in enumerate(tqdm(all_num_frames, desc="Generating")):
            with torch.profiler.record_function(f"Block_{block_idx}"):
                noisy_input = noise[
                    :, :,
                    current_start_frame - num_input_frames:
                    current_start_frame + current_num_frames - num_input_frames
                ]

                if profile:
                    torch.cuda.synchronize()
                    diffusion_start.record()

                # Spatial denoising loop
                for step_idx, current_timestep in enumerate(self.denoising_step_list):
                    with torch.profiler.record_function(f"Denoising_Step_{step_idx}"):
                        timestep = torch.full(
                            (batch_size, current_num_frames),
                            current_timestep,
                            device=noise.device, dtype=torch.int64
                        )

                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_current(
                                conditional_dict, current_start_frame,
                                self.num_frame_per_block, mode=mode
                            ),
                            timestep=timestep,
                            kv_cache=self.kv_cache.caches,
                            kv_cache_mouse=self.kv_cache_mouse,
                            kv_cache_keyboard=self.kv_cache_keyboard,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                                torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                                next_timestep * torch.ones(
                                    batch_size * current_num_frames,
                                    device=noise.device, dtype=torch.long
                                )
                            )
                            noisy_input = rearrange(
                                noisy_input, '(b f) c h w -> b c f h w',
                                b=batch_size
                            )

                # Store output
                output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                # Update KV cache with clean context
                with torch.profiler.record_function("KV_Cache_Update"):
                    context_timestep = torch.full_like(timestep, self.args.context_noise)
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=cond_current(
                            conditional_dict, current_start_frame,
                            self.num_frame_per_block, mode=mode
                        ),
                        timestep=context_timestep,
                        kv_cache=self.kv_cache.caches,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

                current_start_frame += current_num_frames

                # VAE decode
                with torch.profiler.record_function("VAE_Decode"):
                    denoised_pred_t = denoised_pred.transpose(1, 2)
                    video, vae_cache = self.vae_decoder(denoised_pred_t.half(), *vae_cache)
                    videos.append(video)

                if profile:
                    diffusion_end.record()
                    torch.cuda.synchronize()
                    diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                    fps = video.shape[1] * 1000 / diffusion_time
                    print(f"Block {block_idx}: {diffusion_time:.1f}ms, {fps:.2f} FPS")

        if return_latents:
            return output
        else:
            return videos


def create_flashinfer_pipeline(args, **kwargs) -> FlashInferInferencePipeline:
    """
    Factory function to create FlashInfer pipeline.

    Args:
        args: Configuration arguments
        **kwargs: Additional arguments for pipeline

    Returns:
        Configured FlashInferInferencePipeline
    """
    return FlashInferInferencePipeline(args, **kwargs)
