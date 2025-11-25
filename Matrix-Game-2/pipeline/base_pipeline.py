"""
Base inference pipeline for causal video generation.

This module provides the common infrastructure shared between
batch inference and streaming inference modes.
"""

from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import torch
import copy

from einops import rearrange
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.cache_manager import CacheManager
from pipeline.condition_processor import ConditionProcessor
from demo_utils.constant import ZERO_VAE_CACHE
from utils.scheduler import FlowMatchingScheduler
from wan.modules.cuda_graph_inference import CUDAGraphSingleStepRunner


class BaseCausalInferencePipeline(torch.nn.Module, ABC):
    """
    Base class for causal inference pipelines.

    This class provides:
    - Model initialization
    - Scheduler setup
    - KV cache management
    - Common inference loop structure

    Subclasses implement specific inference modes (batch vs. streaming).
    """

    def __init__(
        self,
        config: PipelineConfig,
        generator,
        vae_decoder,
        device: str = "cuda",
        use_cuda_graph: bool = False,
        page_size: int = 16,
    ):
        """
        Initialize the base pipeline.

        Args:
            config: Pipeline configuration
            generator: WanDiffusionWrapper instance
            vae_decoder: VAE decoder wrapper
            device: Device to run on
            use_cuda_graph: Whether to use CUDA Graph for inference
            page_size: Page size for PagedCache
        """
        super().__init__()

        self.config = config
        self.device = torch.device(device)
        self.generator = generator
        self.vae_decoder = vae_decoder
        self.use_cuda_graph = use_cuda_graph
        self.page_size = page_size

        # CUDA Graph runner (lazy initialization)
        self._cuda_graph_runner: Optional[CUDAGraphSingleStepRunner] = None
        self._cuda_graph_captured_start: Optional[int] = None

        # Pipeline owns the scheduler (not generator)
        self.scheduler = FlowMatchingScheduler(
            num_train_timesteps=1000,
            shift=config.inference.timestep_shift,
            sigma_min=0.0,
            extra_one_step=True
        )
        self.scheduler.set_timesteps(num_inference_steps=1000)

        # Get denoising timesteps from scheduler (all logic centralized)
        self.denoising_step_list = self.scheduler.get_inference_timesteps(
            custom_steps=config.inference.denoising_steps,
            warp=config.inference.warp_denoising_step
        )

        # Initialize cache manager
        self.cache_manager = None  # Lazy initialization

        # Initialize condition processor
        self.condition_processor = ConditionProcessor(
            vae_config=config.vae,
            mode=config.mode
        )

        print(f"Initialized {self.__class__.__name__} with {config.inference.num_frame_per_block} frames per block")

        # Set num_frame_per_block on the model if needed
        if config.inference.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = config.inference.num_frame_per_block

    def _ensure_cache_initialized(self, batch_size: int, dtype: torch.dtype) -> None:
        """
        Ensure cache manager is initialized with PagedCache.

        Args:
            batch_size: Batch size for caches
            dtype: Data type for cache tensors
        """
        if self.cache_manager is None:
            self.cache_manager = CacheManager(
                model_config=self.config.model,
                cache_config=self.config.cache,
                device=self.device,
                dtype=dtype,
                page_size=self.page_size,
            )
            self.cache_manager.initialize_all_caches(batch_size)
        elif not self.cache_manager.is_initialized():
            self.cache_manager.initialize_all_caches(batch_size)
        else:
            # Already initialized, just reset
            self.cache_manager.reset_all_caches()

    def _initialize_vae_cache(self):
        """
        Initialize the VAE decoder cache.

        Returns:
            VAE cache list
        """
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        return vae_cache

    def _cache_initial_frames(
        self,
        initial_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        batch_size: int
    ) -> tuple[torch.Tensor, int]:
        """
        Cache the initial reference frames (if provided).

        Args:
            initial_latent: Initial latent frames [batch, channels, frames, h, w]
            conditional_dict: Conditional dictionary
            batch_size: Batch size

        Returns:
            Tuple of (output tensor with initial frames filled, next start frame index)
        """
        num_channels, num_frames, height, width = initial_latent.shape[1:]
        num_input_frames = initial_latent.shape[2]

        # Initialize output tensor
        output = torch.zeros(
            [batch_size, num_channels, num_frames, height, width],
            device=self.device,
            dtype=initial_latent.dtype
        )

        # Process initial frames in blocks
        assert num_input_frames % self.config.inference.num_frame_per_block == 0
        num_input_blocks = num_input_frames // self.config.inference.num_frame_per_block

        timestep = torch.zeros([batch_size, 1], device=self.device, dtype=torch.int64)
        current_start = 0

        visual_cache, mouse_cache, keyboard_cache, crossattn_cache = self.cache_manager.get_caches()

        for _ in range(num_input_blocks):
            current_block = initial_latent[
                :, :, current_start:current_start + self.config.inference.num_frame_per_block
            ]
            output[:, :, current_start:current_start + self.config.inference.num_frame_per_block] = current_block

            # Get conditions for this block
            block_cond, _ = self.condition_processor.slice_block_conditions(
                conditional_dict,
                current_start,
                self.config.inference.num_frame_per_block
            )

            # Run generator to cache KV
            self.generator(
                noisy_image_or_video=current_block,
                conditional_dict=block_cond,
                timestep=timestep,
                kv_cache=visual_cache,
                kv_cache_mouse=mouse_cache,
                kv_cache_keyboard=keyboard_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start * self.config.model.frame_seq_length
            )

            current_start += self.config.inference.num_frame_per_block

        return output, current_start

    def _denoise_block(
        self,
        noisy_input: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        current_start_frame: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Denoise a single block through the diffusion process.

        Args:
            noisy_input: Noisy latent input
            conditional_dict: Conditional dictionary
            current_start_frame: Start frame index of this block
            batch_size: Batch size

        Returns:
            Denoised latent
        """
        current_num_frames = noisy_input.shape[2]
        visual_cache, mouse_cache, keyboard_cache, crossattn_cache = self.cache_manager.get_caches()
        current_start = current_start_frame * self.config.model.frame_seq_length

        # Initialize CUDA Graph if enabled and not yet captured for this start position
        use_graph = self.use_cuda_graph and self._should_use_cuda_graph(current_start)

        if use_graph and self._cuda_graph_runner is None:
            self._cuda_graph_runner = CUDAGraphSingleStepRunner(
                generator=self.generator,
                device=self.device,
                dtype=noisy_input.dtype,
            )

        for index, current_timestep in enumerate(self.denoising_step_list):
            # Set timestep
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=self.device,
                dtype=torch.int64
            ) * current_timestep

            # Forward pass - use CUDA Graph if available
            if use_graph:
                flow_pred, _ = self._run_with_cuda_graph(
                    noisy_input=noisy_input,
                    timestep=timestep,
                    conditional_dict=conditional_dict,
                    visual_cache=visual_cache,
                    mouse_cache=mouse_cache,
                    keyboard_cache=keyboard_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                )
            else:
                flow_pred, _ = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=visual_cache,
                    kv_cache_mouse=mouse_cache,
                    kv_cache_keyboard=keyboard_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                )

            # Convert flow prediction to x0 using scheduler
            denoised_pred = self.scheduler.convert_flow_to_x0(
                flow_pred=rearrange(flow_pred, 'b c f h w -> (b f) c h w'),
                xt=rearrange(noisy_input, 'b c f h w -> (b f) c h w'),
                timestep=timestep.flatten(0, 1)
            )
            denoised_pred = rearrange(denoised_pred, '(b f) c h w -> b c f h w', b=batch_size)

            # If not the last step, add noise for next step
            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                    torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames],
                        device=self.device,
                        dtype=torch.long
                    )
                )
                noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)

        return denoised_pred

    def _should_use_cuda_graph(self, current_start: int) -> bool:
        """
        Determine if CUDA Graph should be used for this forward pass.

        CUDA Graph requires static current_start. We capture once and reuse
        only when current_start matches the captured value.

        For simplicity, we only use CUDA Graph when current_start == 0
        (first block of each inference). This covers the warmup case well.
        """
        # For now, only use graph for first block to avoid recapture complexity
        return current_start == 0

    def _run_with_cuda_graph(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        visual_cache: List[Dict],
        mouse_cache: List[Dict],
        keyboard_cache: List[Dict],
        crossattn_cache: List[Dict],
        current_start: int,
    ):
        """Run forward pass using CUDA Graph."""
        runner = self._cuda_graph_runner

        # Capture graph if not yet captured for this current_start
        if not runner.is_captured or self._cuda_graph_captured_start != current_start:
            if runner.is_captured:
                runner.reset()
            runner.capture(
                noisy_input=noisy_input,
                timestep=timestep,
                conditional_dict=conditional_dict,
                kv_cache=visual_cache,
                kv_cache_mouse=mouse_cache,
                kv_cache_keyboard=keyboard_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )
            self._cuda_graph_captured_start = current_start

        # Run using graph
        return runner.run(noisy_input, timestep, conditional_dict)

    def _update_kv_cache_with_clean_context(
        self,
        denoised_pred: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        current_start_frame: int,
        batch_size: int
    ) -> None:
        """
        Update KV cache using the clean (denoised) latent.

        Args:
            denoised_pred: Clean denoised latent
            conditional_dict: Conditional dictionary
            current_start_frame: Start frame index
            batch_size: Batch size
        """
        current_num_frames = denoised_pred.shape[2]
        context_timestep = torch.ones(
            [batch_size, current_num_frames],
            device=self.device,
            dtype=torch.int64
        ) * self.config.inference.context_noise

        visual_cache, mouse_cache, keyboard_cache, crossattn_cache = self.cache_manager.get_caches()

        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=visual_cache,
            kv_cache_mouse=mouse_cache,
            kv_cache_keyboard=keyboard_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start_frame * self.config.model.frame_seq_length
        )

    def _decode_latent_to_video(
        self,
        latent: torch.Tensor,
        vae_cache: List
    ) -> tuple[torch.Tensor, List]:
        """
        Decode latent to video using VAE decoder.

        Args:
            latent: Latent tensor [batch, channels, frames, h, w]
            vae_cache: VAE cache from previous decode

        Returns:
            Tuple of (decoded video, updated cache)
        """
        # Transpose for VAE decoder: [batch, frames, channels, h, w]
        latent = latent.transpose(1, 2)
        video, vae_cache = self.vae_decoder(latent.half(), *vae_cache)
        return video, vae_cache

    @abstractmethod
    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        **kwargs
    ):
        """
        Run inference to generate video.

        This method must be implemented by subclasses.

        Args:
            noise: Random noise tensor
            conditional_dict: Conditional inputs
            initial_latent: Optional initial frames
            return_latents: Whether to return latents instead of decoded video
            **kwargs: Additional subclass-specific arguments
        """
        pass
