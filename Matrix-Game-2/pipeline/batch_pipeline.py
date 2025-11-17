"""
Batch inference pipeline for causal video generation.

This pipeline generates all frames in a batch (non-interactive mode).
"""

from typing import Optional, Dict, List
import torch

from einops import rearrange
from tqdm import tqdm

from pipeline.base_pipeline import BaseCausalInferencePipeline
from pipeline.config import PipelineConfig


class BatchCausalInferencePipeline(BaseCausalInferencePipeline):
    """
    Batch inference pipeline (non-interactive).

    This generates a fixed sequence of frames based on pre-specified actions.
    Suitable for:
    - Benchmark evaluation
    - Dataset generation
    - Non-interactive video generation
    """

    def __init__(
        self,
        config: PipelineConfig,
        generator,
        vae_decoder,
        device: str = "cuda"
    ):
        """
        Initialize batch inference pipeline.

        Args:
            config: Pipeline configuration
            generator: WanDiffusionWrapper instance
            vae_decoder: VAE decoder wrapper
            device: Device to run on
        """
        super().__init__(config, generator, vae_decoder, device)

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Perform batch inference on the given noise and conditions.

        Args:
            noise: Input noise tensor [batch, channels, num_frames, h, w]
            conditional_dict: Dictionary containing:
                - cond_concat: Concatenated conditioning
                - visual_context: CLIP visual features
                - mouse_cond: Mouse action conditioning (if not templerun)
                - keyboard_cond: Keyboard action conditioning
            initial_latent: Optional initial latent frames for video extension
            return_latents: Whether to return latents instead of decoded videos
            profile: Whether to enable profiling (for FPS measurement)

        Returns:
            If return_latents=True: output latent tensor
            Otherwise: list of decoded video tensors
        """
        # Validate input
        assert noise.shape[1] == self.config.vae.latent_channels
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.config.inference.num_frame_per_block == 0
        num_blocks = num_frames // self.config.inference.num_frame_per_block

        # Initialize cache
        self._ensure_cache_initialized(batch_size, noise.dtype)

        # Calculate frame counts
        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        # Initialize output tensor
        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=self.device,
            dtype=noise.dtype
        )

        # Initialize VAE cache
        vae_cache = self._initialize_vae_cache()
        videos = []

        # Cache initial frames if provided
        current_start_frame = 0
        if initial_latent is not None:
            output, current_start_frame = self._cache_initial_frames(
                initial_latent,
                conditional_dict,
                batch_size
            )

        # Profiling setup
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)

        # Main generation loop
        all_num_frames = [self.config.inference.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(tqdm(all_num_frames)):
            # Get noisy input for this block
            noisy_input = noise[
                :, :,
                current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames
            ]

            # Get conditions for this block
            block_cond, _ = self.condition_processor.slice_block_conditions(
                conditional_dict,
                current_start_frame,
                current_num_frames
            )

            # Profile if requested
            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()

            # Denoise the block
            denoised_pred = self._denoise_block(
                noisy_input,
                block_cond,
                current_start_frame,
                batch_size
            )

            # Store output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Update KV cache with clean context
            self._update_kv_cache_with_clean_context(
                denoised_pred,
                block_cond,
                current_start_frame,
                batch_size
            )

            # Decode to video
            video, vae_cache = self._decode_latent_to_video(denoised_pred, vae_cache)
            videos.append(video)

            # Profile if requested
            if profile:
                diffusion_end.record()
                torch.cuda.synchronize()
                diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                fps = video.shape[1] * 1000 / diffusion_time
                print(f"Block {block_idx}: {diffusion_time:.2f}ms, FPS: {fps:.2f}")

            # Update start frame
            current_start_frame += current_num_frames

        if return_latents:
            return output
        else:
            return videos
