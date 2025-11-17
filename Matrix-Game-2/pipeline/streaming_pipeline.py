"""
Streaming (interactive) inference pipeline for causal video generation.

This pipeline generates frames one block at a time, requesting user actions
between blocks for interactive control.
"""

from typing import Optional, Dict, Callable
import torch
import numpy as np

from einops import rearrange

from pipeline.base_pipeline import BaseCausalInferencePipeline
from pipeline.config import PipelineConfig
from pipeline.action_strategies import ActionStrategy, ActionDict, CLIActionStrategy
from utils.visualize import process_video


class StreamingCausalInferencePipeline(BaseCausalInferencePipeline):
    """
    Streaming inference pipeline (interactive mode).

    This pipeline generates frames incrementally, obtaining user actions
    between blocks. This enables real-time interactive control.

    Key differences from batch pipeline:
    - Uses an ActionStrategy to obtain actions dynamically
    - Optionally saves intermediate videos
    - Supports user-controlled continuation
    """

    def __init__(
        self,
        config: PipelineConfig,
        generator,
        vae_decoder,
        device: str = "cuda"
    ):
        """
        Initialize streaming inference pipeline.

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
        action_strategy: Optional[ActionStrategy] = None,
        output_folder: Optional[str] = None,
        video_name: str = "streaming",
        continuation_callback: Optional[Callable[[int], bool]] = None
    ) -> torch.Tensor | np.ndarray:
        """
        Perform streaming inference with interactive control.

        Args:
            noise: Input noise tensor [batch, channels, num_frames, h, w]
            conditional_dict: Dictionary containing conditional inputs
            initial_latent: Optional initial latent frames
            return_latents: Whether to return latents instead of video
            action_strategy: Strategy for obtaining actions (defaults to CLI)
            output_folder: Folder to save intermediate videos (optional)
            video_name: Base name for saved videos
            continuation_callback: Function called after each block to decide continuation
                                   Signature: (block_idx) -> bool (True to continue)

        Returns:
            If return_latents=True: output latent tensor
            Otherwise: final video as numpy array
        """
        # Validate input
        assert noise.shape[1] == self.config.vae.latent_channels
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.config.inference.num_frame_per_block == 0
        num_blocks = num_frames // self.config.inference.num_frame_per_block

        # Initialize action strategy
        if action_strategy is None:
            action_strategy = CLIActionStrategy(mode=self.config.mode, device=str(self.device))

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

        # Main generation loop
        all_num_frames = [self.config.inference.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(all_num_frames):
            # Get user action for this block
            current_action = action_strategy.get_action(
                frame_index=current_start_frame,
                block_idx=block_idx
            )

            # Get noisy input for this block
            noisy_input = noise[
                :, :,
                current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames
            ]

            # Get conditions with updated actions
            block_cond, conditional_dict = self.condition_processor.slice_block_conditions(
                conditional_dict,
                current_start_frame,
                current_num_frames,
                replace_action=current_action
            )

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

            # Save intermediate video if requested
            if output_folder is not None:
                self._save_intermediate_video(
                    video,
                    videos,
                    conditional_dict,
                    current_start_frame,
                    output_folder,
                    video_name
                )

            # Update frame counter
            current_start_frame += current_num_frames

            # Check continuation
            if continuation_callback is not None:
                if not continuation_callback(block_idx):
                    print(f"Stopping at block {block_idx}")
                    break

        # Final video assembly
        if return_latents:
            return output
        else:
            return self._assemble_final_video(
                videos,
                conditional_dict,
                current_start_frame,
                output_folder,
                video_name
            )

    def _save_intermediate_video(
        self,
        latest_video: torch.Tensor,
        all_videos: list[torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor],
        current_frame: int,
        output_folder: str,
        video_name: str
    ) -> None:
        """
        Save intermediate video with visualization overlays.

        Args:
            latest_video: Most recently decoded video block
            all_videos: All previously decoded blocks
            conditional_dict: Conditional dictionary
            current_frame: Current frame index
            output_folder: Output folder path
            video_name: Base name for video
        """
        # Convert to numpy
        video_np = rearrange(latest_video, "B T C H W -> B T H W C")
        video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video_np = np.ascontiguousarray(video_np)

        # Prepare action configuration for visualization
        action_seq_len = self.condition_processor.get_action_sequence_length(
            current_frame + self.config.inference.num_frame_per_block
        )

        if self.config.mode != 'templerun':
            config = (
                conditional_dict["keyboard_cond"][0, :action_seq_len].float().cpu().numpy(),
                conditional_dict["mouse_cond"][0, :action_seq_len].float().cpu().numpy()
            )
        else:
            config = (
                conditional_dict["keyboard_cond"][0, :action_seq_len].float().cpu().numpy(),
            )

        # Save video
        mouse_icon = 'assets/images/mouse.png'
        output_path = f'{output_folder}/{video_name}_current.mp4'

        process_video(
            video_np.astype(np.uint8),
            output_path,
            config,
            mouse_icon,
            mouse_scale=0.1,
            process_icon=False,
            mode=self.config.mode
        )

    def _assemble_final_video(
        self,
        videos: list[torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor],
        final_frame: int,
        output_folder: Optional[str],
        video_name: str
    ) -> np.ndarray:
        """
        Assemble final video from all blocks and save with visualizations.

        Args:
            videos: List of video blocks
            conditional_dict: Conditional dictionary
            final_frame: Final frame index
            output_folder: Output folder (if None, skip saving)
            video_name: Base name for video

        Returns:
            Final video as numpy array
        """
        # Concatenate all videos
        videos_tensor = torch.cat(videos, dim=1)
        videos_np = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos_np = ((videos_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos_np)

        if output_folder is not None:
            # Prepare configuration
            action_seq_len = self.condition_processor.get_action_sequence_length(
                final_frame + self.config.inference.num_frame_per_block - 1
            )

            if self.config.mode != 'templerun':
                config = (
                    conditional_dict["keyboard_cond"][0, :action_seq_len].float().cpu().numpy(),
                    conditional_dict["mouse_cond"][0, :action_seq_len].float().cpu().numpy()
                )
            else:
                config = (
                    conditional_dict["keyboard_cond"][0, :action_seq_len].float().cpu().numpy(),
                )

            mouse_icon = 'assets/images/mouse.png'

            # Save with and without icons
            process_video(
                video.astype(np.uint8),
                f'{output_folder}/{video_name}_icon.mp4',
                config,
                mouse_icon,
                mouse_scale=0.1,
                mode=self.config.mode
            )

            process_video(
                video.astype(np.uint8),
                f'{output_folder}/{video_name}.mp4',
                config,
                mouse_icon,
                mouse_scale=0.1,
                process_icon=False,
                mode=self.config.mode
            )

        return video


def default_cli_continuation(block_idx: int) -> bool:
    """
    Default continuation callback using CLI input.

    Args:
        block_idx: Current block index

    Returns:
        True to continue, False to stop
    """
    response = input("Continue? (Press 'n' to break): ").strip().lower()
    return response != 'n'
