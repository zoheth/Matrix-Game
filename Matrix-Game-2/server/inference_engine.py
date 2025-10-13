"""
Interactive inference engine for real-time frame generation
"""
import asyncio
import copy
import time
import numpy as np
import torch
from typing import Dict, Optional
from collections import deque
from PIL import Image

from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange

from server.config import GameModeConfig
from server.action_mapper import ActionMapper
from server.model_manager import ModelManager
from server.streaming_server import StreamingServer
from demo_utils.constant import ZERO_VAE_CACHE


class InferenceEngine:
    """Real-time interactive inference engine"""

    def __init__(
        self,
        model_manager: ModelManager,
        action_mapper: ActionMapper,
        server: StreamingServer,
        max_latent_frames: int = 300,
        profiler: Optional[torch.profiler.profile] = None,
        profile_steps: int = 4
    ):
        """
        Initialize inference engine

        Args:
            model_manager: Loaded model manager
            action_mapper: Action parsing/mapping handler
            server: Streaming server for I/O
            max_latent_frames: Maximum latent frames to generate before reset
            profiler: Optional torch profiler for performance analysis
            profile_steps: Number of inference steps to profile before stopping
        """
        self.model_manager = model_manager
        self.action_mapper = action_mapper
        self.server = server
        self.max_latent_frames = max_latent_frames
        self.profiler = profiler
        self.profile_steps = profile_steps
        self.inference_count = 0

        self.pipeline = model_manager.get_pipeline()
        self.vae = model_manager.get_vae()
        self.device = model_manager.device
        self.dtype = model_manager.dtype

        # Frame preprocessing
        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    async def run(self, image_path: Optional[str] = None):
        """
        Run interactive inference loop

        Args:
            image_path: Path to initial image (prompts user if None)
        """
        print("=" * 60)
        print("🎮 Starting Interactive Inference Engine")
        print("=" * 60)

        # Load initial image
        initial_image = self._load_initial_image(image_path)

        # Prepare conditional dictionary
        conditional_dict = self._prepare_conditional_dict(initial_image)

        # Run inference loop
        await self._inference_loop(conditional_dict)

    def _load_initial_image(self, image_path: Optional[str] = None) -> torch.Tensor:
        """Load and preprocess initial image"""
        if image_path is None:
            img_path = input("Please input the image path: ").strip()
            image = load_image(img_path)
        else:
            image = load_image(image_path)

        # Resize/crop to target dimensions
        image = self._resize_crop(image, 352, 640)

        # Convert to tensor
        image_tensor = self.frame_process(image)[None, :, None, :, :]
        return image_tensor.to(dtype=self.dtype, device=self.device)

    def _resize_crop(self, image: Image.Image, th: int, tw: int) -> Image.Image:
        """Resize and center crop image to target dimensions"""
        w, h = image.size
        if h / w > th / tw:
            new_w, new_h = int(w), int(w * th / tw)
        else:
            new_h, new_w = int(h), int(h * tw / th)

        return image.crop((
            (w - new_w) / 2,
            (h - new_h) / 2,
            (w + new_w) / 2,
            (h + new_h) / 2
        ))

    def _prepare_conditional_dict(self, initial_image: torch.Tensor) -> Dict:
        """Prepare conditional dictionary with initial frame encoding"""
        num_frames_total = 4 * self.max_latent_frames + 1

        print(f"Allocating buffer for {self.max_latent_frames} latent frames "
              f"({num_frames_total} actual frames)")

        # Encode first frame with zero padding
        padding_video = torch.zeros_like(initial_image).repeat(
            1, 1, 4 * (self.max_latent_frames - 1), 1, 1
        )
        img_cond = torch.cat([initial_image, padding_video], dim=2)

        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs)

        # Create mask (only first frame is valid)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0

        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.vae.clip.encode_video(initial_image)

        # Build conditional dict
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.dtype)
        }

        # Add action conditioning tensors
        keyboard_dim = self.action_mapper.action_config.keyboard_dim
        conditional_dict['keyboard_cond'] = torch.zeros(
            [1, num_frames_total, keyboard_dim],
            device=self.device,
            dtype=self.dtype
        )

        if self.action_mapper.action_config.has_mouse:
            conditional_dict['mouse_cond'] = torch.zeros(
                [1, num_frames_total, 2],
                device=self.device,
                dtype=self.dtype
            )

        return conditional_dict

    async def _inference_loop(self, conditional_dict: Dict):
        """Main inference loop - generates frames on demand"""
        batch_size = 1
        num_frame_per_block = self.pipeline.num_frame_per_block

        # Initialize caches
        self._initialize_caches(batch_size)

        vae_cache = self._initialize_vae_cache()
        current_start_frame = 0
        frame_times = deque(maxlen=30)

        print("✅ Ready! Waiting for actions from client...")
        print(f"📡 Open browser at http://{{host}}:{{port}}")
        print("=" * 60)

        with torch.no_grad():
            while True:
                # Check if client uploaded a new image
                if self.server.has_pending_image():
                    new_image = await self.server.get_next_image(timeout=0.01)
                    if new_image is not None:
                        print("🔄 Resetting with new image from client...")
                        conditional_dict = await self._reset_with_new_image(
                            new_image,
                            batch_size
                        )
                        current_start_frame = 0
                        vae_cache = self._initialize_vae_cache()
                        print("✅ Reset complete with new image")

                # Wait for next action from client
                action_data = await self.server.get_next_action()

                if action_data is None:
                    continue

                # Parse action
                current_action = self.action_mapper.parse_action(action_data)
                if current_action is None:
                    continue

                # Check frame limit and reset if needed
                if current_start_frame >= self.max_latent_frames:
                    print(f"⚠️  Reached frame limit ({self.max_latent_frames}), resetting...")
                    current_start_frame = 0
                    vae_cache = self._initialize_vae_cache()
                    print("✅ Reset complete")

                # Generate new frame block
                inference_start = time.time()

                # Update conditions with current action
                self._update_conditional_dict(
                    conditional_dict,
                    current_action,
                    current_start_frame,
                    num_frame_per_block
                )

                # Run inference
                video_frames = await self._generate_frame_block(
                    conditional_dict,
                    current_start_frame,
                    num_frame_per_block,
                    batch_size,
                    vae_cache
                )

                # Measure performance
                inference_time = time.time() - inference_start
                frame_times.append(inference_time)
                avg_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                latency_ms = avg_time * 1000

                # Send frames to client
                for frame in video_frames:
                    await self.server.add_frame(frame, fps=fps, latency=latency_ms)

                current_start_frame += num_frame_per_block

                # Profiler step logic
                self.inference_count += 1
                profile_status = ""
                if self.profiler is not None:
                    if self.inference_count <= self.profile_steps:
                        self.profiler.step()
                        profile_status = f" | Profile: {self.inference_count}/{self.profile_steps}"
                    elif self.inference_count == self.profile_steps + 1:
                        # Just finished profiling, notify user
                        profile_status = " | ✓ Profiling complete"

                print(f"✅ Frame {current_start_frame} | FPS: {fps:.1f} | "
                      f"Latency: {latency_ms:.0f}ms{profile_status}")

    def _initialize_caches(self, batch_size: int):
        """Initialize model KV caches"""
        self.pipeline.kv_cache1 = None
        self.pipeline.kv_cache_keyboard = None
        self.pipeline.kv_cache_mouse = None
        self.pipeline.crossattn_cache = None

        self.pipeline._initialize_kv_cache(
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device
        )
        self.pipeline._initialize_kv_cache_mouse_and_keyboard(
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device
        )
        self.pipeline._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device
        )

    def _initialize_vae_cache(self):
        """Initialize VAE decoder cache"""
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        return vae_cache

    async def _generate_frame_block(
        self,
        conditional_dict: Dict,
        current_start_frame: int,
        num_frame_per_block: int,
        batch_size: int,
        vae_cache: list
    ) -> np.ndarray:
        """Generate a block of frames using the diffusion model"""
        # Generate noise
        noise = torch.randn(
            [batch_size, 16, num_frame_per_block, 44, 80],
            device=self.device,
            dtype=self.dtype
        )

        # Get current conditions
        curr_cond = self._get_current_conditions(
            conditional_dict,
            current_start_frame,
            num_frame_per_block
        )

        # Denoising loop
        noisy_input = noise
        for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
            timestep = torch.ones(
                [batch_size, num_frame_per_block],
                device=self.device,
                dtype=torch.int64
            ) * current_timestep

            _, denoised_pred = self.pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=curr_cond,
                timestep=timestep,
                kv_cache=self.pipeline.kv_cache1,
                kv_cache_mouse=self.pipeline.kv_cache_mouse,
                kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=current_start_frame * self.pipeline.frame_seq_length
            )

            # Add noise for next step
            if index < len(self.pipeline.denoising_step_list) - 1:
                next_timestep = self.pipeline.denoising_step_list[index + 1]
                noisy_input = self.pipeline.scheduler.add_noise(
                    rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                    torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                    next_timestep * torch.ones(
                        [batch_size * num_frame_per_block],
                        device=self.device,
                        dtype=torch.long
                    )
                )
                noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)

        # Update KV cache with clean context
        context_timestep = torch.ones(
            [batch_size, num_frame_per_block],
            device=self.device,
            dtype=torch.int64
        ) * self.pipeline.args.context_noise

        self.pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=curr_cond,
            timestep=context_timestep,
            kv_cache=self.pipeline.kv_cache1,
            kv_cache_mouse=self.pipeline.kv_cache_mouse,
            kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=current_start_frame * self.pipeline.frame_seq_length
        )

        # Decode to video
        denoised_pred = denoised_pred.transpose(1, 2)
        video, vae_cache[:] = self.pipeline.vae_decoder(denoised_pred.half(), *vae_cache)

        # Convert to numpy
        video_np = rearrange(video, "B T C H W -> B T H W C")
        video_np = ((video_np.float() + 1) * 127.5).clip(0, 255)
        video_np = video_np.cpu().numpy().astype(np.uint8)[0]
        video_np = np.ascontiguousarray(video_np)

        return video_np

    def _update_conditional_dict(
        self,
        conditional_dict: Dict,
        action: Dict[str, torch.Tensor],
        current_start_frame: int,
        num_frame_per_block: int
    ):
        """Update conditional dict with new action"""
        # Calculate frame range to fill
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block

        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block - 1)
        start_idx = final_frame - last_frame_num
        end_idx = final_frame

        # Update mouse condition
        if 'mouse' in action and 'mouse_cond' in conditional_dict:
            conditional_dict["mouse_cond"][:, start_idx:end_idx] = \
                action['mouse'][None, None, :].repeat(1, last_frame_num, 1)

        # Update keyboard condition
        if 'keyboard' in action:
            conditional_dict["keyboard_cond"][:, start_idx:end_idx] = \
                action['keyboard'][None, None, :].repeat(1, last_frame_num, 1)

    def _get_current_conditions(
        self,
        conditional_dict: Dict,
        current_start_frame: int,
        num_frame_per_block: int
    ) -> Dict:
        """Get current conditional slice for inference"""
        end_frame_idx = 1 + 4 * (current_start_frame + num_frame_per_block - 1)

        new_cond = {
            "cond_concat": conditional_dict["cond_concat"][
                :, :, current_start_frame:current_start_frame + num_frame_per_block
            ],
            "visual_context": conditional_dict["visual_context"],
            "keyboard_cond": conditional_dict["keyboard_cond"][:, :end_frame_idx]
        }

        if "mouse_cond" in conditional_dict:
            new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, :end_frame_idx]

        return new_cond

    async def _reset_with_new_image(
        self,
        image: Image.Image,
        batch_size: int
    ) -> Dict:
        """
        Reset inference with a new initial image from client

        Args:
            image: PIL Image from client upload
            batch_size: Batch size for inference

        Returns:
            New conditional dictionary
        """
        # Resize/crop to target dimensions
        image = self._resize_crop(image, 352, 640)

        # Convert to tensor
        image_tensor = self.frame_process(image)[None, :, None, :, :]
        image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

        # Prepare new conditional dictionary
        conditional_dict = self._prepare_conditional_dict(image_tensor)

        # Re-initialize KV caches
        self._initialize_caches(batch_size)

        return conditional_dict
