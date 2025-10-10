"""
Integrated WebRTC Streaming Inference for Matrix Game

This module combines the inference engine with WebRTC streaming server
to provide real-time visualization of the game inference process.
"""

import os
import argparse
import asyncio
import torch
import numpy as np
import copy
from typing import Optional, Callable

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange

from pipeline import CausalInferenceStreamingPipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file
from webrtc_server import create_server
from demo_utils.constant import ZERO_VAE_CACHE


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix Game WebRTC Streaming Inference")
    parser.add_argument("--config_path", type=str,
                       default="configs/inference_yaml/inference_universal.yaml",
                       help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="",
                       help="Path to the checkpoint")
    parser.add_argument("--output_folder", type=str, default="outputs/",
                       help="Output folder for saving videos")
    parser.add_argument("--max_num_output_frames", type=int, default=360,
                       help="Max number of output latent frames")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str,
                       default="Matrix-Game-2.0",
                       help="Path to the VAE model folder")
    parser.add_argument("--server_host", type=str, default="0.0.0.0",
                       help="WebRTC server host")
    parser.add_argument("--server_port", type=int, default=8000,
                       help="WebRTC server port")
    parser.add_argument("--enable_streaming", action="store_true",
                       help="Enable real-time WebRTC streaming")
    parser.add_argument("--img_path", type=str, default=None,
                       help="Path to initial image (if not provided, will prompt)")
    args = parser.parse_args()
    return args


class WebRTCStreamingInference:
    """
    Enhanced inference engine with WebRTC streaming support.

    Extends the base inference pipeline to support real-time frame
    streaming to connected WebRTC clients.
    """

    def __init__(self, args, frame_callback: Optional[Callable] = None):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16
        self.frame_callback = frame_callback

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        """Load configuration from YAML file."""
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        """Initialize all models: generator, VAE encoder/decoder."""
        print("Initializing models...")

        # Initialize generator
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        # Initialize VAE decoder
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"),
            map_location="cpu"
        )
        decoder_state_dict = {
            key: value for key, value in vae_state_dict.items()
            if 'decoder.' in key or 'conv2' in key
        }
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Initialize pipeline
        pipeline = CausalInferenceStreamingPipeline(
            self.config,
            generator=generator,
            vae_decoder=current_vae_decoder
        )

        # Load checkpoint if provided
        if self.args.checkpoint_path:
            print(f"Loading checkpoint from {self.args.checkpoint_path}")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        # Initialize VAE encoder
        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

        print("Models initialized successfully")

    def _resizecrop(self, image, th, tw):
        """Resize and center crop image to target dimensions."""
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    async def _send_frame(self, frame: np.ndarray):
        """Send a frame to WebRTC clients if callback is set."""
        if self.frame_callback is not None:
            await self.frame_callback(frame)

    async def generate_videos_streaming(self, mode='universal', image_path=None):
        """
        Generate videos with real-time streaming support.

        Args:
            mode: Game mode ('universal', 'gta_drive', 'templerun')
            image_path: Path to initial image (if None, will prompt user)
        """
        # Load initial image
        if image_path is None:
            while True:
                try:
                    img_path = input("Please input the image path: ")
                    image = load_image(img_path.strip())
                    break
                except Exception as e:
                    print(f"Failed to load image from {img_path}: {e}")
        else:
            image = load_image(image_path)
            img_path = image_path

        # Prepare image
        image = self._resizecrop(image, 352, 640)
        image_tensor = self.frame_process(image)[None, :, None, :, :].to(
            dtype=self.weight_dtype, device=self.device)

        # Encode initial image
        padding_video = torch.zeros_like(image_tensor).repeat(
            1, 1, 4 * (self.args.max_num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image_tensor, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.vae.clip.encode_video(image_tensor)

        # Prepare noise
        sampled_noise = torch.randn(
            [1, 16, self.args.max_num_output_frames, 44, 80],
            device=self.device,
            dtype=self.weight_dtype
        )
        num_frames = (self.args.max_num_output_frames - 1) * 4 + 1

        # Prepare conditions
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }

        if mode == 'universal':
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        elif mode == 'gta_drive':
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(
            device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition

        # Run inference with streaming
        print("Starting streaming inference...")
        with torch.no_grad():
            await self._inference_with_streaming(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                mode=mode,
                output_folder=self.args.output_folder,
                name=os.path.basename(img_path)
            )

    async def _inference_with_streaming(
        self,
        noise: torch.Tensor,
        conditional_dict,
        mode='universal',
        output_folder=None,
        name=None
    ):
        """
        Modified inference loop with frame streaming.

        This is an async version of the pipeline inference that sends
        frames to WebRTC clients as they are generated.
        """
        batch_size, num_channels, num_frames, height, width = noise.shape
        num_frame_per_block = self.pipeline.num_frame_per_block

        assert num_frames % num_frame_per_block == 0
        num_blocks = num_frames // num_frame_per_block

        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        # Initialize caches
        self.pipeline.kv_cache1 = None
        self.pipeline.kv_cache_keyboard = None
        self.pipeline.kv_cache_mouse = None
        self.pipeline.crossattn_cache = None

        if self.pipeline.kv_cache1 is None:
            self.pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self.pipeline._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self.pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )

        current_start_frame = 0

        # Inference loop
        print(f"Generating {num_blocks} blocks...")
        for block_idx in range(num_blocks):
            print(f"Processing block {block_idx + 1}/{num_blocks}")

            noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

            # Denoising loop
            for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, num_frame_per_block],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                _, denoised_pred = self.pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=self._cond_current(
                        conditional_dict, current_start_frame, num_frame_per_block, mode=mode),
                    timestep=timestep,
                    kv_cache=self.pipeline.kv_cache1,
                    kv_cache_mouse=self.pipeline.kv_cache_mouse,
                    kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=current_start_frame * self.pipeline.frame_seq_length
                )

                if index < len(self.pipeline.denoising_step_list) - 1:
                    next_timestep = self.pipeline.denoising_step_list[index + 1]
                    noisy_input = self.pipeline.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * num_frame_per_block], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)

            # Update KV cache
            context_timestep = torch.ones_like(timestep) * self.pipeline.args.context_noise
            self.pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=self._cond_current(
                    conditional_dict, current_start_frame, num_frame_per_block, mode=mode),
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                kv_cache_mouse=self.pipeline.kv_cache_mouse,
                kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=current_start_frame * self.pipeline.frame_seq_length,
            )

            # Decode to video
            denoised_pred = denoised_pred.transpose(1, 2)
            video, vae_cache = self.pipeline.vae_decoder(denoised_pred.half(), *vae_cache)
            videos.append(video)

            # Convert to numpy and stream
            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video_np = np.ascontiguousarray(video_np)

            # Send each frame to WebRTC clients
            for frame_idx in range(video_np.shape[0]):
                frame = video_np[frame_idx]
                await self._send_frame(frame)
                await asyncio.sleep(0.03)  # ~30 FPS

            current_start_frame += num_frame_per_block

        print("Inference complete!")

    def _cond_current(self, conditional_dict, current_start_frame, num_frame_per_block, mode='universal'):
        """Extract current condition slice."""
        new_cond = {}
        new_cond["cond_concat"] = conditional_dict["cond_concat"][
            :, :, current_start_frame:current_start_frame + num_frame_per_block]
        new_cond["visual_context"] = conditional_dict["visual_context"]

        if mode != 'templerun':
            new_cond["mouse_cond"] = conditional_dict["mouse_cond"][
                :, :1 + 4 * (current_start_frame + num_frame_per_block - 1)]
        new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][
            :, :1 + 4 * (current_start_frame + num_frame_per_block - 1)]

        return new_cond


async def main():
    """Main entry point for WebRTC streaming inference."""
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)

    if args.enable_streaming:
        print("=" * 60)
        print("Matrix Game WebRTC Streaming Server")
        print("=" * 60)
        print(f"Server will start on: http://{args.server_host}:{args.server_port}")
        print("Open this URL in your browser to view the stream")
        print("=" * 60)

        # Create WebRTC server
        server = create_server()
        frame_callback = server.get_frame_callback()

        # Create inference engine with streaming callback
        inference = WebRTCStreamingInference(args, frame_callback=frame_callback)
        mode = inference.config.pop('mode', 'universal')

        # Start server in background
        server_task = asyncio.create_task(
            server.start(host=args.server_host, port=args.server_port))

        # Wait a bit for server to start
        await asyncio.sleep(2)

        print("\nServer started! Waiting for client connection...")
        print("Once connected, inference will begin automatically.\n")

        # Wait for client to connect (with optional timeout)
        client_connected = await server.wait_for_client(timeout=None)

        if not client_connected:
            print("No client connected. Exiting...")
            await server.shutdown()
            server_task.cancel()
            return

        print("\n🎬 Client connected! Starting streaming inference...\n")

        # Run inference
        try:
            await inference.generate_videos_streaming(mode=mode, image_path=args.img_path)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await server.shutdown()
            server_task.cancel()
    else:
        # Run without streaming (original mode)
        print("Running in non-streaming mode")
        inference = WebRTCStreamingInference(args, frame_callback=None)
        mode = inference.config.pop('mode', 'universal')
        await inference.generate_videos_streaming(mode=mode, image_path=args.img_path)


if __name__ == "__main__":
    asyncio.run(main())
