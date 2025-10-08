import os
import argparse
import torch
import numpy as np

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange
from pipeline import CausalInferencePipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the checkpoint")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0000.png", help="Path to the image")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=150,
                        help="Number of output latent frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0", help="Path to the VAE model folder")
    parser.add_argument("--enable_profile", action="store_true", help="Enable torch profiling")
    parser.add_argument("--vae_compile_mode", type=str, default="auto", choices=["auto", "force", "none"],
                        help="VAE decoder compile mode: auto (use cache if available), force (recompile), none (no compile)")
    args = parser.parse_args()
    return args

class InteractiveGameInference:
    def __init__(self, args):
        self.args = args
        self.enable_profile = args.enable_profile
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        # Initialize pipeline
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        # Load and optionally compile VAE decoder
        compiled_model_path = os.path.join(self.args.pretrained_model_path, "compiled_vae_decoder.pt")
        compile_mode = self.args.vae_compile_mode

        # Load base model
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()

        # Handle compilation based on mode
        if compile_mode == "none":
            print("VAE decoder compilation skipped (mode=none)")
        elif compile_mode == "force":
            print("Force compiling VAE decoder...")
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
            print(f"Saving compiled model to {compiled_model_path}...")
            torch.save(current_vae_decoder, compiled_model_path)
            print("Compiled model saved!")
        elif compile_mode == "auto":
            if os.path.exists(compiled_model_path):
                print(f"Loading cached compiled model from {compiled_model_path}...")
                current_vae_decoder = torch.load(compiled_model_path, map_location=self.device, weights_only=False)
                print("Cached compiled model loaded!")
            else:
                print(f"No cached compiled model found. Compiling VAE decoder (first run)...")
                current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
                print(f"Saving compiled model to {compiled_model_path}...")
                torch.save(current_vae_decoder, compiled_model_path)
                print("Compiled model saved for future use!")

        pipeline = CausalInferencePipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
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
    
    def generate_videos(self):
        mode = self.config.pop('mode')
        assert mode in ['universal', 'gta_drive', 'templerun']

        with torch.profiler.record_function("1_Data_Preparation"):
            with torch.profiler.record_function("1.1_Image_Loading_Resize"):
                image = load_image(self.args.img_path)
                image = self._resizecrop(image, 352, 640)
                image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

            with torch.profiler.record_function("1.2_VAE_Encode_FirstFrame"):
                # Encode the input image as the first latent
                padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.num_output_frames - 1), 1, 1)
                img_cond = torch.concat([image, padding_video], dim=2)
                tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
                img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

            with torch.profiler.record_function("1.3_Condition_Preparation"):
                mask_cond = torch.ones_like(img_cond)
                mask_cond[:, :, 1:] = 0
                cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)

            with torch.profiler.record_function("1.4_CLIP_Visual_Context"):
                visual_context = self.vae.clip.encode_video(image)

            with torch.profiler.record_function("1.5_Noise_Sampling"):
                sampled_noise = torch.randn(
                    [1, 16,self.args.num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
                )
        num_frames = (self.args.num_output_frames - 1) * 4 + 1

        with torch.profiler.record_function("1.6_Action_Conditions_Setup"):
            conditional_dict = {
                "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
                "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
            }

            if mode == 'universal':
                cond_data = Bench_actions_universal(num_frames)
                mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
                conditional_dict['mouse_cond'] = mouse_condition
            elif mode == 'gta_drive':
                cond_data = Bench_actions_gta_drive(num_frames)
                mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
                conditional_dict['mouse_cond'] = mouse_condition
            else:
                cond_data = Bench_actions_templerun(num_frames)
            keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['keyboard_cond'] = keyboard_condition

        with torch.no_grad():
            with torch.profiler.record_function("2_Pipeline_Inference"):
                videos = self.pipeline.inference(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    return_latents=False,
                    mode=mode,
                    profile=self.enable_profile
                )

        with torch.profiler.record_function("3_Video_Postprocessing"):
            videos_tensor = torch.cat(videos, dim=1)
            videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
            videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(videos)
            mouse_icon = 'assets/images/mouse.png'
            if mode != 'templerun':
                config = (
                    keyboard_condition[0].float().cpu().numpy(),
                    mouse_condition[0].float().cpu().numpy()
                )
            else:
                config = (
                    keyboard_condition[0].float().cpu().numpy()
                )

        with torch.profiler.record_function("4_Video_Export"):
            process_video(video.astype(np.uint8), self.args.output_folder+f'/demo.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
            process_video(video.astype(np.uint8), self.args.output_folder+f'/demo_icon.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=True, mode=mode)
        print("Done")

def main():
    """Main entry point for video generation."""
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    pipeline = InteractiveGameInference(args)

    if args.enable_profile:
        # Profiled run
        print("\nStarting profiled run...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=True,
        ) as prof:
            pipeline.generate_videos()

        # Export trace
        trace_path = os.path.join(args.output_folder, "profile_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\nProfile trace saved to: {trace_path}")
        print(f"File size: {os.path.getsize(trace_path) / 1024 / 1024:.2f} MB")

    else:
        pipeline.generate_videos()

if __name__ == "__main__":
    main()