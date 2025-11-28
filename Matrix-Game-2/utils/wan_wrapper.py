import types
from typing import List, Optional
import torch
from torch import nn
from einops import rearrange
from utils.scheduler import FlowMatchingScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel #, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
# from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.action_context import ActionContext


class WanVAEWrapper(torch.nn.Module): # todo
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="skyreels_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_config="",
            timestep_shift=5.0,
            is_causal=True,
    ):
        super().__init__()
        print(model_config)
        self.model = CausalWanModel.from_config(model_config)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        # Store timestep_shift for pipeline to use when creating scheduler
        self.timestep_shift = timestep_shift

        self.seq_len = 15 * 880 # 32760  # [1, 15, 16, 60, 104]

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    # Conversion methods are now handled by scheduler.convert_flow_to_x0()
    # Keeping these as compatibility wrappers that delegate to scheduler

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        kv_cache_mouse: Optional[List[dict]] = None,
        kv_cache_keyboard: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:

        assert noisy_image_or_video.shape[1] == 16
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Extract action conditions from conditional_dict
        mouse_cond = conditional_dict.get('mouse_cond')
        keyboard_cond = conditional_dict.get('keyboard_cond')

        # Create ActionContext if action conditions exist
        action_context = None
        if mouse_cond is not None or keyboard_cond is not None:
            action_context = ActionContext(
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                # Note: kv_cache lists will be indexed per-block inside the model
                kv_cache_mouse=None,
                kv_cache_keyboard=None,
            )

        # Build model input (exclude action fields)
        model_cond = {
            k: v for k, v in conditional_dict.items()
            if k not in ('mouse_cond', 'keyboard_cond')
        }

        with torch.profiler.record_function("DiffusionModel_Forward"):
            if kv_cache is not None:
                flow_pred = self.model(
                    noisy_image_or_video.to(self.model.dtype),
                    t=input_timestep,
                    **model_cond,
                    action_context=action_context,
                    kv_cache=kv_cache,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    current_start=current_start,
                    cache_start=cache_start
                )
            else:
                flow_pred = self.model(
                    noisy_image_or_video.to(self.model.dtype),
                    t=input_timestep,
                    **model_cond,
                    action_context=action_context
                )

        return flow_pred, flow_pred

