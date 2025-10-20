from typing import Any

import torch
from torch import nn

from wan.modules.posemb_layers import get_nd_rotary_pos_embed
from wan.modules.modular_action.action_config import ActionConfig

class ActionModule(nn.Module):
    def __init__(
        self,
        th: int,
        tw: int,
        action_config: ActionConfig,
        max_video_length: int = 7500,
    ):
        super().__init__()
        
        self.th, self.tw = th, tw
        self.S = th * tw
        self.patch_size = action_config.patch_size
        
        self.action_config = action_config
        
        self.freqs_cos, self.freqs_sin = self.get_rotary_pos_embed(max_video_length, th, tw, action_config.rope_dim_list, start_offset)
        
        self.injectors = nn.ModuleList()
            
        
    def get_rotary_pos_embed(self, video_length: int, height: int, width: int, dim_list: list, start_offset: int):
        num_spatial_dims = 3
        
        total_frames = video_length + start_offset
        latent_size_raw = [total_frames, height, width]
        
        if isinstance(self.patch_size, int):
            patch_sizes = [self.patch_size] * num_spatial_dims
        else:
            patch_sizes = self.patch_size
            
        rope_grid_sizes = [size // patch for size, patch in zip(latent_size_raw, patch_sizes)]
        
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            dim_list,
            rope_grid_sizes,
            theta=self.action_config.rope_theta,
            use_real=True,
            theta_rescale_factor=1.0,
        )
        
        tokens_per_frame = rope_grid_sizes[1] * rope_grid_sizes[2]  # H * W
        frames_per_temporal_patch = self.patch_size[0]
        tokens_to_keep = (video_length * tokens_per_frame) // frames_per_temporal_patch
        
        return freqs_cos[-tokens_to_keep:], freqs_sin[-tokens_to_keep:]