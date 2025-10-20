from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from wan.modules.modular_action.interfaces import IAttentionInjector, IActionPreprocessor, KVCacheManager
from wan.modules.modular_action.action_config import ActionConfig

class MousePreprocessor(IActionPreprocessor):
    def __init__(self, vae_time_compression_ratio: int, windows_size: int):
        super().__init__(vae_time_compression_ratio, windows_size)
        
    def forward(self, condition: torch.Tensor, N_feats: int, is_causal: bool, num_frame_per_block: int) -> torch.Tensor:
        B, N_frames, C = condition.shape
        T_q_or_k = N_frames // self.pat_t
        condition = condition[:, : T_q_or_k * self.pat_t, :].reshape(B, T_q_or_k, self.pat_t * C)
        return condition

class MouseInjector(IAttentionInjector):
    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.preprocessor = MousePreprocessor(action_config.vae_time_compression_ratio, action_config.windows_size)
        
        self.attn_qkv = nn.Linear(action_config.mouse_dim * action_config.windows_size * action_config.vae_time_compression_ratio, action_config.attn_dim * 3)
        
        self.attn_proj = nn.Linear(action_config.attn_dim, action_config.hidden_size)
        self.q_norm = WanRMSNorm(...)
        self.k_norm = WanRMSNorm(...)
        self.kv_cache_manager = KVCacheManager(...)
        self.attn_core = AttentionCore()
        
    def forward(self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        freqs_cis_3d: Tuple[torch.Tensor, torch.Tensor],
        freqs_cis_1d: Tuple[torch.Tensor, torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
        block_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        
        B, T_S, C = x.shape
        T_q = temporal_shape
        S = spatial_shape[0] * spatial_shape[1]
        
        
