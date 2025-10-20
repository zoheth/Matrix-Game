from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

class IActionPreprocessor(nn.Module, ABC):
    """ (B, N_frames, C) -> (B, T_q_or_k, C_windowed)"""
    def __init__(self, vae_time_compression_ratio: int, windows_size: int):
        super().__init__()
        # self.vae_time_compression_ratio = vae_time_compression_ratio
        # self.windows_size = windows_size
        self.pat_t = vae_time_compression_ratio * windows_size
        
    @abstractmethod
    def forward(self, condition: torch.Tensor, N_feats: int, is_causal: bool, num_frame_per_block: int) -> torch.Tensor:
        pass
    
class IAttentionInjector(nn.Module, ABC):
    """"""
    @abstractmethod
    def forward(
        self,
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
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass
    
class KVCacheManager(nn.Module):
    def __init__(self, local_attn_size:int, sink_size: int=0):
        super().__init__()
        self.max_attention_size = local_attn_size
        self.sink_tokens = sink_size
        
    def update_cache(self, kv_cache, k, v, start_frame) -> Tuple[torch.Tensor, torch.Tensor]:
        pass