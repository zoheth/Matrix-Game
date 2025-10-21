from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from einops import rearrange

from wan.modules.modular_action.interfaces import (
    IAttentionInjector,
    IActionPreprocessor,
    KVCacheManager,
    FlashInferAttentionCore,
    WanRMSNorm,
)
from wan.modules.modular_action.action_config import ActionConfig
from wan.modules.posemb_layers import apply_rotary_emb


class MousePreprocessor(IActionPreprocessor):
    """
    Preprocessor for mouse condition data.
    Fuses mouse condition with hidden states using sliding window.
    """

    def __init__(self, vae_time_compression_ratio: int, windows_size: int):
        super().__init__(vae_time_compression_ratio, windows_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mouse_condition: torch.Tensor,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        """
        Fuse hidden states with mouse condition using sliding window.

        Args:
            hidden_states: [BS, T_q, C_hidden] - Reshaped hidden states
            mouse_condition: [B, N_frames, C_mouse] - Raw mouse condition
            is_causal: Whether in causal mode
            num_frame_per_block: Number of frames per block in causal mode

        Returns:
            Fused features [BS, T_q, C_hidden + C_mouse * windows]
        """
        B, N_frames, C_mouse = mouse_condition.shape
        BS, T_q, C_hidden = hidden_states.shape

        # Padding for sliding window
        pad_t = self.pat_t  # vae_time_compression_ratio * windows_size
        pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
        mouse_condition_padded = torch.cat([pad, mouse_condition], dim=1)

        # Extract windows
        if is_causal:
            N_feats = (N_frames - 1) // self.pat_t + 1
            # Start from proper position for causal mode
            start_idx = self.pat_t * (N_feats - num_frame_per_block - self.pat_t // self.pat_t) + pad_t
            mouse_condition_padded = mouse_condition_padded[:, start_idx:, :]
            group_mouse = [
                mouse_condition_padded[
                    :, self.pat_t * (i - self.pat_t // self.pat_t) + pad_t : i * self.pat_t + pad_t, :
                ]
                for i in range(num_frame_per_block)
            ]
        else:
            N_feats = T_q  # Should match temporal shape
            group_mouse = [
                mouse_condition_padded[
                    :, self.pat_t * (i - self.pat_t // self.pat_t) + pad_t : i * self.pat_t + pad_t, :
                ]
                for i in range(N_feats)
            ]

        # Stack and expand: [B, T_q, pad_t, C_mouse]
        group_mouse = torch.stack(group_mouse, dim=1)

        # Expand to match spatial dimension: [B, T_q, pad_t, C_mouse] -> [BS, T_q, pad_t * C_mouse]
        S = BS // B
        group_mouse = group_mouse.unsqueeze(2).expand(B, -1, S, pad_t, C_mouse)
        group_mouse = rearrange(group_mouse, "B T S W C -> (B S) T (W C)")

        # Concatenate with hidden states
        fused = torch.cat([hidden_states, group_mouse], dim=-1)
        return fused


class KeyboardPreprocessor(IActionPreprocessor):
    """Preprocessor for keyboard condition data."""

    def __init__(self, vae_time_compression_ratio: int, windows_size: int, hidden_size: int):
        super().__init__(vae_time_compression_ratio, windows_size)
        # Keyboard embedding layers
        self.keyboard_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(
        self,
        keyboard_condition: torch.Tensor,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        """
        Process keyboard condition into windowed features.

        Args:
            keyboard_condition: [B, N_frames, C_keyboard] - Raw keyboard condition
            is_causal: Whether in causal mode
            num_frame_per_block: Number of frames per block in causal mode

        Returns:
            Windowed keyboard features [B, T_k, C_keyboard * windows]
        """
        B, N_frames, C = keyboard_condition.shape

        # Padding
        pad_t = self.pat_t
        pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
        keyboard_condition_padded = torch.cat([pad, keyboard_condition], dim=1)

        # Embed
        keyboard_condition_embedded = self.keyboard_embed(keyboard_condition_padded)

        # Extract windows
        if is_causal:
            N_feats = (N_frames - 1) // self.pat_t + 1
            start_idx = self.pat_t * (N_feats - num_frame_per_block - self.pat_t // self.pat_t) + pad_t
            keyboard_condition_embedded = keyboard_condition_embedded[:, start_idx:, :]
            group_keyboard = [
                keyboard_condition_embedded[
                    :, self.pat_t * (i - self.pat_t // self.pat_t) + pad_t : i * self.pat_t + pad_t, :
                ]
                for i in range(num_frame_per_block)
            ]
        else:
            N_feats = (N_frames - 1) // self.pat_t + 1
            group_keyboard = [
                keyboard_condition_embedded[
                    :, self.pat_t * (i - self.pat_t // self.pat_t) + pad_t : i * self.pat_t + pad_t, :
                ]
                for i in range(N_feats)
            ]

        # Stack and flatten: [B, T_k, windows * C]
        group_keyboard = torch.stack(group_keyboard, dim=1)
        group_keyboard = group_keyboard.reshape(B, group_keyboard.shape[1], -1)

        return group_keyboard


class MouseInjector(IAttentionInjector):
    """
    Mouse condition injector using self-attention with RoPE.
    Based on the original ActionModule mouse attention implementation.
    """

    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.action_config = action_config

        # Preprocessor
        self.preprocessor = MousePreprocessor(
            action_config.vae_time_compression_ratio, action_config.windows_size
        )

        # MLP to fuse hidden states with mouse condition
        mouse_input_dim = (
            action_config.img_hidden_size
            + action_config.mouse_dim_in
            * action_config.vae_time_compression_ratio
            * action_config.windows_size
        )
        self.mouse_mlp = nn.Sequential(
            nn.Linear(mouse_input_dim, action_config.mouse_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.mouse_hidden_dim, action_config.mouse_hidden_dim),
            nn.LayerNorm(action_config.mouse_hidden_dim),
        )

        # QKV projection
        self.t_qkv = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.mouse_hidden_dim * 3,
            bias=action_config.qkv_bias,
        )

        # QK normalization
        head_dim = action_config.mouse_head_dim
        self.q_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        # Output projection
        self.proj_mouse = nn.Linear(
            action_config.mouse_hidden_dim, action_config.img_hidden_size, bias=action_config.qkv_bias
        )

        # KV Cache manager
        self.kv_cache_manager = KVCacheManager(
            local_attn_size=action_config.local_attn_size, sink_size=0
        )

        # Attention core
        self.attn_core = FlashInferAttentionCore()

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for mouse condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_mouse] - Mouse condition
            freqs_cis: (cos, sin) RoPE frequencies
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            is_causal: Whether to use causal attention
            kv_cache: KV cache for incremental decoding
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block
            block_mask: Optional block mask

        Returns:
            Output hidden states [B, T*S, C_img]
        """
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        # Reshape to (B*S, T, C_img)
        hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=S)

        # Fuse with mouse condition
        fused_features = self.preprocessor(hidden_states, condition, is_causal, num_frame_per_block)

        # MLP
        fused_features = self.mouse_mlp(fused_features)

        # QKV projection and split
        qkv = self.t_qkv(fused_features)  # [BS, T, 3*C_mouse]
        q, k, v = rearrange(
            qkv, "BS T (K H D) -> K BS T H D", K=3, H=self.action_config.heads_num
        )

        # QK normalization
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis, head_first=False, start_offset=start_frame)

        # Attention computation
        if is_causal and kv_cache is not None:
            # Update KV cache and get window
            k_window, v_window, _, _ = self.kv_cache_manager.update_cache(kv_cache, k, v, num_frame_per_block)
            # Compute attention with cached KV
            attn_output = self.attn_core(q, k_window, v_window, causal=False)
        else:
            # Regular attention
            attn_output = self.attn_core(q, k, v, causal=is_causal)

        # Reshape and project: [BS, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_mouse(attn_output)

        # Residual connection
        output = x + attn_output
        return output


class KeyboardInjector(IAttentionInjector):
    """
    Keyboard condition injector using cross-attention with RoPE.
    Based on the original ActionModule keyboard attention implementation.
    """

    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.action_config = action_config

        # Preprocessor
        self.preprocessor = KeyboardPreprocessor(
            action_config.vae_time_compression_ratio,
            action_config.windows_size,
            action_config.hidden_size,
        )

        # Query projection (from hidden states)
        self.mouse_attn_q = nn.Linear(
            action_config.img_hidden_size,
            action_config.keyboard_hidden_dim,
            bias=action_config.qkv_bias,
        )

        # Key-Value projection (from keyboard condition)
        self.keyboard_attn_kv = nn.Linear(
            action_config.hidden_size
            * action_config.windows_size
            * action_config.vae_time_compression_ratio,
            action_config.keyboard_hidden_dim * 2,
            bias=action_config.qkv_bias,
        )

        # QK normalization
        head_dim = action_config.keyboard_head_dim
        self.q_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        # Output projection
        self.proj_keyboard = nn.Linear(
            action_config.keyboard_hidden_dim,
            action_config.img_hidden_size,
            bias=action_config.qkv_bias,
        )

        # KV Cache manager
        self.kv_cache_manager = KVCacheManager(
            local_attn_size=action_config.local_attn_size, sink_size=0
        )

        # Attention core
        self.attn_core = FlashInferAttentionCore()

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for keyboard condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_keyboard] - Keyboard condition
            freqs_cis: (cos, sin) RoPE frequencies
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            is_causal: Whether to use causal attention
            kv_cache: KV cache for incremental decoding
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block
            block_mask: Optional block mask

        Returns:
            Output hidden states [B, T*S, C_img]
        """
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        # Process keyboard condition
        group_keyboard = self.preprocessor(condition, is_causal, num_frame_per_block)

        # Compute Query from hidden states
        q = self.mouse_attn_q(x)  # [B, T*S, C_keyboard]
        q = q.view(B, T_S, self.action_config.heads_num, self.action_config.keyboard_head_dim)

        # Compute Key-Value from keyboard condition
        keyboard_kv = self.keyboard_attn_kv(group_keyboard)  # [B, T_k, 2*C_keyboard]
        k, v = rearrange(
            keyboard_kv,
            "B T (K H D) -> K B T H D",
            K=2,
            H=self.action_config.heads_num,
            D=self.action_config.keyboard_head_dim,
        )

        # QK normalization
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # Apply RoPE to both Q and K
        # For cross-attention with temporal alignment:
        # Reshape Q: [B, T*S, H, D] -> [B*S, T, H, D]
        q = rearrange(q, "B (T S) H D -> (B S) T H D", T=T, S=S)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis, head_first=False, start_offset=start_frame)

        # Expand K, V to match spatial dimension
        # k, v: [B, T_k, H, D] -> [B*S, T_k, H, D]
        k = k.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, k.shape[-2], k.shape[-1])
        v = v.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, v.shape[-2], v.shape[-1])

        # Attention computation
        if is_causal and kv_cache is not None:
            # Update KV cache (only store for first spatial location to save memory)
            k_cache_update = k[: S].mean(dim=0, keepdim=True)  # Average across S or just use first
            v_cache_update = v[: S].mean(dim=0, keepdim=True)

            k_window, v_window, _, _ = self.kv_cache_manager.update_cache(
                kv_cache, k_cache_update, v_cache_update, num_frame_per_block
            )

            # Expand window to all spatial locations
            k_window = k_window.expand(B * S, -1, -1, -1)
            v_window = v_window.expand(B * S, -1, -1, -1)

            # Compute attention with cached KV
            attn_output = self.attn_core(q, k_window, v_window, causal=False)
        else:
            # Regular cross-attention
            attn_output = self.attn_core(q, k, v, causal=False)

        # Reshape and project: [B*S, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_keyboard(attn_output)

        # Residual connection
        output = x + attn_output
        return output
