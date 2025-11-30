"""
Modular Action Module - Clean implementation from first principles
"""

from typing import Dict, Optional

import torch
from torch import nn

from wan.modules.modular_action.action_config import ActionConfig
from wan.modules.modular_action.injectors import MouseInjector, KeyboardInjector


class ActionModule(nn.Module):
    """
    Action condition injector using mouse and keyboard attention.

    Injects action conditions (mouse and keyboard) into hidden states
    via attention mechanisms with RoPE positional encoding.
    """

    def __init__(self, action_config: ActionConfig):
        """
        Initialize ActionModule.

        Args:
            action_config: Complete configuration for the action module
        """
        super().__init__()
        self.config = action_config

        # Initialize injectors based on config
        self.mouse_injector = MouseInjector(action_config) if action_config.enable_mouse else None
        self.keyboard_injector = KeyboardInjector(action_config) if action_config.enable_keyboard else None

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: tuple,
        mouse_condition: Optional[torch.Tensor] = None,
        keyboard_condition: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache_mouse: Optional[Dict[str, torch.Tensor]] = None,
        kv_cache_keyboard: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 3,
    ) -> torch.Tensor:
        """
        Inject action conditions into hidden states.

        Args:
            x: [B, T*H*W, C] - Hidden states
            grid_sizes: (F, H, W) - Latent grid dimensions
            mouse_condition: [B, N_frames, C_mouse] - Mouse condition
            keyboard_condition: [B, N_frames, C_keyboard] - Keyboard condition
            is_causal: Whether to use causal attention
            kv_cache_mouse: Mouse KV cache
            kv_cache_keyboard: Keyboard KV cache
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block

        Returns:
            [B, T*H*W, C] - Processed hidden states
        """
        tt, th, tw = grid_sizes
        B = x.shape[0]
        assert tt * th * tw == x.shape[1], f"Sequence length mismatch: {tt}*{th}*{tw}={tt*th*tw} != {x.shape[1]}"

        hidden_states = x

        # Mouse injection
        if self.mouse_injector is not None and mouse_condition is not None:
            hidden_states = self.mouse_injector(
                hidden_states,
                condition=mouse_condition,
                spatial_shape=(th, tw),
                temporal_shape=tt,
                is_causal=is_causal,
                kv_cache=kv_cache_mouse,
                start_frame=start_frame,
                num_frame_per_block=num_frame_per_block,
            )

        # Keyboard injection
        if self.keyboard_injector is not None and keyboard_condition is not None:
            hidden_states = self.keyboard_injector(
                hidden_states,
                condition=keyboard_condition,
                spatial_shape=(th, tw),
                temporal_shape=tt,
                is_causal=is_causal,
                kv_cache=kv_cache_keyboard,
                start_frame=start_frame,
                num_frame_per_block=num_frame_per_block,
            )

        return hidden_states

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Remap legacy checkpoint keys to new modular structure."""
        # Key mappings: old_key -> new_key
        key_mappings = {
            # Mouse injector
            'mouse_mlp.0.weight': 'mouse_injector.mouse_mlp.0.weight',
            'mouse_mlp.0.bias': 'mouse_injector.mouse_mlp.0.bias',
            'mouse_mlp.2.weight': 'mouse_injector.mouse_mlp.2.weight',
            'mouse_mlp.2.bias': 'mouse_injector.mouse_mlp.2.bias',
            'mouse_mlp.3.weight': 'mouse_injector.mouse_mlp.3.weight',
            'mouse_mlp.3.bias': 'mouse_injector.mouse_mlp.3.bias',
            't_qkv.weight': 'mouse_injector.t_qkv.weight',
            't_qkv.bias': 'mouse_injector.t_qkv.bias',
            'img_attn_q_norm.weight': 'mouse_injector.q_norm.weight',
            'img_attn_k_norm.weight': 'mouse_injector.k_norm.weight',
            'proj_mouse.weight': 'mouse_injector.proj_mouse.weight',
            'proj_mouse.bias': 'mouse_injector.proj_mouse.bias',
            # Keyboard injector
            'keyboard_embed.0.weight': 'keyboard_injector.preprocessor.keyboard_embed.0.weight',
            'keyboard_embed.0.bias': 'keyboard_injector.preprocessor.keyboard_embed.0.bias',
            'keyboard_embed.2.weight': 'keyboard_injector.preprocessor.keyboard_embed.2.weight',
            'keyboard_embed.2.bias': 'keyboard_injector.preprocessor.keyboard_embed.2.bias',
            'mouse_attn_q.weight': 'keyboard_injector.mouse_attn_q.weight',
            'mouse_attn_q.bias': 'keyboard_injector.mouse_attn_q.bias',
            'keyboard_attn_kv.weight': 'keyboard_injector.keyboard_attn_kv.weight',
            'keyboard_attn_kv.bias': 'keyboard_injector.keyboard_attn_kv.bias',
            'key_attn_q_norm.weight': 'keyboard_injector.q_norm.weight',
            'key_attn_k_norm.weight': 'keyboard_injector.k_norm.weight',
            'proj_keyboard.weight': 'keyboard_injector.proj_keyboard.weight',
            'proj_keyboard.bias': 'keyboard_injector.proj_keyboard.bias',
        }

        # Remap keys if legacy format detected
        remapped = 0
        for old_key, new_key in key_mappings.items():
            full_old_key = prefix + old_key
            if full_old_key in state_dict:
                state_dict[prefix + new_key] = state_dict.pop(full_old_key)
                remapped += 1

        if remapped > 0:
            print(f"[ActionModule] Remapped {remapped} keys from legacy format")

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self) -> str:
        """Readable representation."""
        return (
            f"ActionModule(\n"
            f"  mouse={self.config.enable_mouse}, "
            f"keyboard={self.config.enable_keyboard}, "
            f"hidden_size={self.config.img_hidden_size}, "
            f"heads={self.config.heads_num}\n"
            f")"
        )
