"""
Action Context and Block Mask Factory for WAN Model.

This module provides:
1. ActionContext: Encapsulates all action-related state during inference
2. BlockMaskFactory: Centralizes block mask creation logic (moved from Model)

OWNERSHIP:
    Pipeline (creates) → Model (distributes) → Block (routes) → ActionModule (consumes)

DESIGN RATIONALE:
    - Separates mask creation from model logic (Model should not create masks)
    - Bundles action parameters to prevent parameter explosion
    - Allows type-safe access to action configuration
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import math
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


@dataclass
class ActionContext:
    """
    Encapsulates action-related state for inference.

    Clean, minimal design with only essential fields.

    Attributes:
        mouse_cond: Mouse movement conditions [B, N_frames, C_mouse]
        keyboard_cond: Keyboard press conditions [B, N_frames, C_keyboard]
        kv_cache_mouse: Per-block KV cache for mouse attention
        kv_cache_keyboard: Per-block KV cache for keyboard attention
        num_frame_per_block: Number of frames per attention block

    Usage:
        action_ctx = ActionContext(
            mouse_cond=mouse_data,
            keyboard_cond=keyboard_data,
            num_frame_per_block=3,
        )
        model.forward(..., action_context=action_ctx)
    """

    # Input conditions
    mouse_cond: Optional[torch.Tensor] = None
    keyboard_cond: Optional[torch.Tensor] = None

    # KV caches (managed by pipeline)
    kv_cache_mouse: Optional[Dict[str, torch.Tensor]] = None
    kv_cache_keyboard: Optional[Dict[str, torch.Tensor]] = None

    # Runtime configuration
    num_frame_per_block: int = 1

    @property
    def has_any_condition(self) -> bool:
        """Check if any action conditioning is provided."""
        return self.mouse_cond is not None or self.keyboard_cond is not None


class BlockMaskFactory:
    """
    Factory for creating block attention masks.

    This class centralizes all mask creation logic that was previously
    scattered across CausalWanModel static methods.

    RESPONSIBILITY SHIFT:
        Before: Model creates masks lazily in _forward_train()
        After: Pipeline pre-creates masks before calling model

    Usage:
        # In Pipeline initialization
        mask_factory = BlockMaskFactory(device=device)

        # Before each inference block
        visual_mask = mask_factory.create_visual_mask(
            num_frames=3, frame_seqlen=880,
            num_frame_per_block=1, local_attn_size=-1
        )

        action_ctx = ActionContext(
            block_mask_mouse=mask_factory.create_action_mask(num_frames, 'mouse'),
            block_mask_keyboard=mask_factory.create_action_mask(num_frames, 'keyboard'),
            ...
        )
    """

    def __init__(self, device: torch.device | str = "cuda"):
        """
        Initialize mask factory.

        Args:
            device: Device to create masks on
        """
        self.device = torch.device(device) if isinstance(device, str) else device

    def create_visual_mask(
        self,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1
    ) -> BlockMask:
        """
        Create block-wise causal attention mask for visual self-attention.

        This is the primary mask for video latent tokens.

        Args:
            num_frames: Number of frames in the sequence
            frame_seqlen: Sequence length per frame (e.g., 880 for 60x104 patches)
            num_frame_per_block: Number of frames grouped into each causal block
            local_attn_size: Window size for temporal local attention (-1 for global)

        Returns:
            BlockMask for flex_attention

        Example:
            For num_frames=9, frame_seqlen=880, num_frame_per_block=3:
            - Total length: 9 * 880 = 7920 tokens
            - Blocks: [0:2640], [2640:5280], [5280:7920]
            - Each query can attend to all keys in its block (causal within block)
        """
        total_length = num_frames * frame_seqlen

        # Pad to multiple of 128 for flexattention efficiency
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length,
            device=self.device, dtype=torch.long
        )

        # Set block boundaries
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=self.device
        )

        for block_start in frame_indices:
            block_end = block_start + frame_seqlen * num_frame_per_block
            ends[block_start:block_end] = block_end

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                # Global attention within each block
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                # Local attention window
                return ((kv_idx < ends[q_idx]) &
                        (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                       (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=self.device
        )

        return block_mask

    def create_action_mask(
        self,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        action_type: str = 'mouse'
    ) -> BlockMask:
        """
        Create block-wise causal attention mask for action conditioning.

        Action masks have different KV sequence lengths because action embeddings
        are per-frame (not per-token).

        Args:
            num_frames: Number of frames
            frame_seqlen: Query sequence length per frame
            num_frame_per_block: Number of frames per block
            local_attn_size: Attention window size
            action_type: Type of action ('mouse' or 'keyboard')

        Returns:
            BlockMask for action attention

        Notes:
            - Query length: num_frames * frame_seqlen (visual tokens)
            - KV length: num_frames (action embeddings, one per frame)
        """
        total_length = num_frames * frame_seqlen

        # Padding
        padded_length = math.ceil(total_length / 32) * 32 - total_length
        padded_length_kv = math.ceil(num_frames / 32) * 32 - num_frames

        ends = torch.zeros(
            total_length + padded_length,
            device=self.device, dtype=torch.long
        )

        # Set block boundaries (in terms of frame indices for KV)
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=self.device
        )

        cnt = num_frame_per_block
        for block_start in frame_indices:
            block_end = block_start + frame_seqlen * num_frame_per_block
            ends[block_start:block_end] = cnt
            cnt += num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) &
                        (kv_idx >= (ends[q_idx] - local_attn_size))) | \
                       (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=num_frames + padded_length_kv,
            _compile=False,
            device=self.device
        )

        return block_mask


