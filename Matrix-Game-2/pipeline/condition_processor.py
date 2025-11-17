"""
Conditional data processing utilities.

This module handles the preparation and slicing of conditional inputs
(visual context, action conditions) for the diffusion model.
"""

from typing import Dict, Optional
import torch

from pipeline.config import VAEConfig
from pipeline.action_strategies import ActionDict


class ConditionProcessor:
    """
    Processes and manages conditional inputs for the diffusion model.

    This includes:
    - Visual context (CLIP features)
    - Concatenated conditioning (mask + initial frame latents)
    - Action conditioning (mouse + keyboard)
    """

    def __init__(self, vae_config: VAEConfig, mode: str):
        """
        Initialize the condition processor.

        Args:
            vae_config: VAE configuration for temporal calculations
            mode: Game mode ('universal', 'gta_drive', 'templerun')
        """
        self.vae_config = vae_config
        self.mode = mode

    def get_action_sequence_length(self, current_block_end: int) -> int:
        """
        Calculate the action conditioning sequence length up to a given block.

        The action condition has higher temporal resolution than latents.

        Args:
            current_block_end: Index of the last latent frame in the current block

        Returns:
            Action sequence length
        """
        return self.vae_config.get_action_condition_length(current_block_end)

    def slice_block_conditions(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        current_start_frame: int,
        num_frames: int,
        replace_action: Optional[ActionDict] = None
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract conditions for the current block and optionally update actions.

        Args:
            conditional_dict: Full conditional dictionary containing all conditions
            current_start_frame: Start frame index of the current block (in latent space)
            num_frames: Number of frames in the current block
            replace_action: Optional action to replace in the sequence

        Returns:
            Tuple of (sliced_conditions, updated_full_conditions)
        """
        new_cond = {}

        # Slice visual concatenated conditioning
        new_cond["cond_concat"] = conditional_dict["cond_concat"][
            :, :, current_start_frame: current_start_frame + num_frames
        ]

        # Visual context is shared across all blocks
        new_cond["visual_context"] = conditional_dict["visual_context"]

        # Update actions if replacement is provided
        if replace_action is not None:
            conditional_dict = self._update_actions(
                conditional_dict,
                current_start_frame,
                num_frames,
                replace_action
            )

        # Calculate action sequence length
        current_block_end = current_start_frame + num_frames
        action_seq_len = self.get_action_sequence_length(current_block_end)

        # Slice action conditions
        if self.mode != 'templerun':
            new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, :action_seq_len]

        new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, :action_seq_len]

        return new_cond, conditional_dict

    def _update_actions(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        current_start_frame: int,
        num_frames: int,
        replace_action: ActionDict
    ) -> Dict[str, torch.Tensor]:
        """
        Update the action conditioning sequence with new actions.

        This replaces actions for the current block in the full sequence.

        Args:
            conditional_dict: Full conditional dictionary
            current_start_frame: Start frame index (in latent space)
            num_frames: Number of frames in the current block
            replace_action: New actions to insert

        Returns:
            Updated conditional_dict
        """
        # Calculate the range of action frames to update
        if current_start_frame == 0:
            # First block: 1 + 4 * (num_frames - 1) action frames
            action_frame_count = self.vae_config.get_action_condition_length(num_frames)
        else:
            # Subsequent blocks: 4 * num_frames action frames
            action_frame_count = self.vae_config.temporal_compression * num_frames

        # Calculate end position in the full action sequence
        final_frame = self.get_action_sequence_length(current_start_frame + num_frames)

        # Calculate start position for replacement
        start_pos = final_frame - action_frame_count

        # Replace mouse actions (if applicable)
        if self.mode != 'templerun' and 'mouse' in replace_action:
            mouse_action = replace_action['mouse'][None, None, :]  # [1, 1, action_dim]
            conditional_dict["mouse_cond"][:, start_pos:final_frame] = mouse_action.repeat(
                1, action_frame_count, 1
            )

        # Replace keyboard actions
        if 'keyboard' in replace_action:
            keyboard_action = replace_action['keyboard'][None, None, :]  # [1, 1, action_dim]
            conditional_dict["keyboard_cond"][:, start_pos:final_frame] = keyboard_action.repeat(
                1, action_frame_count, 1
            )

        return conditional_dict

    @staticmethod
    def create_initial_conditions(
        cond_concat: torch.Tensor,
        visual_context: torch.Tensor,
        mouse_cond: Optional[torch.Tensor] = None,
        keyboard_cond: Optional[torch.Tensor] = None,
        mode: str = 'universal'
    ) -> Dict[str, torch.Tensor]:
        """
        Create the initial conditional dictionary.

        Args:
            cond_concat: Concatenated conditioning [batch, channels, frames, h, w]
            visual_context: Visual context from CLIP [batch, seq_len, dim]
            mouse_cond: Mouse action conditioning [batch, seq_len, action_dim]
            keyboard_cond: Keyboard action conditioning [batch, seq_len, action_dim]
            mode: Game mode

        Returns:
            Conditional dictionary
        """
        conditional_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context
        }

        if mode != 'templerun':
            if mouse_cond is None:
                raise ValueError("mouse_cond is required for non-templerun modes")
            conditional_dict["mouse_cond"] = mouse_cond

        if keyboard_cond is None:
            raise ValueError("keyboard_cond is required for all modes")

        conditional_dict["keyboard_cond"] = keyboard_cond

        return conditional_dict
