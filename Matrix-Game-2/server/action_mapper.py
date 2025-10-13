"""
Action parsing and mapping for different game modes
"""
from typing import Dict, Optional
import torch

from server.config import GameModeConfig


class ActionMapper:
    """
    Maps client input to model-compatible action tensors.

    IMPORTANT: This follows the behavior of inference.py where keyboard and mouse (camera)
    actions are PAIRED (both can be active simultaneously).
    """

    def __init__(self, config: GameModeConfig, device: torch.device, dtype: torch.dtype):
        """
        Initialize action mapper

        Args:
            config: Game mode configuration
            device: PyTorch device
            dtype: PyTorch dtype for tensors
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.action_config = config.action_config

        # Track the last action state for pairing (like in inference.py)
        self.last_keyboard_key: Optional[str] = None
        self.last_camera_key: Optional[str] = None

    def parse_action(self, action_data: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Parse client action to model input format.

        IMPORTANT: Each action is used ONCE then resets to default (no movement).
        This matches the behavior in inference_streaming.py where each frame uses
        the provided action then returns to neutral state (like 'u' key).

        For example, in universal mode:
        - Pressing 'w' (forward) creates keyboard=[1,0,0,0] for ONE frame
        - Next frame automatically returns to keyboard=[0,0,0,0] (no movement)
        - Pressing 'w' + 'l' together creates keyboard=[1,0,0,0] + mouse=[0, 0.1] for ONE frame

        Args:
            action_data: Raw action from client (e.g., {"type": "keyboard", "key": "w"})

        Returns:
            Dictionary with 'mouse' and 'keyboard' tensors for this single action,
            or None if invalid. State is consumed and reset after this call.
        """
        if action_data['type'] != 'keyboard':
            return None

        key = action_data['key']
        has_mouse = self.action_config.has_mouse
        camera_map = self.action_config.camera_map
        keyboard_map = self.action_config.keyboard_map

        # Set the state for THIS action only
        if has_mouse and camera_map and key in camera_map:
            # This is a camera control key
            self.last_camera_key = key
        elif key in keyboard_map:
            # This is a keyboard movement key
            self.last_keyboard_key = key
        else:
            # Unknown key
            return None

        # Create the action tensor for this frame
        action_tensor = self._create_paired_action()

        # IMPORTANT: Reset state immediately after creating the action
        # so next frame defaults to no movement (like 'u' key behavior)
        self.reset_action_state()

        return action_tensor

    def _create_paired_action(self) -> Dict[str, torch.Tensor]:
        """
        Create a paired action combining current keyboard + camera state.
        This matches the behavior in inference.py where actions are paired.
        """
        result = {}

        # Create keyboard component
        keyboard_tensor = torch.zeros(
            self.action_config.keyboard_dim,
            device=self.device,
            dtype=self.dtype
        )

        if self.last_keyboard_key is not None:
            idx = self.action_config.keyboard_map.get(self.last_keyboard_key, -1)
            if idx >= 0:
                keyboard_tensor[idx] = 1.0

        result['keyboard'] = keyboard_tensor

        # Create mouse component (if mode supports it)
        if self.action_config.has_mouse:
            mouse_tensor = torch.zeros(2, device=self.device, dtype=self.dtype)

            if self.last_camera_key is not None and self.action_config.camera_map:
                mouse_values = self.action_config.camera_map.get(
                    self.last_camera_key,
                    [0.0, 0.0]
                )
                mouse_tensor = torch.tensor(
                    mouse_values,
                    device=self.device,
                    dtype=self.dtype
                )

            result['mouse'] = mouse_tensor

        return result

    def reset_action_state(self):
        """Reset the action state to no action (useful for explicit resets)"""
        self.last_keyboard_key = None
        self.last_camera_key = None

    def create_zero_action(self) -> Dict[str, torch.Tensor]:
        """Create a zero (no-op) action"""
        result = {
            'keyboard': torch.zeros(
                self.action_config.keyboard_dim,
                device=self.device,
                dtype=self.dtype
            )
        }

        if self.action_config.has_mouse:
            result['mouse'] = torch.zeros(2, device=self.device, dtype=self.dtype)

        return result
