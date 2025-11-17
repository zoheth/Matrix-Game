"""
Action input strategies for different game modes.

This module implements the Strategy Pattern to decouple action input
from the inference pipeline, enabling flexible input sources (CLI, WebSocket, API, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch


class ActionDict(Dict[str, torch.Tensor]):
    """Type-safe action dictionary."""

    def __init__(self, **kwargs):
        """
        Initialize action dictionary.

        Valid keys depend on the game mode:
        - universal/gta_drive: 'mouse', 'keyboard'
        - templerun: 'keyboard'
        """
        super().__init__(**kwargs)

    def validate(self, mode: str) -> None:
        """Validate that the action dict has required keys for the mode."""
        if mode == 'templerun':
            required = {'keyboard'}
        else:
            required = {'mouse', 'keyboard'}

        if not required.issubset(self.keys()):
            raise ValueError(f"Missing required keys for mode {mode}: {required - self.keys()}")


class ActionStrategy(ABC):
    """
    Abstract base class for action input strategies.

    Subclasses implement different ways to obtain user actions,
    such as keyboard input, network requests, pre-recorded sequences, etc.
    """

    def __init__(self, mode: str):
        """
        Initialize the action strategy.

        Args:
            mode: Game mode - 'universal', 'gta_drive', or 'templerun'
        """
        if mode not in ['universal', 'gta_drive', 'templerun']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: universal, gta_drive, templerun")
        self.mode = mode

    @abstractmethod
    def get_action(self, frame_index: int, **context) -> ActionDict:
        """
        Get the action for the current frame.

        Args:
            frame_index: Current frame index
            **context: Additional context (e.g., current state, frame data)

        Returns:
            ActionDict containing action tensors
        """
        pass


class CLIActionStrategy(ActionStrategy):
    """
    Interactive command-line input strategy (original behavior).

    This blocks execution and waits for user input via stdin.
    Suitable for demo/debug purposes but not for production.
    """

    # Camera sensitivity constant
    CAM_VALUE = 0.1

    # Mode-specific input mappings
    CAMERA_MAPS = {
        'universal': {
            "i": [CAM_VALUE, 0],
            "k": [-CAM_VALUE, 0],
            "j": [0, -CAM_VALUE],
            "l": [0, CAM_VALUE],
            "u": [0, 0]
        },
        'gta_drive': {
            "a": [0, -CAM_VALUE],
            "d": [0, CAM_VALUE],
            "q": [0, 0]
        }
    }

    KEYBOARD_MAPS = {
        'universal': {
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0],
            "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        },
        'gta_drive': {
            "w": [1, 0], "s": [0, 1], "q": [0, 0]
        },
        'templerun': {
            "w": [0, 1, 0, 0, 0, 0, 0],  # jump
            "s": [0, 0, 1, 0, 0, 0, 0],  # slide
            "a": [0, 0, 0, 0, 0, 1, 0],  # left side
            "d": [0, 0, 0, 0, 0, 0, 1],  # right side
            "z": [0, 0, 0, 1, 0, 0, 0],  # turn left
            "c": [0, 0, 0, 0, 1, 0, 0],  # turn right
            "q": [1, 0, 0, 0, 0, 0, 0]   # no move
        }
    }

    def __init__(self, mode: str, device: str = 'cuda'):
        """
        Initialize CLI action strategy.

        Args:
            mode: Game mode
            device: Device to place tensors on
        """
        super().__init__(mode)
        self.device = device
        self._print_instructions()

    def _print_instructions(self) -> None:
        """Print user instructions based on mode."""
        print()
        print('-' * 30)

        if self.mode == 'universal':
            print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM")
            print(" (I: up, K: down, J: left, L: right, U: no move)")
            print("PRESS [W, S, A, D, Q] FOR MOVEMENT")
            print(" (W: forward, S: back, A: left, D: right, Q: no move)")
        elif self.mode == 'gta_drive':
            print("PRESS [W, S, A, D, Q] FOR MOVEMENT")
            print(" (W: forward, S: back, A: left, D: right, Q: no move)")
        elif self.mode == 'templerun':
            print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS")
            print(" (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")

        print('-' * 30)

    def _get_universal_input(self) -> Tuple[str, str]:
        """Get input for universal mode."""
        while True:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()

                if idx_mouse in self.CAMERA_MAPS['universal'] and idx_keyboard in self.KEYBOARD_MAPS['universal']:
                    return idx_mouse, idx_keyboard
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception:
                pass

    def _get_gta_drive_input(self) -> Tuple[str, str]:
        """Get input for GTA drive mode."""
        while True:
            try:
                indexes = input(
                    'Please input the actions (split with ` `):\n'
                    '(e.g. `W` for forward, `W A` for forward and left)\n'
                ).strip().lower().split(' ')

                idx_mouse = []
                idx_keyboard = []

                for i in indexes:
                    if i in self.CAMERA_MAPS['gta_drive']:
                        idx_mouse.append(i)
                    elif i in self.KEYBOARD_MAPS['gta_drive']:
                        idx_keyboard.append(i)

                if len(idx_mouse) == 0:
                    idx_mouse.append('q')
                if len(idx_keyboard) == 0:
                    idx_keyboard.append('q')

                # Validate combination
                if idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]:
                    return idx_mouse[0], idx_keyboard[0]
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception:
                pass

    def _get_templerun_input(self) -> str:
        """Get input for Temple Run mode."""
        while True:
            try:
                idx_keyboard = input(
                    'Please input the action:\n'
                    '(e.g. `W` for forward, `Z` for turning left)\n'
                ).strip().lower()

                if idx_keyboard in self.KEYBOARD_MAPS['templerun']:
                    return idx_keyboard
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception:
                pass

    def get_action(self, frame_index: int, **context) -> ActionDict:
        """
        Get action from command-line input.

        Args:
            frame_index: Current frame index (unused in CLI mode)
            **context: Additional context (unused)

        Returns:
            ActionDict with mouse and/or keyboard tensors
        """
        if self.mode == 'universal':
            idx_mouse, idx_keyboard = self._get_universal_input()
            return ActionDict(
                mouse=torch.tensor(self.CAMERA_MAPS['universal'][idx_mouse], device=self.device),
                keyboard=torch.tensor(self.KEYBOARD_MAPS['universal'][idx_keyboard], device=self.device)
            )

        elif self.mode == 'gta_drive':
            idx_mouse, idx_keyboard = self._get_gta_drive_input()
            return ActionDict(
                mouse=torch.tensor(self.CAMERA_MAPS['gta_drive'][idx_mouse], device=self.device),
                keyboard=torch.tensor(self.KEYBOARD_MAPS['gta_drive'][idx_keyboard], device=self.device)
            )

        elif self.mode == 'templerun':
            idx_keyboard = self._get_templerun_input()
            return ActionDict(
                keyboard=torch.tensor(self.KEYBOARD_MAPS['templerun'][idx_keyboard], device=self.device)
            )


class PrerecordedActionStrategy(ActionStrategy):
    """
    Strategy that plays back a pre-recorded sequence of actions.

    Useful for:
    - Automated testing
    - Deterministic reproduction
    - Benchmark evaluation
    """

    def __init__(self, mode: str, action_sequence: list[ActionDict], device: str = 'cuda'):
        """
        Initialize with a pre-recorded action sequence.

        Args:
            mode: Game mode
            action_sequence: List of ActionDict objects, one per frame
            device: Device to place tensors on
        """
        super().__init__(mode)
        self.action_sequence = action_sequence
        self.device = device

        # Validate sequence
        for idx, action in enumerate(action_sequence):
            try:
                action.validate(mode)
            except ValueError as e:
                raise ValueError(f"Invalid action at index {idx}: {e}")

    def get_action(self, frame_index: int, **context) -> ActionDict:
        """
        Get action from the pre-recorded sequence.

        Args:
            frame_index: Current frame index
            **context: Additional context (unused)

        Returns:
            ActionDict for the given frame
        """
        if frame_index >= len(self.action_sequence):
            raise IndexError(
                f"Frame index {frame_index} exceeds action sequence length {len(self.action_sequence)}"
            )

        action = self.action_sequence[frame_index]
        # Ensure tensors are on the correct device
        return ActionDict(**{k: v.to(self.device) for k, v in action.items()})


class CallbackActionStrategy(ActionStrategy):
    """
    Strategy that delegates to a user-provided callback function.

    This is the most flexible strategy and enables:
    - WebSocket-based real-time control
    - API-based control
    - ML agent control
    - Any custom input source
    """

    def __init__(self, mode: str, callback_fn, device: str = 'cuda'):
        """
        Initialize with a callback function.

        Args:
            mode: Game mode
            callback_fn: Function with signature (frame_index, **context) -> ActionDict
            device: Device to place tensors on
        """
        super().__init__(mode)
        self.callback_fn = callback_fn
        self.device = device

    def get_action(self, frame_index: int, **context) -> ActionDict:
        """
        Get action from the callback function.

        Args:
            frame_index: Current frame index
            **context: Additional context passed to callback

        Returns:
            ActionDict from the callback
        """
        action = self.callback_fn(frame_index, **context)

        if not isinstance(action, ActionDict):
            raise TypeError(f"Callback must return ActionDict, got {type(action)}")

        action.validate(self.mode)

        # Ensure tensors are on the correct device
        return ActionDict(**{k: v.to(self.device) for k, v in action.items()})
