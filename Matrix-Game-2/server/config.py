"""
Configuration management for different game modes
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class GameMode(Enum):
    """Supported game modes"""
    UNIVERSAL = "universal"
    GTA_DRIVE = "gta_drive"
    TEMPLE_RUN = "templerun"


@dataclass
class ActionConfig:
    """Configuration for action mappings"""
    camera_map: Optional[Dict[str, List[float]]]
    keyboard_map: Dict[str, int]
    keyboard_dim: int
    has_mouse: bool = True


@dataclass
class GameModeConfig:
    """Complete configuration for a game mode"""
    mode: GameMode
    action_config: ActionConfig
    control_instructions_html: str


class GameConfig:
    """Central configuration manager for all game modes"""

    CAM_VALUE = 0.1

    @classmethod
    def get_config(cls, mode: str) -> GameModeConfig:
        """Get configuration for specified game mode"""
        mode_enum = GameMode(mode)

        configs = {
            GameMode.UNIVERSAL: cls._get_universal_config(),
            GameMode.GTA_DRIVE: cls._get_gta_drive_config(),
            GameMode.TEMPLE_RUN: cls._get_temple_run_config(),
        }

        return configs[mode_enum]

    @classmethod
    def _get_universal_config(cls) -> GameModeConfig:
        """Universal mode configuration"""
        action_config = ActionConfig(
            camera_map={
                "i": [cls.CAM_VALUE, 0],
                "k": [-cls.CAM_VALUE, 0],
                "j": [0, -cls.CAM_VALUE],
                "l": [0, cls.CAM_VALUE],
                "u": [0, 0]
            },
            keyboard_map={
                "w": 0, "s": 1, "a": 2, "d": 3, "q": -1
            },
            keyboard_dim=4,
            has_mouse=True
        )

        control_instructions = """
        <div class="info">
            <h3>🎮 Universal Mode</h3>
            <p><strong>Camera:</strong></p>
            <p>• <kbd>I</kbd> = Up</p>
            <p>• <kbd>K</kbd> = Down</p>
            <p>• <kbd>J</kbd> = Left</p>
            <p>• <kbd>L</kbd> = Right</p>
            <p>• <kbd>U</kbd> = No move</p>
            <p><strong>Movement:</strong></p>
            <p>• <kbd>W</kbd> = Forward</p>
            <p>• <kbd>S</kbd> = Back</p>
            <p>• <kbd>A</kbd> = Left</p>
            <p>• <kbd>D</kbd> = Right</p>
            <p>• <kbd>Q</kbd> = No move</p>
        </div>
        """

        return GameModeConfig(
            mode=GameMode.UNIVERSAL,
            action_config=action_config,
            control_instructions_html=control_instructions
        )

    @classmethod
    def _get_gta_drive_config(cls) -> GameModeConfig:
        """GTA Drive mode configuration"""
        action_config = ActionConfig(
            camera_map={
                "a": [0, -cls.CAM_VALUE],
                "d": [0, cls.CAM_VALUE],
                "q": [0, 0]
            },
            keyboard_map={
                "w": 0, "s": 1, "q": -1
            },
            keyboard_dim=2,
            has_mouse=True
        )

        control_instructions = """
        <div class="info">
            <h3>🚗 GTA Drive Mode</h3>
            <p><strong>Steering:</strong></p>
            <p>• <kbd>A</kbd> = Left</p>
            <p>• <kbd>D</kbd> = Right</p>
            <p>• <kbd>Q</kbd> = Straight</p>
            <p><strong>Acceleration:</strong></p>
            <p>• <kbd>W</kbd> = Forward</p>
            <p>• <kbd>S</kbd> = Back</p>
            <p>• <kbd>Q</kbd> = Coast</p>
        </div>
        """

        return GameModeConfig(
            mode=GameMode.GTA_DRIVE,
            action_config=action_config,
            control_instructions_html=control_instructions
        )

    @classmethod
    def _get_temple_run_config(cls) -> GameModeConfig:
        """Temple Run mode configuration"""
        action_config = ActionConfig(
            camera_map=None,  # No camera control
            keyboard_map={
                "q": 0,  # no action
                "w": 1,  # jump
                "s": 2,  # slide
                "z": 3,  # turn left
                "c": 4,  # turn right
                "a": 5,  # left side
                "d": 6   # right side
            },
            keyboard_dim=7,
            has_mouse=False
        )

        control_instructions = """
        <div class="info">
            <h3>🏃 Temple Run Mode</h3>
            <p>• <kbd>W</kbd> = Jump</p>
            <p>• <kbd>S</kbd> = Slide</p>
            <p>• <kbd>A</kbd> = Left side</p>
            <p>• <kbd>D</kbd> = Right side</p>
            <p>• <kbd>Z</kbd> = Turn left</p>
            <p>• <kbd>C</kbd> = Turn right</p>
            <p>• <kbd>Q</kbd> = No move</p>
        </div>
        """

        return GameModeConfig(
            mode=GameMode.TEMPLE_RUN,
            action_config=action_config,
            control_instructions_html=control_instructions
        )
