"""
Modular Action Module Package

Terminology:
    - View Control: Camera/view angle control (input: mouse, gamepad right stick)
    - Movement: Character movement (input: keyboard WASD, gamepad left stick)
"""

from wan.modules.modular_action.action_config import (
    ActionConfig,
    DEFAULT_ACTION_CONFIG,
    SMALL_ACTION_CONFIG,
    WAN_1_3B_ACTION_CONFIG,
    get_action_config,
)

from wan.modules.modular_action.action_module import ActionModule

from wan.modules.modular_action.view_control import (
    ViewControlInjector,
    ViewControlPreprocessor,
)
from wan.modules.modular_action.movement_control import (
    MovementInjector,
    MovementPreprocessor,
)

from wan.modules.modular_action.interfaces import (
    ActionInjector,
    AttentionKernel,
    WanRMSNorm,
)

__all__ = [
    # Main module
    "ActionModule",

    # Configuration
    "ActionConfig",
    "DEFAULT_ACTION_CONFIG",
    "SMALL_ACTION_CONFIG",
    "WAN_1_3B_ACTION_CONFIG",
    "get_action_config",

    # Injectors
    "ViewControlInjector",
    "MovementInjector",
    "ViewControlPreprocessor",
    "MovementPreprocessor",

    # Core components
    "ActionInjector",
    "AttentionKernel",
    "WanRMSNorm",
]
