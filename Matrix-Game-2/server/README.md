# Server Module Documentation

This module contains the refactored interactive streaming server for Matrix Game inference. The code has been reorganized for better modularity, maintainability, and extensibility.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│           interactive_inference_server_refactored.py    │
│                  (Main Entry Point)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ GameConfig     │  → Game mode configurations
            └────────────────┘
                     │
        ┌────────────┼────────────┬────────────┐
        ▼            ▼            ▼            ▼
┌──────────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
│ModelManager  │ │ActionMap│ │Streaming│ │Inference     │
│              │ │per      │ │Server   │ │Engine        │
└──────────────┘ └─────────┘ └─────────┘ └──────────────┘
```

## Module Structure

### 1. `config.py` - Configuration Management
**Purpose**: Centralized game mode configuration
- `GameMode`: Enum of supported modes (universal, gta_drive, templerun)
- `ActionConfig`: Action mapping configuration (camera, keyboard dimensions)
- `GameModeConfig`: Complete mode configuration
- `GameConfig`: Factory for creating mode-specific configs

**Benefits**:
- Easy to add new game modes
- Type-safe configuration
- Single source of truth

**Example - Adding a new game mode**:
```python
@classmethod
def _get_minecraft_config(cls) -> GameModeConfig:
    action_config = ActionConfig(
        camera_map={"i": [cls.CAM_VALUE, 0], ...},
        keyboard_map={"w": 0, "s": 1, ...},
        keyboard_dim=6,
        has_mouse=True
    )
    control_instructions = """<div>...</div>"""
    return GameModeConfig(...)
```

### 2. `action_mapper.py` - Action Parsing
**Purpose**: Convert client input to model tensors
- `ActionMapper`: Maps keyboard/mouse events to tensors
- Handles one-hot encoding for keyboard
- Manages camera (mouse) actions
- Creates zero/no-op actions

**Benefits**:
- Isolated action parsing logic
- Easy to modify input handling
- Testable in isolation

### 3. `model_manager.py` - Model Loading
**Purpose**: Encapsulate all model initialization
- `ModelManager`: Loads generator, VAE, decoder
- Handles checkpoint loading
- Manages device/dtype configuration
- Compiles models for performance

**Benefits**:
- Clean separation of model concerns
- Easy to swap model implementations
- Reusable across different scripts

### 4. `streaming_server.py` - HTTP/WebSocket Server
**Purpose**: Handle network communication
- `StreamingServer`: FastAPI-based server
- MJPEG streaming endpoint
- WebSocket for real-time input
- Frame queue management
- Statistics broadcasting

**Benefits**:
- Independent of inference logic
- Can be tested separately
- Easy to add new endpoints

### 5. `templates.py` - HTML Generation
**Purpose**: Generate web interface
- `generate_html_page()`: Creates interactive HTML
- Embedded JavaScript for WebSocket
- Responsive UI with stats display

**Benefits**:
- Separates presentation from logic
- Easy to customize UI
- Can be replaced with static files

### 6. `inference_engine.py` - Core Inference
**Purpose**: Real-time frame generation
- `InferenceEngine`: Main inference loop
- KV cache management
- Frame preprocessing
- Conditional dict updates
- Performance tracking

**Benefits**:
- Focused on inference logic only
- Clean async/await structure
- Easy to optimize

## Key Improvements Over Original Code

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Changes to one component don't affect others
- Easier to understand and debug

### 2. **Extensibility**
- Adding new game modes: Just add to `config.py`
- New action types: Extend `ActionMapper`
- Different models: Implement new `ModelManager`
- Alternative streaming: Swap `StreamingServer`

### 3. **Testability**
```python
# Each component can be tested independently
def test_action_mapper():
    mapper = ActionMapper(config, device, dtype)
    action = mapper.parse_action({"type": "keyboard", "key": "w"})
    assert action['keyboard'][0] == 1.0
```

### 4. **Type Safety**
- Uses Python type hints throughout
- Dataclasses for structured configuration
- Enums for game modes
- Better IDE support and error catching

### 5. **Documentation**
- Clear docstrings for all classes/methods
- Structured module documentation
- Examples for common tasks

## Usage

### Basic Usage
```bash
python interactive_inference_server_refactored.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --pretrained_model_path Matrix-Game-2.0 \
    --server_port 8000
```

### Advanced Usage
```python
from server import (
    GameConfig, ModelManager, ActionMapper,
    StreamingServer, InferenceEngine
)

# Custom configuration
config = GameConfig.get_config("universal")
model_manager = ModelManager(...)
action_mapper = ActionMapper(config, device, dtype)
server = StreamingServer(config)
engine = InferenceEngine(model_manager, action_mapper, server)

# Run
await engine.run(image_path="path/to/image.png")
```

## Comparison: Original vs Refactored

| Aspect | Original | Refactored |
|--------|----------|------------|
| **Lines of Code** | ~876 | ~350 (main) + ~550 (modules) |
| **Classes** | 2 giant classes | 6 focused classes |
| **Configuration** | Hardcoded in methods | Centralized in `config.py` |
| **Testability** | Difficult | Easy (isolated modules) |
| **Adding new mode** | Modify 3+ methods | Add 1 config method |
| **Code reuse** | Low | High |
| **Type safety** | Minimal | Full type hints |

## Migration Guide

The refactored version is **backward compatible** in terms of functionality. To migrate:

1. **Use the new entry point**:
   ```bash
   # Old
   python interactive_inference_server.py [args]

   # New
   python interactive_inference_server_refactored.py [args]
   ```

2. **Same arguments**: All command-line arguments remain the same

3. **Same API**: HTTP/WebSocket endpoints unchanged

4. **Same behavior**: Identical inference output

## Future Enhancements

With this modular structure, easy additions include:

1. **Multi-user support**: Add session management to `StreamingServer`
2. **Recording**: Add `RecordingManager` to save sessions
3. **Model hot-swapping**: Extend `ModelManager` with dynamic loading
4. **Custom actions**: Subclass `ActionMapper` for gamepad support
5. **API mode**: Add REST endpoints for programmatic control
6. **Metrics dashboard**: Extend `templates.py` with real-time graphs

## Contributing

When adding features:
1. Choose the appropriate module (or create a new one)
2. Follow existing patterns (type hints, docstrings, etc.)
3. Keep classes focused and single-purpose
4. Add tests if possible
5. Update this documentation
