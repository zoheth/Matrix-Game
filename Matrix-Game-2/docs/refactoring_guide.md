# Pipeline Refactoring Guide

## Overview

The causal inference pipeline has been refactored to address several design issues:

1. **Separation of Concerns**: Input handling, inference, and visualization are now separate
2. **Eliminated Code Duplication**: Common logic moved to base class
3. **Type Safety**: Magic numbers replaced with configuration dataclasses
4. **Flexibility**: Pluggable action input strategies
5. **Maintainability**: Clear module boundaries and responsibilities

## Architecture

```
pipeline/
├── config.py                  # Configuration dataclasses
├── action_strategies.py       # Input abstraction (Strategy Pattern)
├── cache_manager.py          # KV cache management
├── condition_processor.py    # Conditional data processing
├── base_pipeline.py          # Base inference pipeline
├── batch_pipeline.py         # Batch (non-interactive) inference
├── streaming_pipeline.py     # Streaming (interactive) inference
└── causal_inference.py       # Legacy code (for backward compatibility)
```

## Migration Guide

### Old Code (Still Works!)

```python
from pipeline import CausalInferencePipeline

# Old initialization from OmegaConf
pipeline = CausalInferencePipeline(args, generator=generator, vae_decoder=vae)

# Old inference call
videos = pipeline.inference(noise, conditional_dict, mode='universal')
```

**This code still works!** The legacy interface is preserved through aliases.

### New Code (Recommended)

#### 1. Using Batch Pipeline

```python
from pipeline import BatchCausalInferencePipeline, PipelineConfig

# Create configuration
config = PipelineConfig.from_legacy_args(args)
# Or create manually:
config = PipelineConfig(
    mode='universal',
    inference=InferenceConfig(
        num_frame_per_block=1,
        denoising_steps=[1000, 750, 500, 250]
    )
)

# Initialize pipeline
pipeline = BatchCausalInferencePipeline(
    config=config,
    generator=generator,
    vae_decoder=vae_decoder
)

# Run inference (same as before)
videos = pipeline.inference(
    noise=noise,
    conditional_dict=conditional_dict,
    profile=True  # Optional profiling
)
```

#### 2. Using Streaming Pipeline with Custom Actions

```python
from pipeline import (
    StreamingCausalInferencePipeline,
    PipelineConfig,
    PrerecordedActionStrategy,
    ActionDict
)
import torch

# Prepare pre-recorded actions
actions = [
    ActionDict(
        mouse=torch.tensor([0.0, 0.1]),
        keyboard=torch.tensor([1, 0, 0, 0])
    ),
    ActionDict(
        mouse=torch.tensor([0.1, 0.0]),
        keyboard=torch.tensor([0, 0, 0, 1])
    ),
    # ... more actions
]

strategy = PrerecordedActionStrategy(
    mode='universal',
    action_sequence=actions
)

config = PipelineConfig(mode='universal')
pipeline = StreamingCausalInferencePipeline(config, generator, vae_decoder)

# Run with custom actions
video = pipeline.inference(
    noise=noise,
    conditional_dict=conditional_dict,
    action_strategy=strategy,
    output_folder='outputs',
    video_name='my_game'
)
```

#### 3. Using Callback Strategy (for Web API)

```python
from pipeline import CallbackActionStrategy

def get_action_from_websocket(frame_index, **context):
    """Get action from WebSocket connection."""
    # Wait for WebSocket message
    action_data = websocket.receive_json()

    return ActionDict(
        mouse=torch.tensor(action_data['mouse']),
        keyboard=torch.tensor(action_data['keyboard'])
    )

strategy = CallbackActionStrategy(
    mode='universal',
    callback_fn=get_action_from_websocket
)

pipeline = StreamingCausalInferencePipeline(config, generator, vae_decoder)
video = pipeline.inference(
    noise=noise,
    conditional_dict=conditional_dict,
    action_strategy=strategy
)
```

## Key Improvements

### 1. No More Magic Numbers

**Before:**
```python
# What does 1 + 4 * (num_frames - 1) mean?
action_len = 1 + 4 * (current_start_frame + num_frame_per_block - 1)
```

**After:**
```python
# Clear semantic meaning
action_len = vae_config.get_action_condition_length(current_block_end)
```

### 2. Separated Input from Logic

**Before:**
```python
# Inference pipeline directly calls input()
current_actions = get_current_action(mode='universal')  # Blocking!
```

**After:**
```python
# Pipeline receives actions from strategy
current_actions = action_strategy.get_action(frame_index)
# Strategy can be CLI, WebSocket, API, ML agent, etc.
```

### 3. Single Responsibility

**Before:**
```python
class CausalInferencePipeline:
    def inference(self):
        # Initializes caches
        # Gets user input
        # Runs diffusion
        # Decodes video
        # Saves video with overlays
        # All in one method!
```

**After:**
```python
class BaseCausalInferencePipeline:
    # Only handles inference loop

class CacheManager:
    # Only handles cache initialization/reset

class ConditionProcessor:
    # Only handles conditional data processing

class ActionStrategy:
    # Only handles action input
```

### 4. Easy Testing

**Before:**
```python
# Hard to test - requires manual keyboard input
pipeline.inference(...)  # Will block waiting for input
```

**After:**
```python
# Easy to test with PrerecordedActionStrategy
test_actions = [...]
strategy = PrerecordedActionStrategy('universal', test_actions)
result = pipeline.inference(..., action_strategy=strategy)
assert result.shape == expected_shape
```

## Configuration Examples

### Minimal Configuration
```python
config = PipelineConfig(mode='universal')
```

### Full Configuration
```python
config = PipelineConfig(
    mode='gta_drive',
    model=ModelConfig(
        num_transformer_blocks=30,
        frame_seq_length=880,
        num_attention_heads=12,
        head_dim=128
    ),
    cache=CacheConfig(
        local_attn_size=15
    ),
    vae=VAEConfig(
        latent_channels=16,
        temporal_compression=4,
        spatial_compression=8
    ),
    inference=InferenceConfig(
        denoising_steps=[1000, 750, 500, 250],
        num_frame_per_block=2,
        context_noise=0
    )
)
```

## Backward Compatibility

All old code continues to work without modification. The legacy classes are still available:

```python
from pipeline import (
    CausalInferencePipeline,           # Legacy batch
    CausalInferenceStreamingPipeline,  # Legacy streaming
)
```

However, we recommend migrating to the new interfaces for better maintainability.

## Future Extensions

The refactored design makes it easy to add:

1. **Distributed Inference**: Replace `CacheManager` with a distributed version
2. **Model Variants**: Create new pipeline classes inheriting from `BaseCausalInferencePipeline`
3. **Custom Actions**: Implement new `ActionStrategy` subclasses
4. **New Visualizations**: Modify `StreamingCausalInferencePipeline._save_intermediate_video`

## Common Pitfalls

### Import Errors

If you get import errors, make sure you're importing from the correct module:

```python
# ✅ Correct
from pipeline import BatchCausalInferencePipeline

# ❌ Wrong
from pipeline.batch_pipeline import BatchCausalInferencePipeline
# (This works but bypasses __init__.py exports)
```

### Action Validation

Actions must match the game mode:

```python
# ✅ Correct for 'universal'
action = ActionDict(mouse=..., keyboard=...)

# ❌ Wrong for 'universal' (missing mouse)
action = ActionDict(keyboard=...)

# ✅ Correct for 'templerun'
action = ActionDict(keyboard=...)
```

## Questions?

See the docstrings in each module for detailed API documentation.
