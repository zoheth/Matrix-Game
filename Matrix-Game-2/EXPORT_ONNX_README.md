# CausalWanModel ONNX Export Guide

This guide explains how to export the CausalWanModel to ONNX format with input shapes aligned to `inference.py`.

## Overview

The `export_onnx.py` script exports the CausalWanModel (from WanDiffusionWrapper) to ONNX format, maintaining the same input/output shapes as used in the inference pipeline.

## Input Shapes (aligned with inference.py)

Based on the inference pipeline, the model expects the following inputs:

### Common Inputs (All Modes)
- **noisy_input**: `[B, 16, F, 44, 80]` - Noisy latent frames
  - B: Batch size (default: 1)
  - F: Number of latent frames (default: 150)
  - 44, 80: Latent spatial dimensions (height/4, width/4)

- **timestep**: `[B, F]` - Diffusion timesteps
  - For causal mode, each frame can have different timesteps

- **visual_context**: `[B, 257, 1280]` - CLIP visual embeddings
  - From CLIP encoder output

- **cond_concat**: `[B, 20, F, 44, 80]` - Concatenated conditioning
  - Contains mask (4 channels) + encoded first frame (16 channels)

### Mode-Specific Inputs

#### Universal Mode
- **mouse_cond**: `[B, P, 2]` - Mouse movement conditions
  - P = 1 + 4 * (F - 1) pixel frames (e.g., 597 for F=150)
  - 2 channels: [delta_y, delta_x]

- **keyboard_cond**: `[B, P, 4]` - Keyboard action conditions
  - 4 channels: [W, S, A, D] key states

#### GTA Drive Mode
- **mouse_cond**: `[B, P, 2]` - Camera steering
  - 2 channels: [vertical, horizontal]

- **keyboard_cond**: `[B, P, 2]` - Drive controls
  - 2 channels: [forward, backward]

#### Temple Run Mode
- **keyboard_cond**: `[B, P, 7]` - Game actions
  - 7 channels: [no_action, jump, slide, turn_left, turn_right, move_left, move_right]

## Output Shapes

- **flow_pred**: `[B, 16, F, 44, 80]` - Flow matching prediction
- **pred_x0**: `[B, 16, F, 44, 80]` - Denoised latent prediction

## Usage

### Basic Export

```bash
python export_onnx.py \
    --checkpoint_path path/to/checkpoint.safetensors \
    --output_path outputs/causal_wan_model.onnx
```

### Export with Custom Settings

```bash
python export_onnx.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path path/to/checkpoint.safetensors \
    --output_path outputs/causal_wan_model.onnx \
    --mode universal \
    --batch_size 1 \
    --num_output_frames 150 \
    --opset_version 17 \
    --simplify
```

### Command-Line Arguments

- `--config_path`: Path to inference config YAML (default: `configs/inference_yaml/inference_universal.yaml`)
- `--checkpoint_path`: Path to model checkpoint (required, `.safetensors` format)
- `--output_path`: Output ONNX file path (default: `outputs/causal_wan_model.onnx`)
- `--mode`: Model mode - `universal`, `gta_drive`, or `templerun` (default: `universal`)
- `--batch_size`: Batch size for export (default: 1)
- `--num_output_frames`: Number of latent frames (default: 150)
- `--opset_version`: ONNX opset version (default: 17)
- `--simplify`: Simplify ONNX graph using onnx-simplifier (requires `onnx-simplifier` package)

## Examples

### Export for Universal Mode (Default)
```bash
python export_onnx.py \
    --checkpoint_path Matrix-Game-2.0/checkpoint.safetensors \
    --output_path outputs/wan_universal.onnx
```

### Export for GTA Drive Mode
```bash
python export_onnx.py \
    --config_path configs/inference_yaml/inference_gta_drive.yaml \
    --checkpoint_path Matrix-Game-2.0/checkpoint.safetensors \
    --output_path outputs/wan_gta_drive.onnx \
    --mode gta_drive
```

### Export for Temple Run Mode
```bash
python export_onnx.py \
    --config_path configs/inference_yaml/inference_templerun.yaml \
    --checkpoint_path Matrix-Game-2.0/checkpoint.safetensors \
    --output_path outputs/wan_templerun.onnx \
    --mode templerun
```

### Export with Simplification
```bash
# First install onnx-simplifier
pip install onnx-simplifier

# Then export with simplification
python export_onnx.py \
    --checkpoint_path Matrix-Game-2.0/checkpoint.safetensors \
    --output_path outputs/wan_model.onnx \
    --simplify
```

## Requirements

### Core Requirements
- Python 3.8+
- PyTorch 2.0+
- ONNX
- safetensors
- omegaconf

### Optional Requirements
- onnx-simplifier (for model simplification)
- onnxruntime (for ONNX inference verification)

Install requirements:
```bash
pip install torch onnx safetensors omegaconf
pip install onnx-simplifier  # optional
pip install onnxruntime-gpu  # optional, for verification
```

## Notes

1. **KV Cache**: The current export does NOT include KV cache support. This exports the non-cached training forward path. For inference with KV caching, additional modifications are needed.

2. **Model Size**: The exported ONNX model is typically 2-3GB depending on the model configuration.

3. **Dynamic Shapes**: The export includes dynamic axes for batch size and number of frames, allowing flexibility in input dimensions.

4. **Precision**: The model is exported in bfloat16 precision (matching the inference.py default).

5. **FlexAttention**: ONNX export may not support all PyTorch operations (like FlexAttention). Consider using alternative attention mechanisms or exporting specific attention blocks separately if issues occur.

## Troubleshooting

### Issue: ONNX export fails with FlexAttention error
**Solution**: FlexAttention is not ONNX-compatible. You may need to:
- Use standard attention implementation
- Export with `torch.onnx.export(..., operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)`

### Issue: Model too large
**Solution**:
- Use `--simplify` flag to reduce model size
- Export with lower precision (though this may affect quality)
- Consider exporting only specific model components

### Issue: Dynamic shape errors
**Solution**: Export with fixed shapes by removing `dynamic_axes` parameter or specifying fixed dimensions

## Related Files

- `inference.py`: Main inference script (reference for input shapes)
- `pipeline/causal_inference.py`: Causal inference pipeline implementation
- `wan/modules/causal_model.py`: CausalWanModel definition
- `utils/wan_wrapper.py`: WanDiffusionWrapper implementation

## Contact

For issues or questions about ONNX export, please refer to the main project repository.
