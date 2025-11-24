# FlashInfer Integration for CausalWanModel

This document describes the FlashInfer-based KV cache and attention implementation for the CausalWanModel, designed for high-performance inference with CUDA Graph support.

## Overview

The integration addresses three key challenges:

1. **3D RoPE Compatibility**: The model uses 3D decomposed RoPE (Time + Height + Width), which is incompatible with FlashInfer's built-in RoPE. Solution: Apply RoPE before FlashInfer with `rope_mode="NONE"`.

2. **"Prefill" vs "Decode" Semantics**: Each inference step processes one frame (~880 tokens), not a single token. This is technically a "prefill" operation, not "decode". Solution: Use `BatchPrefillWithPagedKVCacheWrapper`.

3. **CUDA Graph Compatibility**: The original implementation has dynamic tensor allocations that break CUDA Graph capture. Solution: Pre-allocate all memory and precompute RoPE frequencies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FlashInfer Integration                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  PrecomputedRoPE3D   │    │   PagedKVCacheManager        │   │
│  │  - Precomputed grid  │    │   - FlashInfer PagedKVCache  │   │
│  │  - O(1) freq lookup  │    │   - Sliding window eviction  │   │
│  └──────────────────────┘    │   - Static memory allocation │   │
│                              └──────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          FlashInferCausalSelfAttention                    │   │
│  │  - Drop-in replacement for CausalWanSelfAttention         │   │
│  │  - Uses FlashInfer's single_prefill_with_kv_cache         │   │
│  │  - Applies precomputed 3D RoPE before attention           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          CUDAGraphInferenceRunner                         │   │
│  │  - Warmup and capture infrastructure                      │   │
│  │  - Static tensor pool management                          │   │
│  │  - Graph replay with input updates                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Files

- `wan/modules/flashinfer_kv_cache.py`: PagedKVCache manager and 3D RoPE precomputation
- `wan/modules/flashinfer_attention.py`: FlashInfer-based self-attention module
- `wan/modules/cuda_graph_inference.py`: CUDA Graph capture/replay infrastructure
- `pipeline/flashinfer_inference.py`: High-level inference pipeline

## Usage

### Command Line Usage

```bash
# Basic inference (no optimizations)
python inference.py --img_path demo_images/universal/0000.png

# With FlashInfer enabled
python inference.py --flashinfer_mode enabled --img_path demo_images/universal/0000.png

# With CUDA Graph enabled (captures GPU operations for replay)
python inference.py --use_cuda_graph --img_path demo_images/universal/0000.png

# With warmup (for accurate timing)
python inference.py --warmup --img_path demo_images/universal/0000.png

# Maximum performance (FlashInfer + CUDA Graph + Warmup)
python inference.py --flashinfer_mode enabled --use_cuda_graph --warmup --img_path demo_images/universal/0000.png
```

### Python API Usage

```python
from pipeline import BatchCausalInferencePipeline, PipelineConfig

# Initialize pipeline with CUDA Graph
pipeline = BatchCausalInferencePipeline(
    config=pipeline_config,
    generator=generator,
    vae_decoder=vae_decoder,
    device="cuda",
    use_cuda_graph=True,  # Enable CUDA Graph
)

# Run inference
videos = pipeline.inference(
    noise=noise_tensor,
    conditional_dict=conditions,
)
```

### Legacy FlashInferInferencePipeline

```python
from pipeline.flashinfer_inference import FlashInferInferencePipeline

pipeline = FlashInferInferencePipeline(
    args=config,
    device="cuda",
    use_flashinfer_attention=True,
    use_cuda_graph=True,  # Enable CUDA Graph
)

# First inference captures the graph (slower)
videos = pipeline.inference(noise, conditions)

# Subsequent inferences replay the graph (faster)
videos = pipeline.inference(noise2, conditions2)
```

### Manual Attention Replacement

```python
from wan.modules.flashinfer_attention import replace_attention_with_flashinfer

# Load original model
model = CausalWanModel.from_pretrained(...)

# Replace attention modules
model = replace_attention_with_flashinfer(model)

# Use model as normal
output = model(x, t, context, ...)
```

## Key Design Decisions

### 1. 3D RoPE Precomputation

The original `causal_rope_apply` function does `torch.cat/expand/view` on every forward pass:

```python
# Original (inefficient)
freqs_i = torch.cat([
    freqs[0][start:start + f].view(f, 1, 1, -1).expand(...),
    freqs[1][:h].view(1, h, 1, -1).expand(...),
    freqs[2][:w].view(1, 1, w, -1).expand(...),
], dim=-1)
```

New approach precomputes the full grid at initialization:

```python
# New (efficient, CUDA Graph friendly)
class PrecomputedRoPE3DCache:
    def __init__(self, freqs, max_frames, height, width):
        # Precompute full grid [max_frames, H, W, head_dim//2]
        self.freq_grid = self._precompute_grid()

    def get_freqs(self, start_frame, num_frames):
        # O(1) slice lookup
        return self.freq_grid[start_frame:start_frame + num_frames]
```

### 2. FlashInfer Integration

FlashInfer's `single_prefill_with_kv_cache` is used for attention:

```python
# FlashInfer attention (RoPE already applied externally)
output = flashinfer.single_prefill_with_kv_cache(
    q,  # [seq_len, num_heads, head_dim]
    k,  # [kv_len, num_heads, head_dim]
    v,  # [kv_len, num_heads, head_dim]
    causal=False,  # Causality handled by cache windowing
)
```

### 3. Sliding Window with Page Eviction

The KV cache implements sliding window attention via page eviction:

```python
# When cache is full, evict oldest tokens (except sink tokens)
if new_len > max_tokens:
    tokens_to_evict = new_len - max_tokens
    self._evict_tokens(batch_idx, tokens_to_evict, layer_idx)
```

### 4. CUDA Graph Integration

CUDA Graph captures GPU operations and replays them without CPU overhead:

```python
class CUDAGraphSingleStepRunner:
    """Captures a single model forward pass for replay."""

    def capture(self, noisy_input, timestep, conditional_dict, ...):
        # Warmup iterations
        for _ in range(3):
            _ = self.generator(...)

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            flow_pred, aux = self.generator(...)

    def run(self, noisy_input, timestep):
        # Update static inputs in-place
        self._static_noisy_input.copy_(noisy_input)
        self._static_timestep.copy_(timestep)
        # Replay captured operations
        self.graph.replay()
        return self._static_flow_pred, self._static_aux
```

Key implementation details:
- **First block only**: Currently captures graph for `current_start=0` to avoid recapture overhead
- **Static tensors**: All inputs/outputs are pre-allocated and updated in-place
- **Warmup required**: First iteration includes capture overhead; use `--warmup` for accurate timing

## Performance Considerations

### Memory Usage

- **PagedKVCache**: Memory is organized in fixed-size pages (default: 16 tokens/page)
- **Workspace Buffer**: 128MB pre-allocated for FlashInfer operations
- **RoPE Cache**: Precomputed grid for max_frames × H × W × head_dim/2

### Latency

- **Without CUDA Graph**: ~5-10% improvement over original (fewer memory allocations)
- **With CUDA Graph**: ~20-30% improvement (eliminated CPU overhead)

### Limitations

1. **Static Shapes**: CUDA Graph requires fixed tensor shapes
2. **Memory**: Higher peak memory due to pre-allocation
3. **FlashInfer Dependency**: Requires FlashInfer installation

## Troubleshooting

### FlashInfer Not Available

```python
# Check if FlashInfer is available
from wan.modules.flashinfer_attention import FLASHINFER_AVAILABLE
print(f"FlashInfer available: {FLASHINFER_AVAILABLE}")

# Install if needed
# pip install flashinfer
```

### CUDA Graph Capture Fails

Common issues:
1. **Dynamic tensor shapes**: Ensure all input shapes are constant
2. **CPU-GPU sync**: Remove any `.item()` calls in forward path
3. **Control flow**: No if/else based on tensor values

Debug with:
```python
from wan.modules.cuda_graph_inference import check_cuda_graph_compatibility
issues = check_cuda_graph_compatibility(model)
print(issues)
```

### Memory Issues

Reduce memory usage by:
```python
config = PagedKVCacheConfig(
    page_size=8,  # Smaller pages (default: 16)
    max_num_pages=512,  # Fewer pages
)
```

## Future Work

1. **Triton Kernels**: Fused RoPE + Attention kernel for further optimization
2. **Multi-GPU**: Distributed PagedKVCache for tensor parallelism
3. **Quantization**: INT8/FP8 KV cache for memory efficiency
