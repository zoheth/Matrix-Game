# Cache Design Guide

This document explains the three different cache implementations in the codebase and when to use each one.

## Overview

We have **three cache types for three different purposes**:

| Cache Type | Purpose | Location | CUDA Graph |
|------------|---------|----------|------------|
| `CacheState` | VAE conv layer temporal caching | `demo_utils/new_vae.py` | ❌ |
| `PagedCache` | Transformer visual attention KV cache | `wan/modules/paged_cache.py` | ❌ |
| `RingBufferActionCache` | Transformer action attention KV cache | `wan/modules/ring_buffer_action_cache.py` | ✅ |

**Key Principle**: These caches are **intentionally different** because they solve different problems. Do not try to unify them.

---

## 1. CacheState - VAE Temporal Caching

### Use Case
Temporal caching for causal 3D convolutions in the VAE decoder during streaming video generation.

### Design
```python
class CacheState:
    """Manages temporal caching for streaming video decoding."""
    def __init__(self, size):
        self.feat_map: List[Optional[torch.Tensor]] = [None] * size
        self.idx = 0
```

**Characteristics**:
- ✅ Extremely simple (~30 lines of code)
- ✅ Easy to understand and maintain
- ❌ Not CUDA Graph compatible (uses Python list/int)
- 🎯 Specialized for conv layer caching

### When to Use
Use `CacheState` when:
- Working with VAE decoder streaming inference
- Need to cache temporal context for causal convolutions
- Simplicity is more important than CUDA Graph compatibility

### Example Usage

```python
from demo_utils.new_vae import VaeDecoder3d, CacheState

# Initialize decoder
vae_decoder = VaeDecoder3d(...)

# Create cache states (one per major layer group)
cache_states = [
    CacheState(size=num_layers),  # conv1
    CacheState(size=num_layers),  # middle
    CacheState(size=num_layers),  # upsamples
    CacheState(size=num_layers),  # head_conv
]

# Streaming inference
for latent_chunk in latent_stream:
    # CacheState handles temporal dependencies automatically
    video_chunk = vae_decoder(latent_chunk, cache_states)

# Reset for next sequence
for cache in cache_states:
    cache.reset_index()
```

### Access Pattern
```python
# Sequential slot access
idx = cache_state.get_and_increment()     # Get index and advance
past_tensor = cache_state.feat_map[idx]   # Read from slot
cache_state.feat_map[idx] = new_tensor    # Write to slot
```

---

## 2. PagedCache - Visual Attention KV Cache

### Use Case
KV cache for visual self-attention in the Transformer, optimized for long sequences and memory efficiency.

### Design
```python
class PagedCache:
    """Paged KV Cache for efficient memory management with FlashInfer."""
    def __init__(self, max_total_tokens, page_size, num_heads, head_dim, ...):
        # Memory: [max_pages, page_size, num_heads, head_dim]
        self.k_cache = torch.zeros((max_pages, page_size, num_heads, head_dim), ...)
        self.v_cache = torch.zeros((max_pages, page_size, num_heads, head_dim), ...)

        # Page management (CPU state)
        self.active_page_indices: list[int] = []
        self.free_page_pool: list[int] = []
```

**Characteristics**:
- ✅ Memory efficient (page-granular allocation and eviction)
- ✅ FlashInfer integration for fast paged attention
- ✅ Supports denoising (update_or_append)
- ❌ Only supports batch_size=1
- ❌ Not CUDA Graph compatible (CPU control flow)
- 🎯 Optimized for long video sequences

### When to Use
Use `PagedCache` when:
- Working with visual self-attention in Transformer
- Need to handle long sequences (14080+ tokens)
- Memory efficiency is critical
- FlashInfer acceleration is available
- Batch size is 1

### Example Usage

```python
from wan.modules.paged_cache import PagedCache
from wan.modules.flashinfer_attention import CausalSelfAttention, FlashInferPlanner

# Create cache (one per transformer layer)
num_layers = 30
visual_cache = [
    PagedCache(
        max_total_tokens=14080,
        page_size=16,
        num_heads=16,
        head_dim=64,
        sink_size=0,
        dtype=torch.bfloat16,
        device="cuda"
    )
    for _ in range(num_layers)
]

# Create FlashInfer planner
planner = FlashInferPlanner(num_heads=16, head_dim=64)

# In attention layer
for layer_idx, attn_layer in enumerate(attention_layers):
    cache = visual_cache[layer_idx]

    # Compute Q, K, V
    q, k, v = compute_qkv(x)

    # Update cache (handles both append and denoising)
    cache.update_or_append(k, v, current_start, current_end)

    # Evict old pages if needed
    cache.evict(max_allowed_tokens)

    # Plan FlashInfer execution (once per step)
    if not planner.is_planned:
        planner.plan(cache, q_len=q.shape[0], ...)

    # Run paged attention
    output = planner.run(q, cache)

# Reset for new sequence
for cache in visual_cache:
    cache.reset()
```

### Access Pattern
```python
# Multi-step workflow
cache.update_or_append(k, v, start, end)  # Append or overwrite (denoising)
cache.evict(max_tokens)                   # Explicit page eviction
indices, indptr, last_len = cache.get_flashinfer_meta()  # FlashInfer metadata
output = planner.run(q, cache)            # Paged attention
```

---

## 3. RingBufferActionCache - Action Attention KV Cache

### Use Case
KV cache for action conditioning (mouse/keyboard) in the Transformer, optimized for CUDA Graph compatibility.

### Design
```python
class RingBufferActionCache:
    """CUDA Graph-compatible KV Cache using Ring Buffer mechanism."""
    def __init__(self, batch_size, max_seq_len, num_heads, head_dim, ...):
        # Memory: [batch_size, max_seq_len, num_heads, head_dim]
        self.k_cache = torch.zeros((batch_size, max_seq_len, num_heads, head_dim), ...)
        self.v_cache = torch.zeros((batch_size, max_seq_len, num_heads, head_dim), ...)

        # State (GPU tensors for CUDA Graph)
        self.pos_ptr = torch.zeros(1, dtype=torch.long, device="cuda")
        self.total_tokens = torch.zeros(1, dtype=torch.long, device="cuda")
```

**Characteristics**:
- ✅ Fully CUDA Graph compatible (no CPU control flow)
- ✅ Supports arbitrary batch sizes
- ✅ Supports spatial batching (B × S for mouse)
- ✅ Fixed output shapes + attention mask
- ✅ Simple single-call API
- ❌ Fixed-size pre-allocation (higher memory)
- 🎯 Optimized for execution speed

### When to Use
Use `RingBufferActionCache` when:
- Working with action conditioning (mouse/keyboard)
- Need CUDA Graph compatibility
- Support for spatial batching (mouse: B×S)
- Execution speed is critical
- Cache size is relatively small (~1024 tokens)

### Example Usage

```python
from wan.modules.ring_buffer_action_cache import RingBufferActionCache
from wan.modules.modular_action import ViewControlInjector, MovementInjector

# Create caches (one per transformer layer)
num_layers = 30
batch_size = 1
frame_seq_length = 880  # Spatial dimension

# Mouse cache (spatial batching: B × S)
mouse_cache = [
    RingBufferActionCache(
        batch_size=batch_size * frame_seq_length,  # B × S
        max_seq_len=1024,
        num_heads=16,
        head_dim=64,
        device="cuda",
        dtype=torch.bfloat16
    )
    for _ in range(num_layers)
]

# Keyboard cache (simple batching: B)
keyboard_cache = [
    RingBufferActionCache(
        batch_size=batch_size,  # B only
        max_seq_len=1024,
        num_heads=16,
        head_dim=64,
        device="cuda",
        dtype=torch.bfloat16
    )
    for _ in range(num_layers)
]

# In action injector
for layer_idx, injector in enumerate(action_injectors):
    cache = mouse_cache[layer_idx]  # or keyboard_cache[layer_idx]

    # Single-call update and get window
    k_window, v_window, attention_mask = cache.update_and_get_window(
        k=new_keys,
        v=new_values,
        num_new_tokens=num_frames,
        max_attention_size=1024
    )

    # Use window in attention
    output = attention(q, k_window, v_window, mask=attention_mask)

# Reset for new sequence
for cache in mouse_cache + keyboard_cache:
    cache.reset()
```

### Access Pattern
```python
# Single integrated call
k_window, v_window, mask = cache.update_and_get_window(
    k=new_k,
    v=new_v,
    num_new_tokens=T,
    max_attention_size=1024
)
# Returns: fixed-shape window + validity mask
# Eviction is implicit via ring buffer wrap-around
```

---

## Comparison Table

| Feature | CacheState | PagedCache | RingBufferActionCache |
|---------|------------|------------|----------------------|
| **Layer Type** | Conv3d | Attention | Attention |
| **Use Case** | VAE temporal caching | Visual KV cache | Action KV cache |
| **Memory Layout** | `List[Tensor]` | `[pages, page_size, H, D]` | `[B, max_seq, H, D]` |
| **State Storage** | Python `int`/`list` | Python `list[int]` | GPU tensors |
| **Batch Size** | N/A (per-layer) | 1 only | Arbitrary (B or B×S) |
| **Eviction** | Overwrite | Page-granular | Ring buffer wrap |
| **CUDA Graph** | ❌ | ❌ | ✅ |
| **Complexity** | Simple (~30 LOC) | Complex (~400 LOC) | Medium (~200 LOC) |
| **Update API** | `get_and_increment()` | `update_or_append()` + `evict()` | `update_and_get_window()` |
| **Output** | Index | In-place update | `(k, v, mask)` |
| **FlashInfer** | N/A | ✅ | ❌ |

---

## Design Rationale

### Why Not Unify These Caches?

**They are intentionally different** because:

1. **Different layers**: Conv vs Attention
2. **Different access patterns**: Slot indexing vs Page management vs Ring buffer
3. **Different optimization goals**: Simplicity vs Memory vs Execution
4. **Different interfaces**: Incompatible method signatures
5. **Different usage contexts**: Never used interchangeably

### Attempting to unify would result in:

❌ Awkward abstractions that hide important differences
❌ Increased complexity without benefit
❌ Loss of specialized optimizations
❌ Confusion for developers

### The Right Approach:

✅ Keep them separate with clear documentation
✅ Each cache is optimized for its specific use case
✅ Developers choose the right cache based on context
✅ Maintainers can optimize each cache independently

---

## Quick Selection Guide

**Start here** - What layer are you working with?

```
┌─────────────────────────────────────┐
│   What layer needs caching?         │
└─────────────┬───────────────────────┘
              │
         ┌────┴────┐
         │         │
    Conv3d?    Attention?
         │         │
         │    ┌────┴────┐
         │    │         │
    CacheState  Visual?  Action?
                  │         │
             PagedCache  RingBufferActionCache
```

### Decision Tree

```python
if layer_type == "Conv3d" and context == "VAE":
    use CacheState

elif layer_type == "Attention" and context == "Visual":
    use PagedCache

elif layer_type == "Attention" and context == "Action":
    use RingBufferActionCache

else:
    raise ValueError("Unknown cache requirement")
```

---

## Common Patterns

### Pattern 1: Cache Manager (Unified Creation)

While the caches themselves are different, they can be **created** by a unified manager:

```python
class CacheManager:
    """Manages all cache types for the model."""

    def initialize_all_caches(self):
        # Visual cache (PagedCache)
        self.visual_cache = self._create_paged_visual_cache()

        # Action caches (RingBufferActionCache)
        self.mouse_cache = self._create_action_mouse_cache()
        self.keyboard_cache = self._create_action_keyboard_cache()

    def reset_all_caches(self):
        """Unified reset interface."""
        for cache in self.visual_cache:
            cache.reset()
        for cache in self.mouse_cache:
            cache.reset()
        for cache in self.keyboard_cache:
            cache.reset()
```

**Key insight**: Unified **management** is good, unified **interface** is not.

### Pattern 2: Cache Reset

All caches support `reset()` - this is the only common interface needed:

```python
# All caches can be reset
cache_state.reset_index()          # CacheState
paged_cache.reset()                # PagedCache
ring_buffer_cache.reset()          # RingBufferActionCache
```

### Pattern 3: Type-Specific Usage

Use type annotations to make cache usage clear:

```python
from typing import Optional
from wan.modules.paged_cache import PagedCache
from wan.modules.ring_buffer_action_cache import RingBufferActionCache
from demo_utils.new_vae import CacheState

def attention_layer(
    x: torch.Tensor,
    kv_cache: Optional[PagedCache] = None  # Type makes it clear
) -> torch.Tensor:
    ...

def action_injector(
    x: torch.Tensor,
    kv_cache: Optional[RingBufferActionCache] = None  # Different type
) -> torch.Tensor:
    ...

def conv_layer(
    x: torch.Tensor,
    cache_state: Optional[CacheState] = None  # Different type
) -> torch.Tensor:
    ...
```

---

## Migration Guide

### If You're Using Dict-based Caches

Old dict-based caches should migrate to:
- **Visual attention** → `PagedCache`
- **Action attention** → `RingBufferActionCache`

### Migration Checklist

- [ ] Identify cache usage context (visual vs action)
- [ ] Choose appropriate cache type
- [ ] Update initialization in `CacheManager`
- [ ] Update forward pass to use new API
- [ ] Update reset logic
- [ ] Test CUDA Graph compatibility (if using RingBufferActionCache)

---

## Performance Considerations

### CacheState
- ✅ Minimal overhead (~1-2% of VAE decode time)
- ❌ Not optimized for large-scale caching

### PagedCache
- ✅ Memory efficient (saves ~30% vs dense allocation)
- ✅ FlashInfer speedup (~20-30% faster attention)
- ❌ Page eviction overhead (~5% of attention time)
- ❌ Not CUDA Graph compatible

### RingBufferActionCache
- ✅ CUDA Graph compatible (20-30% speedup with Graph)
- ✅ Zero CPU overhead (pure GPU operations)
- ✅ Fast window extraction (index_select)
- ❌ Fixed-size allocation (higher memory than needed)

---

## Debugging Tips

### CacheState Issues
```python
# Check if cache is being used correctly
print(f"Cache idx: {cache_state.idx}")
print(f"Cache size: {len(cache_state.feat_map)}")
for i, feat in enumerate(cache_state.feat_map):
    print(f"Slot {i}: {feat.shape if feat is not None else None}")
```

### PagedCache Issues
```python
# Check cache state
print(f"Active pages: {cache.active_page_indices}")
print(f"Free pages: {cache.free_page_pool}")
print(f"Seq len: {cache.seq_len}")
print(f"Current page offset: {cache.current_page_offset}")
```

### RingBufferActionCache Issues
```python
# Check GPU state (no .item() calls!)
print(f"Pos ptr: {cache.pos_ptr}")  # GPU tensor
print(f"Total tokens: {cache.total_tokens}")  # GPU tensor
print(f"K cache shape: {cache.k_cache.shape}")
```

---

## Summary

**Three caches, three purposes, three designs.**

- **CacheState**: Simple temporal caching for VAE convolutions
- **PagedCache**: Memory-efficient KV cache for visual attention
- **RingBufferActionCache**: CUDA Graph-compatible KV cache for action attention

**Remember**: The differences are **features, not bugs**. Each cache is optimized for its specific use case. Choose based on your context, not on a desire for uniformity.

**When in doubt**: Check the layer type and data flow, not the cache interface.
