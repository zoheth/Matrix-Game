# Action Module 正确配置 - 最终版本

## 核心概念理解

### 维度流程

```
原始视频 (57 frames)
    ↓ VAE 编码 (时间压缩 4x)
Latent 视频 (15 frames, 44x80)
    ↓ DiT 处理 (无时间 patching, patch_size=[1,2,2])
时间 tokens (15 frames, 每帧 880 tokens)
    ↓ Action Module
增强的 tokens (15 frames, 每帧 880 tokens)
```

### 关键参数

1. **VAE 时间压缩**: `vae_time_compression_ratio = 4`
   - `latent_frames = (raw_frames - 1) // 4 + 1`
   - `raw_frames = (latent_frames - 1) * 4 + 1`

2. **空间 patching**: `patch_size = [1, 2, 2]`
   - 时间维度: 不压缩 (patch=1)
   - 空间维度: 2x2 压缩
   - `spatial_tokens = (H // 2) * (W // 2)`

3. **时间 tokens**:
   - DiT 中没有时间 patching
   - `num_feats = latent_frames`

## 实际推理配置

### 从 inference_universal.yaml

```yaml
image_or_video_shape: [1, 16, 15, 44, 80]
# [Batch, Channels, LatentFrames, Height, Width]
num_frame_per_block: 3  # 流式推理时每个 chunk 的大小
```

### 计算结果

```python
# Latent 空间
B = 1
latent_frames = 15
H_latent = 44
W_latent = 80

# 空间 tokens (patching后)
patch_size = [1, 2, 2]
H_tokens = H_latent // patch_size[1]  # 22
W_tokens = W_latent // patch_size[2]  # 40
S = H_tokens * W_tokens  # 880

# 时间维度
num_feats = latent_frames  # 15 (无时间 patching)

# Action condition 维度
vae_ratio = 4
num_frames = (latent_frames - 1) * vae_ratio + 1  # 57

# 输入形状
x: [B, num_feats * S, C] = [1, 15 * 880, 1536] = [1, 13200, 1536]
mouse_condition: [B, num_frames, 2] = [1, 57, 2]
keyboard_condition: [B, num_frames, 6] = [1, 57, 6]
```

## 正确的测试代码

### 完整非因果测试

```python
import torch
from wan.modules.action_module import ActionModule

# 配置
B = 1
latent_frames = 15
vae_ratio = 4
num_feats = latent_frames  # 15
num_frames = (latent_frames - 1) * vae_ratio + 1  # 57
H, W = 44, 80  # Latent space dimensions
H_tokens, W_tokens = H // 2, W // 2  # 22, 40 (patched)
S = H_tokens * W_tokens  # 880
img_hidden_size = 1536
dtype = torch.bfloat16  # 重要！Flash Attention 需要 fp16 或 bf16

# 创建输入
device = 'cuda'
x = torch.randn(B, num_feats * S, img_hidden_size, device=device, dtype=dtype)
mouse_cond = torch.randn(B, num_frames, 2, device=device, dtype=dtype)
keyboard_cond = torch.randn(B, num_frames, 6, device=device, dtype=dtype)

# 创建模块
action_module = ActionModule(
    mouse_dim_in=2,
    keyboard_dim_in=6,
    hidden_size=128,
    img_hidden_size=img_hidden_size,
    enable_keyboard=True,
    enable_mouse=True,
).to(device=device, dtype=dtype).eval()

# 运行 (非因果模式)
with torch.no_grad():
    output = action_module(
        x,
        tt=num_feats,           # 15
        th=H_tokens,            # 22
        tw=W_tokens,            # 40
        mouse_condition=mouse_cond,
        keyboard_condition=keyboard_cond,
        is_causal=False,
        num_frame_per_block=num_feats  # 非因果: 处理所有 frames
    )

print(f"Output: {output.shape}")  # [1, 13200, 1536]
```

### 流式因果推理测试

```python
# 首个 chunk
num_frame_per_block = 3  # 从配置文件
start_frame = 0
chunk_num_feats = num_frame_per_block  # 3
chunk_num_frames = (chunk_num_feats - 1) * vae_ratio + 1  # 9

# 创建 chunk 输入
x_chunk = torch.randn(B, chunk_num_feats * S, img_hidden_size, device='cuda')
mouse_chunk = torch.randn(B, chunk_num_frames, 2, device='cuda')
keyboard_chunk = torch.randn(B, chunk_num_frames, 6, device='cuda')

# 初始化 KV cache
cache_mouse = {
    "k": torch.zeros(B * S, 20, 16, 64, device='cuda'),
    "v": torch.zeros(B * S, 20, 16, 64, device='cuda'),
    "global_end_index": torch.tensor([0], device='cuda'),
    "local_end_index": torch.tensor([0], device='cuda'),
}
cache_keyboard = {
    "k": torch.zeros(B * S, 20, 16, 64, device='cuda'),
    "v": torch.zeros(B * S, 20, 16, 64, device='cuda'),
    "global_end_index": torch.tensor([0], device='cuda'),
    "local_end_index": torch.tensor([0], device='cuda'),
}

# 运行 (因果模式 - 首个 chunk)
with torch.no_grad():
    output_chunk = action_module(
        x_chunk,
        tt=chunk_num_feats,     # 3
        th=H_tokens,            # 22
        tw=W_tokens,            # 40
        mouse_condition=mouse_chunk,
        keyboard_condition=keyboard_chunk,
        is_causal=True,
        kv_cache_mouse=cache_mouse,
        kv_cache_keyboard=cache_keyboard,
        start_frame=start_frame,  # 0
        num_frame_per_block=chunk_num_feats  # 3
    )

# 后续 chunk
start_frame = 3
x_chunk2 = torch.randn(B, chunk_num_feats * S, img_hidden_size, device='cuda')
mouse_chunk2 = torch.randn(B, chunk_num_frames, 2, device='cuda')
keyboard_chunk2 = torch.randn(B, chunk_num_frames, 6, device='cuda')

with torch.no_grad():
    output_chunk2 = action_module(
        x_chunk2,
        tt=chunk_num_feats,
        th=H_tokens,
        tw=W_tokens,
        mouse_condition=mouse_chunk2,
        keyboard_condition=keyboard_chunk2,
        is_causal=True,
        kv_cache_mouse=cache_mouse,  # 复用 cache
        kv_cache_keyboard=cache_keyboard,
        start_frame=start_frame,  # 3
        num_frame_per_block=chunk_num_feats
    )
```

## 简化测试配置 (用于单元测试)

如果想要更小的测试维度：

```python
# 简化配置
latent_frames = 3
vae_ratio = 4
num_feats = latent_frames  # 3
num_frames = (latent_frames - 1) * vae_ratio + 1  # 9
H, W = 44, 80
H_tokens, W_tokens = 22, 40
S = 880
img_hidden_size = 1536

# 输入形状
x: [1, 3 * 880, 1536] = [1, 2640, 1536]
mouse_condition: [1, 9, 2]
keyboard_condition: [1, 9, 6]
```

## 关键注意事项

1. **数据类型 (dtype)**
   - **必须使用 `torch.bfloat16` 或 `torch.float16`**
   - Flash Attention 不支持 fp32
   - 模型和输入都要转换: `.to(device=device, dtype=dtype)`

2. **必须提供 keyboard_condition**
   - 即使 `enable_keyboard=False`
   - 原始代码在 line 202 无条件访问 `keyboard_condition.shape`

3. **tt, th, tw 参数是 patched 后的维度**
   - `tt = num_feats` (latent frames, 无时间 patching)
   - `th = H_latent // patch_size[1]`
   - `tw = W_latent // patch_size[2]`

4. **断言检查**
   - `tt * th * tw == x.shape[1]` 必须成立
   - 非因果: `N_feats == tt`
   - 因果: `(N_frames-1)//vae_ratio+1 == start_frame + num_frame_per_block`

5. **num_frame_per_block**
   - 非因果模式: 设置为 `num_feats` (处理所有 frames)
   - 因果模式: 设置为 chunk 大小 (如 3)

6. **RoPE 维度**
   - 默认 `rope_dim_list=[8, 28, 28]` (sum=64)
   - 但 `head_dim=96`
   - ActionConfig 会警告并自动调整

## 测试验证清单

- [ ] 空间维度正确: `tt * th * tw == num_feats * S`
- [ ] Action 帧数正确: `num_frames = (num_feats-1) * vae_ratio + 1`
- [ ] 提供了 keyboard_condition
- [ ] num_frame_per_block 设置正确
- [ ] 因果模式下初始化了 KV cache
- [ ] start_frame 在后续 chunk 中正确递增
