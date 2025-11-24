# 3D RoPE 内存优化方案

## 问题背景

在 FlashInfer 集成中，我们需要为 3D RoPE（Rotary Position Embedding）预计算频率。原始实现会预计算完整的 3D 频率网格，导致严重的 OOM（Out of Memory）问题。

### 原始实现的问题

```python
# 原始实现：预计算完整网格
self.freq_grid = torch.zeros(
    max_frames, height, width, head_dim_half,
    dtype=self.dtype, device=self.device
)

# 使用三重循环填充
for f in range(max_frames):
    for h in range(height):
        for w in range(width):
            self.freq_grid[f, h, w, :c_time] = self.freqs_time[f]
            self.freq_grid[f, h, w, c_time:c_time+c_height] = self.freqs_height[h]
            self.freq_grid[f, h, w, c_time+c_height:] = self.freqs_width[w]
```

**内存占用计算**：
- `max_frames = 256`
- `height = 22`（352 / 16 patchify后）
- `width = 40`（640 / 16 patchify后）
- `head_dim_half ≈ 64`

总元素数 = 256 × 22 × 40 × 64 = **14,417,920 个元素**

以 bfloat16 计算 = 14M × 2 bytes ≈ **28MB per layer**

对于 30+ 层的模型，总内存占用 = 28MB × 30 ≈ **840MB 额外显存**

## 优化方案

### 核心思路：按需计算（On-demand Computation）

不再预计算完整的 3D 网格，而是只存储 1D 频率分量，在需要时使用向量化操作动态计算。

### 存储结构对比

| 数据 | 原始实现 | 优化后 |
|------|---------|--------|
| `freq_grid` | [256, 22, 40, 64] = 14M 元素 | 不存储 |
| `freqs_time` | [1024, ~21] | [1024, ~21] |
| `freqs_height` | [1024, ~21] | [1024, ~21] |
| `freqs_width` | [1024, ~21] | [1024, ~21] |

**优化后内存**：3 × 1024 × 21 × 2 bytes ≈ **126KB per layer**

**内存节省**：28MB → 126KB = **减少 99.5%**

### 实现代码

```python
class PrecomputedRoPE3DCache:
    """
    按需计算 3D RoPE 频率，而非预计算完整网格。
    内存高效且 CUDA Graph 兼容。
    """

    def __init__(self, freqs, max_frames=1024, height=22, width=40, device=None):
        # 只存储 1D 频率分量（内存极小）
        head_dim_half = freqs.shape[1]
        c = head_dim_half
        self.c_time = c - 2 * (c // 3)
        self.c_height = c // 3
        self.c_width = c // 3
        self.head_dim_half = head_dim_half

        # 仅存储 1D 分量
        self.freqs_time = freqs[:, :self.c_time].to(device)       # [1024, c_time]
        self.freqs_height = freqs[:, self.c_time:self.c_time + self.c_height].to(device)
        self.freqs_width = freqs[:, self.c_time + self.c_height:].to(device)

    def get_freqs_for_frame_range(self, start_frame, num_frames):
        """按需计算指定帧范围的 3D 频率"""
        f, h, w = num_frames, self.height, self.width

        # 向量化操作：与 causal_rope_apply 相同逻辑，但批量处理
        # 时间分量: [F, 1, 1, c_time] -> [F, H, W, c_time]
        time_freqs = self.freqs_time[start_frame:start_frame+f].view(f, 1, 1, -1).expand(f, h, w, -1)

        # 高度分量: [1, H, 1, c_height] -> [F, H, W, c_height]
        height_freqs = self.freqs_height[:h].view(1, h, 1, -1).expand(f, h, w, -1)

        # 宽度分量: [1, 1, W, c_width] -> [F, H, W, c_width]
        width_freqs = self.freqs_width[:w].view(1, 1, w, -1).expand(f, h, w, -1)

        # 拼接: [F, H, W, head_dim_half]
        freqs = torch.cat([time_freqs, height_freqs, width_freqs], dim=-1)

        # 重塑为注意力所需格式: [F*H*W, 1, head_dim_half]
        return freqs.reshape(-1, 1, self.head_dim_half)
```

### 计算复杂度

| 操作 | 原始实现 | 优化后 |
|------|---------|--------|
| 初始化 | O(F × H × W × D) 三重循环 | O(D) 仅复制 |
| 每次查询 | O(1) 切片 | O(F × H × W) 向量化 |

虽然查询时需要计算，但：
1. 使用 PyTorch 向量化操作，GPU 上极快
2. `expand()` 不实际复制数据，只改变 stride
3. 只在 `cat()` 时才真正分配临时内存
4. 临时内存仅为当前帧的大小（3帧 × 22 × 40 × 64 ≈ 168K），用完即释放

## CUDA Graph 兼容性

此优化保持 CUDA Graph 兼容：
- 无动态内存分配（`expand` 使用 stride，`cat` 输出形状固定）
- 无条件分支依赖张量值
- 无 CPU-GPU 同步（如 `.item()`）

## 使用方式

```bash
# 启用 FlashInfer（自动使用优化后的 RoPE）
python inference.py --flashinfer_mode enabled --img_path demo_images/universal/0000.png

# 配合 CUDA Graph 获得最佳性能
python inference.py --flashinfer_mode enabled --use_cuda_graph --warmup
```

## 总结

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 每层 RoPE 缓存 | ~28MB | ~126KB |
| 30层总额外显存 | ~840MB | ~3.8MB |
| 内存减少 | - | **99.5%** |
| 初始化时间 | 慢（三重循环） | 快（仅复制） |
| 查询性能 | O(1) | O(F×H×W) 向量化 |
| CUDA Graph 兼容 | 是 | 是 |
