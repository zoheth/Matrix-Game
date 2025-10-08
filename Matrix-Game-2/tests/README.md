# CausalWanModel Test Suite

完整的单元测试套件，用于验证 CausalWanModel 的正确性和性能对比。

## 快速开始

### 安装依赖

```bash
pip install pytest pytest-benchmark numpy
```

### 运行测试

```bash
# 运行所有测试
pytest tests/test_causal_wan_model.py -v

# 运行单个测试
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_correctness -v

# 显示详细输出
pytest tests/test_causal_wan_model.py -v -s
```

## 测试说明

### 1. `test_forward_correctness` - 前向传播正确性测试

验证模型在训练模式（无KV cache）下的基本功能：
- 输出 shape 正确性
- 无 NaN/Inf
- 输出数值范围合理

```bash
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_correctness -v -s
```

### 2. `test_forward_with_kv_cache` - KV Cache 推理测试

验证模型在推理模式（有KV cache）下的功能：
- KV cache 正确更新
- Cross-attention cache 初始化
- 输出正确性

```bash
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_with_kv_cache -v -s
```

### 3. `test_performance_benchmark` - 性能基准测试

测量原始模型的性能，作为优化的基准：
- 平均延迟
- P95/P99 延迟
- 标准差

```bash
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark -v -s
```

性能基准会保存到 `tests/baseline_performance.txt`

### 4. `test_model_comparison` - 模型对比测试（模板）

这是一个模板测试，用于对比原始模型和优化后的模型。

**使用步骤：**

1. 编写你的优化版本 `OptimizedCausalWanModel`
2. 在测试中导入并加载优化模型
3. 取消注释对比代码
4. 运行测试验证等价性和性能提升

```bash
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_model_comparison -v -s
```

## 优化工作流

### 第一步：建立基准

```bash
# 运行性能基准测试，建立原始模型的性能基准
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark -v -s

# 查看基准结果
cat tests/baseline_performance.txt
```

### 第二步：实现优化模型

在 `wan/modules/optimized_causal_model.py` 中实现你的优化版本：

```python
class OptimizedCausalWanModel(CausalWanModel):
    """优化版本的 CausalWanModel"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 你的优化初始化

    def _forward_inference(self, *args, **kwargs):
        # 你的优化实现
        pass
```

### 第三步：集成到测试

修改 `test_model_comparison` 方法，加载你的优化模型：

```python
def test_model_comparison(self, original_model, test_inputs, kv_cache_structure, device, dtype):
    from wan.modules.optimized_causal_model import OptimizedCausalWanModel

    # 加载优化模型
    optimized_model = OptimizedCausalWanModel.from_config("configs/distilled_model/universal")
    optimized_model.eval()
    optimized_model.to(device=device, dtype=dtype)
    optimized_model.requires_grad_(False)

    # 运行对比...
```

### 第四步：验证等价性和性能

```bash
# 验证数值等价性
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_model_comparison -v -s

# 对比性能
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark -v -s
```

## 测试输入规格

测试使用以下标准输入：

| 参数 | Shape | 说明 |
|------|-------|------|
| `x` | `[1, 16, 3, 44, 80]` | 主输入 latent (B, C, F, H, W) |
| `t` | `[1, 3]` | Timestep (B, F) |
| `cond_concat` | `[1, 20, 3, 44, 80]` | Concat 条件 |
| `visual_context` | `[1, 257, 1280]` | CLIP 视觉特征 |
| `mouse_cond` | `[1, 9, 2]` | 鼠标动作条件 |
| `keyboard_cond` | `[1, 9, 4]` | 键盘动作条件 |

## KV Cache 规格

- **Self-Attention KV Cache**: `[1, 13200, 12, 128]` per block
  - 13200 = 15 frames × 880 tokens/frame
- **Keyboard KV Cache**: `[1, 15, 16, 64]` per block
- **Mouse KV Cache**: `[880, 15, 16, 64]` per block
- **Cross-Attention Cache**: `[1, 257, 12, 128]` per block

## 常见问题

### Q: 测试失败，提示找不到配置文件

确保在项目根目录运行测试：
```bash
cd /home/ykx/dev/Matrix-Game/Matrix-Game-2
pytest tests/test_causal_wan_model.py -v
```

### Q: CUDA OOM

减小 batch size 或 num_frames（默认是 3）：
```python
# 在 test_inputs fixture 中修改
num_frames = 1  # 从 3 改为 1
```

### Q: 如何调整性能测试的迭代次数

使用 pytest 参数化：
```bash
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark[3-20] -v -s
# [warmup-iterations] 格式
```

### Q: 数值差异的可接受阈值是多少

- **Max difference < 1e-2**: 使用 bfloat16 的可接受范围
- **Mean difference < 1e-3**: 理想范围
- 如果使用更激进的优化（如 TF32），可以放宽到 `1e-1`

## 扩展测试

可以添加更多测试场景：

```python
def test_multi_block_inference(self):
    """测试多个 block 连续推理"""
    pass

def test_different_dtypes(self):
    """测试不同精度：fp32, fp16, bf16"""
    pass

def test_batch_sizes(self):
    """测试不同 batch size"""
    pass
```

## 性能优化目标

基于当前 Matrix-Game 的性能分析，优化目标：

- **目标1**: DiffusionModel_Forward 延迟降低 30-50%
- **目标2**: 保持数值精度 (max_diff < 1e-2)
- **目标3**: 支持 CUDA Graph (可选，高级目标)

## 许可证

遵循 Matrix-Game 项目的许可证。
