# 快速开始指南

## 已创建的文件

为了帮助你优化 CausalWanModel，我已经创建了完整的测试和优化框架：

### 测试文件
- ✅ `tests/test_causal_wan_model.py` - 完整的单元测试套件
- ✅ `tests/run_tests.sh` - 便捷的测试运行脚本
- ✅ `tests/README.md` - 详细的测试文档

### 优化模板
- ✅ `wan/modules/optimized_causal_model.py` - 优化实现模板
- ✅ `OPTIMIZATION_GUIDE.md` - 完整的优化指南

## 立即开始

### 1. 运行基准测试

```bash
cd /home/ykx/dev/Matrix-Game/Matrix-Game-2

# 方式1：使用便捷脚本
./tests/run_tests.sh benchmark

# 方式2：直接使用 pytest
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark -v -s
```

### 2. 验证测试通过

```bash
# 测试正确性
./tests/run_tests.sh correctness

# 或单独运行
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_correctness -v -s
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_with_kv_cache -v -s
```

### 3. 开始优化

编辑 `wan/modules/optimized_causal_model.py`：

```python
class OptimizedCausalWanModel(CausalWanModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在这里添加你的优化

    def _forward_inference(self, x, t, ...):
        # 重写这个方法实现优化
        pass
```

### 4. 测试你的优化

修改 `tests/test_causal_wan_model.py` 中的 `test_model_comparison` 方法：

```python
def test_model_comparison(self, original_model, test_inputs, ...):
    from wan.modules.optimized_causal_model import OptimizedCausalWanModel

    # 加载优化模型
    optimized_model = OptimizedCausalWanModel.from_config("configs/distilled_model/universal")
    optimized_model.eval()
    optimized_model.to(device=device, dtype=dtype)
    optimized_model.num_frame_per_block = 3  # 重要！

    # 运行对比...
```

然后运行：

```bash
./tests/run_tests.sh compare
```

## 关键参数说明

### num_frame_per_block = 3

这是从 `configs/inference_yaml/inference_universal.yaml` 中获取的关键参数：
- 模型每次处理 3 个 latent frames
- Action conditions 需要 `1 + 4 * (num_frames - 1) = 9` 个时间步
- 这影响 KV cache 的断言检查

### 输入形状

```python
x: [B=1, C=16, F=3, H=44, W=80]
t: [B=1, F=3]
visual_context: [B=1, 257, 1280]
mouse_cond: [B=1, 9, 2]      # 1 + 4*(3-1) = 9
keyboard_cond: [B=1, 9, 4]
```

## 常见问题

### Q1: AssertionError in action_module.py:210

这通常是因为 `num_frame_per_block` 没有设置为 3。确保：

```python
model.num_frame_per_block = 3
```

### Q2: CUDA OOM

如果显存不足，可以：
- 减小 `num_frames` 从 3 到 1
- 使用 `torch.float16` 而不是 `bfloat16`
- 减小 batch size（虽然已经是 1 了）

### Q3: 测试运行很慢

第一次运行会：
- 编译 FlexAttention
- 初始化 KV cache
- 加载模型权重

后续运行会快很多。

## 优化目标

根据你的问题描述，主要目标是优化 `DiffusionModel_Forward` 中的小算子开销。

### 推荐的优化顺序

1. **第一周**：基础优化（15-20% 收益）
   - Torch.compile 局部模块
   - 减少类型转换
   - 融合 LayerNorm + Linear

2. **第二周**：中级优化（累计 30-35% 收益）
   - 静态 KV Cache（环形缓冲区）
   - FlashAttention 替换

3. **第三周+**：高级优化（累计 50-70% 收益）
   - Triton 自定义 kernel
   - CUDA Graph

详见 `OPTIMIZATION_GUIDE.md`。

## 文件结构

```
Matrix-Game-2/
├── tests/
│   ├── test_causal_wan_model.py    # 主测试文件
│   ├── run_tests.sh                # 测试运行脚本
│   ├── README.md                   # 测试文档
│   ├── QUICK_START.md              # 本文件
│   └── baseline_performance.txt    # 性能基准（运行后生成）
├── wan/modules/
│   ├── causal_model.py            # 原始模型
│   └── optimized_causal_model.py  # 优化模板
└── OPTIMIZATION_GUIDE.md          # 完整优化指南
```

## 下一步

1. ✅ 运行 `./tests/run_tests.sh benchmark` 建立基准
2. ✅ 阅读 `OPTIMIZATION_GUIDE.md` 了解优化策略
3. ✅ 编辑 `wan/modules/optimized_causal_model.py` 实现优化
4. ✅ 运行 `./tests/run_tests.sh compare` 验证

祝优化顺利！🚀
