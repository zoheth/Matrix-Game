# CausalWanModel 优化指南

本指南帮助你系统地优化 CausalWanModel，提升 DiffusionModel_Forward 的性能。

## 📋 目录

1. [快速开始](#快速开始)
2. [性能分析](#性能分析)
3. [优化策略](#优化策略)
4. [测试流程](#测试流程)
5. [常见陷阱](#常见陷阱)

## 🚀 快速开始

### 第一步：建立性能基准

```bash
cd /home/ykx/dev/Matrix-Game/Matrix-Game-2

# 运行基准测试
./tests/run_tests.sh benchmark

# 查看基准结果
cat tests/baseline_performance.txt
```

预期输出：
```
Original Model Baseline
Mean: XXX.XX ms
Median: XXX.XX ms
P95: XXX.XX ms
```

### 第二步：运行正确性测试

```bash
# 确保原始模型测试通过
./tests/run_tests.sh correctness
```

### 第三步：开始优化

编辑 `wan/modules/optimized_causal_model.py`，在 TODO 标记处实现你的优化。

### 第四步：验证优化

```bash
# 运行对比测试
./tests/run_tests.sh compare
```

## 📊 性能分析

根据 profiling 数据，主要瓶颈在 `DiffusionModel_Forward` ([utils/wan_wrapper.py:160](utils/wan_wrapper.py#L160))：

### 瓶颈分解

```
DiffusionModel_Forward (总时间: ~XXX ms)
├── Attention (30-40%)
│   ├── FlexAttention computation
│   ├── RoPE application
│   └── KV cache indexing
├── LayerNorm (15-20%)
│   ├── Self-attention norm
│   ├── Cross-attention norm
│   └── FFN norm
├── FFN (20-25%)
│   └── Linear + GeLU + Linear
├── KV Cache Update (10-15%)
│   ├── Dynamic indexing
│   ├── Memory copy
│   └── Index calculation
└── Misc operations (10-15%)
    ├── Embeddings
    ├── Time projection
    └── Unpatchify
```

### 小算子开销

原始实现中存在大量小算子：
- `unflatten` / `flatten` 频繁调用
- 多次 `to()` / `type_as()` 转换
- 动态 shape 计算
- 频繁的 CPU-GPU 同步

## 🎯 优化策略

### 优先级 1：低成本高收益 (1-2天)

#### 1.1 算子融合

**目标文件**: `wan/modules/optimized_causal_model.py`

```python
# 融合 LayerNorm + Linear
class FusedLayerNormLinear(nn.Module):
    def forward(self, x):
        # 单次 kernel 调用
        return F.linear(F.layer_norm(x, ...), self.weight, self.bias)
```

**预期收益**: 5-10% 延迟降低

#### 1.2 减少类型转换

```python
# 坏例子（当前代码）
x = x.to(dtype)
y = y.type_as(x)
z = z.to(device)

# 好例子
# 在初始化时统一设置，避免运行时转换
```

**预期收益**: 3-5% 延迟降低

#### 1.3 Torch.compile 局部编译

```python
# 编译时间嵌入
self.time_embedding = torch.compile(
    self.time_embedding,
    mode="reduce-overhead"
)

# 编译 FFN
for block in self.blocks:
    block.ffn = torch.compile(block.ffn, mode="reduce-overhead")
```

**预期收益**: 10-15% 延迟降低

### 优先级 2：中等成本中等收益 (3-5天)

#### 2.1 静态 KV Cache

**问题**：当前 KV cache 使用动态索引，导致：
- 无法使用 CUDA Graph
- 频繁的索引计算开销

**解决方案**：环形缓冲区

```python
class RingBufferKVCache:
    def __init__(self, max_size, ...):
        # 预分配固定大小 buffer
        self.buffer = torch.zeros(max_size, ...)
        self.head = 0  # 写入位置

    def update(self, new_kv):
        # 固定位置写入，避免动态索引
        size = new_kv.shape[0]
        self.buffer[self.head:self.head+size] = new_kv
        self.head = (self.head + size) % self.max_size
```

**预期收益**: 15-20% 延迟降低 + 支持 CUDA Graph

#### 2.2 FlashAttention 3

替换 FlexAttention：

```python
from flash_attn import flash_attn_func

class OptimizedAttention(nn.Module):
    def forward(self, q, k, v, ...):
        return flash_attn_func(q, k, v, causal=True)
```

**预期收益**: 20-30% Attention 部分加速

### 优先级 3：高成本高收益 (1-2周)

#### 3.1 Triton 自定义 Kernel

针对特定模式手写融合 kernel：

```python
import triton
import triton.language as tl

@triton.jit
def fused_attn_norm_ffn_kernel(...):
    # 融合 Attention + Norm + FFN
    pass
```

**预期收益**: 30-40% 整体加速

#### 3.2 CUDA Graph

完成静态 KV cache 后，封装整个前向传播：

```python
class CUDAGraphWrapper:
    def __init__(self, model):
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # 录制计算图
            self.static_output = model(self.static_input, ...)

    def __call__(self, input):
        # 更新输入，重放图
        self.static_input.copy_(input)
        self.graph.replay()
        return self.static_output
```

**预期收益**: 2-3x 整体加速（最大收益）

## 🧪 测试流程

### 正确性验证

```bash
# 每次修改后运行
pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_model_comparison -v -s
```

**接受标准**：
- Max difference < 1e-2 (bfloat16)
- Mean difference < 1e-3

### 性能验证

```bash
# 对比基准
./tests/run_tests.sh benchmark
```

**记录收益**：
```
优化版本 vs 基准
- 延迟降低: XX%
- P95 改善: XX%
```

### 完整测试

```bash
# 运行所有测试
./tests/run_tests.sh all
```

## ⚠️ 常见陷阱

### 陷阱 1：过早优化

❌ **错误做法**：
```python
# 一次性重写所有代码
class OptimizedCausalWanModel:
    def forward(self, ...):
        # 1000 行全新实现
```

✅ **正确做法**：
```python
# 逐步优化，每步测试
def _forward_inference(self, ...):
    # 步骤1：优化 embedding (测试通过)
    x = self._optimized_embed(x)

    # 步骤2：优化 attention (测试通过)
    for block in self.blocks:
        x = self._optimized_block(block, x)

    # 步骤3：优化 output (测试通过)
    return self._optimized_output(x)
```

### 陷阱 2：忽视数值精度

使用 bfloat16 时要注意：
- Softmax 容易溢出
- 小数累加误差

**解决方案**：关键路径使用 float32
```python
# Attention 中使用更高精度
attn_weights = F.softmax(scores.float(), dim=-1).to(dtype)
```

### 陷阱 3：CUDA Graph 的限制

CUDA Graph 要求：
- ✅ 静态 shape
- ✅ 固定控制流
- ✅ 无 CPU 同步
- ❌ 不支持动态索引
- ❌ 不支持 `if`/`for` 动态分支

**检查方法**：
```python
# 如果看到这些，CUDA Graph 不可用
if kv_cache is not None:  # 动态分支
    ...

kv_cache["k"][:, start:end]  # 动态索引 (start/end 是变量)
```

### 陷阱 4：编译开销

torch.compile 首次运行很慢（编译时间）：
- 预热：3-10 次迭代
- 编译缓存：保存到磁盘

```python
# 首次编译后保存
compiled_model = torch.compile(model)
torch.save(compiled_model, "compiled_model.pt")

# 后续直接加载
compiled_model = torch.load("compiled_model.pt")
```

## 📈 优化路线图

### Week 1: 基础优化
- [ ] 建立测试基准
- [ ] 实现算子融合 (LayerNorm + Linear)
- [ ] 减少类型转换
- [ ] 局部 torch.compile

**目标**: 15-20% 延迟降低

### Week 2: 中级优化
- [ ] 实现静态 KV Cache (环形缓冲区)
- [ ] 集成 FlashAttention 3
- [ ] 优化 RoPE 计算

**目标**: 累计 30-35% 延迟降低

### Week 3-4: 高级优化（可选）
- [ ] 手写 Triton kernel (Attention + Norm + FFN)
- [ ] CUDA Graph 封装
- [ ] 多流并行（如果适用）

**目标**: 累计 50-70% 延迟降低，或 2-3x 加速

## 🔧 调试技巧

### 性能分析

```bash
# 使用 PyTorch Profiler
python inference.py --enable_profile
# 查看 Chrome trace
```

### 数值调试

```python
# 在优化代码中插入检查点
def forward(self, x):
    x_orig = self._original_forward(x)
    x_opt = self._optimized_forward(x)

    diff = (x_orig - x_opt).abs().max()
    if diff > 1e-2:
        print(f"Warning: numerical diff = {diff}")

    return x_opt
```

### CUDA Graph 调试

```python
# 检查是否可以用 CUDA Graph
torch.cuda.make_graphed_callables(model, sample_args)
# 如果报错，说明有不兼容的操作
```

## 📚 参考资源

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [CUDA Graphs Tutorial](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Triton Documentation](https://triton-lang.org/)

## 💡 提示

1. **增量开发**：每次优化一个模块，立即测试
2. **性能监控**：使用 `PerformanceMonitor` 追踪每个模块的耗时
3. **版本控制**：每个优化阶段提交 git
4. **文档记录**：记录每次优化的收益和问题

---

**Good luck with your optimization!** 🚀

如有问题，参考 `tests/README.md` 或查看测试代码示例。
