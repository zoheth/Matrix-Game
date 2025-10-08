# Matrix-Game Profiling 标签索引

本文档列出了所有添加到代码中的 `torch.profiler.record_function` 标签，帮助你在 Chrome Trace Viewer 中快速定位关键操作。

---

## 📍 使用方法

1. **运行推理并生成 trace 文件：**
   ```bash
   python inference.py --enable_profile
   ```

2. **在 Chrome 中查看：**
   - 打开 `chrome://tracing`
   - 加载生成的 `outputs/profile_trace.json`
   - 使用 `Ctrl+F` 搜索下面的标签名称

---

## 🏷️ 标签层级结构

### **Level 1: 顶层流程** ([inference.py](inference.py))

| 标签名称 | 位置 | 描述 |
|---------|------|------|
| `1_Data_Preparation` | inference.py:129 | 数据准备总阶段 |
| `2_Pipeline_Inference` | inference.py:176 | 推理主循环（最耗时） |
| `3_Video_Postprocessing` | inference.py:185 | 视频后处理 |
| `4_Video_Export` | inference.py:201 | 视频导出到文件 |

---

### **Level 2: 数据准备子阶段** ([inference.py](inference.py))

| 标签名称 | 位置 | 描述 |
|---------|------|------|
| `1.1_Image_Loading_Resize` | inference.py:130 | 加载并裁剪输入图像 (640×352) |
| `1.2_VAE_Encode_FirstFrame` | inference.py:135 | VAE编码首帧 → Latent (16×150×44×80) |
| `1.3_Condition_Preparation` | inference.py:142 | 准备 cond_concat mask |
| `1.4_CLIP_Visual_Context` | inference.py:147 | CLIP提取视觉特征 (257 tokens) |
| `1.5_Noise_Sampling` | inference.py:150 | 采样初始噪声 |
| `1.6_Action_Conditions_Setup` | inference.py:156 | 构建键盘/鼠标条件 |

---

### **Level 3: Pipeline 推理阶段** ([pipeline/causal_inference.py](pipeline/causal_inference.py))

#### 3.1 初始化

| 标签名称 | 位置 | 描述 |
|---------|------|------|
| `2.1_KV_Cache_Initialization` | causal_inference.py:218 | 初始化4种 KV Cache (主/鼠标/键盘/交叉注意力) |

#### 3.2 逐块生成循环 (每块重复)

**块级标签格式：** `2.3_Block_{block_idx}` (例如 `2.3_Block_0`, `2.3_Block_1` ...)

每个块包含以下子标签：

| 标签名称模式 | 位置 | 描述 |
|------------|------|------|
| `2.3.{i}_Denoising_Step_{j}_t{timestep}` | causal_inference.py:295 | 第j个去噪步骤 (例如 `2.3.0_Denoising_Step_0_t999`) |
| ↳ `Generator_Forward` | causal_inference.py:303 | Diffusion 模型前向传播 |
| ↳ `Scheduler_AddNoise` | causal_inference.py:314 | 添加噪声到下一步 |
| ↳ `Generator_Forward_Final` | causal_inference.py:325 | 最后一步去噪（不加噪声） |
| `2.3.{i}_Update_KV_Cache` | causal_inference.py:341 | 使用干净预测更新 KV Cache |
| `2.3.{i}_VAE_Decode` | causal_inference.py:358 | VAE 解码 latent → 视频帧 |

---

### **Level 4: Diffusion 模型内部** ([utils/wan_wrapper.py](utils/wan_wrapper.py))

| 标签名称 | 位置 | 描述 |
|---------|------|------|
| `DiffusionModel_Forward` | wan_wrapper.py:160 | CausalWanModel 前向传播（30层Transformer） |
| `Flow_to_X0_Conversion` | wan_wrapper.py:179 | Flow Matching → X0 转换 |

**模型内部结构（未显式标记，但可在 trace 中看到）：**
- Patch Embedding
- Timestep Embedding
- Self-Attention × 30 layers (使用 kv_cache1)
- Cross-Attention × 30 layers (使用 crossattn_cache)
- Keyboard-Attention × 30 layers (使用 kv_cache_keyboard)
- Mouse-Attention × 30 layers (使用 kv_cache_mouse)
- Feed-Forward Networks × 30 layers

---

### **Level 5: VAE Decoder 内部** ([demo_utils/vae_block3.py](demo_utils/vae_block3.py))

#### 5.1 Wrapper 层级

| 标签名称 | 位置 | 描述 |
|---------|------|------|
| `VAE_Decode_Preprocessing` | vae_block3.py:152 | Latent 反归一化 |
| `VAE_Conv2_Input` | vae_block3.py:169 | 输入卷积层 (16→16 channels) |
| `VAE_Decode_Frame_{i}` | vae_block3.py:174 | 逐帧解码（例如 `VAE_Decode_Frame_0`） |
| `VAE_Decode_Postprocessing` | vae_block3.py:185 | Clamp 和维度重排 |

#### 5.2 Decoder 网络层级

| 标签名称 | 位置 | 描述 |
|---------|------|------|
| `VAE_Dec_Conv1` | vae_block3.py:255 | 第一层 3D 卷积 (16→384) |
| `VAE_Dec_Middle_Blocks` | vae_block3.py:271 | 中间层 (ResBlock + Attention) |
| `VAE_Dec_Upsample_Blocks` | vae_block3.py:279 | 上采样块 (4阶段，16倍放大) |
| `VAE_Dec_Head` | vae_block3.py:284 | 输出头 (48→3 RGB) |

---

## 🔍 常见性能分析场景

### 1️⃣ **找出最慢的阶段**
搜索标签：
```
1_Data_Preparation
2_Pipeline_Inference
3_Video_Postprocessing
```
→ 通常 `2_Pipeline_Inference` 占用 99% 时间

---

### 2️⃣ **分析单个块的去噪时间**
搜索标签：
```
2.3_Block_0
2.3_Block_1
```
查看每个块包含的：
- 去噪步数（默认 ~50 步）
- 每步的 `DiffusionModel_Forward` 时间
- VAE 解码时间

---

### 3️⃣ **对比不同去噪步的耗时**
搜索标签：
```
2.3.0_Denoising_Step_0_t999
2.3.0_Denoising_Step_1_t950
...
```
→ 通常第一步最慢（无 cache），后续步骤加速

---

### 4️⃣ **分析 Diffusion 模型内部**
搜索标签：
```
DiffusionModel_Forward
```
展开后可以看到：
- `aten::linear` (FFN 层)
- `aten::scaled_dot_product_attention` (注意力计算)
- `aten::layer_norm` (归一化)

---

### 5️⃣ **分析 VAE Decoder 瓶颈**
搜索标签：
```
VAE_Dec_Conv1
VAE_Dec_Middle_Blocks
VAE_Dec_Upsample_Blocks
VAE_Dec_Head
```
→ 通常 `VAE_Dec_Upsample_Blocks` 最慢（4次上采样）

---

### 6️⃣ **查看 KV Cache 更新耗时**
搜索标签：
```
2.3.0_Update_KV_Cache
2.3.1_Update_KV_Cache
```
→ 这步用干净预测更新历史上下文

---

## 📊 预期时间分布（参考）

基于默认配置（150 latent frames，50 denoising steps）：

| 阶段 | 占比 | 子操作 |
|-----|------|--------|
| **数据准备** | ~1% | VAE Encode 首帧 + CLIP 编码 |
| **推理循环** | ~98% | ↓ |
| ↳ KV Cache 初始化 | <0.1% | 仅首次 |
| ↳ 去噪循环 | ~60% | Diffusion × 150块 × 50步 |
| ↳ VAE 解码 | ~38% | Decoder × 150块 |
| ↳ KV Cache 更新 | ~2% | 每块后更新 |
| **后处理** | <1% | Tensor → NumPy 转换 |

---

## 🎯 优化建议

根据 profiling 结果，可以针对性优化：

1. **Diffusion Forward 慢？**
   - 检查 KV Cache 是否正常工作
   - 考虑减少 Transformer layers (30→20)
   - 使用 FlashAttention

2. **VAE Decode 慢？**
   - 确认 `torch.compile` 生效
   - 检查 `vae_compile_mode` 参数
   - 考虑降低输出分辨率

3. **内存不足？**
   - 减少 `num_output_frames`
   - 降低 `local_attn_size` (15→10)
   - 使用 gradient checkpointing

---

## 📝 查看示例

运行后，在 Chrome Trace Viewer 中：

1. **整体视图：** 可以看到 150 个 `2.3_Block_X` 依次执行
2. **单块放大：** 可以看到 50 个去噪步 + 1 次 VAE 解码
3. **单步放大：** 可以看到 `DiffusionModel_Forward` 内的层级调用
4. **算子级别：** 可以看到 GPU kernel 调用（matmul, conv 等）

---

## 🔧 自定义标签

如果需要添加更多标签，使用：

```python
with torch.profiler.record_function("Your_Custom_Tag"):
    your_operation()
```

建议命名规则：
- 使用下划线分隔单词
- 使用层级编号（如 `2.3.1`）
- 包含关键参数（如 `_t999` 表示 timestep=999）

---

**生成时间：** 2025-10-08
**适用版本：** Matrix-Game-2
**Profiler：** PyTorch Profiler + Chrome Trace Viewer
