# Batch vs Streaming 推理对比

## 📌 核心结论

**是的，推理核心几乎完全一样！** 主要区别在于：

1. ✅ **推理算法：** 完全相同（Diffusion 去噪 + KV Cache + VAE 解码）
2. ❌ **条件获取：** Batch 是静态预设，Streaming 是动态交互
3. ❌ **输出时机：** Batch 最后输出，Streaming 每块实时输出
4. ❌ **用户交互：** Batch 无交互，Streaming 每块可控制/停止

---

## 🔍 逐行对比核心差异

### **差异 1: 条件准备方式**

#### Batch 模式 ([inference.py](inference.py)):
```python
# 一次性生成所有动作条件
if mode == 'universal':
    cond_data = Bench_actions_universal(num_frames)  # ← 预生成 597 帧动作
    mouse_condition = cond_data['mouse_condition'].unsqueeze(0)
    conditional_dict['mouse_cond'] = mouse_condition

keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0)
conditional_dict['keyboard_cond'] = keyboard_condition

# 所有 block 共用同一个 conditional_dict
for block in range(50):
    self.generator(..., conditional_dict=cond_current(conditional_dict, ...))
```

#### Streaming 模式 ([pipeline/causal_inference.py:593](pipeline/causal_inference.py#L593)):
```python
# 初始化时也是预生成（作为默认值）
cond_data = Bench_actions_universal(num_frames)
conditional_dict['mouse_cond'] = ...
conditional_dict['keyboard_cond'] = ...

# 但每个 block 前会动态替换！
for block in range(max_blocks):
    current_actions = get_current_action(mode=mode)  # 🎮 实时用户输入
    new_act, conditional_dict = cond_current(
        conditional_dict,
        current_start_frame,
        num_frame_per_block,
        replace=current_actions,  # ← 关键：动态替换当前块的条件
        mode=mode
    )
    self.generator(..., conditional_dict=new_act)  # 使用更新后的条件
```

**关键函数 `cond_current` 的差异：**

| 参数 | Batch 模式 | Streaming 模式 |
|------|-----------|---------------|
| `replace` | `None` | `current_actions` (用户输入) |
| 行为 | 只切片，不修改 | 切片 + 替换指定区间 |

---

### **差异 2: 每块完成后的处理**

#### Batch 模式 ([pipeline/causal_inference.py:352-362](pipeline/causal_inference.py#L352-L362)):
```python
# 去噪完成
denoised_pred = ...

# 更新 KV Cache
self.generator(denoised_pred, ..., timestep=context_noise)

# VAE 解码
with torch.profiler.record_function(f"2.3.{block_idx}_VAE_Decode"):
    denoised_pred = denoised_pred.transpose(1,2)
    video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
    videos += [video]  # ← 只是加入列表

current_start_frame += current_num_frames
# ❌ 没有保存视频
# ❌ 没有用户交互
# 直接进入下一个 block
```

#### Streaming 模式 ([pipeline/causal_inference.py:654-673](pipeline/causal_inference.py#L654-L673)):
```python
# 去噪完成
denoised_pred = ...

# 更新 KV Cache
self.generator(denoised_pred, ..., timestep=context_noise)

# VAE 解码
denoised_pred = denoised_pred.transpose(1,2)
video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
videos += [video]

# ✅ 立即转换为视频格式
video = rearrange(video, "B T C H W -> B T H W C")
video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
video = np.ascontiguousarray(video)

# ✅ 准备可视化配置
if mode != 'templerun':
    config = (
        conditional_dict["keyboard_cond"][...].float().cpu().numpy(),
        conditional_dict["mouse_cond"][...].float().cpu().numpy()
    )

# ✅ 实时保存当前进度
mouse_icon = 'assets/images/mouse.png'
process_video(
    video.astype(np.uint8),
    output_folder + f'/{name}_current.mp4',  # ← 每块都覆盖保存
    config, mouse_icon,
    mouse_scale=0.1,
    process_icon=False,
    mode=mode
)

current_start_frame += current_num_frames

# ✅ 用户交互：询问是否继续
if input("Continue? (Press `n` to break)").strip() == "n":
    break  # 可以提前终止
```

---

### **差异 3: 最终输出**

#### Batch 模式 ([inference.py:185-203](inference.py#L185-L203)):
```python
# 所有 block 完成后，一次性处理
with torch.profiler.record_function("3_Video_Postprocessing"):
    videos_tensor = torch.cat(videos, dim=1)  # 拼接所有 block
    videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
    videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
    video = np.ascontiguousarray(videos)
    # 准备配置...

# 最后保存两个版本
with torch.profiler.record_function("4_Video_Export"):
    process_video(..., 'demo.mp4', ...)         # 无图标版本
    process_video(..., 'demo_icon.mp4', ...)    # 带图标版本
```

#### Streaming 模式 ([pipeline/causal_inference.py:676-690](pipeline/causal_inference.py#L676-L690)):
```python
# 循环结束后（用户停止或达到上限）
videos_tensor = torch.cat(videos, dim=1)
videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
video = np.ascontiguousarray(videos)

# 保存完整视频（带/不带图标）
process_video(..., output_folder + f'/{name}.mp4', ...)       # 无图标
process_video(..., output_folder + f'/{name}_icon.mp4', ...)  # 带图标

# 注意：此时 {name}_current.mp4 已经存在（最后一个 block 的版本）
```

---

### **差异 4: Profiling 支持**

#### Batch 模式:
```python
def inference(self, ..., profile=False):
    if profile:
        diffusion_start = torch.cuda.Event(enable_timing=True)
        diffusion_end = torch.cuda.Event(enable_timing=True)

    for block_idx, current_num_frames in enumerate(tqdm(all_num_frames)):
        if profile:
            torch.cuda.synchronize()
            diffusion_start.record()

        # ... 推理过程 ...

        if profile:
            torch.cuda.synchronize()
            diffusion_end.record()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            print(f"diffusion_time: {diffusion_time}", flush=True)
            fps = video.shape[1] * 1000 / diffusion_time
            print(f"  - FPS: {fps:.2f}")
```

#### Streaming 模式:
```python
def inference(self, ...):
    # ❌ 没有 profile 参数
    # ❌ 没有 CUDA Events 计时
    # 专注于交互性，不关心性能分析
```

---

## 📊 完整对比表

| 特性 | **CausalInferencePipeline** (Batch) | **CausalInferenceStreamingPipeline** |
|------|-------------------------------------|-------------------------------------|
| **类名** | `CausalInferencePipeline` | `CausalInferenceStreamingPipeline` |
| **文件位置** | [pipeline/causal_inference.py:134](pipeline/causal_inference.py#L134) | [pipeline/causal_inference.py:434](pipeline/causal_inference.py#L434) |
| **推理算法** | ✅ 完全相同 | ✅ 完全相同 |
| **KV Cache** | ✅ 完全相同 | ✅ 完全相同 |
| **去噪步骤** | ✅ 完全相同 | ✅ 完全相同 |
| **VAE 解码** | ✅ 完全相同 | ✅ 完全相同 |
| | | |
| **条件获取** | 预设静态动作 | 🎮 每块实时输入 |
| **条件更新** | 不更新 | `replace=current_actions` |
| **实时输出** | ❌ 无 | ✅ 每块保存 `_current.mp4` |
| **用户交互** | ❌ 无 | ✅ 每块询问是否继续 |
| **提前终止** | ❌ 不支持 | ✅ 按 'n' 停止 |
| **Profiling** | ✅ 支持 CUDA Events | ❌ 不支持 |
| **进度条** | ✅ tqdm | ❌ 无 |
| | | |
| **输出文件** | `demo.mp4`, `demo_icon.mp4` | `{name}.mp4`, `{name}_icon.mp4`, `{name}_current.mp4` |
| **适用场景** | 离线渲染、性能测试 | 交互式 Demo、实时游戏 |

---

## 🧪 代码复用情况

### **共享部分（100% 相同）：**

1. ✅ `_initialize_kv_cache()` - KV Cache 初始化
2. ✅ `_initialize_kv_cache_mouse_and_keyboard()` - 动作条件 Cache
3. ✅ `_initialize_crossattn_cache()` - 交叉注意力 Cache
4. ✅ Diffusion 去噪循环逻辑
5. ✅ Scheduler 添加噪声
6. ✅ VAE Decoder 调用

### **独有部分：**

#### Batch 模式独有：
- ✅ `profile` 参数和 CUDA Events 计时
- ✅ tqdm 进度条
- ✅ torch.profiler.record_function 标签（我们添加的）

#### Streaming 模式独有：
- ✅ `get_current_action()` 用户输入
- ✅ `cond_current(..., replace=current_actions)` 条件替换
- ✅ 每块后的视频保存
- ✅ 每块后的用户交互询问
- ✅ `output_folder` 和 `name` 参数

---

## 🎯 关键差异总结

### **1. 条件控制方式**

```python
# Batch: 静态条件
conditional_dict = {...}  # 固定不变
for block in blocks:
    generator(..., conditional_dict=cond_current(conditional_dict, block, 3))
    # ↑ cond_current 只是切片，不修改

# Streaming: 动态条件
conditional_dict = {...}  # 初始化
for block in blocks:
    user_input = get_current_action()  # 🎮 实时输入
    new_cond, conditional_dict = cond_current(
        conditional_dict, block, 3,
        replace=user_input  # ← 修改 conditional_dict
    )
    generator(..., conditional_dict=new_cond)
```

### **2. 输出时机**

```python
# Batch: 最后输出
for block in blocks:
    video = process_block()
    videos.append(video)  # 只存储
final_video = concat(videos)
save_video(final_video)  # ← 一次性保存

# Streaming: 实时输出
for block in blocks:
    video = process_block()
    videos.append(video)
    save_video(video, "current.mp4")  # ← 每次都保存当前进度
    if user_says_stop():
        break
final_video = concat(videos)
save_video(final_video)  # ← 再保存完整版本
```

### **3. 交互性**

```python
# Batch: 无交互
for block in blocks:
    process_block()
    # 自动进入下一块

# Streaming: 有交互
for block in blocks:
    process_block()
    if input("Continue?") == 'n':
        break  # 用户可随时停止
```

---

## 💡 设计思想

两个 Pipeline 的设计体现了：

1. **代码复用：** 核心推理逻辑完全一致，避免重复
2. **职责分离：** Batch 关注性能和完整性，Streaming 关注交互性
3. **灵活扩展：** 通过 `replace` 参数优雅地支持动态条件

### **为什么不合并成一个类？**

虽然逻辑相似，但分开有好处：
- ✅ Batch 可以专注优化性能（profiling, batching）
- ✅ Streaming 可以专注用户体验（实时反馈）
- ✅ 避免"瑞士军刀"类（一个类做太多事）
- ✅ 用户根据场景选择合适的工具

---

## 🔬 验证实验

### **验证条件替换：**

在 `cond_current` 函数中添加打印：

```python
def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    if replace != None:
        print(f"🎮 Replacing conditions for block {current_start_frame // num_frame_per_block}")
        print(f"  Keyboard: {replace['keyboard']}")
        if mode != 'templerun':
            print(f"  Mouse: {replace['mouse']}")
    # ...
```

**预期输出（Streaming）：**
```
🎮 Replacing conditions for block 0
  Keyboard: tensor([1, 0, 0, 0])  # W键
  Mouse: tensor([0, 0.1])         # 右转
...
```

**预期输出（Batch）：**
```
# 无输出（replace=None）
```

---

## 📝 总结

你的观察是正确的！两个 Pipeline 的推理核心**几乎一模一样**，主要区别在于：

| 方面 | 区别程度 |
|------|---------|
| **推理算法** | 0% - 完全相同 |
| **条件获取** | 100% - 完全不同 |
| **输出时机** | 100% - 完全不同 |
| **用户交互** | 100% - 完全不同 |
| **代码结构** | ~10% - 微小差异 |

核心推理流程可以用同一张流程图表示，差异只在**输入来源**和**输出时机**两个外围环节。
