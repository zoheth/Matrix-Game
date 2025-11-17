## Scheduler 架构重构

### 当前设计的问题

#### 1. 破碎的继承体系

```python
# 旧设计
class SchedulerInterface(ABC):  # "接口"
    def convert_x0_to_noise(self, ...):  # 但包含具体实现！
        # 200 行实现代码
        pass

class FlowMatchScheduler():  # 不继承接口！
    pass

# 在 WanDiffusionWrapper 中
scheduler = FlowMatchScheduler(...)
scheduler.convert_x0_to_noise = types.MethodType(  # 动态绑定黑魔法
    SchedulerInterface.convert_x0_to_noise, scheduler
)
```

**问题：**
- `SchedulerInterface` 名为接口，实为混合体
- `FlowMatchScheduler` 不继承接口，违反多态性
- 通过 `types.MethodType` 动态绑定方法是 **反模式**
- 无法利用 Python 的类型检查和 IDE 支持

#### 2. 所有权混乱

```python
# Scheduler 由 WanDiffusionWrapper 创建
class WanDiffusionWrapper:
    def __init__(self):
        self.scheduler = FlowMatchScheduler(shift=5.0)

# 但被 Pipeline 获取和使用
class CausalInferencePipeline:
    def __init__(self, generator):
        self.scheduler = generator.get_scheduler()  # 获取 Scheduler

        # Pipeline 还要处理 timestep warping
        self.denoising_step_list = self._prepare_denoising_steps()
```

**问题：**
- **谁真正拥有 Scheduler？** Generator 还是 Pipeline？
- Timestep 逻辑分散在两处：
  - Pipeline: `_prepare_denoising_steps()` 处理 warping
  - Scheduler: `set_timesteps()` 生成 timesteps
- 违反单一职责原则

#### 3. Pipeline 中的 Timestep 处理逻辑

```python
# 在 base_pipeline.py
def _prepare_denoising_steps(self) -> torch.Tensor:
    steps = torch.tensor(self.config.inference.denoising_steps, dtype=torch.long)

    if self.config.inference.warp_denoising_step:
        timesteps = torch.cat([
            self.scheduler.timesteps.cpu(),
            torch.tensor([0], dtype=torch.float32)
        ])
        steps = timesteps[1000 - steps]  # 这个 warping 逻辑应该在 Scheduler 里！

    return steps
```

**问题：**
- Pipeline 需要了解 Scheduler 的内部实现（`timesteps`，warping 公式）
- 破坏封装性
- 难以替换不同的 Scheduler

---

### 重构后的设计

#### 核心原则

1. **清晰的继承层次** - 所有 Scheduler 继承统一基类
2. **单一职责** - Scheduler 拥有 **所有** timestep 相关逻辑
3. **明确所有权** - Scheduler 由 Pipeline 拥有和管理
4. **封装** - Pipeline 不需要知道 Scheduler 内部实现

#### 新的类层次结构

```python
class DiffusionScheduler(ABC):
    """所有 Scheduler 的抽象基类"""

    @abstractmethod
    def set_timesteps(self, num_inference_steps, denoising_strength):
        """设置推理时的离散时间步"""
        pass

    @abstractmethod
    def add_noise(self, original_samples, noise, timestep):
        """前向扩散：添加噪声"""
        pass

    @abstractmethod
    def step(self, model_output, timestep, sample):
        """反向扩散：单步去噪"""
        pass

    def get_inference_timesteps(self, custom_steps=None, warp=False):
        """
        获取推理用的 timesteps（可选自定义或 warp）

        这是新增的关键方法，将 Pipeline 中的 timestep 逻辑移入 Scheduler
        """
        if custom_steps is None:
            return self.timesteps.clone()

        steps_tensor = torch.tensor(custom_steps, dtype=torch.long)

        if not warp:
            return steps_tensor

        # Warping 逻辑现在在 Scheduler 内部
        timesteps_extended = torch.cat([
            self.timesteps.cpu(),
            torch.tensor([0], dtype=torch.float32)
        ])
        warped_steps = timesteps_extended[1000 - steps_tensor]

        return warped_steps


class FlowMatchingScheduler(DiffusionScheduler):
    """Flow Matching 的具体实现"""

    def __init__(self, num_train_timesteps=1000, shift=3.0, ...):
        super().__init__(num_train_timesteps)
        self.shift = shift
        # ...

    def convert_flow_to_x0(self, flow_pred, xt, timestep):
        """Flow 特有的转换方法"""
        # x_0 = x_t - sigma_t * v
        pass
```

#### 新的所有权模型

```python
# Scheduler 由 Pipeline 直接创建和拥有
class BatchCausalInferencePipeline:
    def __init__(self, config, generator, vae_decoder):
        self.generator = generator

        # Pipeline 拥有 Scheduler
        self.scheduler = FlowMatchingScheduler(
            shift=config.model.timestep_shift,
            num_inference_steps=1000
        )

        # 使用 Scheduler 的封装方法
        self.denoising_step_list = self.scheduler.get_inference_timesteps(
            custom_steps=config.inference.denoising_steps,
            warp=config.inference.warp_denoising_step
        )
```

**优点：**
- 明确的所有权：Scheduler 属于 Pipeline
- 所有 timestep 逻辑在 Scheduler 内
- Generator 不再关心 Scheduler（解耦）

---

### 关键改进

#### 1. Timestep 逻辑集中化

**之前（分散）：**
```python
# 在 Pipeline 中
def _prepare_denoising_steps(self):
    steps = torch.tensor(self.config.inference.denoising_steps)
    if self.config.inference.warp_denoising_step:
        # Warping 逻辑在 Pipeline
        timesteps = torch.cat([self.scheduler.timesteps.cpu(), ...])
        steps = timesteps[1000 - steps]
    return steps

# 在 Scheduler 中
def set_timesteps(self, num_inference_steps):
    self.timesteps = self.sigmas * self.num_train_timesteps
```

**之后（集中）：**
```python
# 全部在 Scheduler 中
class DiffusionScheduler:
    def get_inference_timesteps(self, custom_steps=None, warp=False):
        """一个方法处理所有 timestep 逻辑"""
        if custom_steps is None:
            return self.timesteps.clone()

        steps_tensor = torch.tensor(custom_steps)

        if warp:
            # Warping 逻辑现在也在 Scheduler 内部
            warped_steps = self._warp_timesteps(steps_tensor)
            return warped_steps

        return steps_tensor
```

#### 2. 消除动态方法绑定

**之前（黑魔法）：**
```python
# 在 WanDiffusionWrapper
def get_scheduler(self):
    scheduler = self.scheduler
    # 动态绑定方法（反模式！）
    scheduler.convert_x0_to_noise = types.MethodType(
        SchedulerInterface.convert_x0_to_noise, scheduler
    )
    return scheduler
```

**之后（正常继承）：**
```python
class FlowMatchingScheduler(DiffusionScheduler):
    """直接继承，方法是类的一部分"""

    def convert_flow_to_x0(self, flow_pred, xt, timestep):
        """作为类方法定义，不需要动态绑定"""
        sigma_t = self._get_sigma_at_timestep(timestep)
        return xt - sigma_t * flow_pred
```

#### 3. Generator 和 Scheduler 解耦

**之前（耦合）：**
```python
class WanDiffusionWrapper:
    def __init__(self):
        # Generator 创建和管理 Scheduler
        self.scheduler = FlowMatchScheduler(shift=5.0)

    def get_scheduler(self):
        # Pipeline 从 Generator 获取 Scheduler
        return self.scheduler
```

**之后（解耦）：**
```python
class WanDiffusionWrapper:
    """Generator 不再持有 Scheduler"""

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep, scheduler):
        """需要时通过参数传入 Scheduler"""
        return scheduler.convert_flow_to_x0(flow_pred, xt, timestep)

class Pipeline:
    def __init__(self, generator, ...):
        # Pipeline 直接创建 Scheduler
        self.scheduler = FlowMatchingScheduler(...)
        self.generator = generator  # Generator 不知道 Scheduler
```

**为什么这样更好？**
- Generator 职责：执行模型前向传播
- Scheduler 职责：管理噪声调度
- 它们是正交的关注点，不应耦合

---

### 迁移策略

#### Phase 1: 添加新接口（向后兼容）

1. 创建 `utils/scheduler_refactored.py`
2. 保留旧的 `utils/scheduler.py`
3. 在 Pipeline 中提供两种初始化方式：

```python
class Pipeline:
    def __init__(self, use_new_scheduler=False):
        if use_new_scheduler:
            from utils.scheduler_refactored import FlowMatchingScheduler
            self.scheduler = FlowMatchingScheduler(...)
        else:
            # 保持向后兼容
            self.scheduler = self.generator.get_scheduler()
```

#### Phase 2: 逐步迁移

1. 更新 `BatchCausalInferencePipeline` 使用新 Scheduler
2. 更新 `StreamingCausalInferencePipeline`
3. 运行测试确保功能一致

#### Phase 3: 弃用旧接口

1. 添加弃用警告
2. 6-12 个月后移除旧代码

---

### 对比总结

| 方面 | 旧设计 | 新设计 |
|------|--------|--------|
| **继承** | 破碎（不继承接口） | 清晰（单一基类） |
| **方法绑定** | 动态 `types.MethodType` | 静态类方法 |
| **所有权** | Generator 拥有 | Pipeline 拥有 |
| **Timestep 逻辑** | 分散在 Pipeline 和 Scheduler | 集中在 Scheduler |
| **耦合度** | 高（Generator ↔ Scheduler） | 低（正交分离） |
| **可测试性** | 困难 | 容易 |
| **可扩展性** | 困难 | 容易（新 Scheduler 只需继承） |

---

### 使用示例

#### 基础使用

```python
from utils.scheduler_refactored import FlowMatchingScheduler
from pipeline import BatchCausalInferencePipeline

# 创建 Scheduler
scheduler = FlowMatchingScheduler(
    num_train_timesteps=1000,
    shift=3.0,
    sigma_min=0.003 / 1.002
)

# 设置推理步数
scheduler.set_timesteps(num_inference_steps=50)

# 获取推理用的 timesteps
# 选项 1: 使用默认 timesteps
timesteps = scheduler.get_inference_timesteps()

# 选项 2: 自定义 timesteps
timesteps = scheduler.get_inference_timesteps(
    custom_steps=[1000, 750, 500, 250]
)

# 选项 3: 自定义 + warping
timesteps = scheduler.get_inference_timesteps(
    custom_steps=[1000, 750, 500, 250],
    warp=True  # 应用 Scheduler 的 warping 逻辑
)
```

#### 在 Pipeline 中使用

```python
class BatchCausalInferencePipeline(BaseCausalInferencePipeline):
    def __init__(self, config, generator, vae_decoder):
        super().__init__(config, generator, vae_decoder)

        # Pipeline 拥有 Scheduler
        self.scheduler = FlowMatchingScheduler(
            shift=config.model.get('timestep_shift', 3.0)
        )

        # 获取 denoising timesteps（所有逻辑在 Scheduler 内）
        self.denoising_step_list = self.scheduler.get_inference_timesteps(
            custom_steps=config.inference.denoising_steps,
            warp=config.inference.warp_denoising_step
        )

    def _denoise_block(self, noisy_input, conditional_dict, ...):
        for timestep in self.denoising_step_list:
            # 使用 Scheduler 进行去噪
            denoised = self.scheduler.step(
                model_output=flow_pred,
                timestep=timestep,
                sample=noisy_input
            )
```

---

### 结论

重构后的 Scheduler 设计：

✅ **清晰的职责边界** - Scheduler 管理所有 timestep 逻辑
✅ **标准的 OOP 设计** - 正常的继承，不需要黑魔法
✅ **低耦合** - Generator 和 Scheduler 正交分离
✅ **易于测试** - 每个组件独立可测
✅ **易于扩展** - 新的 Scheduler 只需继承基类

这是一个**真正符合软件工程原则的设计**，而不是"能跑就行"的研究原型。
