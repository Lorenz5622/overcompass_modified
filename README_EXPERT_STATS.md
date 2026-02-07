# 专家调用统计功能说明

## 功能概述

在评测过程中自动记录 **Predict MoE** 模型的专家调用统计信息，包括：
- **全局平均每个 token 调用的专家个数**
- **每层调用专家个数的分布**

**关键特性：**
- ✅ 仅在 Predict MoE（使用 `modeling_moe_infer`）中生效
- ✅ Dynamic MoE（使用 `modeling_moe_ori`）不受影响
- ✅ 自动检测模型类型，无需手动区分
- ✅ 零侵入式设计，不影响评测流程

---

## 已修改的文件

### 1. `/home/cyx/opencompass/opencompass/models/huggingface.py`

**修改内容：**
- 在 `HuggingFaceDynamicMoE.__init__` 中添加模型类型检测
- 添加 `get_expert_usage_stats()` 方法：收集所有层的统计
- 添加 `save_expert_usage_stats()` 方法：保存统计到 JSON 文件
- 添加 `reset_expert_usage_stats()` 方法：重置统计信息

**关键逻辑：**
```python
# 自动检测是否为 Predict MoE
self.is_predict_moe = "modeling_moe_infer" in moe_module.__name__
```

### 2. `/home/cyx/opencompass/opencompass/configs/models/dynamic_moe/dynamic_moe.py`

**修改内容：**
```python
model_kwargs=dict(
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    enable_expert_stats=True,  # 启用专家统计
    expert_stats_path='./expert_usage_stats.json',  # 统计结果保存路径
),
```

### 3. `/home/cyx/Predict_MoE/Predict_MoE/modeling/modeling_moe_infer.py`（需要手动应用）

**需要添加的代码详见：** `PATCH_modeling_moe_infer.py`

**三处修改点：**
1. 在 `SwitchMLP.__init__` 末尾添加统计累积器
2. 在 `SwitchMLP.forward` 中记录每次调用的 k 值
3. 在 `SwitchMLP` 类末尾添加 `get_expert_usage_stats()` 和 `reset_expert_usage_stats()` 方法

---

## 使用方法

### 方式 1：评测结束后手动保存（推荐）

在评测脚本末尾添加：

```python
# 在你的评测脚本（如 eval_single_dynamic_moe.py）末尾添加
from opencompass.models import HuggingFaceDynamicMoE

# 假设你的模型实例名为 model
if hasattr(model, 'save_expert_usage_stats'):
    model.save_expert_usage_stats('./expert_stats_result.json')
```

### 方式 2：在推理过程中实时获取

```python
from opencompass.models import HuggingFaceDynamicMoE
import torch

model = HuggingFaceDynamicMoE(
    path="/data/cyx/models/Predict_MoE_k25/",
    model_kwargs=dict(
        enable_expert_stats=True,
        expert_stats_path='./stats.json',
    ),
)

# 进行推理
outputs = model.generate(inputs, max_out_len=50)

# 获取统计
stats = model.get_expert_usage_stats()
print(f"Average experts per token: {stats['global_avg_k']:.2f}")

# 保存到文件
model.save_expert_usage_stats()
```

### 方式 3：使用环境变量配置

```bash
export OC_MODEL_PATH="/data/cyx/models/Predict_MoE_k25/"
export OC_MODEL_NAME="predict_moe"

python run.py configs/eval_xxx.py \
    --model configs/models/dynamic_moe/dynamic_moe.py
```

---

## 输出格式

统计结果保存为 JSON 格式：

```json
{
  "model_type": "predict_moe",
  "global_avg_k": 3.45,
  "total_tokens": 25600,
  "per_layer": [
    {
      "layer_idx": 0,
      "avg_k": 3.2,
      "total_tokens": 25600,
      "distribution": {
        "2": 5120,
        "3": 12800,
        "4": 7680
      }
    },
    {
      "layer_idx": 1,
      "avg_k": 3.8,
      "total_tokens": 25600,
      "distribution": {
        "3": 10240,
        "4": 15360
      }
    }
    // ... 其他层
  ]
}
```

**字段说明：**
- `model_type`: 模型类型标识
- `global_avg_k`: 全局平均每个 token 调用的专家数
- `total_tokens`: 处理的总 token 数
- `per_layer`: 每层的详细统计
  - `layer_idx`: 层索引
  - `avg_k`: 该层平均每个 token 调用的专家数
  - `total_tokens`: 该层处理的 token 数
  - `distribution`: k 值分布 `{k值: 出现次数}`

---

## 统计分析

使用提供的分析脚本：

```bash
python example_expert_stats_usage.py expert_usage_stats.json
```

输出示例：
```
============================================================
Model Type: predict_moe
Global Average Experts per Token: 3.45
Total Tokens Processed: 25600
============================================================

Per-layer statistics:

  Layer 0:
    Average k: 3.20
    Total tokens: 25600
    Distribution: {'2': 5120, '3': 12800, '4': 7680}

  Layer 1:
    Average k: 3.80
    Total tokens: 25600
    Distribution: {'3': 10240, '4': 15360}

[INFO] Visualization saved to expert_usage_analysis.png
```

---

## 实现原理

### 1. 模型检测机制

通过模块名称自动区分两种模型：

```python
# 在 HuggingFaceDynamicMoE 初始化时
moe_module = importlib.import_module("Predict_MoE.modeling.modeling_moe_infer")
self.is_predict_moe = "modeling_moe_infer" in moe_module.__name__

# modeling_moe_infer -> Predict MoE -> 启用统计
# modeling_moe_ori   -> Dynamic MoE  -> 不启用统计
```

### 2. 统计收集流程

```
┌─────────────────────────────────────────────────────┐
│  SwitchMLP.__init__()                               │
│  - 读取 config.enable_expert_stats                  │
│  - 初始化 _k_call_counts 和 _total_tokens           │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  SwitchMLP.forward()                                │
│  - 每次前向传播记录 dynamic_k                       │
│  - 累积到 _k_call_counts 字典                       │
│  - 更新 _total_tokens                               │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  HuggingFaceDynamicMoE.get_expert_usage_stats()     │
│  - 遍历所有层的 layer.mlp                           │
│  - 调用 get_expert_usage_stats() 收集统计           │
│  - 计算全局平均值                                   │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  HuggingFaceDynamicMoE.save_expert_usage_stats()    │
│  - 将统计结果保存为 JSON                            │
└─────────────────────────────────────────────────────┘
```

### 3. 零侵入设计

- Dynamic MoE 模型：没有统计方法，`get_expert_usage_stats()` 返回 `None`
- Predict MoE 模型：自动检测到统计方法，正常收集数据
- 评测流程：通过 `hasattr()` 检查，兼容两种模型

---

## 常见问题

### Q1: 如何确认统计功能已启用？

**A:** 模型加载时会输出：
```
[INFO] Detected Predict MoE model, enabling expert usage statistics
[Layer 0] Expert usage statistics enabled
[Layer 1] Expert usage statistics enabled
...
```

### Q2: Dynamic MoE 会收集统计吗？

**A:** 不会。代码会自动检测模型类型，仅在 Predict MoE 中启用。

### Q3: 如何修改统计结果保存路径？

**A:** 在配置文件中修改：
```python
model_kwargs=dict(
    enable_expert_stats=True,
    expert_stats_path='/path/to/your/stats.json',
),
```

### Q4: 评测过程中统计会影响性能吗？

**A:** 影响极小。统计仅涉及：
- 字典累加操作（O(1)）
- CPU 端的轻量计算
- 不影响 GPU 计算流程

### Q5: 如何在评测结束后自动保存统计？

**A:** 在评测脚本末尾添加保存逻辑（见"使用方法 - 方式 1"）

---

## 文件清单

- ✅ `/home/cyx/opencompass/opencompass/models/huggingface.py` - 已修改
- ✅ `/home/cyx/opencompass/opencompass/configs/models/dynamic_moe/dynamic_moe.py` - 已修改
- 📝 `PATCH_modeling_moe_infer.py` - modeling_moe_infer 修改补丁
- 📝 `example_expert_stats_usage.py` - 使用示例脚本
- 📝 `README_EXPERT_STATS.md` - 本文档

---

## 下一步操作

1. **应用 modeling_moe_infer 补丁**
   ```bash
   # 参照 PATCH_modeling_moe_infer.py 中的说明
   # 在 /home/cyx/Predict_MoE/Predict_MoE/modeling/modeling_moe_infer.py 中添加代码
   ```

2. **测试统计功能**
   ```bash
   python example_expert_stats_usage.py
   ```

3. **运行评测**
   ```bash
   export OC_MODEL_PATH="/data/cyx/models/Predict_MoE_k25/"
   python run.py your_eval_config.py --model configs/models/dynamic_moe/dynamic_moe.py
   ```

4. **保存和分析结果**
   ```python
   # 在评测脚本末尾
   model.save_expert_usage_stats('./expert_stats.json')
   ```
   ```bash
   python example_expert_stats_usage.py expert_stats.json
   ```

---

## 技术支持

如有问题，请检查：
1. `modeling_moe_infer.py` 是否已正确应用补丁
2. 配置文件中 `enable_expert_stats=True` 是否设置
3. 模型加载日志中是否出现统计启用提示
