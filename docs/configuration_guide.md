# 配置指南

## 目录
1. [概述](#概述)
2. [模型参数详解](#模型参数详解)
3. [超参数调优建议](#超参数调优建议)
4. [常见配置场景](#常见配置场景)
5. [性能与资源权衡](#性能与资源权衡)
6. [故障排除](#故障排除)

---

## 概述

本文档详细解释 `AdvancedModel` 的所有配置参数，并提供调优建议。

### 配置文件示例

```python
model_params = {
    # ===== 数据维度 =====
    'static_feature_dim': 50,      # 静态流域属性维度
    'dynamic_feature_dim': 5,      # 动态气象特征维度

    # ===== 网络架构 =====
    'gnn_hidden_dim': 64,          # GNN 隐藏层维度
    'rnn_hidden_dim': 128,         # RNN 隐藏层维度
    'rnn_num_layers': 2,           # RNN 层数
    'rnn_type': 'lstm',            # 'lstm' 或 'gru'
    'output_lead_times': 10,       # 预测前导时间数量
    'dropout': 0.2,                # Dropout 率

    # ===== 训练参数 =====
    'learning_rate': 0.001,        # 学习率
    'batch_size': 32,              # 批大小
    'num_epochs': 50,              # 训练轮数

    # ===== 数据参数 =====
    'seq_length': 365,             # RNN 输入序列长度（天）
    'samples_per_gauge': 10,       # 每个站点的训练样本数

    # ===== 数据路径 =====
    'meteorology_file': None,      # 气象数据文件路径（可选）
}
```

---

## 模型参数详解

### 1. 数据维度参数

#### `static_feature_dim` (int)

- **描述**：静态流域属性的特征数量
- **默认值**：50
- **物理意义**：决定 GNN 输入层的维度
- **如何设置**：
  ```python
  # 检查实际特征数
  from backend import loading_utils
  attrs = loading_utils.load_attributes_file()
  print(f"实际特征数: {attrs.shape[1]}")

  # 设置为实际特征数（或略大以容纳填充）
  model_params['static_feature_dim'] = attrs.shape[1]
  ```

**注意事项**：
- 如果设置得太小，后面的特征会被截断
- 如果设置得太大，会用 0 填充
- 建议与数据集保持一致

---

#### `dynamic_feature_dim` (int)

- **描述**：动态气象特征的数量
- **默认值**：5
- **物理意义**：决定 LSTM 输入层的维度
- **常见特征**：
  1. `precip`（降水）
  2. `temp`（温度）
  3. `pet`（潜在蒸散发）
  4. `soil_moisture`（土壤湿度）
  5. `snow`（积雪）

**扩展建议**：
- 添加 `radiation`（辐射）提高融雪预测
- 添加 `wind_speed`（风速）提高蒸散发估算
- 添加 `humidity`（湿度）改进蒸散发计算

---

### 2. 网络架构参数

#### `gnn_hidden_dim` (int)

- **描述**：GNN 隐藏层和输出层的神经元数量
- **默认值**：64
- **物理意义**：控制流域嵌入向量的维度
- **影响**：
  - **太小**（< 32）：无法充分表示复杂的流域特性
  - **太大**（> 256）：过拟合风险增加，训练时间延长
  - **推荐范围**：32 - 128

**调优指南**：
```python
# 小规模实验（< 100 个站点）
model_params['gnn_hidden_dim'] = 32

# 中等规模（100 - 1000 个站点）
model_params['gnn_hidden_dim'] = 64  # 推荐

# 大规模（> 1000 个站点）
model_params['gnn_hidden_dim'] = 128
```

---

#### `rnn_hidden_dim` (int)

- **描述**：LSTM/GRU 隐藏层的神经元数量
- **默认值**：128
- **物理意义**：控制时间序列嵌入向量的维度
- **影响**：
  - **太小**（< 64）：无法记住长期依赖关系
  - **太大**（> 256）：训练时间显著增加，可能过拟合
  - **推荐范围**：64 - 256

**与序列长度的关系**：
- `seq_length = 365`（一年）→ 建议 `rnn_hidden_dim = 128`
- `seq_length = 180`（半年）→ 可用 `rnn_hidden_dim = 64`
- `seq_length = 730`（两年）→ 建议 `rnn_hidden_dim = 256`

---

#### `rnn_num_layers` (int)

- **描述**：堆叠的 RNN 层数
- **默认值**：2
- **物理意义**：更深的网络可以学习更抽象的时间模式
- **影响**：
  - **1 层**：简单模式，训练快，容易欠拟合
  - **2 层**：平衡性能和复杂度（推荐）
  - **3+ 层**：复杂模式，但容易梯度消失/爆炸

**梯度裁剪的重要性**：
```python
# 对于深层 RNN（> 2 层），必须使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

#### `rnn_type` (str)

- **描述**：循环神经网络类型
- **可选值**：`'lstm'` 或 `'gru'`
- **默认值**：`'lstm'`

**LSTM vs. GRU 比较**：

| 特性 | LSTM | GRU |
|------|------|-----|
| **门控机制** | 3 个门（输入、遗忘、输出） | 2 个门（重置、更新） |
| **参数数量** | 更多（4 × hidden_dim²） | 更少（3 × hidden_dim²） |
| **训练速度** | 较慢 | 较快（约 20-30% 快） |
| **长期记忆** | 更强（有独立的细胞状态） | 稍弱 |
| **适用场景** | 长序列（> 365 天），复杂模式 | 中等序列，实时应用 |

**推荐**：
- 默认使用 `'lstm'`（更强的长期记忆能力）
- 如果训练时间是瓶颈，尝试 `'gru'`

---

#### `output_lead_times` (int)

- **描述**：预测未来多少天的流量
- **默认值**：10
- **物理意义**：洪水预报的前瞻期
- **影响**：
  - **短期预报**（1-3 天）：精度高，适合应急响应
  - **中期预报**（7-10 天）：平衡精度和前瞻期（推荐）
  - **长期预报**（> 14 天）：精度低，不确定性大

**与评估指标的对应**：
```python
# 在 Nearing et al. (2024) 中：
# - 重现期评估使用 lead_time = 0, 1, 2, ..., 9
# - 即预测未来 10 天的极端洪水概率
model_params['output_lead_times'] = 10  # 与论文保持一致
```

---

#### `dropout` (float)

- **描述**：Dropout 正则化比率
- **默认值**：0.2
- **物理意义**：训练时随机丢弃 20% 的神经元，防止过拟合
- **推荐范围**：0.1 - 0.5

**设置指南**：
- **训练集很大**（> 1000 个站点）：`dropout = 0.1`
- **训练集中等**（100 - 1000 个站点）：`dropout = 0.2`（推荐）
- **训练集很小**（< 100 个站点）：`dropout = 0.3 - 0.5`

**应用位置**（在 `common.py` 中）：
```python
# GNN 层后
h = self.dropout(h)

# LSTM 层间（PyTorch 自动应用）
self.rnn = nn.LSTM(..., dropout=dropout)

# 融合层后
combined = self.dropout(combined)
```

---

### 3. 训练参数

#### `learning_rate` (float)

- **描述**：Adam 优化器的初始学习率
- **默认值**：0.001
- **物理意义**：每次梯度下降时的步长

**学习率调度策略**（未来改进）：
```python
# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# 指数衰减
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)

# ReduceLROnPlateau（基于验证损失）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

**调优指南**：
- **Loss 震荡不收敛** → 降低学习率（0.0005 或 0.0001）
- **Loss 下降太慢** → 提高学习率（0.002 或 0.005）
- **Loss 开始下降后趋于平缓** → 使用学习率衰减

---

#### `batch_size` (int)

- **描述**：每个训练批次的样本数量
- **默认值**：32
- **物理意义**：影响梯度估计的稳定性和内存占用

**批大小的权衡**：

| Batch Size | 优点 | 缺点 |
|-----------|------|------|
| **小**（8-16） | 内存占用少，梯度噪声大（更好的泛化） | 训练慢，梯度不稳定 |
| **中等**（32-64） | 平衡性能和稳定性（推荐） | 需要中等 GPU 内存 |
| **大**（128-256） | 训练快，梯度稳定 | 内存占用大，可能过拟合 |

**根据硬件调整**：
```python
# CPU
model_params['batch_size'] = 8

# GPU (8GB VRAM)
model_params['batch_size'] = 32  # 推荐

# GPU (16GB+ VRAM)
model_params['batch_size'] = 64

# 多 GPU
model_params['batch_size'] = 128  # 每 GPU 32-64
```

---

#### `num_epochs` (int)

- **描述**：训练的总轮数
- **默认值**：50
- **物理意义**：模型看到整个训练集的次数

**早停法**（未来改进）：
```python
best_val_loss = float('inf')
patience = 10  # 如果 10 个 epoch 验证损失没有改进，则停止
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(...)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("早停：验证损失不再改进")
        break
```

---

### 4. 数据参数

#### `seq_length` (int)

- **描述**：LSTM 输入序列的长度（天数）
- **默认值**：365（一年）
- **物理意义**：模型的"记忆窗口"

**设置指南**：

| 流域类型 | 响应时间 | 推荐 seq_length |
|---------|---------|----------------|
| **山区小流域** | 几小时 - 1 天 | 90 天 |
| **丘陵中等流域** | 3-7 天 | 180 天 |
| **平原大流域** | 1-2 周 | 365 天（推荐） |
| **冰川/融雪流域** | 数月 | 730 天（两年） |

**计算成本**：
- `seq_length` 加倍 → 训练时间增加约 30-50%
- 对于长序列（> 500 天），考虑使用 GRU 代替 LSTM

---

#### `samples_per_gauge` (int)

- **描述**：从每个站点的时间序列中采样的训练样本数
- **默认值**：10
- **物理意义**：控制数据增强的程度

**采样策略**（在 `my_advanced_model.py:119-132` 实现）：
- 当前实现：**穷举所有可能的窗口**
  ```python
  windows = [(start, start + seq_length) for start in range(max_start + 1)]
  ```
- 未来改进：**随机采样** $N$ 个窗口，减少训练时间

**设置建议**：
- **快速实验**：`samples_per_gauge = 5`
- **标准训练**：`samples_per_gauge = 10`（推荐）
- **充分训练**：`samples_per_gauge = 20`（或使用全部窗口）

---

### 5. 数据路径参数

#### `meteorology_file` (Path or None)

- **描述**：气象数据 NetCDF 文件的路径
- **默认值**：`None`（使用模拟数据）
- **推荐格式**：NetCDF (.nc)

**数据要求**：
```python
# 必须包含以下维度和坐标：
# - 维度: (time, gauge_id)
# - 坐标: time (datetime64), gauge_id (str)
# - 变量: precip, temp, pet, soil_moisture, snow

import xarray as xr
met_data = xr.open_dataset(meteorology_file)
print(met_data)

# 预期输出：
# <xarray.Dataset>
# Dimensions:        (time: 14610, gauge_id: 5420)
# Coordinates:
#   * time           (time) datetime64[ns] ...
#   * gauge_id       (gauge_id) object ...
# Data variables:
#     precip         (time, gauge_id) float32 ...
#     temp           (time, gauge_id) float32 ...
#     pet            (time, gauge_id) float32 ...
#     soil_moisture  (time, gauge_id) float32 ...
#     snow           (time, gauge_id) float32 ...
```

---

## 超参数调优建议

### 网格搜索

```python
from itertools import product

# 定义超参数搜索空间
param_grid = {
    'gnn_hidden_dim': [32, 64, 128],
    'rnn_hidden_dim': [64, 128, 256],
    'rnn_num_layers': [1, 2],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0005, 0.001, 0.002],
}

# 网格搜索
best_val_loss = float('inf')
best_params = None

for params in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    model = AdvancedModel(model_params=config)
    val_loss = train_and_evaluate(model, ...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = config

print(f"最佳参数: {best_params}")
```

### 贝叶斯优化（推荐）

使用 [Optuna](https://optuna.org/) 进行更高效的超参数搜索：

```python
import optuna

def objective(trial):
    # 定义搜索空间
    model_params = {
        'gnn_hidden_dim': trial.suggest_categorical('gnn_hidden_dim', [32, 64, 128]),
        'rnn_hidden_dim': trial.suggest_categorical('rnn_hidden_dim', [64, 128, 256]),
        'rnn_num_layers': trial.suggest_int('rnn_num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }

    # 训练模型
    model = AdvancedModel(model_params=model_params)
    val_loss = train_and_evaluate(model, ...)

    return val_loss

# 创建研究并优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"最佳参数: {study.best_params}")
print(f"最佳验证损失: {study.best_value}")
```

---

## 常见配置场景

### 场景 1: 快速原型验证（笔记本电脑）

```python
model_params = {
    'static_feature_dim': 50,
    'dynamic_feature_dim': 5,
    'gnn_hidden_dim': 32,          # 减小网络规模
    'rnn_hidden_dim': 64,          # 减小网络规模
    'rnn_num_layers': 1,           # 减少层数
    'rnn_type': 'gru',             # GRU 更快
    'output_lead_times': 10,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 8,               # 小批量
    'num_epochs': 10,              # 少轮数
    'seq_length': 90,              # 短序列
    'samples_per_gauge': 5,        # 少样本
}

# 只使用少量站点
training_gauge_ids = all_gauge_ids[:10]
```

**预期训练时间**：10-20 分钟（CPU）

---

### 场景 2: 标准训练（单 GPU）

```python
model_params = {
    'static_feature_dim': 50,
    'dynamic_feature_dim': 5,
    'gnn_hidden_dim': 64,
    'rnn_hidden_dim': 128,
    'rnn_num_layers': 2,
    'rnn_type': 'lstm',
    'output_lead_times': 10,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'seq_length': 365,
    'samples_per_gauge': 10,
}

# 使用全部训练集
training_gauge_ids = train_set  # 约 3000 个站点
```

**预期训练时间**：4-8 小时（NVIDIA RTX 3090）

---

### 场景 3: 高精度模型（多 GPU）

```python
model_params = {
    'static_feature_dim': 50,
    'dynamic_feature_dim': 5,
    'gnn_hidden_dim': 128,         # 增大网络
    'rnn_hidden_dim': 256,         # 增大网络
    'rnn_num_layers': 3,           # 更深
    'rnn_type': 'lstm',
    'output_lead_times': 10,
    'dropout': 0.3,                # 更强的正则化
    'learning_rate': 0.0005,       # 降低学习率
    'batch_size': 128,             # 大批量（分布式）
    'num_epochs': 100,             # 更多轮数
    'seq_length': 365,
    'samples_per_gauge': 20,       # 更多样本
}

# 使用 PyTorch DDP（分布式数据并行）
# 需要额外代码配置多 GPU 训练
```

**预期训练时间**：8-16 小时（4× NVIDIA A100）

---

### 场景 4: 实时预报（推理优化）

```python
# 训练时使用标准配置，但优化推理速度

# 1. 模型量化（减少模型大小）
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model.model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# 2. ONNX 导出（跨平台推理）
torch.onnx.export(
    model.model,
    (dummy_rnn_input, dummy_gnn_input),
    "model.onnx",
    opset_version=14
)

# 3. TorchScript（JIT 编译）
scripted_model = torch.jit.script(model.model)
scripted_model.save("model_scripted.pt")
```

---

## 性能与资源权衡

### 内存占用估算

**公式**（近似）：
```
GPU 内存 (GB) ≈ 4 × (
    batch_size × seq_length × dynamic_feature_dim × 4 bytes +  # 输入数据
    batch_size × rnn_hidden_dim × rnn_num_layers × 4 × 4 bytes +  # LSTM 状态
    总参数数量 × 4 bytes  # 模型权重
)
```

**参数数量估算**：
```python
def estimate_params(gnn_hidden, rnn_hidden, rnn_layers, static_dim, dynamic_dim, output_dim):
    # GNN 参数
    gnn_params = static_dim * gnn_hidden + gnn_hidden * gnn_hidden

    # LSTM 参数（4 个门）
    lstm_params = 4 * (dynamic_dim * rnn_hidden + rnn_hidden * rnn_hidden) * rnn_layers

    # FC 参数
    fc_params = (gnn_hidden + rnn_hidden) ** 2 + (gnn_hidden + rnn_hidden) * output_dim

    total = gnn_params + lstm_params + fc_params
    return total

# 示例
params = estimate_params(64, 128, 2, 50, 5, 10)
print(f"估计参数数量: {params:,} ({params * 4 / 1e6:.1f} MB)")
```

### 训练时间估算

**公式**（经验值）：
```
训练时间 (小时) ≈ (
    num_epochs × num_samples × seq_length
) / (GPU 算力 × 并行效率)
```

**示例**：
- GPU: NVIDIA RTX 3090 (约 35 TFLOPS)
- 配置: 50 epochs, 30,000 样本, seq_length=365
- 预期: 4-6 小时

---

## 故障排除

### 问题 1: Loss 为 NaN

**原因**：
- 学习率过大导致梯度爆炸
- 数据中存在未处理的 NaN 或 Inf

**解决方案**：
```python
# 1. 降低学习率
model_params['learning_rate'] = 0.0001

# 2. 检查数据
print(f"训练数据中的 NaN 数量: {torch.isnan(train_data).sum()}")

# 3. 使用梯度裁剪（已在代码中实现）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. 检查 Huber Loss 的 delta 参数
criterion = nn.HuberLoss(delta=0.5)  # 减小 delta
```

---

### 问题 2: 过拟合（训练损失低，验证损失高）

**解决方案**：
```python
# 1. 增加 Dropout
model_params['dropout'] = 0.3

# 2. 减少模型复杂度
model_params['gnn_hidden_dim'] = 32
model_params['rnn_hidden_dim'] = 64

# 3. 增加数据（更多站点）
training_gauge_ids = all_gauge_ids[:1000]  # 增加到 1000 个

# 4. 早停法（在验证损失开始上升时停止）
# （参考前面的早停法代码）
```

---

### 问题 3: 欠拟合（训练损失和验证损失都很高）

**解决方案**：
```python
# 1. 增加模型复杂度
model_params['gnn_hidden_dim'] = 128
model_params['rnn_hidden_dim'] = 256
model_params['rnn_num_layers'] = 3

# 2. 增加训练轮数
model_params['num_epochs'] = 100

# 3. 调整学习率
model_params['learning_rate'] = 0.002

# 4. 检查数据质量
# - 是否有足够的动态特征？
# - 静态特征是否有效？
```

---

### 问题 4: 内存不足 (OOM)

**解决方案**：
```python
# 1. 减少批大小
model_params['batch_size'] = 16

# 2. 减少序列长度
model_params['seq_length'] = 180

# 3. 减少网络规模
model_params['rnn_hidden_dim'] = 64

# 4. 使用梯度累积（模拟大批量）
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 5. 使用混合精度训练（AMP）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    predictions = model(inputs)
    loss = criterion(predictions, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### 问题 5: 训练速度太慢

**解决方案**：
```python
# 1. 使用 GRU 代替 LSTM
model_params['rnn_type'] = 'gru'

# 2. 增加 DataLoader 的 num_workers
train_loader = DataLoader(..., num_workers=4)

# 3. 使用固定内存（pinned memory）
train_loader = DataLoader(..., pin_memory=True)

# 4. 检查是否使用了 GPU
print(f"使用设备: {model.device}")
assert model.device.type == 'cuda', "请使用 GPU！"

# 5. 减少日志频率
# 不要每个 batch 都打印，改为每 100 个 batch 打印一次
if batch_idx % 100 == 0:
    print(f"Batch {batch_idx}, Loss: {loss.item()}")
```

---

## 参考资源

1. **PyTorch 官方文档**
   - [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
   - [优化器](https://pytorch.org/docs/stable/optim.html)

2. **PyTorch Geometric 文档**
   - [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)

3. **超参数调优**
   - [Optuna](https://optuna.org/)
   - [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

4. **性能优化**
   - [PyTorch 性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**最后更新**: 2025-12-04
**版本**: 1.0.0
