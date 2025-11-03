# GNN-LSTM 混合模型实现

## 概述

本目录包含了基于 **Graph Neural Network (GNN)** 和 **LSTM/GRU** 的混合深度学习模型实现，用于全球未测流域的极端洪水预测。该模型是对 Nearing et al. (2024) "Global prediction of extreme floods in ungauged watersheds" (Nature) 论文中原始模型的扩展。

## 模型架构

### HybridGNN_RNN 模型

混合模型结合了两种互补的神经网络架构：

1. **GNN 部分**（处理静态流域属性）
   - 使用 Graph Convolutional Networks (GCN) 处理流域静态特征
   - 在 "per-gauge" 方法中，每个流域被建模为单个节点
   - 提取流域特征嵌入（土壤类型、坡度、面积等）

2. **RNN 部分**（处理动态气象输入）
   - 使用 LSTM 或 GRU 处理时间序列气象数据
   - 捕获降水、温度等动态特征的时间依赖关系
   - 可配置的序列长度和隐藏层维度

3. **融合层**
   - 将 GNN 和 RNN 的嵌入连接起来
   - 通过全连接层进行特征融合
   - 输出多个前导时间的径流预测

## 主要组件

### 1. `HybridGNN_RNN` 类
混合神经网络模型的核心实现。

**关键参数：**
- `static_feature_dim`: 静态属性特征数（默认：50）
- `dynamic_feature_dim`: 动态气象特征数（默认：5）
- `gnn_hidden_dim`: GNN 隐藏层维度（默认：64）
- `rnn_hidden_dim`: RNN 隐藏层维度（默认：128）
- `rnn_num_layers`: RNN 层数（默认：2）
- `rnn_type`: 'lstm' 或 'gru'
- `output_lead_times`: 预测前导时间数量（默认：10）

### 2. `HydroDataset` 类
自定义 PyTorch Dataset，用于处理水文预测数据。

**功能：**
- 懒加载站点数据
- 时间序列窗口采样
- 自动处理 NaN 值
- 每个站点生成多个训练样本

### 3. `AdvancedModel` 类
模型训练和预测的高级接口。

**主要方法：**
- `__init__`: 初始化模型、优化器和损失函数
- `_prepare_data_for_gauge`: 加载单个站点的数据
- `train`: 训练模型
- `predict`: 生成预测并保存为 NetCDF 文件

## 使用方法

### 基本使用

```python
from models.my_advanced_model import AdvancedModel

# 1. 定义模型超参数
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

# 2. 初始化模型
model = AdvancedModel(
    model_params=model_params,
    experiment_name="gnn_lstm_hybrid_v1"
)

# 3. 训练模型
training_gauge_ids = ['GRDC_6335020', 'GRDC_6335075', ...]
model.train(training_gauge_ids)

# 4. 生成预测
prediction_gauge_ids = ['GRDC_6335020', ...]
model.predict(prediction_gauge_ids)
```

### 运行示例

```bash
cd /home/user/prediction-of-extreme-floods-in-ungauged-watersheds
python models/my_advanced_model.py
```

## 数据流程

### 训练流程

1. **数据准备**
   - 为每个站点加载 GRDC 观测数据
   - 加载流域静态属性（HydroATLAS）
   - 创建模拟或加载真实气象输入

2. **Dataset 构建**
   - 从完整时间序列中随机采样窗口
   - 创建 RNN 输入序列（例如 365 天）
   - 创建 GNN 图数据（单节点图）
   - 提取对应的目标值

3. **训练循环**
   - 批处理样本
   - 前向传播通过 GNN 和 RNN
   - 计算 Huber Loss（对异常值鲁棒）
   - 反向传播和优化
   - 定期保存检查点

### 预测流程

1. **加载最佳模型**
   - 从检查点恢复训练好的权重

2. **滑动窗口推理**
   - 为每个站点加载完整数据
   - 使用滑动窗口生成连续预测
   - 对齐到模板时间坐标

3. **格式化输出**
   - 创建与原始 Google 模型兼容的 xarray.Dataset
   - 保存为 NetCDF 文件（每个站点一个文件）

## 与评估框架的兼容性

### 输出格式

预测输出严格遵循原始评估框架的格式要求：

- **文件格式**: NetCDF (.nc)
- **文件命名**: `{gauge_id}.nc`
- **维度**: `[time, lead_time]`
- **坐标**: `time`, `lead_time`, `gauge_id`
- **变量名**: 与模板文件保持一致（通常是 'sim' 或 'google_prediction'）

### 评估步骤

生成预测后，可以使用以下 Notebooks 进行评估：

1. **标准水文指标**（NSE, KGE, Bias 等）
   ```
   notebooks/calculate_hydrograph_metrics.ipynb
   ```

2. **极端事件指标**（重现期的 Precision, Recall, F1）
   ```
   notebooks/calculate_return_period_metrics.ipynb
   ```

在评估脚本中设置：
```python
experiment = 'gnn_lstm_hybrid_v1'  # 您的实验名称
```

## 依赖项

### 必需的 Python 包

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
xarray>=2023.1.0
tqdm>=4.65.0
```

### 安装

```bash
pip install torch torch-geometric numpy pandas xarray tqdm
```

## 重要注意事项

### 1. 气象数据

⚠️ **当前实现使用模拟气象数据**

在实际应用中，您需要：
- 在 `_prepare_data_for_gauge` 方法中加载真实的气象驱动数据
- 确保气象数据与 GRDC 观测数据的时间索引对齐
- 常见气象特征：降水、温度、潜在蒸散发、土壤湿度、积雪等

### 2. 静态特征维度

确保 `static_feature_dim` 与 HydroATLAS 属性文件的列数匹配：

```python
# 检查实际特征数
attrs = loading_utils.load_attributes_file()
print(f"Number of static features: {attrs.shape[1]}")
```

### 3. 超参数调优

关键超参数建议：

- **seq_length**: 根据流域响应时间调整（通常 180-365 天）
- **rnn_hidden_dim**: 根据数据复杂度调整（64-256）
- **learning_rate**: 从 0.001 开始，根据训练曲线调整
- **batch_size**: 根据 GPU 内存调整（16-64）
- **samples_per_gauge**: 平衡训练时间和数据多样性（5-20）

### 4. 计算资源

- **GPU**: 强烈推荐（训练速度可提升 10-50 倍）
- **内存**: 至少 16GB RAM
- **存储**: 每个站点的 NetCDF 文件约 1-10 MB

### 5. 训练时间估计

假设：
- 100 个站点
- 每个站点 10 个样本
- 50 个 epochs
- 批大小 32

**预计时间：**
- GPU (NVIDIA RTX 3090): 2-4 小时
- CPU (Intel i9): 24-48 小时

## 文件结构

```
models/
├── my_advanced_model.py    # 主要实现文件
├── README.md               # 本文件
└── checkpoints/            # 训练检查点（训练后生成）
    ├── best_model.pth
    └── model_epoch_*.pth

output/
└── gnn_lstm_hybrid_v1/    # 预测输出（预测后生成）
    ├── GRDC_6335020.nc
    ├── GRDC_6335075.nc
    └── ...
```

## 扩展和改进

### 建议的改进方向

1. **真实气象数据集成**
   - 集成 ERA5 再分析数据
   - 添加卫星观测数据（如 MODIS）

2. **更复杂的 GNN 架构**
   - 使用多节点图（包含上下游关系）
   - 尝试 GAT (Graph Attention Networks)
   - 添加边特征（如河流长度、坡度）

3. **注意力机制**
   - 在 RNN 输出上添加注意力层
   - 跨站点注意力机制

4. **物理约束**
   - 添加水量平衡约束
   - 集成 PUB (Prediction in Ungauged Basins) 理论

5. **不确定性量化**
   - 实现贝叶斯神经网络
   - 集成学习（Ensemble）

## 引用

如果您使用本实现，请引用原始论文：

```
Nearing, G. S., et al. (2024).
Global prediction of extreme floods in ungauged watersheds.
Nature, [DOI].
```

## 许可证

本实现遵循与原始代码库相同的许可证。

## 联系

如有问题或建议，请通过 GitHub Issues 提交。

---

**最后更新**: 2025-11-03
**版本**: 1.0.0
