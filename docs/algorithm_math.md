# 算法原理与数学基础

## 目录
1. [概述](#概述)
2. [GNN (图神经网络) 数学原理](#gnn-图神经网络-数学原理)
3. [LSTM (长短期记忆网络) 数学原理](#lstm-长短期记忆网络-数学原理)
4. [混合架构融合机制](#混合架构融合机制)
5. [损失函数：Huber Loss](#损失函数huber-loss)
6. [优化算法](#优化算法)
7. [参考文献](#参考文献)

---

## 概述

本文档详细描述了 GNN-LSTM 混合模型的数学原理。该模型用于全球未测流域的极端洪水预测，是对 Nearing et al. (2024) 原始 LSTM 模型的扩展。

**核心思想**：
- **GNN** 处理静态流域属性（空间特征）
- **LSTM** 处理动态气象序列（时间特征）
- **融合层** 将两者结合，输出多步径流预测

---

## GNN (图神经网络) 数学原理

### 1. 图的定义

在我们的模型中，每个流域被建模为一个图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中：
- $\mathcal{V}$ 是节点集合（在 per-gauge 方法中，$|\mathcal{V}| = 1$，即每个流域只有一个节点）
- $\mathcal{E}$ 是边集合（在当前实现中，边集为空；未来版本可添加上下游关系）

### 2. 节点特征

每个节点 $v \in \mathcal{V}$ 有一个特征向量 $\mathbf{x}_v \in \mathbb{R}^{d_s}$，其中：
- $d_s = 50$（静态特征维度）
- $\mathbf{x}_v$ 包含流域的静态属性（见 [data_dictionary.md](data_dictionary.md)）

### 3. 特征归一化

在输入 GNN 之前，静态特征被标准化：

$$
\mathbf{x}_v^{\text{norm}} = \frac{\mathbf{x}_v - \boldsymbol{\mu}_s}{\boldsymbol{\sigma}_s + \epsilon}
$$

其中：
- $\boldsymbol{\mu}_s \in \mathbb{R}^{d_s}$ 是训练集静态特征的均值向量
- $\boldsymbol{\sigma}_s \in \mathbb{R}^{d_s}$ 是训练集静态特征的标准差向量
- $\epsilon = 10^{-8}$ 防止除零

**实现位置**：`my_advanced_model.py:306-328`

### 4. GCN 层

我们使用图卷积网络 (GCN, Kipf & Welling, 2017) 来提取流域嵌入。

对于单个 GCN 层：

$$
\mathbf{h}_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{d_u d_v}} \mathbf{W}^{(l)} \mathbf{h}_u^{(l)} \right)
$$

其中：
- $\mathbf{h}_v^{(l)} \in \mathbb{R}^{d_l}$ 是节点 $v$ 在第 $l$ 层的嵌入
- $\mathcal{N}(v)$ 是节点 $v$ 的邻居集合
- $d_u$, $d_v$ 是节点度数
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d_{l+1} \times d_l}$ 是可学习权重矩阵
- $\sigma(\cdot)$ 是激活函数（ReLU）

**初始条件**：$\mathbf{h}_v^{(0)} = \mathbf{x}_v^{\text{norm}}$

### 5. 我们的 GNN 架构

```python
GNN 模块（在 common.py 中定义）：
  输入: x ∈ ℝ^(N × d_s), edge_index

  层1: GCNConv(d_s → h_g)
       h^(1) = ReLU(GCN(x, edge_index))
       h^(1) = Dropout(h^(1), p=0.2)

  层2: GCNConv(h_g → h_g)
       h^(2) = ReLU(GCN(h^(1), edge_index))
       h^(2) = Dropout(h^(2), p=0.2)

  全局池化: z_g = mean(h^(2))  ∈ ℝ^(h_g)

  输出: z_g（流域嵌入向量）
```

其中：
- $d_s = 50$（静态特征维度）
- $h_g = 64$（GNN 隐藏维度，可配置）
- $N$ 是图中的节点数（当前实现中 $N=1$）

**实现位置**：`models/common.py:15-49`

### 6. Per-Gauge vs. Multi-Node 图

**当前实现**（Per-Gauge）：
- 每个流域是一个独立的单节点图
- 边集为空：$\mathcal{E} = \emptyset$
- GCN 退化为简单的 MLP：$\mathbf{h}^{(l+1)} = \sigma(\mathbf{W}^{(l)} \mathbf{h}^{(l)})$

**未来扩展**（Multi-Node）：
- 可以构建包含上下游关系的多节点图
- 边特征可以包括：河流长度、坡度、汇流时间等
- 参考：`models/GNN_ARCHITECTURE_IMPROVEMENTS.md`

---

## LSTM (长短期记忆网络) 数学原理

### 1. LSTM 单元

LSTM 通过三个门控机制来控制信息的流动：

#### 遗忘门 (Forget Gate)
$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)
$$

#### 输入门 (Input Gate)
$$
\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)
$$

#### 候选细胞状态 (Candidate Cell State)
$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)
$$

#### 细胞状态更新 (Cell State Update)
$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

#### 输出门 (Output Gate)
$$
\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)
$$

#### 隐藏状态 (Hidden State)
$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{d_d}$ 是时刻 $t$ 的输入（动态气象特征）
- $\mathbf{h}_t \in \mathbb{R}^{h_r}$ 是隐藏状态
- $\mathbf{c}_t \in \mathbb{R}^{h_r}$ 是细胞状态
- $\sigma(\cdot)$ 是 sigmoid 函数：$\sigma(z) = \frac{1}{1 + e^{-z}}$
- $\odot$ 是逐元素乘法（Hadamard product）
- $\mathbf{W}_*, \mathbf{U}_* \in \mathbb{R}^{h_r \times \cdot}$ 是权重矩阵
- $\mathbf{b}_* \in \mathbb{R}^{h_r}$ 是偏置向量

### 2. 序列处理

给定输入序列 $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\}$，其中 $T = 365$（可配置）：

1. **初始化**：$\mathbf{h}_0 = \mathbf{0}$, $\mathbf{c}_0 = \mathbf{0}$
2. **前向传播**：对于 $t = 1, 2, \ldots, T$，计算 $\mathbf{h}_t$ 和 $\mathbf{c}_t$
3. **输出**：最终隐藏状态 $\mathbf{h}_T \in \mathbb{R}^{h_r}$

### 3. 多层 LSTM

我们使用 $L = 2$ 层堆叠的 LSTM：

$$
\begin{aligned}
\mathbf{h}_t^{(1)} &= \text{LSTM}^{(1)}(\mathbf{x}_t, \mathbf{h}_{t-1}^{(1)}, \mathbf{c}_{t-1}^{(1)}) \\
\mathbf{h}_t^{(2)} &= \text{LSTM}^{(2)}(\mathbf{h}_t^{(1)}, \mathbf{h}_{t-1}^{(2)}, \mathbf{c}_{t-1}^{(2)})
\end{aligned}
$$

最终输出：$\mathbf{z}_r = \mathbf{h}_T^{(2)}$

### 4. 输入特征归一化

在输入 LSTM 之前，动态特征被标准化（**逐样本标准化**）：

$$
\mathbf{x}_t^{\text{norm}} = \frac{\mathbf{x}_t - \boldsymbol{\mu}_{\text{seq}}}{\boldsymbol{\sigma}_{\text{seq}} + \epsilon}
$$

其中：
- $\boldsymbol{\mu}_{\text{seq}} = \frac{1}{T} \sum_{t=1}^T \mathbf{x}_t$（序列均值）
- $\boldsymbol{\sigma}_{\text{seq}}^2 = \frac{1}{T} \sum_{t=1}^T (\mathbf{x}_t - \boldsymbol{\mu}_{\text{seq}})^2$（序列标准差）

**实现位置**：`my_advanced_model.py:105-108`

**重要**：这种逐样本标准化与 GNN 的全局标准化不同。未来版本应统一为全局标准化（参考 `hydro_data_loader.py:343-350`）。

---

## 混合架构融合机制

### 1. 特征拼接

GNN 和 LSTM 的输出被拼接：

$$
\mathbf{z} = [\mathbf{z}_g; \mathbf{z}_r] \in \mathbb{R}^{h_g + h_r}
$$

其中：
- $\mathbf{z}_g \in \mathbb{R}^{h_g}$ 是 GNN 输出（流域静态嵌入）
- $\mathbf{z}_r \in \mathbb{R}^{h_r}$ 是 LSTM 输出（气象序列嵌入）
- $[\cdot; \cdot]$ 表示向量拼接

**实现位置**：`models/common.py:87-88`

### 2. 全连接预测层

拼接后的特征经过两个全连接层进行预测：

$$
\begin{aligned}
\mathbf{z}^{(1)} &= \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) \\
\mathbf{z}^{(1)} &= \text{Dropout}(\mathbf{z}^{(1)}, p=0.2) \\
\hat{\mathbf{y}} &= \mathbf{W}_2 \mathbf{z}^{(1)} + \mathbf{b}_2
\end{aligned}
$$

其中：
- $\mathbf{W}_1 \in \mathbb{R}^{(h_g + h_r) \times (h_g + h_r)}$
- $\mathbf{W}_2 \in \mathbb{R}^{L_{\text{out}} \times (h_g + h_r)}$
- $\hat{\mathbf{y}} \in \mathbb{R}^{L_{\text{out}}}$ 是预测的径流序列
- $L_{\text{out}} = 10$（预测前导时间数量）

**实现位置**：`models/common.py:89-91`

### 3. 完整的前向传播

```
输入:
  - 动态序列: X_dyn ∈ ℝ^(B × T × d_d)
  - 静态图: G = (V, E, X_static)

GNN 分支:
  z_g = GNN(X_static, E)  ∈ ℝ^(B × h_g)

LSTM 分支:
  z_r = LSTM(X_dyn)  ∈ ℝ^(B × h_r)

融合:
  z = [z_g; z_r]  ∈ ℝ^(B × (h_g + h_r))

预测:
  ŷ = FC2(ReLU(FC1(z)))  ∈ ℝ^(B × L_out)

输出: ŷ（未来 L_out 天的径流预测）
```

其中 $B$ 是批大小。

---

## 损失函数：Huber Loss

### 1. 定义

Huber Loss 是均方误差 (MSE) 和平均绝对误差 (MAE) 的组合：

$$
L_{\delta}(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

其中：
- $y$ 是真实观测值
- $\hat{y}$ 是模型预测值
- $\delta = 1.0$ 是阈值参数（可配置）

### 2. 为什么使用 Huber Loss？

| 损失函数 | 优点 | 缺点 |
|---------|------|------|
| **MSE** | 梯度连续，优化平滑 | 对异常值（洪峰）过于敏感，容易被极端值主导 |
| **MAE** | 对异常值鲁棒 | 在零点不可导，优化不稳定 |
| **Huber** | 结合两者优点：小误差时像 MSE（平滑），大误差时像 MAE（鲁棒） | 需要调整 $\delta$ 参数 |

### 3. 在洪水预测中的重要性

洪水预测的核心挑战是**极端事件**（洪峰）：
- 洪峰流量可能是平均流量的 10-100 倍
- MSE 会过度关注这些极端值，导致平时预测不准
- Huber Loss 在保持对洪峰敏感的同时，不会完全忽略常态流量

### 4. 批量损失计算

对于一个批次：

$$
\mathcal{L} = \frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \sum_{t=1}^{L_{\text{out}}} L_{\delta}(y_{i,t}, \hat{y}_{i,t})
$$

其中：
- $N_{\text{valid}}$ 是批次中有效样本数（排除 NaN）
- $y_{i,t}$ 是样本 $i$ 在前导时间 $t$ 的真实流量
- $\hat{y}_{i,t}$ 是对应的预测值

**实现位置**：`my_advanced_model.py:574`

**重要**：我们在计算损失前过滤掉 NaN 值（`my_advanced_model.py:570-572`）。

---

## 优化算法

### 1. Adam 优化器

我们使用 Adam (Adaptive Moment Estimation) 优化器：

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla_{\theta} \mathcal{L}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla_{\theta} \mathcal{L}_t)^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}
$$

其中：
- $\theta$ 是模型参数
- $\alpha = 0.001$ 是学习率（learning rate）
- $\beta_1 = 0.9$ 是一阶矩估计的指数衰减率
- $\beta_2 = 0.999$ 是二阶矩估计的指数衰减率
- $\epsilon = 10^{-8}$ 防止除零
- $\mathbf{m}_t$ 是梯度的一阶矩（均值）
- $\mathbf{v}_t$ 是梯度的二阶矩（未中心化方差）

**实现位置**：`my_advanced_model.py:214`

### 2. 梯度裁剪 (Gradient Clipping)

为了防止梯度爆炸，我们使用梯度范数裁剪：

$$
\nabla_{\theta} \mathcal{L} \leftarrow
\begin{cases}
\nabla_{\theta} \mathcal{L} & \text{if } \|\nabla_{\theta} \mathcal{L}\|_2 \leq \tau \\
\tau \cdot \frac{\nabla_{\theta} \mathcal{L}}{\|\nabla_{\theta} \mathcal{L}\|_2} & \text{otherwise}
\end{cases}
$$

其中 $\tau = 1.0$ 是裁剪阈值。

**实现位置**：`my_advanced_model.py:580`

### 3. 训练流程

```
for epoch in 1..num_epochs:
    for batch in DataLoader:
        # 1. 前向传播
        ŷ = model(X_dyn, G_static)

        # 2. 计算损失（忽略 NaN）
        valid_mask = ~isnan(y)
        loss = HuberLoss(ŷ[valid_mask], y[valid_mask])

        # 3. 反向传播
        loss.backward()

        # 4. 梯度裁剪
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. 参数更新
        optimizer.step()
        optimizer.zero_grad()
```

---

## 参考文献

1. **Kipf, T. N., & Welling, M. (2017).**
   *Semi-Supervised Classification with Graph Convolutional Networks.*
   ICLR 2017.
   [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)

2. **Hochreiter, S., & Schmidhuber, J. (1997).**
   *Long Short-Term Memory.*
   Neural Computation, 9(8), 1735-1780.
   [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

3. **Huber, P. J. (1964).**
   *Robust Estimation of a Location Parameter.*
   Annals of Mathematical Statistics, 35(1), 73-101.

4. **Kingma, D. P., & Ba, J. (2015).**
   *Adam: A Method for Stochastic Optimization.*
   ICLR 2015.
   [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

5. **Nearing, G. S., et al. (2024).**
   *Global prediction of extreme floods in ungauged watersheds.*
   Nature.
   [DOI: TBD]

6. **Colah's Blog: Understanding LSTMs**
   [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

7. **Distill: A Gentle Introduction to Graph Neural Networks**
   [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)

---

## 附录：符号表

| 符号 | 描述 | 维度 |
|------|------|------|
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | 图结构 | - |
| $\mathbf{x}_v$ | 节点 $v$ 的静态特征 | $\mathbb{R}^{d_s}$ |
| $d_s$ | 静态特征维度 | 50 |
| $d_d$ | 动态特征维度 | 5 |
| $h_g$ | GNN 隐藏维度 | 64 |
| $h_r$ | RNN 隐藏维度 | 128 |
| $T$ | 输入序列长度 | 365 |
| $L_{\text{out}}$ | 输出前导时间数 | 10 |
| $\mathbf{z}_g$ | GNN 输出（流域嵌入） | $\mathbb{R}^{h_g}$ |
| $\mathbf{z}_r$ | LSTM 输出（气象嵌入） | $\mathbb{R}^{h_r}$ |
| $\hat{\mathbf{y}}$ | 模型预测 | $\mathbb{R}^{L_{\text{out}}}$ |
| $\mathbf{y}$ | 真实观测 | $\mathbb{R}^{L_{\text{out}}}$ |
| $\delta$ | Huber Loss 阈值 | 1.0 |
| $\alpha$ | 学习率 | 0.001 |
| $\tau$ | 梯度裁剪阈值 | 1.0 |

---

**最后更新**: 2025-12-04
**版本**: 1.0.0
