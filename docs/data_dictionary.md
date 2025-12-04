# 数据字典

## 目录
1. [概述](#概述)
2. [静态特征 (Static Attributes)](#静态特征-static-attributes)
3. [动态特征 (Dynamic Features)](#动态特征-dynamic-features)
4. [目标变量 (Target Variable)](#目标变量-target-variable)
5. [数据来源与引用](#数据来源与引用)
6. [数据预处理](#数据预处理)

---

## 概述

本文档详细描述了 GNN-LSTM 混合模型使用的所有数据字段。数据分为三类：
1. **静态特征**：流域的固定属性（地形、土壤、植被等）
2. **动态特征**：随时间变化的气象输入（降雨、温度等）
3. **目标变量**：需要预测的河流流量

---

## 静态特征 (Static Attributes)

### 数据来源

静态特征来自 **HydroATLAS** 数据库（Linke et al., 2019）。
- 数据集：全球流域属性数据集
- 空间分辨率：基于河网单元（subcatchments）
- 覆盖范围：全球所有主要流域

### 特征维度

- **总特征数**：50（`static_feature_dim = 50`）
- **数据格式**：每个站点一个特征向量 $\mathbf{x}_s \in \mathbb{R}^{50}$
- **加载函数**：`loading_utils.load_attributes_file()`

### 主要特征类别

HydroATLAS 包含数百个属性，我们选择了 50 个最相关的特征。以下是按类别分组的关键特征：

#### 1. 地形特征 (Topography)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `slp_mean` | 平均坡度 | % | 坡度越大，径流速度越快，洪峰更尖锐 |
| `slp_max` | 最大坡度 | % | 极端坡度影响快速汇流 |
| `ele_mean` | 平均海拔 | m | 高海拔地区降雪多，融雪洪水 |
| `ele_min` | 最低海拔 | m | 流域出口位置 |
| `ele_max` | 最高海拔 | m | 影响温度梯度和降水类型 |
| `ele_range` | 海拔范围 | m | 反映流域起伏程度 |

**物理意义**：
- 陡峭流域（高 `slp_mean`）：快速汇流，短滞后时间，洪峰高且尖锐
- 平坦流域（低 `slp_mean`）：慢速汇流，长滞后时间，洪峰低且宽阔

#### 2. 流域形态 (Basin Geometry)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `area_sqkm` | 流域面积 | km² | 面积越大，汇集降雨越多，但响应时间越长 |
| `length_km` | 主河道长度 | km | 长河道意味着更长的汇流时间 |
| `width_km` | 流域宽度 | km | 影响流域形状因子 |
| `shape_factor` | 形状因子 | - | 圆形流域洪峰更集中，长条形流域洪峰更分散 |
| `stream_density` | 河网密度 | km/km² | 高河网密度导致快速汇流 |

**形状因子**：
$$
\text{shape\_factor} = \frac{\text{area}}{\text{length}^2}
$$
- 接近 1：圆形流域，所有降雨几乎同时到达出口，洪峰高
- 远小于 1：长条形流域，降雨分批到达，洪峰相对分散

#### 3. 土壤特征 (Soil Properties)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `soil_clay_pc` | 黏土含量 | % | 黏土保水能力强，渗透慢，产流快 |
| `soil_sand_pc` | 沙土含量 | % | 沙土渗透快，产流慢 |
| `soil_silt_pc` | 粉土含量 | % | 介于黏土和沙土之间 |
| `soil_org_pc` | 有机质含量 | % | 有机质增加持水能力 |
| `soil_depth_cm` | 土壤厚度 | cm | 厚土壤可以储存更多水分 |
| `soil_permeability` | 土壤渗透率 | mm/h | 直接影响下渗和地表径流 |

**土壤质地对径流的影响**：
- **高黏土**（`soil_clay_pc > 40%`）：
  - 饱和后几乎不再吸水
  - 产生大量地表径流
  - 洪水风险高
- **高沙土**（`soil_sand_pc > 60%`）：
  - 快速下渗，补给地下水
  - 地表径流少
  - 洪水风险低（但可能有基流补给）

#### 4. 土地覆盖 / 植被特征 (Land Cover / Vegetation)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `for_pc_sse` | 森林覆盖率 | % | 森林减缓径流，降低洪峰 |
| `crop_pc_sse` | 农田覆盖率 | % | 农田产流快于森林，慢于城市 |
| `urban_pc_sse` | 城市覆盖率 | % | 不透水面产流最快，洪峰最高 |
| `grass_pc_sse` | 草地覆盖率 | % | 中等产流速度 |
| `wetland_pc_sse` | 湿地覆盖率 | % | 湿地蓄洪，削减洪峰 |
| `LAI_mean` | 叶面积指数 | m²/m² | 反映植被密度，影响截留和蒸散发 |
| `NDVI_mean` | 归一化植被指数 | - | 植被健康程度，影响产流 |

**城市化的影响**：
- 不透水面（混凝土、沥青）占比增加
- 下渗几乎为零
- 产流系数从 0.1-0.3（自然流域）提高到 0.7-0.9（城市流域）
- 洪峰显著增加，响应时间缩短

#### 5. 气候特征 (Climate)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `pre_mean_year` | 年平均降水量 | mm/year | 湿润区和干旱区的径流机制不同 |
| `temp_mean_year` | 年平均温度 | °C | 影响蒸散发和降水形态（雨/雪） |
| `pet_mean_year` | 年平均潜在蒸散发 | mm/year | 蒸散发是水量平衡的重要组成部分 |
| `arid_index` | 干旱指数 | - | PET/降水，反映气候干旱程度 |
| `snow_days_year` | 年均积雪日数 | days | 积雪融化形成春季洪水 |

**干旱指数**：
$$
\text{arid\_index} = \frac{\text{PET}}{\text{Precipitation}}
$$
- AI < 0.05：极端湿润
- 0.05 < AI < 0.2：湿润
- 0.2 < AI < 0.5：半湿润
- 0.5 < AI < 0.65：半干旱
- AI > 0.65：干旱

#### 6. 水文地质特征 (Hydrogeology)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `karst_pc` | 喀斯特地貌占比 | % | 喀斯特区域地下径流发达，地表径流少 |
| `aquifer_type` | 含水层类型 | 类别 | 影响基流补给 |
| `groundwater_depth` | 地下水埋深 | m | 浅层地下水容易饱和，产生径流 |

#### 7. 河网特征 (River Network)

| 特征名 | 描述 | 单位 | 对洪水的影响 |
|--------|------|------|-------------|
| `stream_order` | 河流级别（Strahler） | - | 高级别河流汇水范围大 |
| `sinuosity` | 河道弯曲度 | - | 弯曲河道减缓流速 |
| `channel_slope` | 河道坡度 | % | 陡峭河道流速快 |
| `dam_count` | 流域内水坝数量 | count | 水坝调蓄洪水 |
| `reservoir_capacity` | 水库总库容 | m³ | 库容越大，削峰能力越强 |

### 缺失值处理

**策略**（在 `my_advanced_model.py:306-328` 实现）：
1. **填充为全局均值**：
   ```python
   gauge_attrs = gauge_attrs.fillna(static_attr_mean)
   ```
2. **如果仍有缺失，填充为 0**：
   ```python
   gauge_attrs = gauge_attrs.fillna(0.0)
   ```
3. **标准化**（Z-score）：
   ```python
   static_array = (gauge_attrs.values - mean_values) / (std_values + 1e-8)
   ```

### 特征重要性分析（未来工作）

可以使用以下方法评估特征重要性：
1. **SHAP (SHapley Additive exPlanations)**：解释每个特征对预测的贡献
2. **Ablation Study**：逐个移除特征，观察性能下降
3. **Attention Weights**：如果使用注意力机制，权重反映特征重要性

---

## 动态特征 (Dynamic Features)

### 数据来源

动态特征来自气象再分析数据集或模拟数据：
- **推荐数据集**：ERA5（ECMWF 第五代再分析）
- **时间分辨率**：日尺度
- **空间分辨率**：0.25° × 0.25°（约 25 km）

### 特征维度

- **总特征数**：5（`dynamic_feature_dim = 5`）
- **数据格式**：每个站点每天一个向量 $\mathbf{x}_d \in \mathbb{R}^{5}$
- **序列长度**：365 天（`seq_length = 365`）

### 特征详细描述

#### 1. `precip` - 降水量

| 属性 | 值 |
|------|-----|
| **描述** | 日累计降水量 |
| **单位** | mm/day |
| **典型范围** | 0 - 200 mm/day（极端暴雨可达 500+ mm/day） |
| **物理意义** | 河流流量的主要驱动因子 |
| **数据源** | ERA5: `total_precipitation` |

**特殊情况**：
- 降雪会被记录为液态水当量（Snow Water Equivalent, SWE）
- 需要区分降雨和降雪（通过温度判断）

#### 2. `temp` - 气温

| 属性 | 值 |
|------|-----|
| **描述** | 日平均气温 |
| **单位** | °C |
| **典型范围** | -50 °C (极地) 到 +50 °C (沙漠) |
| **物理意义** | 控制蒸散发和降水形态（雨/雪），影响融雪 |
| **数据源** | ERA5: `2m_temperature` |

**作用**：
- **T > 0°C**：降雪融化，产生融雪径流
- **T < 0°C**：降水以雪的形式累积
- **高温**：增加蒸散发，减少径流

#### 3. `pet` - 潜在蒸散发

| 属性 | 值 |
|------|-----|
| **描述** | 日潜在蒸散发量（参考作物蒸散发） |
| **单位** | mm/day |
| **典型范围** | 0 - 15 mm/day |
| **物理意义** | 反映大气蒸发能力，影响土壤水分和径流 |
| **计算方法** | FAO Penman-Monteith 公式 |

**计算公式**（简化版）：
$$
\text{PET} = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}
$$

其中：
- $R_n$：净辐射
- $G$：土壤热通量
- $T$：气温
- $u_2$：2m 风速
- $e_s - e_a$：饱和水汽压差
- $\Delta$：饱和水汽压曲线斜率
- $\gamma$：干湿表常数

#### 4. `soil_moisture` - 土壤湿度

| 属性 | 值 |
|------|-----|
| **描述** | 根区土壤体积含水量 |
| **单位** | m³/m³（或 %） |
| **典型范围** | 0.05 (干旱) 到 0.50 (饱和) |
| **物理意义** | 土壤越湿，产流越快（下渗能力降低） |
| **数据源** | ERA5-Land: `volumetric_soil_water_layer_1` |

**临界状态**：
- **田间持水量**（Field Capacity）：土壤能够长期保持的最大水分
- **凋萎点**（Wilting Point）：植物无法吸取水分的最低土壤湿度
- **饱和点**：所有孔隙被水填满，超过此点的降雨全部产流

**对径流的影响**：
```
if soil_moisture > field_capacity:
    产流系数 → 接近 1（几乎全部降雨变成径流）
elif soil_moisture < wilting_point:
    产流系数 → 接近 0（几乎全部降雨下渗）
```

#### 5. `snow` - 积雪水当量

| 属性 | 值 |
|------|-----|
| **描述** | 地面积雪的液态水当量 |
| **单位** | mm |
| **典型范围** | 0 - 2000 mm（高山地区） |
| **物理意义** | 积雪是延迟的径流，春季融雪形成洪水 |
| **数据源** | ERA5: `snow_depth_water_equivalent` |

**融雪径流**：
- 春季温度升高，积雪融化
- 融雪速度取决于温度和辐射
- 常见的融雪洪水发生在 3-5 月（北半球）

### 时间窗口与滞后

**输入序列**：过去 365 天的动态特征
$$
\mathbf{X}_{\text{dyn}} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{365}] \in \mathbb{R}^{365 \times 5}
$$

**为什么需要 365 天？**
1. **季节性**：捕捉雨季/旱季、夏季/冬季模式
2. **长期记忆**：早春的降雪会影响晚春的融雪径流
3. **土壤水分累积**：多次降雨的累积效应

### 数据预处理

**归一化**（在 `my_advanced_model.py:105-108`）：
```python
feature_mean = dynamic_tensor.mean(dim=0, keepdim=True)  # 每个特征的均值
feature_std = dynamic_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)  # 每个特征的标准差
dynamic_tensor = (dynamic_tensor - feature_mean) / feature_std  # Z-score 标准化
```

**缺失值处理**（在 `my_advanced_model.py:388-399`）：
1. **时间插值**：
   ```python
   features = features.interpolate(method='time', limit_direction='both')
   ```
2. **前向/后向填充**：
   ```python
   features = features.fillna(method='ffill').fillna(method='bfill')
   ```
3. **最终填充为 0**：
   ```python
   features = features.fillna(0.0)
   ```

---

## 目标变量 (Target Variable)

### `obs` - 观测径流

| 属性 | 值 |
|------|-----|
| **描述** | 河流断面的日平均流量 |
| **单位** | m³/s（立方米每秒） |
| **数据源** | GRDC（全球径流数据中心） |
| **变量名** | `metrics_utils.OBS_VARIABLE` |
| **坐标** | `(time, lead_time)` |

### Lead Time

模型预测未来 10 天的流量：
$$
\hat{\mathbf{y}} = [\hat{y}_{t+1}, \hat{y}_{t+2}, \ldots, \hat{y}_{t+10}] \in \mathbb{R}^{10}
$$

其中：
- `lead_time = 0`：当天流量（现时预报）
- `lead_time = 1`：明天流量（1 天预报）
- `lead_time = 9`：10 天后流量（10 天预报）

### 数据质量控制

**GRDC 数据质量标志**：
- `A`：已验证，高质量
- `B`：已检查，中等质量
- `C`：未检查，低质量
- `M`：缺失数据

**我们的处理**（在 `my_advanced_model.py:464`）：
```python
targets = targets.replace([np.inf, -np.inf], np.nan).dropna()
```

**异常值检测**（未来工作）：
- 流量 < 0：物理上不可能，标记为缺失
- 流量突变（相邻日变化 > 100 倍）：可能是错误，需人工审核

---

## 数据来源与引用

### 主要数据集

1. **GRDC（全球径流数据中心）**
   - 网站：[https://www.bafg.de/GRDC](https://www.bafg.de/GRDC)
   - 引用：
     ```
     GRDC (2024). Global Runoff Data Centre.
     Federal Institute of Hydrology (BfG), Koblenz, Germany.
     ```

2. **HydroATLAS**
   - 网站：[https://www.hydrosheds.org/hydroatlas](https://www.hydrosheds.org/hydroatlas)
   - 引用：
     ```
     Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019).
     Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution.
     Scientific Data, 6, 283. https://doi.org/10.1038/s41597-019-0300-6
     ```

3. **ERA5 再分析数据**
   - 网站：[https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
   - 引用：
     ```
     Hersbach, H., Bell, B., Berrisford, P., et al. (2020).
     The ERA5 global reanalysis.
     Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
     https://doi.org/10.1002/qj.3803
     ```

---

## 数据预处理

### 完整数据流水线

```
1. 加载原始数据
   ├─ GRDC 观测数据 (NetCDF)
   ├─ HydroATLAS 属性 (CSV/GeoPackage)
   └─ ERA5 气象数据 (NetCDF)

2. 数据对齐
   ├─ 按站点 ID 匹配
   ├─ 按时间索引对齐（daily）
   └─ 处理时区和日期格式

3. 质量控制
   ├─ 移除物理上不可能的值
   ├─ 标记异常值
   └─ 过滤低质量数据

4. 缺失值处理
   ├─ 时间插值（动态特征）
   ├─ 均值填充（静态特征）
   └─ 前向/后向填充

5. 归一化
   ├─ 静态特征：全局 Z-score
   ├─ 动态特征：逐序列 Z-score
   └─ 目标变量：保持原始尺度（用于评估）

6. 数据分割
   ├─ 训练集（70%）
   ├─ 验证集（15%）
   └─ 测试集（15%）

7. 生成训练样本
   ├─ 滑动窗口采样
   ├─ 每个站点 N 个样本
   └─ 批处理（Batch）
```

### 推荐的数据存储格式

**原始数据**：
- NetCDF（`.nc`）：适合多维时空数据
- GeoPackage（`.gpkg`）：适合空间属性数据

**预处理后的数据**（提高加载速度）：
- **Zarr**：云原生格式，支持懒加载和并行读取
- **Parquet**：列式存储，压缩率高
- **HDF5**：传统的科学计算格式

---

**最后更新**: 2025-12-04
**版本**: 1.0.0
