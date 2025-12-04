# 测试套件

本目录包含 GNN-LSTM 混合模型的测试用例。

## 测试类别

### 1. 边界测试 (`TestHydrologyModelBoundary`)
- 测试极端干旱输入（全零降雨）
- 测试极端洪水输入（极大降雨）
- 测试不同批大小的输出维度一致性
- 测试单样本推理
- 测试零静态特征

### 2. 数据质量测试 (`TestDataQuality`)
- 测试动态特征中的 NaN 处理
- 测试无穷大值的处理
- 测试空时间序列
- 测试过短时间序列

### 3. 性能测试 (`TestModelPerformance`)
- 测试大批量推理性能
- 测试内存使用效率

### 4. 逻辑一致性测试 (`TestLogicConsistency`)
- 测试推理的确定性
- 测试降雨量与流量的单调性
- 测试无负流量预测

### 5. 泛化测试 (`TestGeneralization`)
- 测试无数据泄露

## 运行测试

### 运行所有测试
```bash
python tests/test_scenarios.py
```

### 运行特定测试类
```bash
python -m unittest tests.test_scenarios.TestHydrologyModelBoundary
```

### 运行特定测试方法
```bash
python -m unittest tests.test_scenarios.TestHydrologyModelBoundary.test_extreme_drought_input
```

## 测试依赖

确保已安装以下包：
```bash
pip install torch torch-geometric numpy pandas psutil
```

## 添加新测试

在 `test_scenarios.py` 中添加新的测试类或测试方法：

```python
class TestNewFeature(unittest.TestCase):
    def test_something(self):
        # 测试代码
        self.assertEqual(actual, expected)
```
