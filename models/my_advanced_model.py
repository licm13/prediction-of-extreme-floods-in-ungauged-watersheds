# models/my_advanced_model.py
import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
from tqdm import tqdm

# 确保 backend 在路径中，以便可以从 models/ 目录导入
# (假设 models/ 和 notebooks/ 位于同一父目录下)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
notebooks_backend_dir = os.path.join(parent_dir, 'notebooks', 'backend')
sys.path.append(notebooks_backend_dir)

try:
    from backend import loading_utils 
    from backend import data_paths
    from backend import metrics_utils
except ImportError:
    print("Error: Could not import from notebooks/backend.")
    print(f"Attempted to add path: {notebooks_backend_dir}")
    print("Please ensure 'models' and 'notebooks' directories are siblings.")
    sys.exit(1)

# TODO: 在此导入您选择的ML库
# import torch
# from torch.utils.data import DataLoader, Dataset
# import torch_geometric
# from torch_geometric.data import Data, Batch
# ...

class AdvancedModel:
    """
    一个先进ML模型（如Transformer或GNN）的模板。
    它必须实现 train() 和 predict() 方法。
    """
    
    def __init__(self, model_params: dict, experiment_name: str = "my_advanced_model"):
        """
        初始化模型, 定义架构。
        
        Args:
            model_params (dict): 包含超参数的字典 (e.g., learning_rate, layers)。
            experiment_name (str): 用于保存输出的实验名称。
        """
        self.params = model_params
        self.model = None # TODO: 在此定义您的模型架构
        self.experiment_name = experiment_name
        
        # [CRITICAL] 路径设置，基于 data_paths.py
        # 我们将新模型的输出保存在与 Google 模型相同的父目录中
        self.output_path = data_paths.GOOGLE_MODEL_RUNS_DIR.parent / self.experiment_name
        
        # 确保输出目录存在
        loading_utils.create_remote_folder_if_necessary(self.output_path)
        
        print(f"Initializing AdvancedModel for experiment: {self.experiment_name}")
        print(f"Model outputs will be saved to: {self.output_path}")

    def _prepare_data_for_gauge(self, gauge_id: str) -> (object, object):
        """
        一个辅助函数，用于为单个站点加载和预处理数据。
        
        Args:
            gauge_id (str): 站点ID (e.g., 'GRDC_1101101').
            
        Returns:
            (features, target): 可用于模型训练/预测的已处理数据。
        """
        print(f"Preparing data for {gauge_id}...")
        
        # TODO: 实现数据加载和特征工程
        # 这是一个示例。您需要加载驱动数据（如气象数据）和观测数据
        
        # 1. 加载观测数据 (用于训练/验证)
        # 您需要 GRDC 数据。确保已运行 `concatenate_grdc_downloads.ipynb`
        # 并通过 `loading_utils.load_grdc_data()` 加载
        
        # 2. 加载驱动数据 (气象数据)
        # 原始论文的驱动数据在 Zenodo 存储库中。
        # 您需要自己实现这部分的数据加载逻辑，
        # 因为 `loading_utils.py` 主要关注 *模型输出* 和 *GRDC*。
        
        # 3. 加载静态属性 (用于GNN或作为静态特征)
        # static_attrs = loading_utils.load_attributes_file(gauges=[gauge_id])
        
        # 4. 特征工程
        # features = ... (e.g., 气象数据 + 静态属性)
        # targets = ... (e.g., 观测径流)
        
        # 模拟返回, 您需要替换它
        # features = np.random.rand(1000, 10) # (time, features)
        # targets = np.random.rand(1000, 1)  # (time, target)
        
        # 仅为示例:
        try:
            ds_grdc = loading_utils.load_grdc_data()
            ds_grdc_gauge = ds_grdc.sel(gauge_id=gauge_id)
            targets = ds_grdc_gauge[metrics_utils.OBS_VARIABLE].to_pandas()
            
            # 模拟特征 (应替换为真实的气象输入)
            features = pd.DataFrame(
                np.random.rand(len(targets), 5), 
                index=targets.index,
                columns=['precip', 'temp', 'PET', 'soil_moisture', 'snow']
            )
            return features, targets
            
        except Exception as e:
            print(f"Error loading data for gauge {gauge_id}: {e}")
            return None, None


    def train(self, training_gauge_ids: list[str]):
        """
        在提供的站点列表上训练模型。
        
        Args:
            training_gauge_ids (list[str]): 用于训练的站点ID列表。
        """
        print(f"Starting training on {len(training_gauge_ids)} gauges...")
        
        # TODO: 实现您的训练循环
        # 1. 设置 Dataset 和 DataLoader (如果使用 PyTorch/TF)
        #    (您可能需要一个自定义的 Dataset 类来迭代 `training_gauge_ids`
        #     并在 `__getitem__` 中调用 `_prepare_data_for_gauge`)
        # 2. 迭代 epochs
        # 3. 在每个 epoch 中, 迭代 (tqdm) Dataloader
        # 4.   ... (执行前向和反向传播) ...
        # 5. 保存模型检查点
        
        print("Training complete (simulation).")

    def predict(self, prediction_gauge_ids: list[str]) -> None:
        """
        为提供的站点列表生成预测，并保存为 NetCDF 文件。
        
        这必须匹配 `loading_utils.load_google_model_for_one_gauge` 的输出格式，
        以确保与 `return_period_metrics.py` 兼容。
        
        Args:
            prediction_gauge_ids (list[str]): 需要预测的站点ID列表。
        """
        print(f"Starting prediction on {len(prediction_gauge_ids)} gauges...")
        
        if self.model is None:
            print("Warning: Model is not trained. Using random predictions.")
            # self.model = torch.load('path/to/checkpoint.pth') # 加载模型
            
        for gauge_id in tqdm(prediction_gauge_ids):
            
            # 1. 准备该站点的数据
            # features, _ = self._prepare_data_for_gauge(gauge_id)
            # if features is None:
            #     continue
                
            # TODO: 2. 运行模型推理
            # predictions_array = self.model(features) # (time, lead_time)
            
            # --- 模拟预测 (TODO: 替换为真实预测) ---
            # 我们需要模拟一个与原版 兼容的 xarray.Dataset
            # [CRITICAL] 加载一个现有文件以获取其结构和坐标
            try:
                # 使用 'full_run' 的一个站点作为坐标模板
                #
                ds_template = loading_utils.load_google_model_for_one_gauge(
                    experiment='full_run', 
                    gauge=gauge_id
                )
                if ds_template is None:
                    # 如果该站点在 'full_run' 中不存在, 尝试另一个
                    ds_template = loading_utils.load_google_model_for_one_gauge(
                        'full_run', 'GRDC_6335020' # 找一个已知存在的
                    )
                if ds_template is None:
                     raise FileNotFoundError("Cannot load any template file.")
                    
            except FileNotFoundError:
                print(f"Skipping {gauge_id}: Cannot load template file for structure.")
                continue

            # 创建模拟数据
            # 确保 'time' 和 'lead_time' 坐标是相同的
            # TODO: 您的模型输出需要对齐到 `ds_template['time']`
            sim_data = np.random.rand(
                len(ds_template['time']), 
                len(ds_template['lead_time'])
            )
            
            # 3. [CRITICAL] 创建 xarray.Dataset
            # 这必须与 `evaluation_utils` 和 
            # `return_period_metrics` 兼容
            
            # 变量名必须是 'sim' (来自 metrics_utils.GOOGLE_VARIABLE) 
            # 否则评估脚本 会失败
            sim_variable_name = metrics_utils.GOOGLE_VARIABLE # "sim"
            
            prediction_ds = xr.Dataset(
                {
                    sim_variable_name: (
                        ["time", "lead_time"], 
                        sim_data,
                        {'description': 'Simulated streamflow by my_advanced_model'}
                    )
                },
                coords={
                    "time": ds_template['time'],
                    "lead_time": ds_template['lead_time'],
                    "gauge_id": gauge_id
                }
            )
            
            # 4. 保存为 NetCDF 文件
            output_file_path = self.output_path / f'{gauge_id}.nc'
            try:
                prediction_ds.to_netcdf(output_file_path)
            except Exception as e:
                print(f"Error saving {gauge_id}.nc: {e}")

        print(f"Predictions saved to {self.output_path}")