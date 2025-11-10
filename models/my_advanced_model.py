# models/my_advanced_model.py
import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
from typing import Dict, Optional, Tuple
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

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.utils import k_hop_subgraph

# Import common model architectures
from common import HybridGNN_RNN, collate_fn


# ==================== Custom Dataset ====================

class HydroDataset(Dataset):
    """
    自定义 PyTorch Dataset，用于水文预测。
    为每个站点生成训练样本（时间序列窗口）。
    """

    def __init__(self,
                 gauge_ids: list,
                 data_preparation_fn,
                 seq_length: int = 365,
                 pred_length: int = 10,
                 samples_per_gauge: int = 10):
        """
        Args:
            gauge_ids: 站点 ID 列表
            data_preparation_fn: 数据准备函数（通常是 AdvancedModel._prepare_data_for_gauge）
            seq_length: RNN 输入序列长度（天）
            pred_length: 预测前导时间数量
            samples_per_gauge: 每个站点随机采样的样本数
        """
        self.gauge_ids = gauge_ids
        self.data_preparation_fn = data_preparation_fn
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.samples_per_gauge = samples_per_gauge

        # 预缓存每个站点的数据，避免在 __getitem__ 中重复加载
        self.gauge_cache: dict[str, dict] = {}
        self.samples: list[tuple[str, int, int]] = []  # (gauge_id, start_idx, end_idx)

        for gauge_id in self.gauge_ids:
            prepared = self.data_preparation_fn(gauge_id)

            if not prepared or any(item is None for item in prepared):
                continue

            dynamic_features, static_graph_data, targets = prepared

            if isinstance(dynamic_features, pd.DataFrame):
                dynamic_df = dynamic_features
            else:
                dynamic_df = pd.DataFrame(dynamic_features)

            if isinstance(targets, pd.Series):
                target_series = targets
            else:
                target_series = pd.Series(targets)

            # 对齐并过滤缺失值
            valid_mask = ~(dynamic_df.isna().any(axis=1) | target_series.isna())
            dynamic_df = dynamic_df[valid_mask]
            target_series = target_series[valid_mask]

            if len(target_series) < self.seq_length + self.pred_length:
                continue

            dynamic_tensor = torch.as_tensor(dynamic_df.values, dtype=torch.float32)
            target_tensor = torch.as_tensor(target_series.values, dtype=torch.float32)

            # 对动态特征按特征维度进行归一化（标准化）
            feature_mean = dynamic_tensor.mean(dim=0, keepdim=True)
            feature_std = dynamic_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            dynamic_tensor = (dynamic_tensor - feature_mean) / feature_std

            if not isinstance(static_graph_data, Data):
                raise TypeError("Expected static_graph_data to be a torch_geometric.data.Data instance")

            total_length = target_tensor.shape[0]
            max_start = total_length - (self.seq_length + self.pred_length)

            if max_start < 0:
                continue

            windows = [(start, start + self.seq_length) for start in range(max_start + 1)]

            if not windows:
                continue

            self.gauge_cache[gauge_id] = {
                "dynamic": dynamic_tensor,
                "targets": target_tensor,
                "graph": static_graph_data,
                "windows": windows,
            }

            for start_idx, end_idx in windows:
                self.samples.append((gauge_id, start_idx, end_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个训练样本。

        Returns:
            (rnn_input, gnn_graph_data, target)
        """
        gauge_id, start_idx, end_idx = self.samples[idx]

        cache = self.gauge_cache.get(gauge_id)
        if cache is None:
            return None

        dynamic_tensor = cache["dynamic"][start_idx:end_idx]
        target_tensor = cache["targets"][end_idx:end_idx + self.pred_length]

        # 复制静态图，避免在批处理中出现共享引用问题
        gnn_graph_data = cache["graph"].clone() if hasattr(cache["graph"], "clone") else cache["graph"]

        if dynamic_tensor.shape[0] != self.seq_length or target_tensor.shape[0] != self.pred_length:
            return None

        return dynamic_tensor, gnn_graph_data, target_tensor


# ==================== AdvancedModel Class ====================

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
        self.experiment_name = experiment_name

        # 提取超参数（带默认值）
        self.static_feature_dim = model_params.get('static_feature_dim', 50)
        self.dynamic_feature_dim = model_params.get('dynamic_feature_dim', 5)
        self.gnn_hidden_dim = model_params.get('gnn_hidden_dim', 64)
        self.rnn_hidden_dim = model_params.get('rnn_hidden_dim', 128)
        self.rnn_num_layers = model_params.get('rnn_num_layers', 2)
        self.rnn_type = model_params.get('rnn_type', 'lstm')
        self.output_lead_times = model_params.get('output_lead_times', 10)
        self.dropout = model_params.get('dropout', 0.2)
        self.learning_rate = model_params.get('learning_rate', 0.001)
        self.batch_size = model_params.get('batch_size', 32)
        self.num_epochs = model_params.get('num_epochs', 50)
        self.seq_length = model_params.get('seq_length', 365)
        self.samples_per_gauge = model_params.get('samples_per_gauge', 10)
        self.meteorology_file = model_params.get('meteorology_file', None)

        # 设置设备（GPU 或 CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化混合模型
        self.model = HybridGNN_RNN(
            static_feature_dim=self.static_feature_dim,
            dynamic_feature_dim=self.dynamic_feature_dim,
            gnn_hidden_dim=self.gnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            output_lead_times=self.output_lead_times,
            dropout=self.dropout
        ).to(self.device)

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 初始化损失函数（使用 Huber Loss，对异常值更鲁棒）
        self.criterion = nn.HuberLoss(delta=1.0)

        # [CRITICAL] 路径设置，基于 data_paths.py
        # 我们将新模型的输出保存在与 Google 模型相同的父目录中
        self.output_path = data_paths.GOOGLE_MODEL_RUNS_DIR.parent / self.experiment_name
        self.checkpoint_path = self.output_path / 'checkpoints'

        # 确保输出目录存在
        loading_utils.create_remote_folder_if_necessary(self.output_path)
        loading_utils.create_remote_folder_if_necessary(self.checkpoint_path)

        # 缓存核心数据以避免重复读盘
        self._grdc_data: Optional[xr.Dataset] = None
        self._available_gauges: set[str] = set()
        self._static_attributes_df: Optional[pd.DataFrame] = None
        self._static_attribute_order: Optional[list[str]] = None
        self._static_attr_mean: Optional[pd.Series] = None
        self._static_attr_std: Optional[pd.Series] = None
        self._dynamic_feature_columns: Optional[list[str]] = None
        self._meteorology_data: Optional[xr.Dataset] = None
        self._meteorology_load_failed = False
        self._gauge_cache: Dict[str, Tuple[pd.DataFrame, Data, pd.Series]] = {}

        self._initialize_data_caches()

        print(f"Initializing AdvancedModel for experiment: {self.experiment_name}")
        print(f"Model outputs will be saved to: {self.output_path}")
        print(f"Model architecture: {self.model}")

    def _initialize_data_caches(self) -> None:
        """Load core datasets once and prepare shared statistics."""

        self._grdc_data = loading_utils.load_grdc_data()
        if self._grdc_data is not None and 'gauge_id' in self._grdc_data.coords:
            self._available_gauges = set(self._grdc_data.gauge_id.values.tolist())
        else:
            self._available_gauges = set()

        try:
            attributes_df = loading_utils.load_attributes_file()
            self._static_attributes_df = attributes_df.sort_index()
            self._static_attribute_order = list(self._static_attributes_df.columns)
            self._static_attr_mean = self._static_attributes_df.mean().astype(float).fillna(0.0)
            self._static_attr_std = (
                self._static_attributes_df.std(ddof=0).astype(float).replace(0, np.nan).fillna(1.0)
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Failed to load static attributes: {exc}")
            self._static_attributes_df = None
            self._static_attribute_order = None
            self._static_attr_mean = None
            self._static_attr_std = None

        self._get_meteorology_dataset()
        self._gauge_cache.clear()

    def _get_meteorology_dataset(self) -> Optional[xr.Dataset]:
        """Return cached meteorology dataset, loading if necessary."""

        if self._meteorology_data is None and not self._meteorology_load_failed:
            try:
                self._meteorology_data = loading_utils.load_meteorology(self.meteorology_file)
            except ValueError as exc:
                print(f"Error validating meteorology dataset: {exc}")
                self._meteorology_data = None
                self._meteorology_load_failed = True

            if self._meteorology_data is None:
                self._meteorology_load_failed = True

        return self._meteorology_data

    def invalidate_cache(self, reload_data: bool = False) -> None:
        """Invalidate per-gauge caches, optionally reloading backing data."""

        self._gauge_cache.clear()
        if reload_data:
            self._meteorology_data = None
            self._meteorology_load_failed = False
            self._initialize_data_caches()

    def _prepare_static_graph(self, gauge_id: str) -> Data:
        """Create a PyG Data object with scaled static attributes for a gauge."""

        if (
            self._static_attributes_df is None
            or self._static_attribute_order is None
            or gauge_id not in self._static_attributes_df.index
        ):
            static_array = np.zeros(self.static_feature_dim, dtype=np.float32)
        else:
            gauge_attrs = self._static_attributes_df.loc[gauge_id]
            gauge_attrs = gauge_attrs.reindex(self._static_attribute_order)
            gauge_attrs = gauge_attrs.astype(float)

            if self._static_attr_mean is not None:
                gauge_attrs = gauge_attrs.fillna(self._static_attr_mean.reindex(self._static_attribute_order))
            gauge_attrs = gauge_attrs.fillna(0.0)

            mean_values = (
                self._static_attr_mean.reindex(self._static_attribute_order).fillna(0.0).values
                if self._static_attr_mean is not None
                else np.zeros(len(gauge_attrs))
            )
            std_values = (
                self._static_attr_std.reindex(self._static_attribute_order).fillna(1.0).values
                if self._static_attr_std is not None
                else np.ones(len(gauge_attrs))
            )

            static_array = (gauge_attrs.values - mean_values) / (std_values + 1e-8)
            static_array = np.nan_to_num(static_array, nan=0.0, posinf=0.0, neginf=0.0)

            if len(static_array) < self.static_feature_dim:
                padding = np.zeros(self.static_feature_dim - len(static_array), dtype=np.float32)
                static_array = np.concatenate([static_array, padding])
            elif len(static_array) > self.static_feature_dim:
                static_array = static_array[:self.static_feature_dim]

        static_tensor = torch.as_tensor(static_array, dtype=torch.float32).unsqueeze(0)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=static_tensor, edge_index=edge_index)

    def _extract_dynamic_features(
        self,
        gauge_id: str,
        target_index: pd.DatetimeIndex,
    ) -> Optional[pd.DataFrame]:
        """Extract meteorology aligned to gauge and time index."""

        meteorology = self._get_meteorology_dataset()
        if meteorology is None or 'gauge_id' not in meteorology.coords:
            return None

        if gauge_id not in meteorology.gauge_id.values:
            print(f"Warning: {gauge_id} not present in meteorology data")
            return None

        gauge_meteorology = meteorology.sel(gauge_id=gauge_id)

        target_mask = xr.DataArray(
            np.ones(len(target_index), dtype=np.int8),
            coords={'time': target_index},
            dims=['time']
        )

        gauge_meteorology, _ = xr.align(gauge_meteorology, target_mask, join='inner')

        if 'time' not in gauge_meteorology.coords or gauge_meteorology.sizes.get('time', 0) == 0:
            return None

        meteorology_df = gauge_meteorology.to_dataframe().reset_index()
        if 'gauge_id' in meteorology_df.columns:
            meteorology_df = meteorology_df.drop(columns=['gauge_id'])
        meteorology_df = meteorology_df.set_index('time').sort_index()

        return meteorology_df

    def _format_dynamic_features(
        self,
        features: Optional[pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Ensure dynamic features are numeric, aligned, and have expected width."""

        if features is None:
            features = pd.DataFrame(index=index)
        else:
            features = features.reindex(index)

        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.sort_index()
        features = features.reindex(index)

        if len(features) > 0:
            try:
                features = features.interpolate(method='time', limit_direction='both')
            except Exception:
                features = features.interpolate(limit_direction='both')
            features = features.fillna(method='ffill').fillna(method='bfill')

        features = features.fillna(0.0)

        if features.shape[1] == 0:
            columns = [f'feature_{i}' for i in range(self.dynamic_feature_dim)]
            features = pd.DataFrame(0.0, index=index, columns=columns, dtype=np.float32)
            self._dynamic_feature_columns = columns
            return features

        if self._dynamic_feature_columns is None:
            existing_cols = list(features.columns)
            extra_idx = 0
            while len(existing_cols) < self.dynamic_feature_dim:
                candidate = f'extra_{extra_idx}'
                extra_idx += 1
                if candidate in existing_cols:
                    continue
                features[candidate] = 0.0
                existing_cols.append(candidate)

            if len(existing_cols) > self.dynamic_feature_dim:
                existing_cols = existing_cols[:self.dynamic_feature_dim]
                features = features[existing_cols]

            self._dynamic_feature_columns = existing_cols[:self.dynamic_feature_dim]
        else:
            features = features.reindex(columns=self._dynamic_feature_columns, fill_value=0.0)

        return features.astype(np.float32)

    def _prepare_data_for_gauge(self, gauge_id: str) -> tuple:
        """
        为单个站点加载和预处理数据。

        Args:
            gauge_id (str): 站点ID (e.g., 'GRDC_1101101').

        Returns:
            (dynamic_features, static_graph_data, targets):
                - dynamic_features: pd.DataFrame (time, dynamic_feature_dim) - 动态气象输入
                - static_graph_data: PyG Data 对象 - 静态流域属性图
                - targets: pd.Series (time,) - 观测径流
        """
        if gauge_id in self._gauge_cache:
            cached_dynamic, cached_static, cached_targets = self._gauge_cache[gauge_id]
            return (
                cached_dynamic.copy(deep=True),
                cached_static.clone(),
                cached_targets.copy(deep=True),
            )

        if not self._available_gauges:
            self._initialize_data_caches()

        if gauge_id not in self._available_gauges:
            print(f"Gauge {gauge_id} not found in GRDC data.")
            return None, None, None

        try:
            gauge_dataset = self._grdc_data.sel(gauge_id=gauge_id)
            targets_data_array = gauge_dataset[metrics_utils.OBS_VARIABLE].sel(lead_time=0)
            targets = targets_data_array.to_pandas()

            if isinstance(targets, pd.DataFrame):
                targets = targets.iloc[:, 0]

            targets = targets.replace([np.inf, -np.inf], np.nan).dropna()
            targets = targets.sort_index()

            if targets.empty:
                print(f"Warning: No valid observations for {gauge_id}")
                return None, None, None

            dynamic_features = self._extract_dynamic_features(gauge_id, targets.index)

            if dynamic_features is not None and not dynamic_features.empty:
                common_index = targets.index.intersection(dynamic_features.index)
                targets = targets.loc[common_index]
                dynamic_features = dynamic_features.loc[common_index]
            else:
                common_index = targets.index

            if len(common_index) == 0:
                print(f"Warning: No overlapping meteorology for {gauge_id}")
                return None, None, None

            targets = targets.loc[common_index]
            dynamic_features = self._format_dynamic_features(dynamic_features, targets.index)
            targets = targets.astype(np.float32)

            static_graph_data = self._prepare_static_graph(gauge_id)

            cache_entry = (
                dynamic_features.copy(deep=True),
                static_graph_data.clone(),
                targets.copy(deep=True),
            )
            self._gauge_cache[gauge_id] = cache_entry

            static_graph_data = self._get_subgraph_for_gauge(gauge_id)

            if static_graph_data is None:
                node_features = torch.tensor(fallback_static_array, dtype=torch.float32).unsqueeze(0)
                augmented_features, target_mask = self._augment_with_target_indicator(node_features, 0)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                static_graph_data = Data(x=augmented_features, edge_index=edge_index)
                static_graph_data.target_mask = target_mask
                static_graph_data.target_index = torch.tensor(0, dtype=torch.long)

            return dynamic_features, static_graph_data, targets

        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading data for gauge {gauge_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def train(self, training_gauge_ids: list[str]):
        """
        在提供的站点列表上训练模型。

        Args:
            training_gauge_ids (list[str]): 用于训练的站点ID列表。
        """
        print(f"Starting training on {len(training_gauge_ids)} gauges...")
        print(f"Training for {self.num_epochs} epochs with batch size {self.batch_size}")

        # ===== 1. 创建 Dataset 和 DataLoader =====
        train_dataset = HydroDataset(
            gauge_ids=training_gauge_ids,
            data_preparation_fn=self._prepare_data_for_gauge,
            seq_length=self.seq_length,
            pred_length=self.output_lead_times,
            samples_per_gauge=self.samples_per_gauge
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # 设置为 0 避免多进程问题
        )

        print(f"Dataset size: {len(train_dataset)} samples")

        # ===== 2. 训练循环 =====
        self.model.train()
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # 使用 tqdm 显示进度
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')

            for batch_idx, (rnn_input, gnn_input, targets) in enumerate(pbar):
                # 跳过空批次
                if rnn_input is None:
                    continue

                # 移动到设备
                rnn_input = rnn_input.to(self.device)
                gnn_input = gnn_input.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                predictions = self.model(rnn_input, gnn_input)

                # 计算损失（忽略 NaN 目标）
                valid_mask = ~torch.isnan(targets)
                if valid_mask.sum() == 0:
                    continue  # 跳过全是 NaN 的批次

                loss = self.criterion(predictions[valid_mask], targets[valid_mask])

                # 反向传播
                loss.backward()

                # 梯度裁剪（避免梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 记录损失
                epoch_loss += loss.item()
                num_batches += 1

                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 计算平均 epoch 损失
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Average Loss: {avg_epoch_loss:.4f}")

            # ===== 3. 保存检查点 =====
            # 每 10 个 epoch 或最后一个 epoch 保存
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.num_epochs:
                checkpoint_file = self.checkpoint_path / f'model_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, checkpoint_file)
                print(f"Checkpoint saved: {checkpoint_file}")

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_file = self.checkpoint_path / 'best_model.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, best_model_file)
                print(f"Best model updated: {best_model_file} (loss: {best_loss:.4f})")

        print("Training complete!")
        print(f"Best loss: {best_loss:.4f}")

    def predict(self, prediction_gauge_ids: list[str]) -> None:
        """
        为提供的站点列表生成预测，并保存为 NetCDF 文件。

        这必须匹配 `loading_utils.load_google_model_for_one_gauge` 的输出格式，
        以确保与 `return_period_metrics.py` 兼容。

        Args:
            prediction_gauge_ids (list[str]): 需要预测的站点ID列表。
        """
        print(f"Starting prediction on {len(prediction_gauge_ids)} gauges...")

        # ===== 1. 加载最佳模型 =====
        best_model_file = self.checkpoint_path / 'best_model.pth'
        if best_model_file.exists():
            print(f"Loading best model from {best_model_file}")
            checkpoint = torch.load(best_model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        else:
            print("Warning: No trained model found. Using randomly initialized model.")

        self.model.eval()  # 设置为评估模式

        # ===== 2. 为每个站点生成预测 =====
        for gauge_id in tqdm(prediction_gauge_ids, desc='Generating predictions'):

            # --- 2.1 准备该站点的数据 ---
            dynamic_features, static_graph_data, targets = self._prepare_data_for_gauge(gauge_id)

            if dynamic_features is None:
                print(f"Skipping {gauge_id}: Could not load data.")
                continue

            # --- 2.2 加载模板以获取时间坐标 ---
            try:
                template_dataset = loading_utils.load_google_model_for_one_gauge(
                    experiment='full_run',
                    gauge=gauge_id
                )
                if template_dataset is None:
                    print(f"Skipping {gauge_id}: Cannot load template file.")
                    continue

            except Exception as e:
                print(f"Skipping {gauge_id}: Error loading template: {e}")
                continue

            # --- 2.3 运行模型推理（滑动窗口） ---
            # 我们需要为模板中的每个时间步生成预测
            target_times = template_dataset['time'].values
            target_lead_times = template_dataset['lead_time'].values

            # 初始化预测数组
            predictions_array = np.full(
                (len(target_times), len(target_lead_times)),
                np.nan
            )

            # 清理数据（移除 NaN）
            valid_mask = ~(dynamic_features.isna().any(axis=1) | targets.isna())
            clean_dynamic_features = dynamic_features[valid_mask]
            clean_targets = targets[valid_mask]

            if len(clean_dynamic_features) < self.seq_length:
                print(f"Skipping {gauge_id}: Insufficient data after cleaning.")
                continue

            # 滑动窗口预测
            with torch.no_grad():
                for i in range(len(clean_dynamic_features) - self.seq_length):
                    # 提取输入序列
                    rnn_input_seq = clean_dynamic_features.iloc[i:i+self.seq_length].values
                    rnn_input_tensor = torch.FloatTensor(rnn_input_seq).unsqueeze(0).to(self.device)  # (1, seq_length, features)

                    # 静态图数据
                    gnn_input = static_graph_data.to(self.device)
                    # 创建批次（单个样本）
                    gnn_batch = Batch.from_data_list([gnn_input])

                    # 前向传播
                    prediction = self.model(rnn_input_tensor, gnn_batch)  # (1, output_lead_times)
                    prediction_values = prediction.cpu().numpy()[0]  # (output_lead_times,)

                    # 确定预测时间（窗口结束后的时间）
                    prediction_time = clean_dynamic_features.index[i + self.seq_length]

                    # 将预测映射到 target_times
                    if prediction_time in target_times:
                        time_idx = np.where(target_times == prediction_time)[0][0]

                        # 填充前导时间（可能少于 output_lead_times）
                        num_lead_times_to_fill = min(len(prediction_values), len(target_lead_times))
                        predictions_array[time_idx, :num_lead_times_to_fill] = prediction_values[:num_lead_times_to_fill]

            # --- 2.4 处理缺失预测（使用简单插值或填充） ---
            # 对于没有预测的时间步，可以使用插值或填充策略
            # 这里我们简单地保留 NaN（评估脚本会忽略它们）

            # --- 2.5 创建 xarray.Dataset ---
            # 注意：变量名需要与评估脚本兼容
            # 根据模板代码，应该使用与原始 Google 模型相同的变量名
            # 检查模板中的变量名
            if 'sim' in template_dataset.data_vars:
                sim_variable_name = 'sim'
            elif metrics_utils.GOOGLE_VARIABLE in template_dataset.data_vars:
                sim_variable_name = metrics_utils.GOOGLE_VARIABLE
            else:
                # 使用模板中的第一个变量名
                sim_variable_name = list(template_dataset.data_vars.keys())[0]
                print(f"Warning: Using variable name '{sim_variable_name}' from template.")

            prediction_dataset = xr.Dataset(
                {
                    sim_variable_name: (
                        ["time", "lead_time"],
                        predictions_array,
                        {'description': f'Streamflow prediction by {self.experiment_name}'}
                    )
                },
                coords={
                    "time": template_dataset['time'],
                    "lead_time": template_dataset['lead_time'],
                    "gauge_id": gauge_id
                }
            )

            # --- 2.6 保存为 NetCDF 文件 ---
            output_file_path = self.output_path / f'{gauge_id}.nc'
            try:
                prediction_dataset.to_netcdf(output_file_path)
            except Exception as e:
                print(f"Error saving {gauge_id}.nc: {e}")
                import traceback
                traceback.print_exc()

        print(f"Predictions saved to {self.output_path}")
        print("Prediction complete!")


# ==================== Usage Example ====================

if __name__ == "__main__":
    """
    使用示例：训练和预测 GNN-LSTM 混合模型
    """

    print("=" * 80)
    print("GNN-LSTM Hybrid Model for Hydrological Prediction")
    print("=" * 80)

    # ===== 1. 定义模型超参数 =====
    model_params = {
        'static_feature_dim': 50,      # 静态流域属性维度（根据实际数据调整）
        'dynamic_feature_dim': 5,      # 动态气象特征维度
        'gnn_hidden_dim': 64,          # GNN 隐藏层维度
        'rnn_hidden_dim': 128,         # RNN 隐藏层维度
        'rnn_num_layers': 2,           # RNN 层数
        'rnn_type': 'lstm',            # 'lstm' 或 'gru'
        'output_lead_times': 10,       # 预测前导时间数量
        'dropout': 0.2,                # Dropout 率
        'learning_rate': 0.001,        # 学习率
        'batch_size': 32,              # 批大小
        'num_epochs': 50,              # 训练轮数
        'seq_length': 365,             # RNN 输入序列长度（天）
        'samples_per_gauge': 10,       # 每个站点的训练样本数
    }

    # ===== 2. 初始化模型 =====
    model = AdvancedModel(
        model_params=model_params,
        experiment_name="gnn_lstm_hybrid_v1"
    )

    # ===== 3. 定义训练和测试站点 =====
    # 注意：这里使用示例站点 ID，您需要替换为实际的站点列表
    # 可以从 loading_utils.load_grdc_data() 获取可用站点列表

    # 示例站点 ID（您需要替换为实际存在的站点）
    try:
        ds_grdc = loading_utils.load_grdc_data()
        all_gauge_ids = ds_grdc.gauge_id.values.tolist()

        print(f"\nTotal available gauges: {len(all_gauge_ids)}")

        # 使用前 100 个站点进行训练（示例）
        training_gauge_ids = all_gauge_ids[:100] if len(all_gauge_ids) >= 100 else all_gauge_ids[:len(all_gauge_ids)//2]

        # 使用接下来的 20 个站点进行预测（示例）
        prediction_gauge_ids = all_gauge_ids[100:120] if len(all_gauge_ids) >= 120 else all_gauge_ids[len(all_gauge_ids)//2:]

        print(f"Training gauges: {len(training_gauge_ids)}")
        print(f"Prediction gauges: {len(prediction_gauge_ids)}")

    except Exception as e:
        print(f"Error loading gauge IDs: {e}")
        print("Using dummy gauge IDs for demonstration.")
        training_gauge_ids = ['GRDC_6335020', 'GRDC_6335075']
        prediction_gauge_ids = ['GRDC_6335020']

    # ===== 4. 训练模型 =====
    print("\n" + "=" * 80)
    print("TRAINING PHASE")
    print("=" * 80)

    # 取消下面的注释以启动训练
    # model.train(training_gauge_ids)

    # ===== 5. 生成预测 =====
    print("\n" + "=" * 80)
    print("PREDICTION PHASE")
    print("=" * 80)

    # 取消下面的注释以生成预测
    # model.predict(prediction_gauge_ids)

    # ===== 6. 评估模型 =====
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print("""
    预测完成后，您可以使用以下 Notebooks 评估模型性能：

    1. 标准指标（NSE, KGE, etc.）:
       - notebooks/calculate_hydrograph_metrics.ipynb

    2. 极端事件指标（重现期）:
       - notebooks/calculate_return_period_metrics.ipynb

    确保在评估脚本中将 experiment 参数设置为您的模型名称：
       experiment = 'gnn_lstm_hybrid_v1'
    """)

    print("=" * 80)
    print("示例完成！")
    print("=" * 80)

    print("""
    注意事项：
    1. 本实现使用模拟的气象数据。在实际应用中，您需要：
       - 在 _prepare_data_for_gauge 方法中加载真实的气象驱动数据
       - 确保气象数据与 GRDC 观测数据的时间索引对齐

    2. 模型超参数需要根据您的数据和任务进行调优：
       - seq_length: 根据流域的响应时间调整
       - rnn_hidden_dim, gnn_hidden_dim: 根据数据复杂度调整
       - learning_rate, batch_size: 根据训练表现调整

    3. 静态特征维度 (static_feature_dim) 应该与：
       loading_utils.load_attributes_file() 返回的列数匹配

    4. 训练可能需要大量时间和计算资源（GPU 推荐）

    5. 预测输出格式已与原始评估框架兼容，可以直接使用
       calculate_return_period_metrics.ipynb 进行评估
    """)