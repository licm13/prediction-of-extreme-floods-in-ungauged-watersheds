"""
Refactored GNN-LSTM Hybrid Model for Hydrological Prediction.

This refactored version:
1. Uses real data loading instead of simulated data
2. Improves dataset efficiency with preprocessing
3. Integrates evaluation metrics
4. Provides better model evaluation capabilities
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import pathlib

# Ensure backend is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
notebooks_backend_dir = os.path.join(parent_dir, 'notebooks', 'backend')
sys.path.append(notebooks_backend_dir)

try:
    from backend import loading_utils
    from backend import data_paths
    from backend import metrics_utils
    from backend import metrics
    from backend.return_period_calculator import return_period_calculator
    from backend.return_period_calculator import exceptions as rpc_exceptions
except ImportError:
    print("Error: Could not import from notebooks/backend.")
    sys.exit(1)

# Import the new data loader
from hydro_data_loader import HydroDataLoader, PreprocessedHydroDataset

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

# Import common model architectures
from common import HybridGNN_RNN, collate_fn


# ==================== Improved Dataset ====================

class ImprovedHydroDataset(Dataset):
    """
    Improved dataset that uses preprocessed data for efficiency.
    """

    def __init__(self,
                 preprocessed_dataset: PreprocessedHydroDataset,
                 samples_per_gauge: Optional[int] = None):
        """
        Args:
            preprocessed_dataset: PreprocessedHydroDataset instance
            samples_per_gauge: Number of samples per gauge.
                              If None, uses all possible windows.
        """
        self.preprocessed_dataset = preprocessed_dataset
        self.samples_per_gauge = samples_per_gauge

        # Build index: list of (gauge_id, window_start_idx) tuples
        self.samples = []
        for gauge_id in preprocessed_dataset.valid_gauge_ids:
            num_windows = preprocessed_dataset.get_num_valid_windows_per_gauge(gauge_id)

            if samples_per_gauge is None:
                # Use all windows
                window_indices = list(range(num_windows))
            else:
                # Sample randomly
                window_indices = np.random.choice(
                    num_windows,
                    size=min(samples_per_gauge, num_windows),
                    replace=False
                ).tolist()

            for window_idx in window_indices:
                self.samples.append((gauge_id, window_idx))

        print(f"Dataset contains {len(self.samples)} samples from {len(preprocessed_dataset.valid_gauge_ids)} gauges")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            (rnn_input, gnn_graph_data, target)
        """
        gauge_id, window_start = self.samples[idx]
        data = self.preprocessed_dataset.gauge_data[gauge_id]

        seq_length = self.preprocessed_dataset.seq_length
        pred_length = self.preprocessed_dataset.pred_length

        # Extract window
        dynamic_features = data['dynamic']
        targets = data['targets']

        end_idx = window_start + seq_length

        # RNN input
        rnn_input = dynamic_features.iloc[window_start:end_idx].values
        rnn_input = torch.FloatTensor(rnn_input)

        # Target
        target_values = targets.iloc[end_idx:end_idx + pred_length].values
        if len(target_values) < pred_length:
            padding = np.full(pred_length - len(target_values), np.nan)
            target_values = np.concatenate([target_values, padding])
        target = torch.FloatTensor(target_values)

        # GNN input (static graph)
        static_features = data['static']
        x = torch.FloatTensor(static_features).unsqueeze(0)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        gnn_graph_data = Data(x=x, edge_index=edge_index)

        return rnn_input, gnn_graph_data, target


# ==================== StreamflowModel Class ====================

class StreamflowModel:
    """
    Refactored streamflow prediction model with real data loading and evaluation.
    """

    def __init__(self, model_params: dict, experiment_name: str = "streamflow_model"):
        self.params = model_params
        self.experiment_name = experiment_name

        # Extract hyperparameters
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
        self.validation_interval = model_params.get('validation_interval', 5)
        self.validation_metrics = [
            metric for metric in model_params.get(
                'validation_metrics', ['nse', 'kge', 'rmse']
            )
        ]
        self.prediction_metrics = [
            metric for metric in model_params.get(
                'prediction_metrics', ['nse', 'kge', 'rmse', 'pearson-r']
            )
        ]

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
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

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.criterion = nn.HuberLoss(delta=1.0)

        # Initialize data loader
        meteorology_file = model_params.get('meteorology_file', None)
        use_simulated_met = model_params.get('use_simulated_met', True)

        self.data_loader = HydroDataLoader(
            static_feature_dim=self.static_feature_dim,
            dynamic_feature_dim=self.dynamic_feature_dim,
            meteorology_file=meteorology_file,
            use_simulated_met=use_simulated_met
        )

        # Paths
        self.output_path = data_paths.GOOGLE_MODEL_RUNS_DIR.parent / self.experiment_name
        self.checkpoint_path = self.output_path / 'checkpoints'
        self.metrics_path = self.output_path / 'metrics'
        self.predictions_path = self.output_path / 'predictions'
        self.evaluation_path = self.output_path / 'evaluation'
        self.figures_path = self.evaluation_path / 'figures'
        self.extreme_tables_path = self.evaluation_path / 'extreme_tables'
        self.metadata_path = self.output_path / 'metadata'

        for path in [
            self.output_path,
            self.checkpoint_path,
            self.metrics_path,
            self.predictions_path,
            self.evaluation_path,
            self.figures_path,
            self.extreme_tables_path,
            self.metadata_path,
        ]:
            loading_utils.create_remote_folder_if_necessary(path)

        print(f"Initialized StreamflowModel: {self.experiment_name}")
        print(f"Outputs: {self.output_path}")

    def train(self, training_gauge_ids: list[str], validation_gauge_ids: Optional[list[str]] = None):
        """
        Train the model.

        Args:
            training_gauge_ids: List of gauge IDs for training
            validation_gauge_ids: Optional list for validation
        """
        print(f"Starting training on {len(training_gauge_ids)} gauges...")

        # Preload training data
        print("Preprocessing training data...")
        train_preprocessed = PreprocessedHydroDataset(
            data_loader=self.data_loader,
            gauge_ids=training_gauge_ids,
            seq_length=self.seq_length,
            pred_length=self.output_lead_times
        )

        train_dataset = ImprovedHydroDataset(
            preprocessed_dataset=train_preprocessed,
            samples_per_gauge=self.samples_per_gauge
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Validation data (if provided)
        val_loader = None
        if validation_gauge_ids is not None:
            print("Preprocessing validation data...")
            val_preprocessed = PreprocessedHydroDataset(
                data_loader=self.data_loader,
                gauge_ids=validation_gauge_ids,
                seq_length=self.seq_length,
                pred_length=self.output_lead_times
            )

            val_dataset = ImprovedHydroDataset(
                preprocessed_dataset=val_preprocessed,
                samples_per_gauge=self.samples_per_gauge
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )

        # Training loop
        self.model.train()
        best_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': []}
        validation_metric_history: List[Dict[str, float]] = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')

            for rnn_input, gnn_input, targets in pbar:
                if rnn_input is None:
                    continue

                rnn_input = rnn_input.to(self.device)
                gnn_input = gnn_input.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(rnn_input, gnn_input)

                valid_mask = ~torch.isnan(targets)
                if valid_mask.sum() == 0:
                    continue

                loss = self.criterion(predictions[valid_mask], targets[valid_mask])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = epoch_loss / max(num_batches, 1)
            training_history['train_loss'].append(avg_train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                training_history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

            if val_loader is not None and ((epoch + 1) % max(self.validation_interval, 1) == 0):
                validation_metrics = self._compute_validation_metrics(val_loader)
                if validation_metrics:
                    metrics_with_epoch = {'epoch': epoch + 1}
                    metrics_with_epoch.update(validation_metrics)
                    validation_metric_history.append(metrics_with_epoch)
                    formatted_metrics = ", ".join(
                        f"{key}: {value:.4f}" if value is not None else f"{key}: NaN"
                        for key, value in validation_metrics.items()
                    )
                    print(f"Validation metrics at epoch {epoch+1}: {formatted_metrics}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.num_epochs:
                self._save_checkpoint(epoch + 1, avg_train_loss)

            # Save best model
            current_loss = val_loss if val_loss is not None else avg_train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                self._save_best_model(epoch + 1, current_loss)

        # Save training history
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(self.metrics_path / 'training_history.csv', index=False)

        if validation_metric_history:
            validation_df = pd.DataFrame(validation_metric_history)
            validation_df.to_csv(self.metrics_path / 'validation_metrics.csv', index=False)

        print("Training complete!")
        print(f"Best loss: {best_loss:.4f}")

        training_metadata = {
            'experiment_name': self.experiment_name,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'params': self.params,
            'best_loss': best_loss,
            'training_history': history_df.to_dict(orient='list'),
            'validation_metrics': validation_metric_history,
        }
        if validation_metric_history:
            training_metadata['validation_metrics_file'] = os.path.relpath(
                self.metrics_path / 'validation_metrics.csv',
                self.output_path,
            )
        self._write_metadata_file('training_run.json', training_metadata)

    def _evaluate(self, data_loader) -> float:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for rnn_input, gnn_input, targets in data_loader:
                if rnn_input is None:
                    continue

                rnn_input = rnn_input.to(self.device)
                gnn_input = gnn_input.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(rnn_input, gnn_input)

                valid_mask = ~torch.isnan(targets)
                if valid_mask.sum() == 0:
                    continue

                loss = self.criterion(predictions[valid_mask], targets[valid_mask])
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _compute_validation_metrics(self, data_loader) -> Dict[str, float]:
        """Run inference on the validation loader and compute summary metrics."""
        self.model.eval()
        preds: List[torch.Tensor] = []
        trues: List[torch.Tensor] = []

        with torch.no_grad():
            for rnn_input, gnn_input, targets in data_loader:
                if rnn_input is None:
                    continue

                rnn_input = rnn_input.to(self.device)
                gnn_input = gnn_input.to(self.device)
                targets = targets.to(self.device)

                batch_predictions = self.model(rnn_input, gnn_input)
                valid_mask = ~torch.isnan(targets)
                if valid_mask.sum() == 0:
                    continue

                preds.append(batch_predictions[valid_mask].detach().cpu())
                trues.append(targets[valid_mask].detach().cpu())

        self.model.train()

        if not preds:
            return {}

        concatenated_preds = torch.cat(preds)
        concatenated_trues = torch.cat(trues)

        if concatenated_preds.numel() == 0:
            return {}

        coords = np.arange(concatenated_preds.numel())
        sim_da = xr.DataArray(
            concatenated_preds.numpy(),
            dims=['time'],
            coords={'time': coords},
        )
        obs_da = xr.DataArray(
            concatenated_trues.numpy(),
            dims=['time'],
            coords={'time': coords},
        )

        try:
            metric_values = metrics.calculate_metrics(
                obs=obs_da,
                sim=sim_da,
                metrics=self.validation_metrics,
                resolution='1D',
                datetime_coord='time',
                minimum_data_points=len(coords),
            )
        except RuntimeError as err:
            print(f"Validation metric computation failed: {err}")
            return {}

        sanitized_metrics: Dict[str, float] = {}
        for key, value in metric_values.items():
            if value is None:
                sanitized_metrics[key] = None
            elif isinstance(value, float):
                sanitized_metrics[key] = float(value) if not np.isnan(value) else None
            else:
                sanitized_metrics[key] = float(value)

        return sanitized_metrics

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_file = self.checkpoint_path / f'model_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

    def _save_best_model(self, epoch: int, loss: float):
        """Save best model."""
        best_model_file = self.checkpoint_path / 'best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, best_model_file)
        print(f"Best model updated: {best_model_file} (loss: {loss:.4f})")

    def predict(self, prediction_gauge_ids: list[str]) -> None:
        """
        Generate predictions, save NetCDF files, and run evaluation routines.

        Args:
            prediction_gauge_ids: List of gauge IDs to predict
        """
        print(f"Generating predictions for {len(prediction_gauge_ids)} gauges...")

        # Load best model
        best_model_file = self.checkpoint_path / 'best_model.pth'
        if best_model_file.exists():
            checkpoint = torch.load(best_model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']}")
        else:
            print("Warning: No trained model found")

        self.model.eval()

        hydro_metrics_records: List[Dict[str, float]] = []
        extreme_metrics_records: List[Dict[str, float]] = []
        extreme_assets: List[Dict[str, str]] = []
        processed_gauges = 0

        for gauge_id in tqdm(prediction_gauge_ids, desc='Generating predictions'):
            dynamic_features, static_features, targets = self.data_loader.prepare_data_for_gauge(gauge_id)

            if dynamic_features is None:
                print(f"Skipping {gauge_id}: Could not load data")
                continue

            try:
                ds_template = loading_utils.load_google_model_for_one_gauge(
                    experiment='full_run',
                    gauge=gauge_id
                )
                if ds_template is None:
                    print(f"Skipping {gauge_id}: Cannot load template")
                    continue
            except Exception as err:
                print(f"Skipping {gauge_id}: Error loading template ({err})")
                continue

            predictions_array = self._run_inference_for_gauge(
                dynamic_features, static_features, targets, ds_template
            )

            prediction_ds = self._save_predictions_netcdf(
                gauge_id, predictions_array, ds_template
            )
            processed_gauges += 1

            gauge_metrics = self._calculate_prediction_metrics(prediction_ds, gauge_id)
            if gauge_metrics is not None:
                hydro_metrics_records.append(gauge_metrics)

            extremes_result = self._evaluate_extremes_from_dataset(prediction_ds, gauge_id)
            if extremes_result is not None:
                extreme_metrics_records.append(
                    {'gauge_id': gauge_id, **extremes_result['metrics']}
                )
                extreme_assets.append({
                    'gauge_id': gauge_id,
                    'table': os.path.relpath(extremes_result['table_path'], self.output_path),
                    'figure': os.path.relpath(extremes_result['figure_path'], self.output_path),
                })

        hydro_metrics_file = None
        hydro_metrics_json = None
        if hydro_metrics_records:
            hydro_metrics_df = pd.DataFrame(hydro_metrics_records).set_index('gauge_id')
            hydro_metrics_file = self.evaluation_path / 'hydrograph_metrics.csv'
            hydro_metrics_df.to_csv(hydro_metrics_file)
            hydro_metrics_json = self.evaluation_path / 'hydrograph_metrics.json'
            with open(hydro_metrics_json, 'w') as f:
                json.dump(
                    self._serialize_for_json(hydro_metrics_df.to_dict(orient='index')),
                    f,
                    indent=2,
                )

        extreme_metrics_file = None
        extreme_metrics_json = None
        if extreme_metrics_records:
            extreme_metrics_df = pd.DataFrame(extreme_metrics_records).set_index('gauge_id')
            extreme_metrics_file = self.evaluation_path / 'extreme_metrics.csv'
            extreme_metrics_df.to_csv(extreme_metrics_file)
            extreme_metrics_json = self.evaluation_path / 'extreme_metrics.json'
            with open(extreme_metrics_json, 'w') as f:
                json.dump(
                    self._serialize_for_json(extreme_metrics_df.to_dict(orient='index')),
                    f,
                    indent=2,
                )

        evaluation_metadata: Dict[str, object] = {
            'experiment_name': self.experiment_name,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'gauges_requested': len(prediction_gauge_ids),
            'gauges_predicted': processed_gauges,
            'prediction_directory': os.path.relpath(
                self.predictions_path, self.output_path
            ),
        }

        if hydro_metrics_file is not None:
            evaluation_metadata['hydrograph_metrics_file'] = os.path.relpath(
                hydro_metrics_file, self.output_path
            )
        if hydro_metrics_json is not None:
            evaluation_metadata['hydrograph_metrics_json'] = os.path.relpath(
                hydro_metrics_json, self.output_path
            )
        if extreme_metrics_file is not None:
            evaluation_metadata['extreme_metrics_file'] = os.path.relpath(
                extreme_metrics_file, self.output_path
            )
        if extreme_metrics_json is not None:
            evaluation_metadata['extreme_metrics_json'] = os.path.relpath(
                extreme_metrics_json, self.output_path
            )
        if extreme_assets:
            evaluation_metadata['extreme_assets'] = extreme_assets

        self._write_metadata_file('evaluation_summary.json', evaluation_metadata)

        print(f"Predictions saved to {self.predictions_path}")
        if hydro_metrics_file is not None:
            print(f"Hydrograph metrics saved to {hydro_metrics_file}")
        if extreme_metrics_file is not None:
            print(f"Extreme metrics saved to {extreme_metrics_file}")

    def _run_inference_for_gauge(self, dynamic_features, static_features, targets, ds_template):
        """Run model inference for a gauge."""
        target_times = ds_template['time'].values
        target_lead_times = ds_template['lead_time'].values

        predictions_array = np.full((len(target_times), len(target_lead_times)), np.nan)

        # Clean data
        valid_mask = ~(dynamic_features.isna().any(axis=1) | targets.isna())
        dynamic_features_clean = dynamic_features[valid_mask]

        if len(dynamic_features_clean) < self.seq_length:
            return predictions_array

        # Create static graph
        x = torch.FloatTensor(static_features).unsqueeze(0)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        static_graph_data = Data(x=x, edge_index=edge_index)

        # Sliding window prediction
        with torch.no_grad():
            for i in range(len(dynamic_features_clean) - self.seq_length):
                rnn_input_seq = dynamic_features_clean.iloc[i:i+self.seq_length].values
                rnn_input_tensor = torch.FloatTensor(rnn_input_seq).unsqueeze(0).to(self.device)

                gnn_input = static_graph_data.to(self.device)
                gnn_batch = Batch.from_data_list([gnn_input])

                pred = self.model(rnn_input_tensor, gnn_batch)
                pred_values = pred.cpu().numpy()[0]

                pred_time = dynamic_features_clean.index[i + self.seq_length]

                if pred_time in target_times:
                    time_idx = np.where(target_times == pred_time)[0][0]
                    num_lead_times_to_fill = min(len(pred_values), len(target_lead_times))
                    predictions_array[time_idx, :num_lead_times_to_fill] = pred_values[:num_lead_times_to_fill]

        return predictions_array

    def _save_predictions_netcdf(self, gauge_id, predictions_array, ds_template):
        """Save predictions as NetCDF file and return the dataset."""
        prediction_ds = ds_template.copy(deep=True)

        sim_variable_name = self._get_simulation_variable_name(prediction_ds)
        prediction_ds[sim_variable_name] = xr.DataArray(
            predictions_array.astype(np.float32),
            dims=('time', 'lead_time'),
            coords={
                'time': prediction_ds['time'],
                'lead_time': prediction_ds['lead_time'],
            },
            attrs={'description': f'Streamflow prediction by {self.experiment_name}'},
        )

        prediction_ds = prediction_ds.assign_coords({'gauge_id': gauge_id})
        if prediction_ds.attrs is None:
            prediction_ds.attrs = {}
        prediction_ds.attrs['model_id'] = self.experiment_name
        prediction_ds.attrs['generated_at'] = datetime.utcnow().isoformat() + 'Z'

        output_file_path = self.predictions_path / f'{gauge_id}.nc'
        prediction_ds.to_netcdf(output_file_path)

        return prediction_ds

    def _get_simulation_variable_name(self, dataset: xr.Dataset) -> str:
        if metrics_utils.GOOGLE_VARIABLE in dataset.data_vars:
            return metrics_utils.GOOGLE_VARIABLE
        if 'sim' in dataset.data_vars:
            return 'sim'
        return list(dataset.data_vars.keys())[0]

    def _calculate_prediction_metrics(
        self, dataset: xr.Dataset, gauge_id: str
    ) -> Optional[Dict[str, float]]:
        if metrics_utils.OBS_VARIABLE not in dataset.data_vars:
            print(f"Dataset for {gauge_id} is missing observations; skipping metrics.")
            return None

        sim_variable_name = self._get_simulation_variable_name(dataset)
        sim = dataset[sim_variable_name]
        obs = dataset[metrics_utils.OBS_VARIABLE]

        if 'lead_time' in sim.dims:
            sim = sim.sel(lead_time=0).squeeze(drop=True)
        if 'lead_time' in obs.dims:
            obs = obs.sel(lead_time=0).squeeze(drop=True)

        try:
            metric_values = metrics.calculate_metrics(
                obs=obs,
                sim=sim,
                metrics=self.prediction_metrics,
                resolution='1D',
                datetime_coord='time',
            )
        except RuntimeError as err:
            print(f"Could not compute metrics for {gauge_id}: {err}")
            return None

        sanitized_metrics: Dict[str, float] = {}
        for key, value in metric_values.items():
            if value is None:
                sanitized_metrics[key] = None
            elif isinstance(value, float):
                sanitized_metrics[key] = float(value) if not np.isnan(value) else None
            else:
                sanitized_metrics[key] = float(value)

        sanitized_metrics['gauge_id'] = gauge_id
        return sanitized_metrics

    def _evaluate_extremes_from_dataset(
        self, dataset: xr.Dataset, gauge_id: str
    ) -> Optional[Dict[str, object]]:
        if metrics_utils.OBS_VARIABLE not in dataset.data_vars:
            return None

        sim_variable_name = self._get_simulation_variable_name(dataset)
        sim = dataset[sim_variable_name]
        obs = dataset[metrics_utils.OBS_VARIABLE]

        if 'lead_time' in sim.dims:
            sim = sim.sel(lead_time=0).squeeze(drop=True)
        if 'lead_time' in obs.dims:
            obs = obs.sel(lead_time=0).squeeze(drop=True)

        sim_series = sim.to_pandas().dropna()
        obs_series = obs.to_pandas().dropna()

        if sim_series.empty or obs_series.empty:
            return None

        evaluation = self.evaluate_extremes(gauge_id, obs_series, sim_series)
        return evaluation

    def evaluate_extremes(
        self,
        gauge_id: str,
        obs_series: pd.Series,
        sim_series: pd.Series,
        return_periods: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, object]]:
        """Evaluate extreme-flow behaviour and persist diagnostics."""
        obs_series = obs_series.sort_index()
        sim_series = sim_series.sort_index()

        common_index = obs_series.index.intersection(sim_series.index)
        if len(common_index) < 2:
            return None

        obs_series = obs_series.loc[common_index]
        sim_series = sim_series.loc[common_index]

        temporal_resolution = self._infer_temporal_resolution(common_index)

        try:
            obs_rpc = return_period_calculator.ReturnPeriodCalculator(
                hydrograph_series=obs_series,
                hydrograph_series_frequency=temporal_resolution,
                use_simple_fitting=True,
                verbose=False,
            )
            sim_rpc = return_period_calculator.ReturnPeriodCalculator(
                hydrograph_series=sim_series,
                hydrograph_series_frequency=temporal_resolution,
                use_simple_fitting=True,
                verbose=False,
            )
        except rpc_exceptions.NotEnoughDataError:
            print(f"Not enough data to compute return periods for {gauge_id}.")
            return None

        if return_periods is None:
            return_periods = np.array([2, 5, 10, 20, 50, 100])

        obs_flows = obs_rpc.flow_values_from_return_periods(return_periods)
        sim_flows = sim_rpc.flow_values_from_return_periods(return_periods)
        bias = sim_flows - obs_flows
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_bias = np.where(obs_flows != 0, (bias / obs_flows) * 100.0, np.nan)

        table = pd.DataFrame(
            {
                'return_period': return_periods,
                'observed_flow': obs_flows,
                'predicted_flow': sim_flows,
                'bias': bias,
                'relative_bias_pct': relative_bias,
            }
        )

        table_path = self.extreme_tables_path / f'{gauge_id}_return_periods.csv'
        table.to_csv(table_path, index=False)

        peak_obs = obs_series.max()
        peak_sim = sim_series.max()
        peak_bias = float(peak_sim - peak_obs)
        peak_relative_bias = float((peak_bias / peak_obs) * 100.0) if peak_obs != 0 else np.nan

        metrics_summary: Dict[str, float] = {
            'peak_bias': peak_bias,
            'peak_relative_bias_pct': None if np.isnan(peak_relative_bias) else peak_relative_bias,
        }

        for rp, rp_bias, rp_rel_bias in zip(return_periods, bias, relative_bias):
            metrics_summary[f'return_period_{int(rp)}_bias'] = float(rp_bias)
            metrics_summary[f'return_period_{int(rp)}_relative_bias_pct'] = (
                None if np.isnan(rp_rel_bias) else float(rp_rel_bias)
            )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(return_periods, obs_flows, marker='o', label='Observed')
        ax.plot(return_periods, sim_flows, marker='o', label='Predicted')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Return period (years)')
        ax.set_ylabel('Flow')
        ax.set_title(f'Return-period curves for {gauge_id}')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        figure_path = self.figures_path / f'{gauge_id}_return_period_curve.png'
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return {
            'metrics': metrics_summary,
            'table_path': table_path,
            'figure_path': figure_path,
        }

    def _infer_temporal_resolution(self, index: pd.Index) -> pd.Timedelta:
        if len(index) < 2:
            return pd.to_timedelta('1D')
        diffs = np.diff(index.values.astype('datetime64[ns]'))
        median_diff = np.median(diffs)
        return pd.to_timedelta(median_diff)

    def _write_metadata_file(self, filename: str, content: Dict[str, object]) -> None:
        metadata_file = self.metadata_path / filename
        with open(metadata_file, 'w') as f:
            json.dump(self._serialize_for_json(content), f, indent=2)

    def _serialize_for_json(self, value):
        if isinstance(value, dict):
            return {key: self._serialize_for_json(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._serialize_for_json(item) for item in value]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.to_datetime(value).isoformat()
        if isinstance(value, pathlib.Path):
            return str(value)
        return value

    def evaluate_predictions(self, gauge_ids: list[str], lead_time: int = 0) -> pd.DataFrame:
        """
        Evaluate model predictions using hydrological metrics.

        Args:
            gauge_ids: List of gauge IDs to evaluate
            lead_time: Lead time to evaluate (default: 0)

        Returns:
            DataFrame with metrics for each gauge
        """
        print(f"Evaluating predictions for {len(gauge_ids)} gauges at lead_time={lead_time}...")

        results = []

        for gauge_id in tqdm(gauge_ids, desc='Evaluating'):
            try:
                # Load prediction
                pred_file = self.predictions_path / f'{gauge_id}.nc'
                if not pred_file.exists():
                    continue

                ds_pred = xr.load_dataset(pred_file)

                # Load observation
                ds_obs = loading_utils.load_grdc_data()
                ds_obs_gauge = ds_obs.sel(gauge_id=gauge_id)

                # Extract data at specified lead time
                if 'sim' in ds_pred.data_vars:
                    sim = ds_pred['sim'].sel(lead_time=lead_time)
                else:
                    sim_var = list(ds_pred.data_vars.keys())[0]
                    sim = ds_pred[sim_var].sel(lead_time=lead_time)

                obs = ds_obs_gauge[metrics_utils.OBS_VARIABLE].sel(lead_time=lead_time)

                # Calculate metrics
                gauge_metrics = metrics.calculate_all_metrics(
                    obs=obs,
                    sim=sim,
                    resolution='1D',
                    datetime_coord='time'
                )

                gauge_metrics['gauge_id'] = gauge_id
                results.append(gauge_metrics)

            except Exception as e:
                print(f"Error evaluating {gauge_id}: {e}")
                continue

        results_df = pd.DataFrame(results)
        results_df.set_index('gauge_id', inplace=True)

        # Save results
        output_file = self.evaluation_path / f'evaluation_metrics_lead{lead_time}.csv'
        results_df.to_csv(output_file)
        print(f"Evaluation results saved to {output_file}")

        return results_df


if __name__ == "__main__":
    print("=" * 80)
    print("Refactored GNN-LSTM Hybrid Model for Hydrological Prediction")
    print("=" * 80)

    # Model parameters
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
        'use_simulated_met': True,  # Set to False when real met data is available
        'meteorology_file': None,   # Path to meteorology NetCDF file
    }

    # Initialize model
    model = StreamflowModel(
        model_params=model_params,
        experiment_name="gnn_lstm_refactored_v1"
    )

    # Get available gauges
    try:
        all_gauge_ids = model.data_loader.get_available_gauges()
        print(f"\nTotal available gauges: {len(all_gauge_ids)}")

        # Split into train/val/test
        n = len(all_gauge_ids)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        training_gauge_ids = all_gauge_ids[:train_end]
        validation_gauge_ids = all_gauge_ids[train_end:val_end]
        test_gauge_ids = all_gauge_ids[val_end:]

        print(f"Training gauges: {len(training_gauge_ids)}")
        print(f"Validation gauges: {len(validation_gauge_ids)}")
        print(f"Test gauges: {len(test_gauge_ids)}")

    except Exception as e:
        print(f"Error loading gauge IDs: {e}")
        print("Using dummy gauge IDs")
        training_gauge_ids = ['GRDC_6335020']
        validation_gauge_ids = ['GRDC_6335075']
        test_gauge_ids = ['GRDC_6335020']

    print("\n" + "=" * 80)
    print("To train the model, uncomment the following line:")
    print("# model.train(training_gauge_ids, validation_gauge_ids)")
    print("\nTo generate predictions:")
    print("# model.predict(test_gauge_ids)")
    print("\nTo evaluate predictions:")
    print("# results_df = model.evaluate_predictions(test_gauge_ids, lead_time=0)")
    print("=" * 80)
