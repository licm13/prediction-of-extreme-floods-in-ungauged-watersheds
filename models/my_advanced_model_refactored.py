"""
Refactored GNN-LSTM Hybrid Model for Hydrological Prediction.

This refactored version:
1. Uses real data loading instead of simulated data
2. Improves dataset efficiency with preprocessing
3. Integrates evaluation metrics
4. Provides better model evaluation capabilities
"""

import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
from tqdm import tqdm
from typing import Optional, Dict, Tuple
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


# ==================== Model Architecture ====================

class HybridGNN_RNN(nn.Module):
    """
    Hybrid GNN-RNN model for hydrological prediction.

    - GNN: Processes static watershed attributes
    - RNN: Processes dynamic meteorological time series
    - Fusion: Combines both for streamflow prediction
    """

    def __init__(self,
                 static_feature_dim: int,
                 dynamic_feature_dim: int,
                 gnn_hidden_dim: int = 64,
                 rnn_hidden_dim: int = 128,
                 rnn_num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 output_lead_times: int = 10,
                 dropout: float = 0.2):
        super(HybridGNN_RNN, self).__init__()

        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.output_lead_times = output_lead_times

        # GNN - Process static watershed attributes
        self.gnn_conv1 = GCNConv(static_feature_dim, gnn_hidden_dim)
        self.gnn_conv2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.gnn_bn1 = nn.BatchNorm1d(gnn_hidden_dim)
        self.gnn_bn2 = nn.BatchNorm1d(gnn_hidden_dim)

        # RNN - Process dynamic meteorological inputs
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=dynamic_feature_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=dropout if rnn_num_layers > 1 else 0.0
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=dynamic_feature_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=dropout if rnn_num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Fusion layers
        fusion_input_dim = gnn_hidden_dim + rnn_hidden_dim
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_fc2 = nn.Linear(128, 64)

        # Output layer
        self.output_fc = nn.Linear(64, output_lead_times)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, dynamic_features, static_graph_batch):
        """
        Forward pass.

        Args:
            dynamic_features: (batch_size, seq_len, dynamic_feature_dim)
            static_graph_batch: PyG Batch object with static watershed attributes

        Returns:
            (batch_size, output_lead_times) predictions
        """
        # ===== GNN Part =====
        x_static = static_graph_batch.x
        edge_index = static_graph_batch.edge_index
        batch_idx = static_graph_batch.batch

        x_static = self.gnn_conv1(x_static, edge_index)
        x_static = self.gnn_bn1(x_static)
        x_static = self.relu(x_static)
        x_static = self.dropout(x_static)

        x_static = self.gnn_conv2(x_static, edge_index)
        x_static = self.gnn_bn2(x_static)
        x_static = self.relu(x_static)

        gnn_embedding = x_static

        # ===== RNN Part =====
        rnn_out, _ = self.rnn(dynamic_features)
        rnn_embedding = rnn_out[:, -1, :]

        # ===== Fusion =====
        fused = torch.cat([gnn_embedding, rnn_embedding], dim=1)

        x = self.fusion_fc1(fused)
        x = self.fusion_bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fusion_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        output = self.output_fc(x)

        return output


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


def collate_fn(batch):
    """Custom collate function for PyG Data objects."""
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None, None, None

    rnn_inputs = [item[0] for item in batch]
    gnn_graphs = [item[1] for item in batch]
    targets = [item[2] for item in batch]

    rnn_batch = torch.stack(rnn_inputs)
    gnn_batch = Batch.from_data_list(gnn_graphs)
    target_batch = torch.stack(targets)

    return rnn_batch, gnn_batch, target_batch


# ==================== AdvancedModel Class ====================

class AdvancedModel:
    """
    Refactored advanced model with real data loading and evaluation.
    """

    def __init__(self, model_params: dict, experiment_name: str = "my_advanced_model_refactored"):
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

        loading_utils.create_remote_folder_if_necessary(self.output_path)
        loading_utils.create_remote_folder_if_necessary(self.checkpoint_path)
        loading_utils.create_remote_folder_if_necessary(self.metrics_path)

        print(f"Initialized AdvancedModel: {self.experiment_name}")
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

        print("Training complete!")
        print(f"Best loss: {best_loss:.4f}")

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
        Generate predictions and save as NetCDF files.

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

        for gauge_id in tqdm(prediction_gauge_ids, desc='Generating predictions'):
            # Prepare data
            dynamic_features, static_features, targets = self.data_loader.prepare_data_for_gauge(gauge_id)

            if dynamic_features is None:
                print(f"Skipping {gauge_id}: Could not load data")
                continue

            # Load template for time coordinates
            try:
                ds_template = loading_utils.load_google_model_for_one_gauge(
                    experiment='full_run',
                    gauge=gauge_id
                )
                if ds_template is None:
                    print(f"Skipping {gauge_id}: Cannot load template")
                    continue
            except:
                print(f"Skipping {gauge_id}: Error loading template")
                continue

            # Run inference
            predictions_array = self._run_inference_for_gauge(
                dynamic_features, static_features, targets, ds_template
            )

            # Save predictions
            self._save_predictions_netcdf(gauge_id, predictions_array, ds_template)

        print(f"Predictions saved to {self.output_path}")

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
        """Save predictions as NetCDF file."""
        if 'sim' in ds_template.data_vars:
            sim_variable_name = 'sim'
        elif metrics_utils.GOOGLE_VARIABLE in ds_template.data_vars:
            sim_variable_name = metrics_utils.GOOGLE_VARIABLE
        else:
            sim_variable_name = list(ds_template.data_vars.keys())[0]

        prediction_ds = xr.Dataset(
            {
                sim_variable_name: (
                    ["time", "lead_time"],
                    predictions_array,
                    {'description': f'Streamflow prediction by {self.experiment_name}'}
                )
            },
            coords={
                "time": ds_template['time'],
                "lead_time": ds_template['lead_time'],
                "gauge_id": gauge_id
            }
        )

        output_file_path = self.output_path / f'{gauge_id}.nc'
        prediction_ds.to_netcdf(output_file_path)

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
                pred_file = self.output_path / f'{gauge_id}.nc'
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
        results_df.to_csv(self.metrics_path / f'evaluation_metrics_lead{lead_time}.csv')
        print(f"Evaluation results saved to {self.metrics_path}")

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
    model = AdvancedModel(
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
