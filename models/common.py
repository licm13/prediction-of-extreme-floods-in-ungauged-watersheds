"""
Common model architectures and utilities for hydrological prediction.

This module contains shared components used across different model implementations.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch


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
        """
        Args:
            static_feature_dim: Number of static features
            dynamic_feature_dim: Number of dynamic features
            gnn_hidden_dim: GNN hidden dimension
            rnn_hidden_dim: RNN hidden dimension
            rnn_num_layers: Number of RNN layers
            rnn_type: 'lstm' or 'gru'
            output_lead_times: Number of prediction lead times
            dropout: Dropout rate
        """
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

        # GNN layers with optional edge weights
        edge_weight = getattr(static_graph_batch, 'edge_weight', None)

        x_static = self.gnn_conv1(x_static, edge_index, edge_weight)
        x_static = self.gnn_bn1(x_static)
        x_static = self.relu(x_static)
        x_static = self.dropout(x_static)

        x_static = self.gnn_conv2(x_static, edge_index, edge_weight)
        x_static = self.gnn_bn2(x_static)
        x_static = self.relu(x_static)

        # Extract target node embeddings (one target node per graph)
        # Extract target node embeddings (one target node per graph)
        use_target_mask = hasattr(static_graph_batch, 'target_mask')
        if use_target_mask:
            target_mask = static_graph_batch.target_mask
            if target_mask.dtype != torch.bool:
                target_mask = target_mask.bool()
            gnn_embedding = x_static[target_mask]

        # If no target mask or if mask extraction failed, fall back to graph-level pooling
        if not use_target_mask or gnn_embedding.shape[0] != dynamic_features.shape[0]:
            gnn_embedding = global_mean_pool(x_static, batch_idx)

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


def collate_fn(batch):
    """
    Custom collate function for PyG Data objects.

    Filters out None samples and ensures consistent shapes across the batch.

    Args:
        batch: List of (rnn_input, gnn_graph, target) tuples

    Returns:
        Tuple of (rnn_batch, gnn_batch, target_batch) or (None, None, None) if batch is empty
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None, None, None

    expected_seq_len = batch[0][0].shape[0]
    expected_target_len = batch[0][2].shape[0]

    # Filter for consistent shapes
    filtered_batch = []
    for rnn_input, gnn_graph, target in batch:
        if rnn_input.shape[0] != expected_seq_len or target.shape[0] != expected_target_len:
            continue
        filtered_batch.append((rnn_input, gnn_graph, target))

    if len(filtered_batch) == 0:
        return None, None, None

    rnn_inputs, gnn_graphs, targets = zip(*filtered_batch)

    rnn_batch = torch.stack(rnn_inputs)
    gnn_batch = Batch.from_data_list(gnn_graphs)
    target_batch = torch.stack(targets)

    return rnn_batch, gnn_batch, target_batch
