# GNN Architecture Improvements for Hydrological Prediction

## Current Architecture Limitations

The current `HybridGNN_RNN` implementation treats each watershed (gauge) as an **isolated node** in its own single-node graph. While this approach works, it doesn't leverage one of GNN's key advantages: **learning from graph structure**.

### Current Per-Gauge Approach:
```
Gauge A: [Node_A] (no edges)
Gauge B: [Node_B] (no edges)
Gauge C: [Node_C] (no edges)
```

Each gauge is processed independently. The GNN essentially acts as a multi-layer perceptron (MLP) on static features.

---

## Proposed Improvement: River Network Graph

### Concept
Connect gauges based on **upstream-downstream relationships** in the river network. This allows the model to:

1. **Learn spatial routing**: How flow propagates from upstream to downstream
2. **Share information** between connected watersheds
3. **Handle ungauged basins**: Predict for nodes with no observations by aggregating information from neighbors
4. **Capture basin dependencies**: Understand how upstream precipitation affects downstream flow

### Proposed Graph Structure:
```
         [Node_A] (upstream)
              |
              v
         [Node_B] (midstream)
              |
              v
         [Node_C] (downstream)
```

With edges: A→B, B→C representing flow direction.

---

## Implementation Approaches

### Approach 1: HydroATLAS Basin Hierarchy

HydroATLAS provides basin IDs and hierarchical relationships through `HYBAS_ID` (HydroBASINS).

**Data needed:**
- `HYBAS_ID`: Unique basin identifier
- `NEXT_DOWN`: ID of the next downstream basin
- `MAIN_BAS`: Main basin ID for identifying the same river system

**Implementation:**
```python
def build_river_network_graph(gauges: list[str]) -> Data:
    """
    Build a graph connecting gauges by river network topology.

    Returns:
        PyG Data object with:
        - x: Static features for all gauges (num_gauges, static_feature_dim)
        - edge_index: Connections based on upstream-downstream (2, num_edges)
        - gauge_ids: List of gauge IDs corresponding to nodes
    """
    # Load basin info
    hybas_info = loading_utils.load_hydroatlas_info_file()

    # Map gauge_id to HYBAS_ID
    gauge_to_hybas = {}
    for gauge in gauges:
        # Extract HYBAS_ID for this gauge
        # This requires a mapping file or extracting from metadata
        hybas_id = get_hybas_id_for_gauge(gauge)
        gauge_to_hybas[gauge] = hybas_id

    # Build adjacency based on NEXT_DOWN relationships
    edge_list = []
    for i, gauge_i in enumerate(gauges):
        hybas_i = gauge_to_hybas[gauge_i]
        next_down = hybas_info.loc[hybas_i, 'NEXT_DOWN']

        # Find if next_down is in our gauge list
        for j, gauge_j in enumerate(gauges):
            hybas_j = gauge_to_hybas[gauge_j]
            if hybas_j == next_down:
                # Add directed edge: i -> j (upstream to downstream)
                edge_list.append([i, j])

    # Convert to PyG format
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Load static features for all gauges
    static_features = []
    for gauge in gauges:
        attrs_df = loading_utils.load_attributes_file(gauges=[gauge])
        static_features.append(attrs_df.values[0])

    x = torch.FloatTensor(np.array(static_features))

    return Data(x=x, edge_index=edge_index), gauges
```

### Approach 2: Distance-Based Graph

Connect gauges within a certain geographic distance.

**Implementation:**
```python
def build_distance_based_graph(gauges: list[str], max_distance_km: float = 100) -> Data:
    """
    Build graph by connecting nearby gauges.
    """
    # Load gauge locations
    gauge_countries = loading_utils.load_gauge_country_file()

    # Calculate pairwise distances
    coords = []
    for gauge in gauges:
        lat = gauge_countries.loc[gauge, 'latitude']
        lon = gauge_countries.loc[gauge, 'longitude']
        coords.append([lat, lon])

    coords = np.array(coords)

    # Build edges for gauges within max_distance
    edge_list = []
    for i in range(len(gauges)):
        for j in range(i+1, len(gauges)):
            dist = haversine_distance(coords[i], coords[j])
            if dist < max_distance_km:
                # Add bidirectional edge
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Load static features
    static_features = load_all_static_features(gauges)
    x = torch.FloatTensor(static_features)

    return Data(x=x, edge_index=edge_index), gauges
```

### Approach 3: Hybrid - River Network + K-Nearest Neighbors

Combine both approaches:
1. Connect by river network topology (primary)
2. Add edges to k-nearest neighbors (secondary, for information sharing)

---

## Modified Model Architecture

### Updated HybridGNN_RNN for Graph-Level Prediction

```python
class GraphBasedHybridGNN_RNN(nn.Module):
    """
    GNN-RNN model that processes multiple gauges in a connected graph.
    """

    def __init__(self,
                 static_feature_dim: int,
                 dynamic_feature_dim: int,
                 gnn_hidden_dim: int = 64,
                 rnn_hidden_dim: int = 128,
                 num_gnn_layers: int = 3,  # More layers for message passing
                 rnn_num_layers: int = 2,
                 output_lead_times: int = 10,
                 dropout: float = 0.2):
        super().__init__()

        # Multi-layer GNN for message passing
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(static_feature_dim, gnn_hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))

        self.gnn_bns = nn.ModuleList([
            nn.BatchNorm1d(gnn_hidden_dim) for _ in range(num_gnn_layers)
        ])

        # RNN for each node
        self.rnn = nn.LSTM(
            input_size=dynamic_feature_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0
        )

        # Fusion layers (per node)
        fusion_input_dim = gnn_hidden_dim + rnn_hidden_dim
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.output_fc = nn.Linear(64, output_lead_times)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, dynamic_features_dict, static_graph):
        """
        Args:
            dynamic_features_dict: Dict mapping node_idx -> (seq_len, dynamic_feature_dim)
            static_graph: PyG Data object with full graph structure

        Returns:
            Dict mapping node_idx -> predictions (output_lead_times,)
        """
        # === GNN: Process static graph ===
        x = static_graph.x
        edge_index = static_graph.edge_index

        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.gnn_bns)):
            x = gnn_layer(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            if i < len(self.gnn_layers) - 1:  # No dropout on last layer
                x = self.dropout(x)

        gnn_embeddings = x  # (num_nodes, gnn_hidden_dim)

        # === RNN: Process dynamic features for each node ===
        predictions = {}
        for node_idx, dynamic_feat in dynamic_features_dict.items():
            # dynamic_feat: (seq_len, dynamic_feature_dim)
            dynamic_feat = dynamic_feat.unsqueeze(0)  # (1, seq_len, dynamic_feature_dim)

            rnn_out, _ = self.rnn(dynamic_feat)
            rnn_embedding = rnn_out[:, -1, :]  # (1, rnn_hidden_dim)

            # Get GNN embedding for this node
            gnn_emb = gnn_embeddings[node_idx].unsqueeze(0)  # (1, gnn_hidden_dim)

            # Fusion
            fused = torch.cat([gnn_emb, rnn_embedding], dim=1)

            x_fused = self.fusion_fc1(fused)
            x_fused = self.fusion_bn(x_fused)
            x_fused = self.relu(x_fused)
            x_fused = self.dropout(x_fused)

            x_fused = self.fusion_fc2(x_fused)
            x_fused = self.relu(x_fused)
            x_fused = self.dropout(x_fused)

            output = self.output_fc(x_fused)  # (1, output_lead_times)

            predictions[node_idx] = output.squeeze(0)

        return predictions
```

---

## Training Strategy with Graph Structure

### Option 1: Graph-Level Batching
Process the entire graph (or subgraphs) at once.

```python
# Create one large graph with all training gauges
full_graph, gauge_list = build_river_network_graph(training_gauge_ids)

# Load time series for all gauges
dynamic_features_dict = {
    i: load_dynamic_features(gauge_list[i])
    for i in range(len(gauge_list))
}

# Forward pass
predictions = model(dynamic_features_dict, full_graph)
```

### Option 2: Mini-Batch Subgraphing
Sample subgraphs (e.g., a river basin) for each mini-batch.

```python
# Sample a subset of connected gauges
subgraph_gauges = sample_river_basin(all_gauges)
subgraph, gauge_subset = build_river_network_graph(subgraph_gauges)

# Load dynamic features for subset
dynamic_features_dict = {...}

# Forward pass on subgraph
predictions = model(dynamic_features_dict, subgraph)
```

---

## Expected Benefits

1. **Improved Generalization**: Model learns spatial patterns applicable to ungauged basins
2. **Better Flood Propagation**: Understands how upstream events affect downstream flow
3. **Reduced Overfitting**: Shared information across connected nodes acts as regularization
4. **Scalability**: Can predict for new gauges added to the graph without retraining from scratch

---

## Recommended Next Steps

1. **Extract Basin Connectivity**:
   - Parse HydroATLAS `NEXT_DOWN` relationships
   - Create gauge_id ↔ HYBAS_ID mapping

2. **Implement Graph Construction**:
   - Start with `build_river_network_graph()` using Approach 1
   - Validate graph structure (visualize if possible)

3. **Test Single Basin**:
   - Select one river system with multiple gauges
   - Train graph-based model and compare with per-gauge model

4. **Scale Up**:
   - Expand to multiple basins
   - Experiment with different GNN layers (GAT, GraphSAGE)

---

## Code Integration

To use the improved architecture, modify the data loader to:

1. Load all training gauges together
2. Build the river network graph once
3. Sample subgraphs or use the full graph during training

Example:
```python
# In AdvancedModel.__init__()
self.river_graph, self.graph_gauge_list = build_river_network_graph(all_gauge_ids)

# In train()
# Instead of per-gauge sampling, sample from graph
for epoch in range(num_epochs):
    for subgraph_gauges in sample_subgraphs(self.river_graph):
        # Train on subgraph
        ...
```

---

## References

- **HydroATLAS**: Linke et al. (2019) "Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution"
- **GNN for Hydrology**: Fang et al. (2022) "Evaluating graph neural networks for spatial streamflow prediction"
- **Graph Construction**: Cheng et al. (2023) "Graph neural networks for river flow forecasting"

