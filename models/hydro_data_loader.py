"""
Improved data loading module for hydrological prediction.

This module provides efficient data loading and preprocessing for the GNN-LSTM hybrid model.
It replaces the simulated data approach with real meteorological data loading.
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional, Tuple, Dict
import pathlib
import sys
import os

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
notebooks_backend_dir = os.path.join(parent_dir, 'notebooks', 'backend')
sys.path.append(notebooks_backend_dir)

from backend import loading_utils
from backend import metrics_utils


class HydroDataLoader:
    """
    Centralized data loader for hydrological prediction.

    This class handles:
    - Loading GRDC observation data (targets)
    - Loading HydroATLAS static attributes (static features for GNN)
    - Loading meteorological forcings (dynamic features for RNN)
    - Aligning all data sources by gauge_id and time
    - Preprocessing and normalization
    """

    def __init__(self,
                 static_feature_dim: int = 50,
                 dynamic_feature_dim: int = 5,
                 meteorology_file: Optional[pathlib.Path] = None,
                 use_simulated_met: bool = True):
        """
        Initialize the data loader.

        Args:
            static_feature_dim: Number of static features to use
            dynamic_feature_dim: Number of dynamic features to use
            meteorology_file: Path to NetCDF file containing meteorological data
                             Expected structure: (time, gauge_id, features)
                             Features should include: precip, temp, pet, etc.
            use_simulated_met: If True and meteorology_file is None,
                              generate simulated met data (for testing only)
        """
        self.static_feature_dim = static_feature_dim
        self.dynamic_feature_dim = dynamic_feature_dim
        self.meteorology_file = meteorology_file
        self.use_simulated_met = use_simulated_met

        # Cache for loaded data
        self._grdc_data_cache = None
        self._meteorology_cache = None
        self._static_attrs_cache = {}

        # Statistics for normalization (computed on first load)
        self._static_mean = None
        self._static_std = None
        self._dynamic_mean = None
        self._dynamic_std = None

    def load_grdc_data(self) -> xr.Dataset:
        """Load GRDC observation data (cached)."""
        if self._grdc_data_cache is None:
            self._grdc_data_cache = loading_utils.load_grdc_data()
        return self._grdc_data_cache

    def load_meteorology_data(self) -> Optional[xr.Dataset]:
        """
        Load meteorological forcing data.

        Returns:
            xarray.Dataset with dimensions (time, gauge_id) and variables
            for each meteorological feature (precip, temp, pet, etc.)

            If meteorology_file is not provided and use_simulated_met is True,
            returns None (will generate per-gauge simulated data later).
        """
        if self._meteorology_cache is not None:
            return self._meteorology_cache

        if self.meteorology_file is not None and self.meteorology_file.exists():
            try:
                self._meteorology_cache = xr.load_dataset(self.meteorology_file)
                print(f"Loaded meteorology data from {self.meteorology_file}")
                return self._meteorology_cache
            except Exception as e:
                print(f"Error loading meteorology file: {e}")
                if not self.use_simulated_met:
                    raise
                print("Falling back to simulated meteorology data")

        # If no meteorology file provided or failed to load
        if not self.use_simulated_met:
            raise ValueError(
                "No meteorology file provided and use_simulated_met=False. "
                "Please provide a valid meteorology_file or set use_simulated_met=True."
            )

        # Return None - will generate simulated data per gauge
        return None

    def load_static_attributes(self, gauge_id: str) -> pd.DataFrame:
        """
        Load static basin attributes for a gauge.

        Args:
            gauge_id: Gauge identifier (e.g., 'GRDC_6335020')

        Returns:
            DataFrame with static attributes for the gauge
        """
        if gauge_id in self._static_attrs_cache:
            return self._static_attrs_cache[gauge_id]

        try:
            attributes_df = loading_utils.load_attributes_file(gauges=[gauge_id])
            self._static_attrs_cache[gauge_id] = attributes_df
            return attributes_df
        except Exception as e:
            print(f"Error loading attributes for {gauge_id}: {e}")
            return None

    def _generate_simulated_meteorology(self,
                                        time_index: pd.DatetimeIndex,
                                        gauge_id: str) -> pd.DataFrame:
        """
        Generate simulated meteorological data for testing.

        NOTE: This is for demonstration purposes only. In production,
        you should load real meteorological data.

        Args:
            time_index: Time index to generate data for
            gauge_id: Gauge ID (used for reproducible random seed)

        Returns:
            DataFrame with simulated meteorological features
        """
        np.random.seed(hash(gauge_id) % (2**32))

        dynamic_features = pd.DataFrame(
            {
                'precip': np.abs(np.random.randn(len(time_index)) * 5 + 2),
                'temp': np.random.randn(len(time_index)) * 10 + 15,
                'pet': np.abs(np.random.randn(len(time_index)) * 2 + 3),
                'soil_moisture': np.random.rand(len(time_index)) * 0.5 + 0.2,
                'snow': np.abs(np.random.randn(len(time_index)) * 3),
            },
            index=time_index
        )

        return dynamic_features

    def prepare_data_for_gauge(self, gauge_id: str) -> Tuple[
        Optional[pd.DataFrame],  # dynamic_features
        Optional[np.ndarray],     # static_features_array
        Optional[pd.Series]       # targets
    ]:
        """
        Load and preprocess all data for a single gauge.

        This method:
        1. Loads GRDC observations (targets)
        2. Loads static basin attributes
        3. Loads or generates meteorological forcings
        4. Aligns all data by time index
        5. Handles missing values
        6. Returns preprocessed data

        Args:
            gauge_id: Gauge identifier

        Returns:
            Tuple of (dynamic_features, static_features_array, targets)
            - dynamic_features: DataFrame (time, num_dynamic_features)
            - static_features_array: numpy array (num_static_features,)
            - targets: Series (time,) with observed discharge

            Returns (None, None, None) if data cannot be loaded.
        """
        try:
            # ===== 1. Load GRDC observations (targets) =====
            grdc_dataset = self.load_grdc_data()

            if gauge_id not in grdc_dataset.gauge_id.values:
                print(f"Gauge {gauge_id} not found in GRDC data.")
                return None, None, None

            gauge_grdc_data = grdc_dataset.sel(gauge_id=gauge_id)

            # Extract observed discharge at lead_time=0 (synchronous observation)
            targets = gauge_grdc_data[metrics_utils.OBS_VARIABLE].sel(lead_time=0).to_pandas()

            # Clean invalid values
            targets = targets.replace([np.inf, -np.inf], np.nan)

            # ===== 2. Load static basin attributes =====
            static_attributes_df = self.load_static_attributes(gauge_id)

            if static_attributes_df is None or static_attributes_df.empty:
                print(f"Warning: Could not load attributes for {gauge_id}")
                print("Using random static features.")
                static_features_array = np.random.randn(self.static_feature_dim)
            else:
                # Handle missing values
                static_attributes_df = static_attributes_df.fillna(static_attributes_df.mean())
                static_attributes_df = static_attributes_df.fillna(0)

                # Convert to numpy array
                static_features_array = static_attributes_df.values[0]

                # Adjust dimensionality
                if len(static_features_array) < self.static_feature_dim:
                    padding = np.zeros(self.static_feature_dim - len(static_features_array))
                    static_features_array = np.concatenate([static_features_array, padding])
                elif len(static_features_array) > self.static_feature_dim:
                    static_features_array = static_features_array[:self.static_feature_dim]

            # Normalize static features
            static_features_array = self._normalize_static_features(static_features_array)

            # ===== 3. Load meteorological forcings =====
            time_index = targets.index

            meteorology_data = self.load_meteorology_data()

            if meteorology_data is not None:
                # Load real meteorology data
                dynamic_features = self._extract_meteorology_for_gauge(
                    meteorology_data, gauge_id, time_index
                )
            else:
                # Generate simulated data (fallback)
                print(f"Using simulated meteorology for {gauge_id}")
                dynamic_features = self._generate_simulated_meteorology(time_index, gauge_id)

            # Ensure correct number of features
            if dynamic_features.shape[1] != self.dynamic_feature_dim:
                dynamic_features = self._adjust_dynamic_features_dim(dynamic_features)

            # Normalize dynamic features
            dynamic_features = self._normalize_dynamic_features(dynamic_features)

            return dynamic_features, static_features_array, targets

        except Exception as e:
            print(f"Error loading data for gauge {gauge_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _extract_meteorology_for_gauge(self,
                                       meteorology_data: xr.Dataset,
                                       gauge_id: str,
                                       time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Extract meteorology data for a specific gauge and align to time index.

        Args:
            meteorology_data: xarray Dataset with meteorology data
            gauge_id: Gauge identifier
            time_index: Target time index to align to

        Returns:
            DataFrame with meteorological features aligned to time_index
        """
        # Select data for this gauge
        if gauge_id not in meteorology_data.gauge_id.values:
            print(f"Warning: {gauge_id} not in meteorology data, using simulated data")
            return self._generate_simulated_meteorology(time_index, gauge_id)

        gauge_meteorology = meteorology_data.sel(gauge_id=gauge_id)

        # Convert to DataFrame
        meteorology_df = gauge_meteorology.to_dataframe()

        # Align to target time index (using forward fill for missing times)
        meteorology_df = meteorology_df.reindex(time_index, method='ffill')

        return meteorology_df

    def _adjust_dynamic_features_dim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust dynamic features to match expected dimensionality."""
        if df.shape[1] < self.dynamic_feature_dim:
            # Add random features (ideally should use meaningful imputation)
            for i in range(self.dynamic_feature_dim - df.shape[1]):
                df[f'extra_{i}'] = np.random.randn(len(df))
        elif df.shape[1] > self.dynamic_feature_dim:
            # Truncate
            df = df.iloc[:, :self.dynamic_feature_dim]
        return df

    def _normalize_static_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize static features using z-score normalization."""
        return (features - np.mean(features)) / (np.std(features) + 1e-8)

    def _normalize_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dynamic features.

        In production, you should compute statistics on training set
        and apply them consistently to validation/test sets.
        """
        return (df - df.mean()) / (df.std() + 1e-8)

    def get_available_gauges(self) -> list:
        """Get list of all available gauge IDs in GRDC data."""
        grdc_dataset = self.load_grdc_data()
        return grdc_dataset.gauge_id.values.tolist()


class PreprocessedHydroDataset:
    """
    Precomputed dataset that loads all data upfront.

    This is more efficient than loading data in __getitem__ for each sample.
    """

    def __init__(self,
                 data_loader: HydroDataLoader,
                 gauge_ids: list[str],
                 seq_length: int = 365,
                 pred_length: int = 10):
        """
        Initialize dataset by loading all data upfront.

        Args:
            data_loader: HydroDataLoader instance
            gauge_ids: List of gauge IDs to include
            seq_length: Sequence length for RNN input
            pred_length: Number of future time steps to predict
        """
        self.data_loader = data_loader
        self.gauge_ids = gauge_ids
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Preload all data
        print("Preloading data for all gauges...")
        self.gauge_data = {}
        valid_gauges = []

        for gauge_id in gauge_ids:
            dynamic_feat, static_feat, targets = data_loader.prepare_data_for_gauge(gauge_id)

            if dynamic_feat is not None and static_feat is not None and targets is not None:
                # Clean data
                valid_mask = ~(dynamic_feat.isna().any(axis=1) | targets.isna())
                dynamic_feat_clean = dynamic_feat[valid_mask]
                targets_clean = targets[valid_mask]

                if len(targets_clean) >= seq_length + pred_length:
                    self.gauge_data[gauge_id] = {
                        'dynamic': dynamic_feat_clean,
                        'static': static_feat,
                        'targets': targets_clean
                    }
                    valid_gauges.append(gauge_id)

        self.valid_gauge_ids = valid_gauges
        print(f"Successfully loaded data for {len(valid_gauges)}/{len(gauge_ids)} gauges")

    def get_num_valid_windows_per_gauge(self, gauge_id: str) -> int:
        """Calculate number of valid time windows for a gauge."""
        if gauge_id not in self.gauge_data:
            return 0
        targets_len = len(self.gauge_data[gauge_id]['targets'])
        return max(0, targets_len - self.seq_length - self.pred_length + 1)

    def get_total_windows(self) -> int:
        """Get total number of valid time windows across all gauges."""
        return sum(self.get_num_valid_windows_per_gauge(g) for g in self.valid_gauge_ids)
