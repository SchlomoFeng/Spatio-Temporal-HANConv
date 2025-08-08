"""
Dataset and DataLoader classes for Steam Pipeline Anomaly Detection

This module provides:
1. StreamPipelineDataset for handling time series and graph data
2. Custom collate function for batching heterogeneous graph data
3. Data loaders for training, validation, and testing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Batch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import yaml

from src.data.preprocessing import (
    PipelineTopologyParser, SensorDataProcessor, HeteroGraphBuilder, 
    DataSplitter, create_hetero_data
)


class StreamPipelineDataset(Dataset):
    """Dataset class for stream pipeline time series and graph data"""
    
    def __init__(self, sensor_data: pd.DataFrame, nodes_df: pd.DataFrame, 
                 edges_df: pd.DataFrame, window_size: int, stride: int,
                 scaler=None, mode: str = 'train'):
        """
        Initialize dataset
        Args:
            sensor_data: Preprocessed sensor DataFrame
            nodes_df: Node information DataFrame  
            edges_df: Edge information DataFrame
            window_size: Time window size for LSTM
            stride: Sliding window stride
            scaler: Fitted scaler for normalization
            mode: 'train', 'val', or 'test'
        """
        self.sensor_data = sensor_data
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.window_size = window_size
        self.stride = stride
        self.scaler = scaler
        self.mode = mode
        
        # Extract sensor columns (exclude timestamp)
        self.sensor_columns = [col for col in sensor_data.columns if col != 'timestamp']
        
        # Identify Stream nodes
        self.stream_nodes = nodes_df[nodes_df['node_type'] == 'Stream'].copy()
        self.stream_node_indices = self.stream_nodes['node_index'].values
        
        # Create time windows
        self._create_time_windows()
        
        # Build heterogeneous graph structure
        self._build_graph_structure()
        
        print(f"Dataset created: {len(self.time_windows)} samples, mode={mode}")
        
    def _create_time_windows(self):
        """Create sliding time windows from sensor data"""
        sensor_values = self.sensor_data[self.sensor_columns].values
        timestamps = self.sensor_data['timestamp'].values
        
        self.time_windows = []
        self.window_timestamps = []
        
        for i in range(0, len(sensor_values) - self.window_size + 1, self.stride):
            window = sensor_values[i:i + self.window_size]
            timestamp = timestamps[i + self.window_size - 1]  # Last timestamp in window
            
            self.time_windows.append(window)
            self.window_timestamps.append(timestamp)
        
        self.time_windows = np.array(self.time_windows)
        self.window_timestamps = np.array(self.window_timestamps)
        
    def _build_graph_structure(self):
        """Build static heterogeneous graph structure"""
        node_types = ['Stream', 'VavlePro', 'Mixer', 'Tee']
        graph_builder = HeteroGraphBuilder(node_types)
        
        # Create node features
        self.node_features = graph_builder.create_node_features(self.nodes_df)
        
        # Create edge indices
        self.edge_indices = graph_builder.create_edge_indices(self.edges_df, self.nodes_df)
        
        # Store node type mappings
        self.node_type_mapping = {}
        for node_type in node_types:
            type_nodes = self.nodes_df[self.nodes_df['node_type'] == node_type]
            self.node_type_mapping[node_type] = type_nodes['node_index'].values
    
    def __len__(self) -> int:
        return len(self.time_windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample
        Args:
            idx: Sample index
        Returns:
            Dictionary containing time series data and graph structure
        """
        # Get time window data
        window_data = self.time_windows[idx]  # Shape: (window_size, num_sensors)
        timestamp = self.window_timestamps[idx]
        
        # For Stream nodes, we use the time series data
        # For now, assume all Stream nodes observe all sensors (simplified)
        num_stream_nodes = len(self.stream_nodes)
        
        # Expand window data for each Stream node
        stream_sequences = np.tile(window_data[np.newaxis, :, :], (num_stream_nodes, 1, 1))
        stream_sequences = torch.tensor(stream_sequences, dtype=torch.float32)
        
        # Get current sensor readings (last time step) for reconstruction target
        current_readings = window_data[-1]  # Shape: (num_sensors,)
        
        # Create target for each Stream node (simplified - same for all)
        stream_targets = np.tile(current_readings[np.newaxis, :], (num_stream_nodes, 1))
        stream_targets = torch.tensor(stream_targets, dtype=torch.float32)
        
        # Static node features
        static_features = {}
        for node_type, features in self.node_features.items():
            if node_type != 'Stream':
                static_features[node_type] = features.clone()
        
        sample = {
            'stream_sequences': stream_sequences,
            'static_features': static_features,
            'edge_index_dict': {k: v.clone() for k, v in self.edge_indices.items()},
            'targets': {
                'sensor_readings': stream_targets
            },
            'timestamp': timestamp,
            'stream_node_ids': self.stream_node_indices
        }
        
        return sample


def collate_hetero_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching heterogeneous graph data
    Args:
        batch: List of samples from StreamPipelineDataset
    Returns:
        Batched data dictionary
    """
    if len(batch) == 0:
        return {}
    
    # Since the graph structure is the same for all samples,
    # we can batch the time series data and targets
    batch_size = len(batch)
    
    # Stack stream sequences from all samples
    stream_sequences = torch.stack([sample['stream_sequences'] for sample in batch], dim=0)
    # Shape: (batch_size, num_stream_nodes, seq_len, features)
    
    # Stack targets
    targets = {
        'sensor_readings': torch.stack([sample['targets']['sensor_readings'] for sample in batch], dim=0)
    }
    # Shape: (batch_size, num_stream_nodes, features)
    
    # Take static features and edge indices from first sample (same for all)
    static_features = batch[0]['static_features']
    edge_index_dict = batch[0]['edge_index_dict']
    
    # Collect timestamps and node IDs
    timestamps = [sample['timestamp'] for sample in batch]
    stream_node_ids = batch[0]['stream_node_ids']  # Same for all samples
    
    return {
        'stream_sequences': stream_sequences,
        'static_features': static_features,
        'edge_index_dict': edge_index_dict,
        'targets': targets,
        'timestamps': timestamps,
        'stream_node_ids': stream_node_ids,
        'batch_size': batch_size
    }


class PipelineDataModule:
    """Data module for managing train/val/test datasets and loaders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        
        # Initialize components
        self.topology_parser = PipelineTopologyParser(self.data_config['blueprint_path'])
        self.sensor_processor = SensorDataProcessor(
            self.data_config['sensor_data_path'],
            self.data_config['scaler_type']
        )
        self.data_splitter = DataSplitter(
            self.data_config['train_ratio'],
            self.data_config['val_ratio'],
            self.data_config['test_ratio']
        )
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.nodes_df = None
        self.edges_df = None
        self.scaler = None
        
    def setup(self):
        """Setup datasets and preprocessing"""
        print("Setting up data module...")
        
        # 1. Process topology
        print("Processing pipeline topology...")
        nodeDF, edgeDF = self.topology_parser.extract_nodes_and_edges()
        self.nodes_df, node_id_to_index = self.topology_parser.process_nodes(nodeDF)
        self.edges_df = self.topology_parser.process_edges(edgeDF, node_id_to_index)
        
        # 2. Process sensor data
        print("Processing sensor data...")
        sensor_df = self.sensor_processor.load_sensor_data()
        sensor_df = self.sensor_processor.clean_data(sensor_df)
        
        # 3. Split data chronologically
        print("Splitting data...")
        train_df, val_df, test_df = self.data_splitter.split_time_series(sensor_df)
        
        # 4. Fit scaler on training data
        print("Fitting scaler on training data...")
        self.sensor_processor.create_scaler(train_df)
        self.scaler = self.sensor_processor.scaler
        
        # 5. Normalize data
        train_df = self.sensor_processor.normalize_data(train_df)
        val_df = self.sensor_processor.normalize_data(val_df)
        test_df = self.sensor_processor.normalize_data(test_df)
        
        # 6. Create datasets
        print("Creating datasets...")
        window_size = self.data_config['window_size']
        stride = self.data_config['stride']
        
        self.train_dataset = StreamPipelineDataset(
            train_df, self.nodes_df, self.edges_df,
            window_size, stride, self.scaler, mode='train'
        )
        
        self.val_dataset = StreamPipelineDataset(
            val_df, self.nodes_df, self.edges_df,
            window_size, stride, self.scaler, mode='val'
        )
        
        self.test_dataset = StreamPipelineDataset(
            test_df, self.nodes_df, self.edges_df,
            window_size, stride, self.scaler, mode='test'
        )
        
        print("Data module setup completed!")
        
    def train_dataloader(self) -> DataLoader:
        """Create training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=collate_hetero_batch
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=collate_hetero_batch
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=collate_hetero_batch
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if self.train_dataset is None:
            return {}
        
        return {
            'num_nodes': len(self.nodes_df),
            'num_edges': len(self.edges_df),
            'node_type_counts': self.nodes_df['node_type'].value_counts().to_dict(),
            'num_sensors': len(self.sensor_processor.sensor_columns),
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'test_samples': len(self.test_dataset),
            'window_size': self.data_config['window_size'],
            'stride': self.data_config['stride']
        }


def create_data_module(config_path: str) -> PipelineDataModule:
    """Factory function to create data module from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_module = PipelineDataModule(config)
    data_module.setup()
    
    return data_module


if __name__ == "__main__":
    # Test data module
    print("Testing data module...")
    
    # Load config
    config_path = "../../config/config.yaml"
    try:
        data_module = create_data_module(config_path)
        
        # Print statistics
        stats = data_module.get_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test data loaders
        print("\nTesting data loaders...")
        train_loader = data_module.train_dataloader()
        
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        print(f"Sample batch keys: {list(sample_batch.keys())}")
        
        if 'stream_sequences' in sample_batch:
            print(f"Stream sequences shape: {sample_batch['stream_sequences'].shape}")
        
        if 'targets' in sample_batch and 'sensor_readings' in sample_batch['targets']:
            print(f"Target sensor readings shape: {sample_batch['targets']['sensor_readings'].shape}")
        
        print("Data module test completed successfully!")
        
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Please run from project root directory")