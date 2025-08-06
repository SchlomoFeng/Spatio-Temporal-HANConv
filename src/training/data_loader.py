"""
Data Loader
Creates data loaders for training, validation and testing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from typing import List, Dict, Tuple, Optional
import numpy as np
from ..data_preprocessing.graph_builder import GraphBuilder


class HeteroGraphDataset(Dataset):
    """Dataset for heterogeneous graph data"""
    
    def __init__(self, 
                 hetero_data_list: List[HeteroData],
                 sensor_data: torch.Tensor,
                 window_size: int = 60):
        """
        Initialize dataset
        
        Args:
            hetero_data_list: List of HeteroData objects
            sensor_data: Sensor readings [num_samples, num_sensors]
            window_size: Time window size
        """
        self.hetero_data_list = hetero_data_list
        self.sensor_data = sensor_data
        self.window_size = window_size
        
        # Ensure data length consistency
        min_length = min(len(hetero_data_list), len(sensor_data))
        self.hetero_data_list = hetero_data_list[:min_length]
        self.sensor_data = sensor_data[:min_length]
    
    def __len__(self):
        return len(self.hetero_data_list)
    
    def __getitem__(self, idx):
        """Get item at index"""
        hetero_data = self.hetero_data_list[idx]
        sensor_target = self.sensor_data[idx]
        
        return {
            'hetero_data': hetero_data,
            'sensor_target': sensor_target,
            'index': idx
        }


class TimeSeriesDataset(Dataset):
    """Time series dataset for sequential sensor data"""
    
    def __init__(self,
                 sensor_data: torch.Tensor,
                 window_size: int = 60,
                 stride: int = 1,
                 prediction_horizon: int = 1):
        """
        Initialize time series dataset
        
        Args:
            sensor_data: Sensor readings [num_timesteps, num_sensors]
            window_size: Input window size
            stride: Stride between windows
            prediction_horizon: Number of future steps to predict
        """
        self.sensor_data = sensor_data
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        
        # Calculate valid indices
        self.valid_indices = []
        for i in range(0, len(sensor_data) - window_size - prediction_horizon + 1, stride):
            self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get time series window at index"""
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        target_idx = end_idx + self.prediction_horizon - 1
        
        input_window = self.sensor_data[start_idx:end_idx]
        target = self.sensor_data[target_idx]
        
        return {
            'input_window': input_window,
            'target': target,
            'start_idx': start_idx
        }


def collate_hetero_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for heterogeneous graph batches
    
    Args:
        batch: List of batch items
        
    Returns:
        Collated batch dictionary
    """
    # Extract components
    hetero_data_list = [item['hetero_data'] for item in batch]
    sensor_targets = torch.stack([item['sensor_target'] for item in batch])
    indices = torch.tensor([item['index'] for item in batch])
    
    # For simplicity, we'll process samples individually
    # In a more sophisticated setup, we could batch the graphs
    return {
        'hetero_data_list': hetero_data_list,
        'sensor_targets': sensor_targets,
        'indices': indices,
        'batch_size': len(batch)
    }


class GraphDataLoader:
    """Data loader factory for graph-based datasets"""
    
    def __init__(self,
                 graph_builder: GraphBuilder,
                 batch_size: int = 32,
                 window_size: int = 60,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 num_workers: int = 0,
                 shuffle: bool = True):
        """
        Initialize graph data loader factory
        
        Args:
            graph_builder: GraphBuilder instance
            batch_size: Batch size
            window_size: Time window size
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle training data
        """
        self.graph_builder = graph_builder
        self.batch_size = batch_size
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.datasets = None
        self.dataloaders = None
    
    def prepare_datasets(self) -> Dict[str, HeteroGraphDataset]:
        """
        Prepare train/val/test datasets
        
        Returns:
            Dictionary of datasets
        """
        # Build complete dataset
        dataset_dict = self.graph_builder.build_dataset(
            window_size=self.window_size,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )
        
        # Get sensor data
        if self.graph_builder.sensor_data is None:
            self.graph_builder.sensor_data = self.graph_builder.sensor_cleaner.clean_sensor_data()
        
        sensor_data = self.graph_builder.sensor_data[self.graph_builder.sensor_cleaner.sensor_columns].values
        sensor_tensor = torch.tensor(sensor_data, dtype=torch.float32)
        
        # Create datasets
        datasets = {}
        start_idx = 0
        
        for split in ['train', 'val', 'test']:
            end_idx = start_idx + len(dataset_dict[split])
            split_sensor_data = sensor_tensor[start_idx:end_idx]
            
            datasets[split] = HeteroGraphDataset(
                hetero_data_list=dataset_dict[split],
                sensor_data=split_sensor_data,
                window_size=self.window_size
            )
            
            start_idx = end_idx
        
        self.datasets = datasets
        return datasets
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create data loaders
        
        Returns:
            Dictionary of data loaders
        """
        if self.datasets is None:
            self.prepare_datasets()
        
        dataloaders = {}
        
        for split, dataset in self.datasets.items():
            shuffle = self.shuffle if split == 'train' else False
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=collate_hetero_batch,
                pin_memory=torch.cuda.is_available()
            )
        
        self.dataloaders = dataloaders
        return dataloaders
    
    def get_dataloader(self, split: str = 'train') -> DataLoader:
        """
        Get specific data loader
        
        Args:
            split: Data split ('train', 'val', 'test')
            
        Returns:
            DataLoader for the specified split
        """
        if self.dataloaders is None:
            self.create_dataloaders()
        
        return self.dataloaders[split]
    
    def get_data_info(self) -> Dict:
        """
        Get dataset information
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.datasets is None:
            self.prepare_datasets()
        
        info = {
            'num_sensors': len(self.graph_builder.sensor_cleaner.sensor_columns),
            'window_size': self.window_size,
            'batch_size': self.batch_size,
            'splits': {}
        }
        
        for split, dataset in self.datasets.items():
            info['splits'][split] = {
                'size': len(dataset),
                'sensor_data_shape': dataset.sensor_data.shape
            }
        
        return info


class StreamingDataLoader:
    """Data loader for real-time streaming data"""
    
    def __init__(self,
                 graph_builder: GraphBuilder,
                 window_size: int = 60,
                 buffer_size: int = 1000):
        """
        Initialize streaming data loader
        
        Args:
            graph_builder: GraphBuilder instance
            window_size: Time window size
            buffer_size: Maximum buffer size for streaming data
        """
        self.graph_builder = graph_builder
        self.window_size = window_size
        self.buffer_size = buffer_size
        
        # Ring buffer for streaming data
        self.sensor_buffer = None
        self.timestamp_buffer = None
        self.buffer_idx = 0
        self.buffer_full = False
        
        # Initialize buffer
        num_sensors = len(self.graph_builder.sensor_cleaner.sensor_columns)
        self.sensor_buffer = torch.zeros(buffer_size, num_sensors)
        self.timestamp_buffer = torch.zeros(buffer_size)
    
    def add_sample(self, sensor_reading: torch.Tensor, timestamp: float = None):
        """
        Add new sensor reading to buffer
        
        Args:
            sensor_reading: New sensor reading
            timestamp: Optional timestamp
        """
        self.sensor_buffer[self.buffer_idx] = sensor_reading
        if timestamp is not None:
            self.timestamp_buffer[self.buffer_idx] = timestamp
        
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if self.buffer_idx == 0:
            self.buffer_full = True
    
    def get_current_window(self) -> Optional[HeteroData]:
        """
        Get current time window as HeteroData
        
        Returns:
            HeteroData object for current window or None if insufficient data
        """
        if not self.buffer_full and self.buffer_idx < self.window_size:
            return None
        
        # Get recent window
        if self.buffer_full:
            # Buffer is full, get last window_size samples
            if self.buffer_idx >= self.window_size:
                window_data = self.sensor_buffer[self.buffer_idx - self.window_size:self.buffer_idx]
            else:
                # Wrap around
                part1 = self.sensor_buffer[self.buffer_size - (self.window_size - self.buffer_idx):]
                part2 = self.sensor_buffer[:self.buffer_idx]
                window_data = torch.cat([part1, part2], dim=0)
        else:
            # Buffer not full yet
            window_data = self.sensor_buffer[max(0, self.buffer_idx - self.window_size):self.buffer_idx]
        
        # Create HeteroData object
        try:
            hetero_data = self.graph_builder.build_hetero_data(
                window_size=len(window_data),
                current_window_idx=0
            )
            return hetero_data
        except Exception as e:
            print(f"Error creating HeteroData: {e}")
            return None
    
    def get_recent_data(self, num_samples: int = None) -> torch.Tensor:
        """
        Get recent sensor data
        
        Args:
            num_samples: Number of recent samples to get
            
        Returns:
            Recent sensor readings
        """
        if num_samples is None:
            num_samples = self.window_size
        
        num_samples = min(num_samples, self.buffer_idx if not self.buffer_full else self.buffer_size)
        
        if self.buffer_full:
            if self.buffer_idx >= num_samples:
                return self.sensor_buffer[self.buffer_idx - num_samples:self.buffer_idx]
            else:
                # Wrap around
                part1 = self.sensor_buffer[self.buffer_size - (num_samples - self.buffer_idx):]
                part2 = self.sensor_buffer[:self.buffer_idx]
                return torch.cat([part1, part2], dim=0)
        else:
            return self.sensor_buffer[max(0, self.buffer_idx - num_samples):self.buffer_idx]


if __name__ == "__main__":
    # Example usage
    from ..data_preprocessing.topology_parser import TopologyParser
    from ..data_preprocessing.sensor_data_cleaner import SensorDataCleaner
    from ..data_preprocessing.graph_builder import GraphBuilder
    
    print("Testing data loader...")
    
    # Initialize components
    topology_parser = TopologyParser("../../blueprint/0708YTS4.txt")
    sensor_cleaner = SensorDataCleaner("../../data/0708YTS4.csv")
    graph_builder = GraphBuilder(topology_parser, sensor_cleaner)
    
    # Create data loader
    data_loader = GraphDataLoader(
        graph_builder=graph_builder,
        batch_size=8,
        window_size=60
    )
    
    # Prepare datasets
    datasets = data_loader.prepare_datasets()
    print(f"Datasets prepared: {list(datasets.keys())}")
    
    # Get data info
    info = data_loader.get_data_info()
    print(f"Dataset info: {info}")
    
    # Create data loaders
    dataloaders = data_loader.create_dataloaders()
    print(f"Data loaders created: {list(dataloaders.keys())}")
    
    # Test training data loader
    train_loader = dataloaders['train']
    print(f"Training loader: {len(train_loader)} batches")
    
    # Test one batch
    for batch in train_loader:
        print(f"Batch size: {batch['batch_size']}")
        print(f"Sensor targets shape: {batch['sensor_targets'].shape}")
        break
    
    print("Data loader test completed!")