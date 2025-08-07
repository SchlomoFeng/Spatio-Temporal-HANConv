"""
Data preprocessing utilities for S4 Steam Pipeline Anomaly Detection System

This module handles:
1. Pipeline topology parsing from JSON blueprint
2. Sensor data preprocessing and cleaning  
3. Node mapping and feature engineering
4. Heterogeneous graph construction
5. Train/validation/test data splitting
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
from torch_geometric.data import HeteroData
import warnings
warnings.filterwarnings('ignore')


class PipelineTopologyParser:
    """Parse pipeline topology from blueprint JSON file"""
    
    def __init__(self, blueprint_path: str):
        self.blueprint_path = blueprint_path
        self.node_types = ['Stream', 'VavlePro', 'Mixer', 'Tee', 'Pipe', 'Unknown']
        
    def load_blueprint_data(self) -> Dict[str, Any]:
        """Load and parse blueprint JSON data"""
        with open(self.blueprint_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def extract_nodes_and_edges(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract nodes and edges from blueprint data"""
        data = self.load_blueprint_data()
        
        # Extract nodes and edges
        nodelist = data['nodelist']
        edgelist = data['linklist']
        
        # Convert to DataFrames
        nodeDF = pd.DataFrame(nodelist)
        edgeDF = pd.DataFrame(edgelist)
        
        print(f"Loaded {len(nodeDF)} nodes and {len(edgeDF)} edges from blueprint")
        
        return nodeDF, edgeDF
    
    def process_nodes(self, nodeDF: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Process node data and extract features"""
        processed_nodes = []
        node_id_to_index = {}
        
        for i, row in nodeDF.iterrows():
            node_id = row['id']
            node_id_to_index[node_id] = i
            
            # Parse node parameters
            try:
                if isinstance(row['parameter'], str):
                    para = json.loads(row['parameter'])
                else:
                    para = row['parameter']
                
                # Extract node type
                node_type = para.get('type', 'Unknown')
                if node_type not in self.node_types:
                    node_type = 'Unknown'
                
                # Extract position
                position = [0, 0]
                if 'styles' in para and 'position' in para['styles']:
                    pos = para['styles']['position']
                    position = [pos['x'], pos['y']]
                
                # Extract additional properties based on node type
                properties = self._extract_node_properties(para, node_type)
                
                processed_nodes.append({
                    'node_id': node_id,
                    'node_index': i,
                    'node_type': node_type,
                    'x_coord': position[0],
                    'y_coord': position[1],
                    'name': row.get('name', f'Node_{i}'),
                    **properties
                })
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error parsing node {node_id}: {e}")
                processed_nodes.append({
                    'node_id': node_id,
                    'node_index': i,
                    'node_type': 'Unknown',
                    'x_coord': 0,
                    'y_coord': 0,
                    'name': row.get('name', f'Node_{i}')
                })
        
        return pd.DataFrame(processed_nodes), node_id_to_index
    
    def _extract_node_properties(self, para: Dict, node_type: str) -> Dict[str, Any]:
        """Extract type-specific node properties"""
        properties = {}
        
        if node_type == 'Pipe':
            param = para.get('parameter', {})
            properties.update({
                'length': param.get('Length', 0.0),
                'diameter': param.get('Inner_Diameter', 0.0),
                'roughness': param.get('Roughness', 0.0)
            })
        elif node_type == 'VavlePro':
            param = para.get('parameter', {})
            properties.update({
                'cv_value': param.get('Cv', 0.0),
                'opening': param.get('Opening', 1.0)
            })
        
        return properties
    
    def process_edges(self, edgeDF: pd.DataFrame, node_id_to_index: Dict[str, int]) -> pd.DataFrame:
        """Process edge data and create edge index"""
        edge_data = []
        
        for i, row in edgeDF.iterrows():
            source_id = row['sourceid']
            target_id = row['targetid']
            
            # Check if both nodes exist
            if source_id in node_id_to_index and target_id in node_id_to_index:
                source_idx = node_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]
                
                # Parse edge parameters
                try:
                    if isinstance(row['parameter'], str):
                        edge_para = json.loads(row['parameter'])
                    else:
                        edge_para = row['parameter']
                    
                    # Extract edge properties
                    length = edge_para.get('parameter', {}).get('Length', 1.0)
                    if length is None or length <= 0:
                        length = 1.0
                    
                    edge_data.append({
                        'edge_id': row['id'],
                        'source_idx': source_idx,
                        'target_idx': target_idx,
                        'source_id': source_id,
                        'target_id': target_id,
                        'length': float(length),
                        'edge_type': 'pipe_connection'  # Can be extended for different edge types
                    })
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error parsing edge {row['id']}: {e}")
                    edge_data.append({
                        'edge_id': row['id'],
                        'source_idx': source_idx,
                        'target_idx': target_idx,
                        'source_id': source_id,
                        'target_id': target_id,
                        'length': 1.0,
                        'edge_type': 'pipe_connection'
                    })
            else:
                print(f"Warning: Edge {row['id']} endpoints not found in nodes")
        
        return pd.DataFrame(edge_data)


class SensorDataProcessor:
    """Process sensor data for time series analysis"""
    
    def __init__(self, sensor_data_path: str, scaler_type: str = 'StandardScaler'):
        self.sensor_data_path = sensor_data_path
        self.scaler_type = scaler_type
        self.scaler = None
        self.sensor_columns = None
        
    def load_sensor_data(self) -> pd.DataFrame:
        """Load sensor data from CSV"""
        df = pd.read_csv(self.sensor_data_path)
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Identify sensor columns (exclude timestamp)
        self.sensor_columns = [col for col in df.columns if col != 'timestamp']
        
        print(f"Loaded sensor data: {len(df)} records, {len(self.sensor_columns)} sensors")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sensor data - handle missing values, outliers"""
        print("Cleaning sensor data...")
        
        # Handle missing values - forward fill then backward fill
        for col in self.sensor_columns:
            if df[col].isna().any():
                missing_count = df[col].isna().sum()
                print(f"  Column {col}: {missing_count} missing values")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Convert non-numeric columns to numeric
        for col in self.sensor_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any remaining NaN values with column means
        for col in self.sensor_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        print(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def create_scaler(self, train_data: pd.DataFrame) -> None:
        """Create and fit scaler on training data"""
        if self.scaler_type == 'StandardScaler':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'RobustScaler':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Fit scaler on training data only
        self.scaler.fit(train_data[self.sensor_columns])
        print(f"Fitted {self.scaler_type} on training data")
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize sensor data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call create_scaler() first.")
        
        df_normalized = df.copy()
        df_normalized[self.sensor_columns] = self.scaler.transform(df[self.sensor_columns])
        
        return df_normalized
    
    def create_time_windows(self, df: pd.DataFrame, window_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding time windows for LSTM input"""
        sensor_data = df[self.sensor_columns].values
        timestamps = df['timestamp'].values
        
        windows = []
        window_timestamps = []
        
        for i in range(0, len(sensor_data) - window_size + 1, stride):
            windows.append(sensor_data[i:i + window_size])
            window_timestamps.append(timestamps[i + window_size - 1])  # Use last timestamp
        
        return np.array(windows), np.array(window_timestamps)


class HeteroGraphBuilder:
    """Build heterogeneous graph data objects"""
    
    def __init__(self, node_types: List[str]):
        self.node_types = node_types
        
    def create_node_features(self, nodes_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Create node features for each node type"""
        node_features = {}
        
        for node_type in self.node_types:
            # Filter nodes of this type
            type_nodes = nodes_df[nodes_df['node_type'] == node_type]
            
            if len(type_nodes) == 0:
                continue
            
            # Create features based on node type
            if node_type == 'Stream':
                # Stream nodes will receive time series data during training
                # For now, create placeholder features
                features = self._create_stream_features(type_nodes)
            else:
                # Static nodes get engineered features
                features = self._create_static_features(type_nodes)
            
            node_features[node_type] = torch.tensor(features, dtype=torch.float32)
        
        return node_features
    
    def _create_stream_features(self, nodes: pd.DataFrame) -> np.ndarray:
        """Create features for Stream nodes (placeholder for time series)"""
        num_nodes = len(nodes)
        # Basic static features: coordinates + one-hot encoding
        features = np.zeros((num_nodes, 10))  # Placeholder dimension
        
        for i, (_, node) in enumerate(nodes.iterrows()):
            features[i, 0] = node['x_coord'] / 100000  # Normalize coordinates
            features[i, 1] = node['y_coord'] / 100000
            features[i, 2] = 1.0  # Is Stream node
        
        return features
    
    def _create_static_features(self, nodes: pd.DataFrame) -> np.ndarray:
        """Create features for static nodes"""
        num_nodes = len(nodes)
        features = np.zeros((num_nodes, 10))
        
        for i, (_, node) in enumerate(nodes.iterrows()):
            # Basic features
            features[i, 0] = node['x_coord'] / 100000  # Normalize coordinates
            features[i, 1] = node['y_coord'] / 100000
            
            # One-hot encoding for node type
            if node['node_type'] == 'VavlePro':
                features[i, 3] = 1.0
            elif node['node_type'] == 'Mixer':
                features[i, 4] = 1.0
            elif node['node_type'] == 'Tee':
                features[i, 5] = 1.0
            elif node['node_type'] == 'Pipe':
                features[i, 6] = 1.0
            else:  # Unknown
                features[i, 7] = 1.0
            
            # Additional properties if available
            if 'length' in node and not pd.isna(node['length']):
                features[i, 8] = node['length'] / 1000  # Normalize length
            if 'diameter' in node and not pd.isna(node['diameter']):
                features[i, 9] = node['diameter'] / 1000  # Normalize diameter
        
        return features
    
    def create_edge_indices(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Create edge indices for heterogeneous graph"""
        edge_indices = {}
        
        # Group edges by source and target node types
        for _, edge in edges_df.iterrows():
            source_idx = edge['source_idx']
            target_idx = edge['target_idx']
            
            # Get node types
            source_type = nodes_df.iloc[source_idx]['node_type']
            target_type = nodes_df.iloc[target_idx]['node_type']
            
            # Create edge type key
            edge_key = (source_type, 'connects_to', target_type)
            
            if edge_key not in edge_indices:
                edge_indices[edge_key] = []
            
            # Map to type-specific indices
            source_type_nodes = nodes_df[nodes_df['node_type'] == source_type]
            target_type_nodes = nodes_df[nodes_df['node_type'] == target_type]
            
            source_type_idx = source_type_nodes[source_type_nodes['node_index'] == source_idx].index[0]
            target_type_idx = target_type_nodes[target_type_nodes['node_index'] == target_idx].index[0]
            
            # Get position in type-specific ordering
            source_pos = list(source_type_nodes.index).index(source_type_idx)
            target_pos = list(target_type_nodes.index).index(target_type_idx)
            
            edge_indices[edge_key].append([source_pos, target_pos])
        
        # Convert to tensors
        for edge_key in edge_indices:
            if edge_indices[edge_key]:
                edge_indices[edge_key] = torch.tensor(edge_indices[edge_key], dtype=torch.long).t().contiguous()
            else:
                edge_indices[edge_key] = torch.empty((2, 0), dtype=torch.long)
        
        return edge_indices


class DataSplitter:
    """Split data into train/validation/test sets"""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
    
    def split_time_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data chronologically"""
        n_samples = len(df)
        
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df


def create_hetero_data(node_features: Dict[str, torch.Tensor], 
                      edge_indices: Dict[Tuple[str, str, str], torch.Tensor]) -> HeteroData:
    """Create PyTorch Geometric HeteroData object"""
    data = HeteroData()
    
    # Add node features
    for node_type, features in node_features.items():
        data[node_type].x = features
    
    # Add edge indices
    for edge_key, edge_index in edge_indices.items():
        data[edge_key].edge_index = edge_index
    
    return data


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing data preprocessing pipeline...")
    
    # Initialize parsers
    topology_parser = PipelineTopologyParser("../../blueprint/0708YTS4.txt")
    sensor_processor = SensorDataProcessor("../../data/0708YTS4.csv")
    
    # Process topology
    nodeDF, edgeDF = topology_parser.extract_nodes_and_edges()
    nodes_df, node_id_to_index = topology_parser.process_nodes(nodeDF)
    edges_df = topology_parser.process_edges(edgeDF, node_id_to_index)
    
    print(f"Processed topology: {len(nodes_df)} nodes, {len(edges_df)} edges")
    print(f"Node types: {nodes_df['node_type'].value_counts().to_dict()}")
    
    # Process sensor data
    sensor_df = sensor_processor.load_sensor_data()
    sensor_df = sensor_processor.clean_data(sensor_df)
    
    # Split data
    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split_time_series(sensor_df)
    
    # Fit scaler on training data
    sensor_processor.create_scaler(train_df)
    
    # Build heterogeneous graph
    node_types = ['Stream', 'VavlePro', 'Mixer', 'Tee', 'Pipe', 'Unknown']
    graph_builder = HeteroGraphBuilder(node_types)
    
    node_features = graph_builder.create_node_features(nodes_df)
    edge_indices = graph_builder.create_edge_indices(edges_df, nodes_df)
    
    hetero_data = create_hetero_data(node_features, edge_indices)
    
    print("Heterogeneous graph created successfully!")
    print(f"Node types: {list(hetero_data.node_types)}")
    print(f"Edge types: {list(hetero_data.edge_types)}")