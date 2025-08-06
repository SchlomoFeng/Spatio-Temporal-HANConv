"""
Graph Builder
Combines topology and sensor data to build heterogeneous graph data objects
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Any, Optional
from topology_parser import TopologyParser
from sensor_data_cleaner import SensorDataCleaner


class GraphBuilder:
    """Builder for heterogeneous graph data objects"""
    
    def __init__(self, topology_parser: TopologyParser, 
                 sensor_cleaner: SensorDataCleaner):
        """
        Initialize graph builder
        
        Args:
            topology_parser: Parser for network topology
            sensor_cleaner: Cleaner for sensor data
        """
        self.topology_parser = topology_parser
        self.sensor_cleaner = sensor_cleaner
        self.topology_data = None
        self.sensor_data = None
        
    def build_node_mappings(self) -> Dict[str, Dict]:
        """
        Build mappings between different node types and sensor data
        
        Returns:
            Dictionary containing node mappings and metadata
        """
        if self.topology_data is None:
            self.topology_data = self.topology_parser.parse_topology()
            
        # Get sensor column names
        if not self.sensor_cleaner.sensor_columns:
            self.sensor_cleaner.identify_sensor_columns()
        
        sensor_columns = self.sensor_cleaner.sensor_columns
        
        # Try to match sensor columns to stream nodes
        stream_node_mapping = {}
        sensor_to_node_mapping = {}
        
        # Simple heuristic: match by name similarity or ID patterns
        for node_id, node_data in self.topology_parser.nodes.items():
            node_name = node_data['name'].lower()
            node_type = node_data['type']
            
            # For stream nodes, try to find matching sensors
            if 'stream' in node_type.lower():
                # Look for sensor columns that might correspond to this node
                matching_sensors = []
                for sensor_col in sensor_columns:
                    sensor_name = sensor_col.lower()
                    # Simple matching heuristics - can be improved
                    if (any(part in sensor_name for part in node_name.split('-')) or
                        any(part in node_name for part in sensor_col.split('.'))):
                        matching_sensors.append(sensor_col)
                
                if matching_sensors:
                    stream_node_mapping[node_id] = matching_sensors
                    for sensor in matching_sensors:
                        sensor_to_node_mapping[sensor] = node_id
        
        # If no direct mapping found, create artificial stream nodes for sensors
        if not stream_node_mapping:
            print("No direct sensor-node mapping found. Creating artificial stream nodes.")
            for i, sensor_col in enumerate(sensor_columns):
                artificial_node_id = f"stream_node_{i}"
                stream_node_mapping[artificial_node_id] = [sensor_col]
                sensor_to_node_mapping[sensor_col] = artificial_node_id
        
        mapping_info = {
            'stream_node_mapping': stream_node_mapping,  # node_id -> [sensor_columns]
            'sensor_to_node_mapping': sensor_to_node_mapping,  # sensor_col -> node_id
            'num_stream_nodes': len(stream_node_mapping),
            'num_sensors_mapped': len(sensor_to_node_mapping)
        }
        
        print(f"Mapped {mapping_info['num_sensors_mapped']} sensors to {mapping_info['num_stream_nodes']} stream nodes")
        return mapping_info
    
    def create_node_features(self, mapping_info: Dict) -> Dict[str, torch.Tensor]:
        """
        Create node feature tensors for different node types
        
        Args:
            mapping_info: Node mapping information
            
        Returns:
            Dictionary of node features by type
        """
        if self.topology_data is None:
            self.topology_data = self.topology_parser.parse_topology()
        
        node_features = {}
        
        # Stream nodes - these will have time-varying sensor data
        stream_nodes = list(mapping_info['stream_node_mapping'].keys())
        
        if stream_nodes:
            # Static features for stream nodes (type encoding + coordinates)
            stream_static_features = []
            
            for node_id in stream_nodes:
                if node_id in self.topology_parser.nodes:
                    node_data = self.topology_parser.nodes[node_id]
                    # Create feature vector: [x, y, type_encoding]
                    type_idx = self.topology_data['type_mapping'].get(node_data['type'], 0)
                    features = [node_data['x'], node_data['y'], type_idx]
                else:
                    # Artificial stream node
                    features = [0.0, 0.0, 0]  # Default values
                
                stream_static_features.append(features)
            
            node_features['stream'] = torch.tensor(stream_static_features, dtype=torch.float)
        
        # Other node types (static nodes)
        other_nodes = []
        other_features = []
        
        for node_id, node_data in self.topology_parser.nodes.items():
            if node_id not in stream_nodes:
                node_type = node_data['type']
                type_idx = self.topology_data['type_mapping'].get(node_type, 0)
                
                # Create one-hot encoding for node type
                type_vector = [0] * len(self.topology_data['node_types'])
                if type_idx < len(type_vector):
                    type_vector[type_idx] = 1
                
                # Combine with coordinates
                features = [node_data['x'], node_data['y']] + type_vector
                other_features.append(features)
                other_nodes.append(node_id)
        
        if other_features:
            node_features['static'] = torch.tensor(other_features, dtype=torch.float)
        
        return node_features, stream_nodes, other_nodes
    
    def create_edge_indices(self, stream_nodes: List[str], 
                          other_nodes: List[str]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Create edge indices for different edge types in heterogeneous graph
        
        Args:
            stream_nodes: List of stream node IDs
            other_nodes: List of other node IDs
            
        Returns:
            Dictionary of edge indices by edge type
        """
        if self.topology_data is None:
            self.topology_data = self.topology_parser.parse_topology()
        
        # Create node ID to index mappings
        stream_id_to_idx = {node_id: idx for idx, node_id in enumerate(stream_nodes)}
        other_id_to_idx = {node_id: idx for idx, node_id in enumerate(other_nodes)}
        
        edge_indices = {}
        
        # Process edges from topology
        for edge in self.topology_parser.edges:
            source_id = edge['source']
            target_id = edge['target']
            
            # Determine edge type based on node types
            source_is_stream = source_id in stream_id_to_idx
            target_is_stream = target_id in stream_id_to_idx
            source_is_other = source_id in other_id_to_idx
            target_is_other = target_id in other_id_to_idx
            
            # Skip if nodes not found
            if not ((source_is_stream or source_is_other) and (target_is_stream or target_is_other)):
                continue
            
            # Determine edge type
            if source_is_stream and target_is_stream:
                edge_type = ('stream', 'connects', 'stream')
                source_idx = stream_id_to_idx[source_id]
                target_idx = stream_id_to_idx[target_id]
            elif source_is_stream and target_is_other:
                edge_type = ('stream', 'flows_to', 'static')
                source_idx = stream_id_to_idx[source_id]
                target_idx = other_id_to_idx[target_id]
            elif source_is_other and target_is_stream:
                edge_type = ('static', 'feeds', 'stream')
                source_idx = other_id_to_idx[source_id]
                target_idx = stream_id_to_idx[target_id]
            else:  # both static
                edge_type = ('static', 'connected', 'static')
                source_idx = other_id_to_idx[source_id]
                target_idx = other_id_to_idx[target_id]
            
            # Add edge to appropriate type
            if edge_type not in edge_indices:
                edge_indices[edge_type] = []
            
            edge_indices[edge_type].append([source_idx, target_idx])
        
        # Convert to tensors
        for edge_type in edge_indices:
            if edge_indices[edge_type]:
                edge_indices[edge_type] = torch.tensor(edge_indices[edge_type], dtype=torch.long).t().contiguous()
            else:
                # Empty edge index
                edge_indices[edge_type] = torch.empty((2, 0), dtype=torch.long)
        
        return edge_indices
    
    def create_time_windows(self, window_size: int = 60, 
                          step_size: int = 1) -> List[torch.Tensor]:
        """
        Create sliding time windows from sensor data
        
        Args:
            window_size: Size of each time window
            step_size: Step size between windows
            
        Returns:
            List of time window tensors
        """
        if self.sensor_data is None:
            self.sensor_data = self.sensor_cleaner.clean_sensor_data()
        
        sensor_columns = self.sensor_cleaner.sensor_columns
        sensor_values = self.sensor_data[sensor_columns].values
        
        time_windows = []
        num_windows = (len(sensor_values) - window_size) // step_size + 1
        
        for i in range(0, num_windows * step_size, step_size):
            if i + window_size <= len(sensor_values):
                window_data = sensor_values[i:i+window_size]
                time_windows.append(torch.tensor(window_data, dtype=torch.float))
        
        print(f"Created {len(time_windows)} time windows of size {window_size}")
        return time_windows
    
    def build_hetero_data(self, window_size: int = 60, 
                         step_size: int = 1,
                         current_window_idx: int = 0) -> HeteroData:
        """
        Build a HeteroData object for a specific time window
        
        Args:
            window_size: Size of time window for sensor data
            step_size: Step size between windows
            current_window_idx: Index of current time window
            
        Returns:
            HeteroData object
        """
        # Build mappings
        mapping_info = self.build_node_mappings()
        
        # Create node features
        node_features, stream_nodes, other_nodes = self.create_node_features(mapping_info)
        
        # Create edge indices
        edge_indices = self.create_edge_indices(stream_nodes, other_nodes)
        
        # Create time windows for sensor data
        time_windows = self.create_time_windows(window_size, step_size)
        
        # Create HeteroData object
        data = HeteroData()
        
        # Add node features
        if 'stream' in node_features:
            # For stream nodes, combine static features with current time window sensor data
            static_features = node_features['stream']
            
            if current_window_idx < len(time_windows):
                # Get current sensor readings
                current_sensors = time_windows[current_window_idx]  # shape: [window_size, num_sensors]
                
                # Take latest reading for each sensor
                latest_readings = current_sensors[-1]  # shape: [num_sensors]
                
                # Map sensors to stream nodes
                stream_sensor_features = []
                for i, node_id in enumerate(stream_nodes):
                    if node_id in mapping_info['stream_node_mapping']:
                        # Get sensor indices for this node
                        sensor_cols = mapping_info['stream_node_mapping'][node_id]
                        sensor_indices = [self.sensor_cleaner.sensor_columns.index(col) 
                                        for col in sensor_cols if col in self.sensor_cleaner.sensor_columns]
                        
                        if sensor_indices:
                            # Average readings for this node
                            node_reading = latest_readings[sensor_indices].mean()
                            stream_sensor_features.append(node_reading.item())
                        else:
                            stream_sensor_features.append(0.0)
                    else:
                        stream_sensor_features.append(0.0)
                
                # Combine static and sensor features
                sensor_features = torch.tensor(stream_sensor_features, dtype=torch.float).unsqueeze(1)
                combined_features = torch.cat([static_features, sensor_features], dim=1)
            else:
                # No sensor data available, use only static features
                combined_features = static_features
            
            data['stream'].x = combined_features
        
        if 'static' in node_features:
            data['static'].x = node_features['static']
        
        # Add edge indices
        for edge_type, edge_index in edge_indices.items():
            data[edge_type].edge_index = edge_index
        
        # Store additional metadata
        data.mapping_info = mapping_info
        data.stream_nodes = stream_nodes
        data.other_nodes = other_nodes
        data.window_idx = current_window_idx
        data.window_size = window_size
        
        return data
    
    def build_dataset(self, window_size: int = 60, 
                     step_size: int = 1,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Dict[str, List[HeteroData]]:
        """
        Build complete dataset with train/val/test splits
        
        Args:
            window_size: Size of time window
            step_size: Step size between windows
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Dictionary with train/val/test data lists
        """
        # Create all time windows
        time_windows = self.create_time_windows(window_size, step_size)
        num_windows = len(time_windows)
        
        # Calculate split indices
        train_end = int(num_windows * train_ratio)
        val_end = int(num_windows * (train_ratio + val_ratio))
        
        # Build data for each split
        dataset = {'train': [], 'val': [], 'test': []}
        
        print(f"Building dataset with {train_end} train, {val_end - train_end} val, {num_windows - val_end} test samples")
        
        # Training data
        for i in range(train_end):
            data = self.build_hetero_data(window_size, step_size, i)
            dataset['train'].append(data)
        
        # Validation data
        for i in range(train_end, val_end):
            data = self.build_hetero_data(window_size, step_size, i)
            dataset['val'].append(data)
        
        # Test data
        for i in range(val_end, num_windows):
            data = self.build_hetero_data(window_size, step_size, i)
            dataset['test'].append(data)
        
        print(f"Dataset built: {len(dataset['train'])} train, {len(dataset['val'])} val, {len(dataset['test'])} test")
        return dataset


if __name__ == "__main__":
    # Example usage
    from topology_parser import TopologyParser
    from sensor_data_cleaner import SensorDataCleaner
    
    # Initialize components
    topology_parser = TopologyParser("../../blueprint/0708YTS4.txt")
    sensor_cleaner = SensorDataCleaner("../../data/0708YTS4.csv")
    
    # Build graph
    graph_builder = GraphBuilder(topology_parser, sensor_cleaner)
    
    # Build single hetero data object
    hetero_data = graph_builder.build_hetero_data(window_size=60)
    
    print(f"HeteroData object created:")
    print(f"Stream nodes: {hetero_data['stream'].x.shape if 'stream' in hetero_data else 'None'}")
    print(f"Static nodes: {hetero_data['static'].x.shape if 'static' in hetero_data else 'None'}")
    print(f"Edge types: {list(hetero_data.edge_types)}")
    print(f"Metadata: {hetero_data.mapping_info['num_sensors_mapped']} sensors mapped")