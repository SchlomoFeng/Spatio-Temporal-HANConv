"""
Pipeline Network Topology Parser
Parses the 0708YTS4.json file to extract nodes, edges and build network structure
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re


class TopologyParser:
    """Parser for pipeline network topology from JSON blueprint"""
    
    def __init__(self, blueprint_path: str):
        """
        Initialize parser with blueprint file path
        
        Args:
            blueprint_path: Path to the JSON blueprint file
        """
        self.blueprint_path = blueprint_path
        self.raw_data = None
        self.nodes = {}
        self.edges = []
        self.node_types = set()
        
    def load_blueprint(self):
        """Load the JSON blueprint file"""
        with open(self.blueprint_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
            
    def extract_nodes(self) -> Dict[str, Dict]:
        """
        Extract nodes from nodelist in the JSON
        
        Returns:
            Dictionary mapping node_id to node attributes
        """
        if self.raw_data is None:
            self.load_blueprint()
            
        nodes = {}
        
        # Extract from nodelist
        if 'nodelist' in self.raw_data:
            for node in self.raw_data['nodelist']:
                node_id = node.get('id')
                if node_id:
                    # Parse node parameter for additional info
                    parameter_str = node.get('parameter', '{}')
                    try:
                        parameter = json.loads(parameter_str)
                    except:
                        parameter = {}
                    
                    # Extract node type from parameter or modelId
                    node_type = parameter.get('type', 'Unknown')
                    if not node_type or node_type == 'Unknown':
                        # Try to infer from name or other fields
                        name = node.get('name', '')
                        if 'Stream' in name or 'stream' in name:
                            node_type = 'Stream'
                        elif 'Valve' in name or 'valve' in name:
                            node_type = 'Valve'
                        elif 'Mixer' in name or 'mixer' in name:
                            node_type = 'Mixer'
                        elif 'Tee' in name or 'tee' in name:
                            node_type = 'Tee'
                        else:
                            node_type = f"ModelId_{node.get('modelId', 'Unknown')}"
                    
                    self.node_types.add(node_type)
                    
                    # Extract coordinates if available
                    x, y = 0.0, 0.0
                    if 'styles' in parameter:
                        styles = parameter['styles']
                        if 'foldPoint' in styles and styles['foldPoint']:
                            # Use first fold point as coordinate
                            first_point = styles['foldPoint'][0]
                            x = first_point.get('x', 0.0)
                            y = first_point.get('y', 0.0)
                    
                    nodes[node_id] = {
                        'id': node_id,
                        'name': node.get('name', ''),
                        'type': node_type,
                        'model_id': node.get('modelId'),
                        'x': x,
                        'y': y,
                        'parameter': parameter
                    }
        
        self.nodes = nodes
        return nodes
    
    def extract_edges(self) -> List[Tuple[str, str]]:
        """
        Extract edges from linklist in the JSON
        
        Returns:
            List of tuples (source_id, target_id)
        """
        if self.raw_data is None:
            self.load_blueprint()
            
        edges = []
        
        # Extract from linklist
        if 'linklist' in self.raw_data:
            for link in self.raw_data['linklist']:
                source_id = link.get('sourceid')
                target_id = link.get('targetid')
                
                if source_id and target_id:
                    edge_info = {
                        'source': source_id,
                        'target': target_id,
                        'id': link.get('id'),
                        'name': link.get('name', ''),
                        'model_id': link.get('modelId')
                    }
                    
                    # Parse edge parameters for additional info
                    parameter_str = link.get('parameter', '{}')
                    try:
                        parameter = json.loads(parameter_str)
                        edge_info['parameter'] = parameter
                        
                        # Extract pipe properties if available
                        if 'parameter' in parameter:
                            pipe_params = parameter['parameter']
                            edge_info['length'] = pipe_params.get('Length', 0)
                            edge_info['diameter'] = pipe_params.get('Inner_Diameter', 0)
                            edge_info['roughness'] = pipe_params.get('Roughness', 0)
                    except:
                        edge_info['parameter'] = {}
                    
                    edges.append(edge_info)
        
        self.edges = edges
        return edges
    
    def get_node_type_mapping(self) -> Dict[str, int]:
        """
        Create mapping from node types to integer indices
        
        Returns:
            Dictionary mapping node_type to integer index
        """
        if not self.node_types:
            self.extract_nodes()
            
        return {node_type: idx for idx, node_type in enumerate(sorted(self.node_types))}
    
    def create_node_features(self) -> pd.DataFrame:
        """
        Create node feature matrix
        
        Returns:
            DataFrame with node features
        """
        if not self.nodes:
            self.extract_nodes()
            
        node_list = []
        type_mapping = self.get_node_type_mapping()
        
        for node_id, node_data in self.nodes.items():
            features = {
                'node_id': node_id,
                'name': node_data['name'],
                'type': node_data['type'],
                'type_idx': type_mapping[node_data['type']],
                'x': node_data['x'],
                'y': node_data['y'],
                'model_id': node_data['model_id']
            }
            
            # Create one-hot encoding for node type
            for node_type in self.node_types:
                features[f'type_{node_type}'] = 1 if node_data['type'] == node_type else 0
            
            node_list.append(features)
        
        return pd.DataFrame(node_list)
    
    def create_edge_list(self) -> pd.DataFrame:
        """
        Create edge list DataFrame
        
        Returns:
            DataFrame with edge information
        """
        if not self.edges:
            self.extract_edges()
            
        edge_list = []
        for edge in self.edges:
            edge_info = {
                'source': edge['source'],
                'target': edge['target'],
                'edge_id': edge['id'],
                'name': edge['name'],
                'model_id': edge['model_id'],
                'length': edge.get('length', 0),
                'diameter': edge.get('diameter', 0),
                'roughness': edge.get('roughness', 0)
            }
            edge_list.append(edge_info)
        
        return pd.DataFrame(edge_list)
    
    def get_stream_nodes(self) -> List[str]:
        """
        Get list of stream node IDs (nodes that have sensor data)
        
        Returns:
            List of stream node IDs
        """
        if not self.nodes:
            self.extract_nodes()
            
        return [node_id for node_id, node_data in self.nodes.items() 
                if 'stream' in node_data['type'].lower()]
    
    def parse_topology(self) -> Dict[str, Any]:
        """
        Parse complete topology and return structured data
        
        Returns:
            Dictionary containing parsed topology data
        """
        nodes_df = self.create_node_features()
        edges_df = self.create_edge_list()
        type_mapping = self.get_node_type_mapping()
        stream_nodes = self.get_stream_nodes()
        
        topology_data = {
            'nodes': nodes_df,
            'edges': edges_df,
            'node_types': list(self.node_types),
            'type_mapping': type_mapping,
            'stream_nodes': stream_nodes,
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'node_id_to_idx': {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'])}
        }
        
        return topology_data


if __name__ == "__main__":
    # Example usage
    parser = TopologyParser("../../blueprint/0708YTS4.txt")
    topology = parser.parse_topology()
    
    print(f"Parsed {topology['num_nodes']} nodes and {topology['num_edges']} edges")
    print(f"Node types: {topology['node_types']}")
    print(f"Stream nodes: {len(topology['stream_nodes'])}")
    print("\nFirst few nodes:")
    print(topology['nodes'].head())