"""
Heterogeneous Graph Encoder
Implements the encoder part of the autoencoder using HANConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear
from typing import Dict, Tuple, Optional, List


class HeteroEncoder(nn.Module):
    """Heterogeneous Graph Encoder using HANConv"""
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 stream_lstm_hidden: int = 32,
                 stream_lstm_layers: int = 1):
        """
        Initialize heterogeneous encoder
        
        Args:
            node_feature_dims: Dictionary mapping node types to feature dimensions
            metadata: Graph metadata (node_types, edge_types) for HANConv
            hidden_dim: Hidden dimension size
            output_dim: Output embedding dimension
            num_heads: Number of attention heads for HANConv
            num_layers: Number of HANConv layers
            dropout: Dropout rate
            stream_lstm_hidden: Hidden size for stream node LSTM
            stream_lstm_layers: Number of LSTM layers for stream nodes
        """
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input transformation layers for different node types
        self.input_transforms = nn.ModuleDict()
        for node_type, feat_dim in node_feature_dims.items():
            self.input_transforms[node_type] = Linear(feat_dim, hidden_dim)
        
        # LSTM for stream nodes to handle temporal information
        if 'stream' in node_feature_dims:
            self.stream_lstm = nn.LSTM(
                input_size=1,  # Single sensor reading per time step
                hidden_size=stream_lstm_hidden,
                num_layers=stream_lstm_layers,
                batch_first=True,
                dropout=dropout if stream_lstm_layers > 1 else 0
            )
            
            # Combine LSTM output with static features
            self.stream_combine = Linear(
                stream_lstm_hidden + node_feature_dims['stream'] - 1,  # -1 because sensor reading is processed by LSTM
                hidden_dim
            )
        
        # HANConv layers
        self.han_convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_dim if i > 0 else hidden_dim
            self.han_convs.append(
                HANConv(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                    dropout=dropout
                )
            )
        
        # Output projection layers
        self.output_projections = nn.ModuleDict()
        for node_type in node_feature_dims.keys():
            self.output_projections[node_type] = nn.Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_dim // 2, output_dim)
            )
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def encode_stream_features(self, x: torch.Tensor, 
                             batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Encode stream node features with temporal information
        
        Args:
            x: Stream node features [num_nodes, feature_dim]
               Last feature is assumed to be current sensor reading
            batch_size: Batch size (not used in current implementation)
            
        Returns:
            Encoded stream features
        """
        if x.size(1) < 2:
            # No temporal information, just transform
            return self.input_transforms['stream'](x)
        
        # Split static features and sensor reading
        static_features = x[:, :-1]  # [num_nodes, static_dim]
        sensor_readings = x[:, -1:].unsqueeze(1)  # [num_nodes, 1, 1]
        
        # Process sensor readings through LSTM
        # For single time step, we create a sequence of length 1
        lstm_out, _ = self.stream_lstm(sensor_readings)
        lstm_features = lstm_out[:, -1, :]  # Take last (and only) output [num_nodes, lstm_hidden]
        
        # Combine LSTM output with static features
        combined_features = torch.cat([static_features, lstm_features], dim=1)
        encoded = self.stream_combine(combined_features)
        
        return encoded
    
    def forward(self, 
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                batch_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder
        
        Args:
            x_dict: Dictionary of node features by type
            edge_index_dict: Dictionary of edge indices by edge type
            batch_dict: Optional batch indices for each node type
            
        Returns:
            Dictionary of encoded node embeddings by type
        """
        # Transform input features
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type == 'stream' and hasattr(self, 'stream_lstm'):
                h_dict[node_type] = self.encode_stream_features(x)
            else:
                h_dict[node_type] = self.input_transforms[node_type](x)
            
            h_dict[node_type] = F.relu(h_dict[node_type])
            h_dict[node_type] = self.dropout_layer(h_dict[node_type])
        
        # Apply HANConv layers
        for han_conv in self.han_convs:
            h_dict = han_conv(h_dict, edge_index_dict)
            
            # Apply activation and dropout
            for node_type in h_dict.keys():
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = self.dropout_layer(h_dict[node_type])
        
        # Project to output embeddings
        output_dict = {}
        for node_type, h in h_dict.items():
            output_dict[node_type] = self.output_projections[node_type](h)
        
        return output_dict


class StreamLSTMEncoder(nn.Module):
    """Specialized LSTM encoder for stream nodes with time series data"""
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 32,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        """
        Initialize LSTM encoder for stream data
        
        Args:
            input_dim: Input feature dimension (typically 1 for single sensor)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output embedding dimension
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through LSTM encoder
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
            lengths: Sequence lengths (optional)
            
        Returns:
            Encoded representations [batch_size, output_dim]
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate final hidden states from both directions
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            # Use final hidden state
            final_hidden = hidden[-1]
        
        # Project to output dimension
        output = self.output_projection(final_hidden)
        
        return output


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example dimensions
    node_feature_dims = {
        'stream': 4,  # 3 static + 1 sensor reading
        'static': 7   # coordinates + one-hot type encoding
    }
    
    # Create encoder
    encoder = HeteroEncoder(
        node_feature_dims=node_feature_dims,
        hidden_dim=64,
        output_dim=32,
        num_heads=4,
        num_layers=2
    ).to(device)
    
    print(f"Encoder created with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Example input
    batch_size = 1
    num_stream_nodes = 10
    num_static_nodes = 20
    
    x_dict = {
        'stream': torch.randn(num_stream_nodes, 4).to(device),
        'static': torch.randn(num_static_nodes, 7).to(device)
    }
    
    # Example edge indices
    edge_index_dict = {
        ('stream', 'connects', 'stream'): torch.randint(0, num_stream_nodes, (2, 5)).to(device),
        ('stream', 'flows_to', 'static'): torch.tensor([[0, 1], [0, 1]]).to(device),
        ('static', 'feeds', 'stream'): torch.tensor([[0, 1], [0, 1]]).to(device)
    }
    
    # Forward pass
    try:
        output = encoder(x_dict, edge_index_dict)
        print("Forward pass successful!")
        for node_type, emb in output.items():
            print(f"{node_type} embeddings shape: {emb.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")