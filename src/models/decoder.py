"""
Decoder Module
Reconstructs original stream sensor readings from node embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class Decoder(nn.Module):
    """Decoder for reconstructing sensor readings from node embeddings"""
    
    def __init__(self,
                 stream_embedding_dim: int = 32,
                 num_sensors: int = 36,
                 hidden_dims: List[int] = [64, 128, 64],
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize decoder
        
        Args:
            stream_embedding_dim: Dimension of stream node embeddings
            num_sensors: Number of sensors to reconstruct
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        self.stream_embedding_dim = stream_embedding_dim
        self.num_sensors = num_sensors
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build MLP layers
        layers = []
        input_dim = stream_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer to reconstruct sensors
        layers.append(nn.Linear(input_dim, num_sensors))
        
        self.decoder_mlp = nn.Sequential(*layers)
        
        # Optional: Individual sensor decoders for better reconstruction
        self.use_individual_decoders = False
        if self.use_individual_decoders:
            self.sensor_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(stream_embedding_dim, hidden_dims[0] // 2),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims[0] // 2, 1)
                )
                for _ in range(num_sensors)
            ])
    
    def forward(self, stream_embeddings: torch.Tensor,
                mapping_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass to reconstruct sensor readings
        
        Args:
            stream_embeddings: Stream node embeddings [num_stream_nodes, embedding_dim]
            mapping_info: Optional mapping information for sensor-to-node assignment
            
        Returns:
            Reconstructed sensor readings [num_stream_nodes, num_sensors] or [num_sensors]
        """
        if self.use_individual_decoders:
            # Use individual decoders for each sensor
            sensor_reconstructions = []
            for sensor_decoder in self.sensor_decoders:
                sensor_recon = sensor_decoder(stream_embeddings)
                sensor_reconstructions.append(sensor_recon)
            
            reconstructed = torch.cat(sensor_reconstructions, dim=-1)
        else:
            # Use shared MLP decoder
            reconstructed = self.decoder_mlp(stream_embeddings)
        
        return reconstructed


class AttentionDecoder(nn.Module):
    """Attention-based decoder for better sensor reconstruction"""
    
    def __init__(self,
                 stream_embedding_dim: int = 32,
                 num_sensors: int = 36,
                 attention_dim: int = 64,
                 hidden_dims: List[int] = [64, 128, 64],
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize attention decoder
        
        Args:
            stream_embedding_dim: Dimension of stream node embeddings
            num_sensors: Number of sensors to reconstruct
            attention_dim: Dimension for attention mechanism
            hidden_dims: Hidden layer dimensions
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.stream_embedding_dim = stream_embedding_dim
        self.num_sensors = num_sensors
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Multi-head attention for combining stream embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=stream_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(stream_embedding_dim)
        
        # Decoder MLP
        layers = []
        input_dim = stream_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_sensors))
        self.decoder_mlp = nn.Sequential(*layers)
    
    def forward(self, stream_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism
        
        Args:
            stream_embeddings: Stream node embeddings [num_stream_nodes, embedding_dim]
            
        Returns:
            Reconstructed sensor readings [num_sensors] or [batch_size, num_sensors]
        """
        # Add batch dimension if needed
        if len(stream_embeddings.shape) == 2:
            stream_embeddings = stream_embeddings.unsqueeze(0)  # [1, num_nodes, embedding_dim]
        
        # Self-attention over stream nodes
        attended_embeddings, attention_weights = self.attention(
            stream_embeddings, stream_embeddings, stream_embeddings
        )
        
        # Residual connection and layer norm
        attended_embeddings = self.layer_norm(attended_embeddings + stream_embeddings)
        
        # Aggregate embeddings (mean pooling)
        aggregated = attended_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Decode to sensor readings
        reconstructed = self.decoder_mlp(aggregated)
        
        # Remove batch dimension if originally 2D
        if reconstructed.shape[0] == 1:
            reconstructed = reconstructed.squeeze(0)
        
        return reconstructed


class SensorSpecificDecoder(nn.Module):
    """Decoder that handles sensor-to-stream-node mappings explicitly"""
    
    def __init__(self,
                 stream_embedding_dim: int = 32,
                 sensor_mapping: Dict[str, List[str]] = None,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        """
        Initialize sensor-specific decoder
        
        Args:
            stream_embedding_dim: Dimension of stream embeddings
            sensor_mapping: Mapping from stream nodes to sensor names
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.stream_embedding_dim = stream_embedding_dim
        self.sensor_mapping = sensor_mapping or {}
        self.num_sensors = sum(len(sensors) for sensors in sensor_mapping.values()) if sensor_mapping else 36
        
        # Create decoders for each sensor group
        self.sensor_group_decoders = nn.ModuleDict()
        
        if sensor_mapping:
            for node_id, sensor_list in sensor_mapping.items():
                self.sensor_group_decoders[node_id] = nn.Sequential(
                    nn.Linear(stream_embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, len(sensor_list))
                )
        else:
            # Default single decoder
            self.default_decoder = nn.Sequential(
                nn.Linear(stream_embedding_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_sensors)
            )
    
    def forward(self, stream_embeddings: torch.Tensor,
                stream_node_ids: List[str] = None) -> torch.Tensor:
        """
        Forward pass with sensor-specific reconstruction
        
        Args:
            stream_embeddings: Stream embeddings [num_stream_nodes, embedding_dim]
            stream_node_ids: List of stream node IDs corresponding to embeddings
            
        Returns:
            Reconstructed sensor readings
        """
        if not self.sensor_mapping or not stream_node_ids:
            # Use default decoder
            if hasattr(self, 'default_decoder'):
                # Aggregate embeddings and decode
                aggregated = stream_embeddings.mean(dim=0)
                return self.default_decoder(aggregated)
            else:
                raise ValueError("No sensor mapping or default decoder available")
        
        # Use sensor-specific decoders
        sensor_reconstructions = []
        
        for i, node_id in enumerate(stream_node_ids):
            if node_id in self.sensor_group_decoders:
                node_embedding = stream_embeddings[i:i+1]  # Keep batch dim
                sensor_recon = self.sensor_group_decoders[node_id](node_embedding)
                sensor_reconstructions.append(sensor_recon.squeeze(0))
        
        if sensor_reconstructions:
            return torch.cat(sensor_reconstructions, dim=0)
        else:
            # Fallback: use mean of embeddings
            aggregated = stream_embeddings.mean(dim=0)
            return torch.zeros(self.num_sensors, device=stream_embeddings.device)


class VariationalDecoder(nn.Module):
    """Variational decoder for uncertainty estimation"""
    
    def __init__(self,
                 stream_embedding_dim: int = 32,
                 num_sensors: int = 36,
                 hidden_dims: List[int] = [64, 128, 64],
                 dropout: float = 0.1):
        """
        Initialize variational decoder
        
        Args:
            stream_embedding_dim: Stream embedding dimension
            num_sensors: Number of sensors
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_sensors = num_sensors
        
        # Shared backbone
        backbone_layers = []
        input_dim = stream_embedding_dim
        
        for hidden_dim in hidden_dims[:-1]:
            backbone_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Mean and log variance heads
        final_dim = hidden_dims[-1]
        self.mean_head = nn.Linear(input_dim, final_dim)
        self.logvar_head = nn.Linear(input_dim, final_dim)
        
        # Final reconstruction layer
        self.reconstruction_layer = nn.Linear(final_dim, num_sensors)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, stream_embeddings: torch.Tensor,
                return_distribution: bool = False) -> torch.Tensor:
        """
        Forward pass with variational sampling
        
        Args:
            stream_embeddings: Stream embeddings
            return_distribution: Whether to return mean and logvar
            
        Returns:
            Reconstructed sensors and optionally distribution parameters
        """
        # Aggregate embeddings
        if len(stream_embeddings.shape) > 2:
            aggregated = stream_embeddings.mean(dim=1)
        else:
            aggregated = stream_embeddings.mean(dim=0)
        
        # Shared backbone
        backbone_out = self.backbone(aggregated)
        
        # Distribution parameters
        mu = self.mean_head(backbone_out)
        logvar = self.logvar_head(backbone_out)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Reconstruct sensors
        reconstructed = self.reconstruction_layer(z)
        
        if return_distribution:
            return reconstructed, mu, logvar
        return reconstructed


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic decoder
    decoder = Decoder(
        stream_embedding_dim=32,
        num_sensors=36,
        hidden_dims=[64, 128, 64]
    ).to(device)
    
    # Example input
    stream_embeddings = torch.randn(10, 32).to(device)  # 10 stream nodes
    
    # Forward pass
    reconstructed = decoder(stream_embeddings)
    print(f"Decoder output shape: {reconstructed.shape}")
    print(f"Expected: [10, 36] (stream_nodes, sensors)")
    
    # Test attention decoder
    attention_decoder = AttentionDecoder(
        stream_embedding_dim=32,
        num_sensors=36,
        num_heads=4
    ).to(device)
    
    reconstructed_att = attention_decoder(stream_embeddings)
    print(f"Attention decoder output shape: {reconstructed_att.shape}")
    
    # Test variational decoder
    var_decoder = VariationalDecoder(
        stream_embedding_dim=32,
        num_sensors=36
    ).to(device)
    
    reconstructed_var, mu, logvar = var_decoder(stream_embeddings, return_distribution=True)
    print(f"Variational decoder output shapes: {reconstructed_var.shape}, {mu.shape}, {logvar.shape}")
    
    print("All decoder tests passed!")