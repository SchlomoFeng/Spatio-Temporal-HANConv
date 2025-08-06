"""
Heterogeneous Graph Autoencoder
Complete autoencoder model combining encoder and decoder for anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from .hetero_encoder import HeteroEncoder
from .decoder import Decoder, AttentionDecoder, VariationalDecoder


class HeteroAutoencoder(nn.Module):
    """Heterogeneous Graph Autoencoder for pipeline anomaly detection"""
    
    def __init__(self,
                 node_feature_dims: Dict[str, int],
                 num_sensors: int = 36,
                 encoder_hidden_dim: int = 64,
                 encoder_output_dim: int = 32,
                 encoder_num_heads: int = 4,
                 encoder_num_layers: int = 2,
                 decoder_hidden_dims: List[int] = [64, 128, 64],
                 decoder_type: str = 'basic',  # 'basic', 'attention', 'variational'
                 dropout: float = 0.1,
                 stream_lstm_hidden: int = 32,
                 stream_lstm_layers: int = 1):
        """
        Initialize heterogeneous autoencoder
        
        Args:
            node_feature_dims: Dictionary mapping node types to feature dimensions
            num_sensors: Number of sensors to reconstruct
            encoder_hidden_dim: Encoder hidden dimension
            encoder_output_dim: Encoder output dimension (embedding size)
            encoder_num_heads: Number of attention heads in encoder
            encoder_num_layers: Number of encoder layers
            decoder_hidden_dims: Decoder hidden layer dimensions
            decoder_type: Type of decoder ('basic', 'attention', 'variational')
            dropout: Dropout rate
            stream_lstm_hidden: LSTM hidden size for stream nodes
            stream_lstm_layers: Number of LSTM layers for stream nodes
        """
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.num_sensors = num_sensors
        self.encoder_output_dim = encoder_output_dim
        self.decoder_type = decoder_type
        
        # Create metadata for HANConv (node_types, edge_types)
        node_types = list(node_feature_dims.keys())
        # Default edge types - will be updated when we have actual data
        edge_types = [
            ('stream', 'connects', 'stream'),
            ('stream', 'flows_to', 'static'),
            ('static', 'feeds', 'stream'),
            ('static', 'connected', 'static')
        ]
        metadata = (node_types, edge_types)
        
        # Encoder
        self.encoder = HeteroEncoder(
            node_feature_dims=node_feature_dims,
            metadata=metadata,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            num_heads=encoder_num_heads,
            num_layers=encoder_num_layers,
            dropout=dropout,
            stream_lstm_hidden=stream_lstm_hidden,
            stream_lstm_layers=stream_lstm_layers
        )
        
        # Decoder
        if decoder_type == 'attention':
            self.decoder = AttentionDecoder(
                stream_embedding_dim=encoder_output_dim,
                num_sensors=num_sensors,
                attention_dim=encoder_hidden_dim,
                hidden_dims=decoder_hidden_dims,
                num_heads=encoder_num_heads,
                dropout=dropout
            )
        elif decoder_type == 'variational':
            self.decoder = VariationalDecoder(
                stream_embedding_dim=encoder_output_dim,
                num_sensors=num_sensors,
                hidden_dims=decoder_hidden_dims,
                dropout=dropout
            )
        else:  # basic decoder
            self.decoder = Decoder(
                stream_embedding_dim=encoder_output_dim,
                num_sensors=num_sensors,
                hidden_dims=decoder_hidden_dims,
                dropout=dropout
            )
    
    def encode(self, 
               x_dict: Dict[str, torch.Tensor],
               edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
               batch_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Encode the heterogeneous graph
        
        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by type
            batch_dict: Batch indices (optional)
            
        Returns:
            Node embeddings by type
        """
        return self.encoder(x_dict, edge_index_dict, batch_dict)
    
    def decode(self, 
               embeddings_dict: Dict[str, torch.Tensor],
               mapping_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Decode stream embeddings to sensor readings
        
        Args:
            embeddings_dict: Node embeddings by type
            mapping_info: Optional sensor mapping information
            
        Returns:
            Reconstructed sensor readings
        """
        # Extract stream embeddings
        if 'stream' not in embeddings_dict:
            raise ValueError("No stream embeddings found for decoding")
        
        stream_embeddings = embeddings_dict['stream']
        
        if self.decoder_type == 'variational':
            return self.decoder(stream_embeddings, return_distribution=False)
        else:
            return self.decoder(stream_embeddings, mapping_info)
    
    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                batch_dict: Optional[Dict[str, torch.Tensor]] = None,
                mapping_info: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through autoencoder
        
        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by type
            batch_dict: Batch indices (optional)
            mapping_info: Sensor mapping information
            
        Returns:
            Dictionary containing reconstructions and embeddings
        """
        # Encode
        embeddings = self.encode(x_dict, edge_index_dict, batch_dict)
        
        # Decode
        if self.decoder_type == 'variational' and self.training:
            reconstructed, mu, logvar = self.decoder(embeddings, return_distribution=True)
            return {
                'reconstructed': reconstructed,
                'embeddings': embeddings,
                'mu': mu,
                'logvar': logvar
            }
        else:
            reconstructed = self.decode(embeddings, mapping_info)
            return {
                'reconstructed': reconstructed,
                'embeddings': embeddings
            }
    
    def reconstruction_loss(self, 
                          reconstructed: torch.Tensor,
                          target: torch.Tensor,
                          reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate reconstruction loss
        
        Args:
            reconstructed: Reconstructed sensor readings
            target: Target sensor readings
            reduction: Loss reduction ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction loss
        """
        return F.mse_loss(reconstructed, target, reduction=reduction)
    
    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence loss for variational decoder
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            KL divergence loss
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def compute_loss(self,
                     output: Dict[str, torch.Tensor],
                     target_sensors: torch.Tensor,
                     kl_weight: float = 1e-4) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including reconstruction and KL divergence
        
        Args:
            output: Model output dictionary
            target_sensors: Target sensor readings
            kl_weight: Weight for KL divergence loss
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(output['reconstructed'], target_sensors)
        losses['reconstruction'] = recon_loss
        
        # KL divergence loss (for variational decoder)
        if 'mu' in output and 'logvar' in output:
            kl_loss = self.kl_divergence_loss(output['mu'], output['logvar'])
            losses['kl_divergence'] = kl_loss
            losses['total'] = recon_loss + kl_weight * kl_loss
        else:
            losses['total'] = recon_loss
        
        return losses
    
    def get_reconstruction_errors(self,
                                output: Dict[str, torch.Tensor],
                                target_sensors: torch.Tensor) -> torch.Tensor:
        """
        Get per-sensor reconstruction errors
        
        Args:
            output: Model output
            target_sensors: Target sensor readings
            
        Returns:
            Per-sensor reconstruction errors
        """
        reconstructed = output['reconstructed']
        errors = torch.abs(reconstructed - target_sensors)
        return errors
    
    def detect_anomalies(self,
                        output: Dict[str, torch.Tensor],
                        target_sensors: torch.Tensor,
                        threshold: float) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies based on reconstruction error
        
        Args:
            output: Model output
            target_sensors: Target sensor readings
            threshold: Anomaly detection threshold
            
        Returns:
            Dictionary with anomaly information
        """
        errors = self.get_reconstruction_errors(output, target_sensors)
        
        # Overall anomaly score
        total_error = errors.sum()
        is_anomaly = total_error > threshold
        
        # Per-sensor anomaly scores
        per_sensor_errors = errors
        sensor_anomalies = per_sensor_errors > (threshold / len(errors))
        
        return {
            'is_anomaly': is_anomaly,
            'total_error': total_error,
            'per_sensor_errors': per_sensor_errors,
            'sensor_anomalies': sensor_anomalies,
            'anomaly_score': total_error / len(errors)  # Normalized score
        }


class HeteroAutoencoderEnsemble(nn.Module):
    """Ensemble of heterogeneous autoencoders for robust anomaly detection"""
    
    def __init__(self,
                 node_feature_dims: Dict[str, int],
                 num_sensors: int = 36,
                 num_models: int = 3,
                 model_configs: List[Dict] = None):
        """
        Initialize autoencoder ensemble
        
        Args:
            node_feature_dims: Node feature dimensions
            num_sensors: Number of sensors
            num_models: Number of models in ensemble
            model_configs: List of configuration dictionaries for each model
        """
        super().__init__()
        
        self.num_models = num_models
        self.num_sensors = num_sensors
        
        # Default configurations
        if model_configs is None:
            model_configs = [
                {'decoder_type': 'basic', 'encoder_hidden_dim': 64},
                {'decoder_type': 'attention', 'encoder_hidden_dim': 32},
                {'decoder_type': 'variational', 'encoder_hidden_dim': 128}
            ]
        
        # Create ensemble models
        self.models = nn.ModuleList()
        for i in range(num_models):
            config = model_configs[i] if i < len(model_configs) else {}
            
            model = HeteroAutoencoder(
                node_feature_dims=node_feature_dims,
                num_sensors=num_sensors,
                **config
            )
            self.models.append(model)
    
    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                batch_dict: Optional[Dict[str, torch.Tensor]] = None,
                mapping_info: Optional[Dict] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass through ensemble
        
        Returns:
            List of outputs from each model
        """
        outputs = []
        for model in self.models:
            output = model(x_dict, edge_index_dict, batch_dict, mapping_info)
            outputs.append(output)
        return outputs
    
    def ensemble_prediction(self,
                          outputs: List[Dict[str, torch.Tensor]],
                          method: str = 'mean') -> torch.Tensor:
        """
        Combine predictions from ensemble
        
        Args:
            outputs: List of model outputs
            method: Combination method ('mean', 'median', 'vote')
            
        Returns:
            Combined prediction
        """
        reconstructions = [output['reconstructed'] for output in outputs]
        
        if method == 'mean':
            return torch.stack(reconstructions).mean(dim=0)
        elif method == 'median':
            return torch.stack(reconstructions).median(dim=0)[0]
        else:  # mean as default
            return torch.stack(reconstructions).mean(dim=0)
    
    def compute_ensemble_loss(self,
                            outputs: List[Dict[str, torch.Tensor]],
                            target_sensors: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble loss
        
        Args:
            outputs: List of model outputs
            target_sensors: Target sensors
            
        Returns:
            Total ensemble loss
        """
        total_loss = 0
        for i, output in enumerate(outputs):
            model = self.models[i]
            losses = model.compute_loss(output, target_sensors)
            total_loss += losses['total']
        
        return total_loss / len(outputs)


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example configuration
    node_feature_dims = {
        'stream': 4,  # 3 static features + 1 sensor reading
        'static': 7   # coordinates + one-hot encoding
    }
    
    num_sensors = 36
    
    # Test basic autoencoder
    autoencoder = HeteroAutoencoder(
        node_feature_dims=node_feature_dims,
        num_sensors=num_sensors,
        decoder_type='basic'
    ).to(device)
    
    print(f"Autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters())}")
    
    # Example data
    x_dict = {
        'stream': torch.randn(10, 4).to(device),
        'static': torch.randn(20, 7).to(device)
    }
    
    edge_index_dict = {
        ('stream', 'connects', 'stream'): torch.randint(0, 10, (2, 5)).to(device),
        ('static', 'feeds', 'stream'): torch.randint(0, 20, (2, 3)).to(device).t().contiguous()
    }
    
    target_sensors = torch.randn(num_sensors).to(device)
    
    # Forward pass
    try:
        output = autoencoder(x_dict, edge_index_dict)
        print("Forward pass successful!")
        print(f"Reconstructed shape: {output['reconstructed'].shape}")
        
        # Compute loss
        losses = autoencoder.compute_loss(output, target_sensors)
        print(f"Reconstruction loss: {losses['reconstruction'].item():.4f}")
        
        # Test anomaly detection
        anomaly_info = autoencoder.detect_anomalies(output, target_sensors, threshold=1.0)
        print(f"Anomaly detected: {anomaly_info['is_anomaly'].item()}")
        print(f"Anomaly score: {anomaly_info['anomaly_score'].item():.4f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("Autoencoder test completed!")