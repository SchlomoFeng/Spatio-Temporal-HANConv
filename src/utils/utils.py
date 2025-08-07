"""
Utility functions for the Steam Pipeline Anomaly Detection System
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('PipelineSystem')
    logger.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / 'system.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_training_curves(train_losses: List[float], val_losses: List[float], 
                        save_path: str):
    """Save training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_anomaly_summary_report(results_df: pd.DataFrame, 
                                config: Dict[str, Any]) -> str:
    """Create comprehensive anomaly detection summary report"""
    report = []
    
    # Header
    report.append("=" * 60)
    report.append("    S4 STEAM PIPELINE ANOMALY DETECTION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overview
    total_samples = len(results_df)
    anomaly_samples = results_df['is_anomaly'].sum()
    anomaly_rate = (anomaly_samples / total_samples) * 100
    
    report.append("DETECTION OVERVIEW:")
    report.append(f"  Total Samples Analyzed: {total_samples:,}")
    report.append(f"  Anomalies Detected: {anomaly_samples:,}")
    report.append(f"  Anomaly Rate: {anomaly_rate:.2f}%")
    report.append("")
    
    # Time analysis
    if 'timestamp' in results_df.columns:
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        time_range = results_df['timestamp'].max() - results_df['timestamp'].min()
        report.append("TIME ANALYSIS:")
        report.append(f"  Analysis Period: {time_range}")
        report.append(f"  Start Time: {results_df['timestamp'].min()}")
        report.append(f"  End Time: {results_df['timestamp'].max()}")
        report.append("")
    
    # Anomaly statistics
    if anomaly_samples > 0:
        anomaly_data = results_df[results_df['is_anomaly']]
        
        report.append("ANOMALY STATISTICS:")
        report.append(f"  Min Anomaly Score: {anomaly_data['anomaly_score'].min():.6f}")
        report.append(f"  Max Anomaly Score: {anomaly_data['anomaly_score'].max():.6f}")
        report.append(f"  Mean Anomaly Score: {anomaly_data['anomaly_score'].mean():.6f}")
        report.append(f"  Std Anomaly Score: {anomaly_data['anomaly_score'].std():.6f}")
        report.append("")
        
        # Top anomalous periods
        top_anomalies = anomaly_data.nlargest(5, 'anomaly_score')
        report.append("TOP 5 ANOMALOUS PERIODS:")
        for i, (_, row) in enumerate(top_anomalies.iterrows()):
            timestamp = row.get('timestamp', 'N/A')
            score = row['anomaly_score']
            nodes = row.get('top_anomalous_nodes', 'N/A')
            report.append(f"  {i+1}. Time: {timestamp}")
            report.append(f"     Score: {score:.6f}")
            report.append(f"     Nodes: {nodes}")
            report.append("")
    
    # Configuration summary
    report.append("SYSTEM CONFIGURATION:")
    report.append(f"  Model Type: Spatio-Temporal HANConv")
    report.append(f"  Window Size: {config['data']['window_size']}")
    report.append(f"  Stride: {config['data']['stride']}")
    report.append(f"  Threshold Method: {config['anomaly']['threshold_method']}")
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def plot_anomaly_timeline(results_df: pd.DataFrame, save_path: str):
    """Plot anomaly detection timeline"""
    if 'timestamp' not in results_df.columns:
        return
    
    plt.figure(figsize=(15, 8))
    
    # Convert timestamp to datetime
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    
    # Plot anomaly scores
    plt.subplot(2, 1, 1)
    plt.plot(results_df['timestamp'], results_df['anomaly_score'], 
             'b-', alpha=0.7, linewidth=1)
    
    # Highlight anomalies
    anomalies = results_df[results_df['is_anomaly']]
    if len(anomalies) > 0:
        plt.scatter(anomalies['timestamp'], anomalies['anomaly_score'],
                   c='red', s=50, alpha=0.8, zorder=5)
    
    plt.title('Anomaly Score Timeline')
    plt.ylabel('Anomaly Score')
    plt.grid(True, alpha=0.3)
    
    # Plot binary anomaly detection
    plt.subplot(2, 1, 2)
    anomaly_binary = results_df['is_anomaly'].astype(int)
    plt.plot(results_df['timestamp'], anomaly_binary, 'r-', linewidth=2)
    plt.fill_between(results_df['timestamp'], anomaly_binary, alpha=0.3)
    
    plt.title('Anomaly Detection (Binary)')
    plt.xlabel('Time')
    plt.ylabel('Anomaly Detected')
    plt.yticks([0, 1], ['Normal', 'Anomaly'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Check required sections
    required_sections = ['data', 'model', 'training', 'anomaly', 'system']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")
    
    # Check data configuration
    if 'data' in config:
        data_config = config['data']
        
        # Check ratios sum to 1
        ratios = data_config.get('train_ratio', 0) + \
                data_config.get('val_ratio', 0) + \
                data_config.get('test_ratio', 0)
        if abs(ratios - 1.0) > 1e-6:
            issues.append(f"Data split ratios don't sum to 1.0: {ratios}")
        
        # Check window size and stride
        window_size = data_config.get('window_size', 0)
        stride = data_config.get('stride', 0)
        if window_size <= 0:
            issues.append("window_size must be positive")
        if stride <= 0:
            issues.append("stride must be positive")
    
    # Check model dimensions
    if 'model' in config:
        model_config = config['model']
        
        # Check dimension consistency
        stream_dim = model_config.get('stream_input_dim', 0)
        if stream_dim <= 0:
            issues.append("stream_input_dim must be positive")
    
    return issues


def format_time_duration(seconds: float) -> str:
    """Format time duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def calculate_model_flops(model: torch.nn.Module, input_shapes: Dict[str, Tuple]) -> int:
    """Estimate model FLOPs (simplified calculation)"""
    # This is a simplified FLOP calculation
    # In practice, you might want to use libraries like fvcore or thop
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Linear layer: input_dim * output_dim
            total_flops += module.in_features * module.out_features
        elif isinstance(module, torch.nn.LSTM):
            # LSTM: approximate calculation
            hidden_size = module.hidden_size
            input_size = module.input_size
            num_layers = module.num_layers
            # Simplified LSTM FLOP calculation
            total_flops += 4 * hidden_size * (input_size + hidden_size) * num_layers
    
    return total_flops


class MetricsTracker:
    """Track and compute various metrics during training and evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.values = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics with new values"""
        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = 0.0
                self.counts[key] = 0
            
            self.values[key] += value * count
            self.counts[key] += count
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics"""
        averages = {}
        for key in self.values:
            if self.counts[key] > 0:
                averages[key] = self.values[key] / self.counts[key]
            else:
                averages[key] = 0.0
        return averages
    
    def get_summary(self) -> str:
        """Get formatted summary of metrics"""
        averages = self.compute()
        summary = []
        for key, value in averages.items():
            summary.append(f"{key}: {value:.6f}")
        return " | ".join(summary)