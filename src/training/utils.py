"""
Training Utilities
Utility functions and classes for model training
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import time
import os
from pathlib import Path


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if early stopping criteria is met
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if should stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class LearningRateScheduler:
    """Learning rate scheduler with different strategies"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 strategy: str = 'plateau',
                 **kwargs):
        """
        Initialize learning rate scheduler
        
        Args:
            optimizer: PyTorch optimizer
            strategy: Scheduling strategy ('plateau', 'cosine', 'exponential', 'step')
            **kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.strategy = strategy
        
        if strategy == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                verbose=True
            )
        elif strategy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif strategy == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        elif strategy == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler"""
        if self.scheduler is not None:
            if self.strategy == 'plateau' and metric is not None:
                self.scheduler.step(metric)
            elif self.strategy != 'plateau':
                self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rate"""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainingMetrics:
    """Track and manage training metrics"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.best_metrics = {}
    
    def update(self, epoch: int, **kwargs):
        """Update metrics for current epoch"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_best(self, metric: str, mode: str = 'min') -> Tuple[float, int]:
        """
        Get best value for a metric
        
        Args:
            metric: Metric name
            mode: 'min' or 'max'
            
        Returns:
            (best_value, best_epoch)
        """
        if metric not in self.metrics or not self.metrics[metric]:
            return None, -1
        
        values = self.metrics[metric]
        if mode == 'min':
            best_value = min(values)
            best_epoch = values.index(best_value)
        else:
            best_value = max(values)
            best_epoch = values.index(best_value)
        
        return best_value, best_epoch
    
    def plot_metrics(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.metrics['train_loss'] and self.metrics['val_loss']:
            axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(self.metrics['val_loss'], label='Validation Loss', color='red')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Reconstruction loss
        if self.metrics['train_recon_loss'] and self.metrics['val_recon_loss']:
            axes[0, 1].plot(self.metrics['train_recon_loss'], label='Train Recon Loss', color='blue')
            axes[0, 1].plot(self.metrics['val_recon_loss'], label='Val Recon Loss', color='red')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Reconstruction Loss')
            axes[0, 1].set_title('Reconstruction Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        if self.metrics['learning_rate']:
            axes[1, 0].plot(self.metrics['learning_rate'], color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Epoch time
        if self.metrics['epoch_time']:
            axes[1, 1].plot(self.metrics['epoch_time'], color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Training Time per Epoch')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_metrics(self, save_path: str):
        """Save metrics to file"""
        np.savez(save_path, **self.metrics)
    
    def load_metrics(self, load_path: str):
        """Load metrics from file"""
        data = np.load(load_path, allow_pickle=True)
        for key in data.keys():
            self.metrics[key] = data[key].tolist()


class ModelCheckpoint:
    """Save model checkpoints during training"""
    
    def __init__(self,
                 checkpoint_dir: str,
                 model_name: str = 'model',
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_last: bool = True):
        """
        Initialize model checkpoint
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Base name for model files
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitoring
            save_best_only: Only save when monitored metric improves
            save_last: Always save the last epoch
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score
        }
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / f'{self.model_name}_last.pth'
            torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best or not self.save_best_only:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'{self.model_name}_epoch_{epoch}.pth'
        torch.save(checkpoint, epoch_path)
    
    def check_improvement(self, current_score: float) -> bool:
        """
        Check if current score is an improvement
        
        Args:
            current_score: Current metric score
            
        Returns:
            True if score improved
        """
        if self.mode == 'min':
            improved = current_score < self.best_score
        else:
            improved = current_score > self.best_score
        
        if improved:
            self.best_score = current_score
            return True
        
        return False
    
    def load_checkpoint(self, 
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_type: str = 'best') -> Dict:
        """
        Load model checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state
            checkpoint_type: 'best', 'last', or specific epoch number
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_type == 'best':
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
        elif checkpoint_type == 'last':
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_last.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_epoch_{checkpoint_type}.pth'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def print_model_summary(model: nn.Module):
    """Print model summary"""
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    # Test utilities
    import torch.nn as nn
    
    print("Testing training utilities...")
    
    # Test model parameter counting
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    print_model_summary(model)
    
    # Test training metrics
    metrics = TrainingMetrics()
    
    # Simulate some training epochs
    for epoch in range(10):
        train_loss = 1.0 - epoch * 0.1 + np.random.normal(0, 0.02)
        val_loss = 1.2 - epoch * 0.08 + np.random.normal(0, 0.03)
        
        metrics.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=0.001 * (0.9 ** epoch)
        )
    
    # Get best validation loss
    best_val_loss, best_epoch = metrics.get_best('val_loss', mode='min')
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Test early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=3)
    
    # Simulate validation losses
    val_losses = [1.0, 0.9, 0.8, 0.85, 0.87, 0.9]
    for i, val_loss in enumerate(val_losses):
        should_stop = early_stopping(val_loss, model)
        print(f"Epoch {i}: val_loss={val_loss:.3f}, should_stop={should_stop}")
        if should_stop:
            break
    
    print("Training utilities test completed!")