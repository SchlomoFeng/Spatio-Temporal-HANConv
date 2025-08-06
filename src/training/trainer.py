"""
Trainer
Main training pipeline for the heterogeneous graph autoencoder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from pathlib import Path
import numpy as np

from ..models.hetero_autoencoder import HeteroAutoencoder
from .utils import (EarlyStopping, LearningRateScheduler, TrainingMetrics, 
                   ModelCheckpoint, get_device, format_time, print_model_summary)
from .data_loader import GraphDataLoader


class HeteroAutoencoderTrainer:
    """Trainer for heterogeneous graph autoencoder"""
    
    def __init__(self,
                 model: HeteroAutoencoder,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        """
        Initialize trainer
        
        Args:
            model: HeteroAutoencoder model
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader (optional)
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = device if device is not None else get_device()
        self.model.to(self.device)
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.model_checkpoint = None
        self.metrics = TrainingMetrics()
        
        # Configuration
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        print_model_summary(self.model)
        self.logger.info(f"Training on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def setup_training(self,
                      learning_rate: float = 1e-3,
                      weight_decay: float = 1e-5,
                      optimizer_type: str = 'adam',
                      scheduler_config: Optional[Dict] = None,
                      early_stopping_patience: int = 10,
                      checkpoint_monitor: str = 'val_loss'):
        """
        Setup training components
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            scheduler_config: Learning rate scheduler configuration
            early_stopping_patience: Patience for early stopping
            checkpoint_monitor: Metric to monitor for checkpoints
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Setup scheduler
        if scheduler_config is not None:
            self.scheduler = LearningRateScheduler(self.optimizer, **scheduler_config)
        
        # Setup early stopping
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Setup checkpointing
        self.model_checkpoint = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            model_name='hetero_autoencoder',
            monitor=checkpoint_monitor
        )
        
        self.logger.info(f"Training setup complete:")
        self.logger.info(f"  Optimizer: {optimizer_type}")
        self.logger.info(f"  Learning rate: {learning_rate}")
        self.logger.info(f"  Weight decay: {weight_decay}")
        self.logger.info(f"  Early stopping patience: {early_stopping_patience}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Move batch to device
            hetero_data_list = batch['hetero_data_list']
            sensor_targets = batch['sensor_targets'].to(self.device)
            batch_size = batch['batch_size']
            
            batch_losses = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'kl_loss': 0.0
            }
            
            # Process each sample in the batch
            for i in range(batch_size):
                hetero_data = hetero_data_list[i]
                target = sensor_targets[i]
                
                # Move hetero data to device
                x_dict = {}
                edge_index_dict = {}
                
                for node_type in hetero_data.node_types:
                    if hasattr(hetero_data[node_type], 'x'):
                        x_dict[node_type] = hetero_data[node_type].x.to(self.device)
                
                for edge_type in hetero_data.edge_types:
                    edge_index_dict[edge_type] = hetero_data[edge_type].edge_index.to(self.device)
                
                # Forward pass
                output = self.model(x_dict, edge_index_dict, mapping_info=getattr(hetero_data, 'mapping_info', None))
                
                # Compute losses
                losses = self.model.compute_loss(output, target)
                
                # Accumulate batch losses
                for key in batch_losses:
                    if key in losses:
                        batch_losses[key] += losses[key]
            
            # Average batch losses
            for key in batch_losses:
                batch_losses[key] /= batch_size
            
            # Backward pass
            total_loss = batch_losses['total_loss']
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate epoch losses
            for key in epoch_losses:
                epoch_losses[key] += batch_losses[key].item()
            
            num_batches += 1
            
            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {total_loss.item():.4f}")
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                hetero_data_list = batch['hetero_data_list']
                sensor_targets = batch['sensor_targets'].to(self.device)
                batch_size = batch['batch_size']
                
                batch_losses = {
                    'total_loss': 0.0,
                    'recon_loss': 0.0,
                    'kl_loss': 0.0
                }
                
                # Process each sample in the batch
                for i in range(batch_size):
                    hetero_data = hetero_data_list[i]
                    target = sensor_targets[i]
                    
                    # Move hetero data to device
                    x_dict = {}
                    edge_index_dict = {}
                    
                    for node_type in hetero_data.node_types:
                        if hasattr(hetero_data[node_type], 'x'):
                            x_dict[node_type] = hetero_data[node_type].x.to(self.device)
                    
                    for edge_type in hetero_data.edge_types:
                        edge_index_dict[edge_type] = hetero_data[edge_type].edge_index.to(self.device)
                    
                    # Forward pass
                    output = self.model(x_dict, edge_index_dict, mapping_info=getattr(hetero_data, 'mapping_info', None))
                    
                    # Compute losses
                    losses = self.model.compute_loss(output, target)
                    
                    # Accumulate batch losses
                    for key in batch_losses:
                        if key in losses:
                            batch_losses[key] += losses[key]
                
                # Average batch losses
                for key in batch_losses:
                    batch_losses[key] /= batch_size
                
                # Accumulate epoch losses
                for key in epoch_losses:
                    epoch_losses[key] += batch_losses[key].item()
                
                num_batches += 1
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self,
              num_epochs: int = 100,
              training_config: Optional[Dict] = None,
              save_every: int = 10,
              validate_every: int = 1) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            training_config: Training configuration
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
            
        Returns:
            Training history and final metrics
        """
        if self.optimizer is None:
            # Setup with default configuration if not already done
            config = training_config or {}
            self.setup_training(**config)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = None
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()
            
            epoch_time = time.time() - epoch_start_time
            
            # Update metrics
            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            
            self.metrics.update(
                epoch=epoch,
                train_loss=train_metrics['total_loss'],
                train_recon_loss=train_metrics['recon_loss'],
                learning_rate=lr,
                epoch_time=epoch_time
            )
            
            if val_metrics:
                self.metrics.update(
                    val_loss=val_metrics['total_loss'],
                    val_recon_loss=val_metrics['recon_loss']
                )
                current_val_loss = val_metrics['total_loss']
            else:
                current_val_loss = train_metrics['total_loss']
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{num_epochs} - "
            log_msg += f"Train Loss: {train_metrics['total_loss']:.4f}, "
            if val_metrics:
                log_msg += f"Val Loss: {val_metrics['total_loss']:.4f}, "
            log_msg += f"LR: {lr:.6f}, Time: {format_time(epoch_time)}"
            
            self.logger.info(log_msg)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(current_val_loss)
            
            # Checkpointing
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
            
            if epoch % save_every == 0 or is_best:
                self.model_checkpoint.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={'val_loss': current_val_loss} if val_metrics else {'train_loss': train_metrics['total_loss']},
                    is_best=is_best
                )
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(current_val_loss, self.model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {format_time(training_time)}")
        
        # Save final metrics
        self.metrics.save_metrics(self.log_dir / 'training_metrics.npz')
        
        # Plot training curves
        self.metrics.plot_metrics(
            save_path=self.log_dir / 'training_curves.png',
            show=False
        )
        
        # Test evaluation if test loader provided
        test_metrics = None
        if self.test_loader:
            test_metrics = self.evaluate(self.test_loader)
        
        return {
            'training_time': training_time,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'test_metrics': test_metrics,
            'metrics': self.metrics.metrics
        }
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on given data loader
        
        Args:
            data_loader: DataLoader to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        eval_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                hetero_data_list = batch['hetero_data_list']
                sensor_targets = batch['sensor_targets'].to(self.device)
                batch_size = batch['batch_size']
                
                batch_errors = []
                batch_losses = {
                    'total_loss': 0.0,
                    'recon_loss': 0.0,
                    'kl_loss': 0.0
                }
                
                for i in range(batch_size):
                    hetero_data = hetero_data_list[i]
                    target = sensor_targets[i]
                    
                    # Move hetero data to device
                    x_dict = {}
                    edge_index_dict = {}
                    
                    for node_type in hetero_data.node_types:
                        if hasattr(hetero_data[node_type], 'x'):
                            x_dict[node_type] = hetero_data[node_type].x.to(self.device)
                    
                    for edge_type in hetero_data.edge_types:
                        edge_index_dict[edge_type] = hetero_data[edge_type].edge_index.to(self.device)
                    
                    # Forward pass
                    output = self.model(x_dict, edge_index_dict, mapping_info=getattr(hetero_data, 'mapping_info', None))
                    
                    # Compute losses
                    losses = self.model.compute_loss(output, target)
                    
                    # Get reconstruction errors
                    errors = self.model.get_reconstruction_errors(output, target)
                    batch_errors.append(errors.cpu().numpy())
                    
                    # Accumulate batch losses
                    for key in batch_losses:
                        if key in losses:
                            batch_losses[key] += losses[key]
                
                # Average batch losses
                for key in batch_losses:
                    batch_losses[key] /= batch_size
                
                # Accumulate epoch losses
                for key in eval_losses:
                    eval_losses[key] += batch_losses[key].item()
                
                reconstruction_errors.extend(batch_errors)
                num_batches += 1
        
        # Average losses
        for key in eval_losses:
            eval_losses[key] /= num_batches
        
        # Compute reconstruction error statistics
        all_errors = np.concatenate(reconstruction_errors, axis=0)
        error_stats = {
            'mean_recon_error': np.mean(all_errors),
            'std_recon_error': np.std(all_errors),
            'max_recon_error': np.max(all_errors),
            'min_recon_error': np.min(all_errors)
        }
        
        eval_metrics = {**eval_losses, **error_stats}
        
        self.logger.info("Evaluation Results:")
        for key, value in eval_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return eval_metrics


if __name__ == "__main__":
    # Example usage
    from data_preprocessing.topology_parser import TopologyParser
    from data_preprocessing.sensor_data_cleaner import SensorDataCleaner
    from data_preprocessing.graph_builder import GraphBuilder
    from models.hetero_autoencoder import HeteroAutoencoder
    
    print("Testing trainer...")
    
    # Initialize components  
    topology_parser = TopologyParser("../../blueprint/0708YTS4.txt")
    sensor_cleaner = SensorDataCleaner("../../data/0708YTS4.csv")
    graph_builder = GraphBuilder(topology_parser, sensor_cleaner)
    
    # Create data loader
    data_loader_factory = GraphDataLoader(
        graph_builder=graph_builder,
        batch_size=4,
        window_size=30
    )
    
    dataloaders = data_loader_factory.create_dataloaders()
    
    # Create model
    node_feature_dims = {
        'stream': 4,
        'static': 7
    }
    
    model = HeteroAutoencoder(
        node_feature_dims=node_feature_dims,
        num_sensors=36,
        decoder_type='basic'
    )
    
    # Create trainer
    trainer = HeteroAutoencoderTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test']
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=1e-3,
        early_stopping_patience=5
    )
    
    print("Trainer setup complete!")
    print("Ready for training...")