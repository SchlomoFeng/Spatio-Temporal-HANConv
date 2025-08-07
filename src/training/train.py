"""
Training script for Steam Pipeline Anomaly Detection System

This script handles:
1. Model training with early stopping
2. Validation monitoring
3. Checkpoint saving
4. Tensorboard logging
5. Anomaly threshold computation
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import (
    PipelineTopologyParser, SensorDataProcessor, HeteroGraphBuilder,
    DataSplitter, create_hetero_data
)
from src.data.dataset import StreamPipelineDataset, collate_hetero_batch
from src.models.han_autoencoder import create_model
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class PipelineTrainer:
    """Trainer class for steam pipeline anomaly detection model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logger()
        
        # Create directories
        self._create_directories()
        
        # Initialize data module
        self.data_module = self._setup_data()
        
        # Initialize model
        self.model = self._setup_model()
        
        # Initialize training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        self.early_stopping = self._setup_early_stopping()
        
        # Initialize logging
        self.writer = self._setup_tensorboard()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        device_config = self.config['system']['device']
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
            
        self.logger.info(f"Using device: {device}")
        return device
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('PipelineTrainer')
        logger.setLevel(getattr(logging, self.config['logging']['log_level']))
        
        # Create handlers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['logging']['log_dir'],
            self.config['logging']['tensorboard_dir'],
            self.config['checkpoints']['save_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_data(self):
        """Setup data module"""
        self.logger.info("Setting up data module...")
        
        # Initialize components
        topology_parser = PipelineTopologyParser(self.config['data']['blueprint_path'])
        sensor_processor = SensorDataProcessor(
            self.config['data']['sensor_data_path'],
            self.config['data']['scaler_type']
        )
        data_splitter = DataSplitter(
            self.config['data']['train_ratio'],
            self.config['data']['val_ratio'],
            self.config['data']['test_ratio']
        )
        
        # Process topology
        nodeDF, edgeDF = topology_parser.extract_nodes_and_edges()
        nodes_df, node_id_to_index = topology_parser.process_nodes(nodeDF)
        edges_df = topology_parser.process_edges(edgeDF, node_id_to_index)
        
        # Process sensor data
        sensor_df = sensor_processor.load_sensor_data()
        sensor_df = sensor_processor.clean_data(sensor_df)
        
        # Split data
        train_df, val_df, test_df = data_splitter.split_time_series(sensor_df)
        
        # Fit scaler
        sensor_processor.create_scaler(train_df)
        
        # Normalize data
        train_df = sensor_processor.normalize_data(train_df)
        val_df = sensor_processor.normalize_data(val_df)
        test_df = sensor_processor.normalize_data(test_df)
        
        # Create datasets
        window_size = self.config['data']['window_size']
        stride = self.config['data']['stride']
        
        self.train_dataset = StreamPipelineDataset(
            train_df, nodes_df, edges_df, window_size, stride,
            sensor_processor.scaler, mode='train'
        )
        
        self.val_dataset = StreamPipelineDataset(
            val_df, nodes_df, edges_df, window_size, stride,
            sensor_processor.scaler, mode='val'
        )
        
        self.test_dataset = StreamPipelineDataset(
            test_df, nodes_df, edges_df, window_size, stride,
            sensor_processor.scaler, mode='test'
        )
        
        # Store for later use
        self.sensor_processor = sensor_processor
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        
        self.logger.info("Data module setup completed")
        
        return {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
    
    def _setup_model(self):
        """Setup model"""
        self.logger.info("Setting up model...")
        model = create_model(self.config)
        model = model.to(self.device)
        return model
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_config = self.config['training']
        
        if optimizer_config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['optimizer']}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            scheduler = None
            
        return scheduler
    
    def _setup_criterion(self):
        """Setup loss function"""
        loss_config = self.config['training']['loss_function']
        
        if loss_config == 'MSELoss':
            return nn.MSELoss()
        elif loss_config == 'L1Loss':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_config}")
    
    def _setup_early_stopping(self):
        """Setup early stopping"""
        training_config = self.config['training']
        return EarlyStopping(
            patience=training_config['patience'],
            min_delta=training_config['min_delta']
        )
    
    def _setup_tensorboard(self):
        """Setup tensorboard writer"""
        log_dir = Path(self.config['logging']['tensorboard_dir'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return SummaryWriter(log_dir / f'run_{timestamp}')
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=collate_hetero_batch
        )
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            output = self.model(batch)
            
            # Compute loss
            loss = self.model.compute_reconstruction_loss(
                output, batch['targets'], self.criterion
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=collate_hetero_batch
        )
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                output = self.model(batch)
                
                # Compute loss
                loss = self.model.compute_reconstruction_loss(
                    output, batch['targets'], self.criterion
                )
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _move_to_device(self, batch: dict) -> dict:
        """Move batch data to device"""
        if 'stream_sequences' in batch:
            batch['stream_sequences'] = batch['stream_sequences'].to(self.device)
        
        if 'static_features' in batch:
            for node_type, features in batch['static_features'].items():
                batch['static_features'][node_type] = features.to(self.device)
        
        if 'edge_index_dict' in batch:
            for edge_type, edge_index in batch['edge_index_dict'].items():
                batch['edge_index_dict'][edge_type] = edge_index.to(self.device)
        
        if 'targets' in batch:
            for target_type, target_data in batch['targets'].items():
                batch['targets'][target_type] = target_data.to(self.device)
        
        return batch
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoints']['save_dir'])
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save last checkpoint
        if self.config['checkpoints']['save_last']:
            torch.save(checkpoint, checkpoint_dir / 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best and self.config['checkpoints']['save_best']:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        epochs = self.config['training']['epochs']
        save_every = self.config['logging']['save_every_n_epochs']
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            if self.scheduler is not None:
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            self.logger.info(
                f'Epoch {epoch + 1}/{epochs} - '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}'
            )
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoints
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f'Early stopping at epoch {epoch + 1}')
                break
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def compute_anomaly_threshold(self) -> float:
        """Compute anomaly threshold on validation set"""
        self.logger.info("Computing anomaly threshold...")
        
        self.model.eval()
        anomaly_scores = []
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_hetero_batch
        )
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_to_device(batch)
                output = self.model(batch)
                
                scores = self.model.compute_anomaly_scores(output, batch['targets'])
                if len(scores) > 0:
                    anomaly_scores.extend(scores.cpu().numpy())
        
        anomaly_scores = np.array(anomaly_scores)
        
        # Compute threshold based on configuration
        anomaly_config = self.config['anomaly']
        method = anomaly_config['threshold_method']
        
        if method == 'percentile':
            threshold = np.percentile(anomaly_scores, anomaly_config['threshold_percentile'])
        elif method == 'std_multiple':
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            threshold = mean_score + anomaly_config['std_multiple'] * std_score
        elif method == 'iqr':
            q75, q25 = np.percentile(anomaly_scores, [75, 25])
            iqr = q75 - q25
            threshold = q75 + anomaly_config['iqr_factor'] * iqr
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        self.logger.info(f"Anomaly threshold computed: {threshold:.6f}")
        
        # Save threshold
        threshold_path = Path(self.config['checkpoints']['save_dir']) / 'anomaly_threshold.txt'
        with open(threshold_path, 'w') as f:
            f.write(str(threshold))
        
        return threshold


def main():
    parser = argparse.ArgumentParser(description='Train Steam Pipeline Anomaly Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['system']['random_seed'])
    np.random.seed(config['system']['random_seed'])
    
    # Create trainer
    trainer = PipelineTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train model
    trainer.train()
    
    # Compute anomaly threshold
    threshold = trainer.compute_anomaly_threshold()
    
    print(f"Training completed! Anomaly threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()