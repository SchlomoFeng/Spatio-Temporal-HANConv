#!/usr/bin/env python3
"""
Training Script for Heterogeneous Graph Autoencoder
Steam Pipeline Network Anomaly Detection System
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
import logging

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from data_preprocessing.topology_parser import TopologyParser
from data_preprocessing.sensor_data_cleaner import SensorDataCleaner
from data_preprocessing.graph_builder import GraphBuilder
from models.hetero_autoencoder import HeteroAutoencoder
from training.data_loader import GraphDataLoader
from training.utils import set_random_seed, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Setup output directories"""
    base_dir = Path(config['output']['base_dir'])
    
    directories = [
        base_dir,
        base_dir / config['output']['checkpoint_dir'],
        base_dir / config['output']['log_dir'],
        base_dir / config['output']['results_dir'],
        base_dir / config['output']['plots_dir']
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    handlers = [logging.StreamHandler()]
    
    if config['logging']['save_logs']:
        log_file = Path(config['output']['base_dir']) / config['logging']['log_file']
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def prepare_data(config: dict):
    """Prepare data for training"""
    logging.info("Preparing data...")
    
    # Initialize components
    topology_parser = TopologyParser(config['data']['blueprint_path'])
    sensor_cleaner = SensorDataCleaner(config['data']['sensor_data_path'])
    graph_builder = GraphBuilder(topology_parser, sensor_cleaner)
    
    # Parse topology
    logging.info("Parsing pipeline topology...")
    topology_data = topology_parser.parse_topology()
    logging.info(f"Parsed {topology_data['num_nodes']} nodes and {topology_data['num_edges']} edges")
    
    # Clean sensor data
    logging.info("Cleaning sensor data...")
    sensor_cleaner.clean_sensor_data(
        missing_strategy=config['preprocessing']['missing_strategy'],
        outlier_method=config['preprocessing']['outlier_method'],
        outlier_threshold=config['preprocessing']['outlier_threshold'],
        normalize_method=config['preprocessing']['normalize_method'],
        add_time_features=config['preprocessing']['add_time_features']
    )
    logging.info(f"Cleaned sensor data: {len(sensor_cleaner.sensor_columns)} sensors")
    
    # Create data loaders
    logging.info("Creating data loaders...")
    data_loader_factory = GraphDataLoader(
        graph_builder=graph_builder,
        batch_size=config['training']['batch_size'],
        window_size=config['data']['window_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        num_workers=config['training']['num_workers'],
        shuffle=config['training']['shuffle']
    )
    
    dataloaders = data_loader_factory.create_dataloaders()
    data_info = data_loader_factory.get_data_info()
    logging.info(f"Data splits: {data_info['splits']}")
    
    return dataloaders, data_info, topology_data, sensor_cleaner


def create_model(config: dict, data_info: dict, topology_data: dict) -> HeteroAutoencoder:
    """Create the heterogeneous autoencoder model"""
    logging.info("Creating model...")
    
    # Determine node feature dimensions
    num_node_types = len(topology_data['node_types'])
    
    node_feature_dims = {
        'stream': 4,  # x, y coordinates + type encoding + sensor reading
        'static': 2 + num_node_types  # x, y coordinates + one-hot type encoding
    }
    
    # Update if we have more information about actual feature dimensions
    if 'stream' in topology_data['nodes']['type'].values:
        # Actual implementation would determine this from the data
        pass
    
    model = HeteroAutoencoder(
        node_feature_dims=node_feature_dims,
        num_sensors=data_info['num_sensors'],
        encoder_hidden_dim=config['model']['encoder']['hidden_dim'],
        encoder_output_dim=config['model']['encoder']['output_dim'],
        encoder_num_heads=config['model']['encoder']['num_heads'],
        encoder_num_layers=config['model']['encoder']['num_layers'],
        decoder_hidden_dims=config['model']['decoder']['hidden_dims'],
        decoder_type=config['model']['decoder']['type'],
        dropout=config['model']['encoder']['dropout'],
        stream_lstm_hidden=config['model']['encoder']['stream_lstm_hidden'],
        stream_lstm_layers=config['model']['encoder']['stream_lstm_layers']
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model


def train_model(config: dict, model: HeteroAutoencoder, dataloaders: dict) -> dict:
    """Train the model"""
    logging.info("Starting model training...")
    
    from training.trainer import HeteroAutoencoderTrainer
    
    # Setup device
    if config['hardware']['device'] == 'auto':
        device = get_device()
    else:
        device = torch.device(config['hardware']['device'])
    
    logging.info(f"Using device: {device}")
    
    # Create trainer
    checkpoint_dir = Path(config['output']['base_dir']) / config['output']['checkpoint_dir']
    log_dir = Path(config['output']['base_dir']) / config['output']['log_dir']
    
    trainer = HeteroAutoencoderTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders.get('test'),
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir)
    )
    
    # Setup training configuration
    training_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'optimizer_type': config['training']['optimizer'],
        'scheduler_config': config['training'].get('scheduler'),
        'early_stopping_patience': config['training']['early_stopping']['patience'],
        'checkpoint_monitor': config['training']['early_stopping']['monitor']
    }
    
    trainer.setup_training(**training_config)
    
    # Train model
    training_results = trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_every=config['training']['save_every'],
        validate_every=config['training']['validate_every']
    )
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    logging.info(f"Training time: {training_results['training_time']:.2f}s")
    
    return training_results


def save_results(config: dict, training_results: dict, model: HeteroAutoencoder):
    """Save training results and model"""
    results_dir = Path(config['output']['base_dir']) / config['output']['results_dir']
    
    # Save training metrics
    if config['output']['save_metrics']:
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in training_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if hasattr(v, 'tolist'):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                if hasattr(value, 'tolist'):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
        
        with open(results_dir / 'training_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Heterogeneous Graph Autoencoder")
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing on pre-trained model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories and logging
    setup_directories(config)
    setup_logging(config)
    
    logging.info("Starting Steam Pipeline Network Anomaly Detection Training")
    logging.info(f"Configuration loaded from: {args.config}")
    
    # Set random seed for reproducibility
    if 'random_seed' in config:
        set_random_seed(config['random_seed'])
        logging.info(f"Random seed set to: {config['random_seed']}")
    
    try:
        # Prepare data
        dataloaders, data_info, topology_data, sensor_cleaner = prepare_data(config)
        
        # Create model
        model = create_model(config, data_info, topology_data)
        
        if args.test_only:
            logging.info("Test-only mode: loading pre-trained model...")
            # Load pre-trained model weights
            if args.resume:
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Loaded model from {args.resume}")
            else:
                logging.error("Test-only mode requires --resume argument")
                return
            
            # Run evaluation
            from training.trainer import HeteroAutoencoderTrainer
            trainer = HeteroAutoencoderTrainer(
                model=model,
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                test_loader=dataloaders.get('test')
            )
            
            if 'test' in dataloaders:
                test_results = trainer.evaluate(dataloaders['test'])
                logging.info("Test Results:")
                for key, value in test_results.items():
                    logging.info(f"  {key}: {value:.4f}")
        else:
            # Train model
            from training.trainer import HeteroAutoencoderTrainer
            training_results = train_model(config, model, dataloaders)
            
            # Save results
            save_results(config, training_results, model)
            
        logging.info("Process completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())