#!/usr/bin/env python3
"""
Main entry point for S4 Steam Pipeline Anomaly Detection System

This script provides a unified interface for:
1. Data preprocessing and validation
2. Model training
3. Anomaly detection (real-time and batch)
4. Visualization and reporting
"""

import argparse
import yaml
import sys
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.training.train import PipelineTrainer
from src.inference.anomaly_detection import AnomalyDetector, BatchAnomalyDetector
from src.data.preprocessing import PipelineTopologyParser, SensorDataProcessor


def validate_data(config: dict):
    """Validate data integrity and structure"""
    print("=== DATA VALIDATION ===")
    
    # Check files exist
    blueprint_path = config['data']['blueprint_path']
    sensor_path = config['data']['sensor_data_path']
    
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    if not Path(sensor_path).exists():
        raise FileNotFoundError(f"Sensor data file not found: {sensor_path}")
    
    print(f"✓ Blueprint file found: {blueprint_path}")
    print(f"✓ Sensor data file found: {sensor_path}")
    
    # Validate topology data
    topology_parser = PipelineTopologyParser(blueprint_path)
    nodeDF, edgeDF = topology_parser.extract_nodes_and_edges()
    nodes_df, node_id_to_index = topology_parser.process_nodes(nodeDF)
    edges_df = topology_parser.process_edges(edgeDF, node_id_to_index)
    
    print(f"✓ Topology loaded: {len(nodes_df)} nodes, {len(edges_df)} edges")
    print(f"✓ Node types: {nodes_df['node_type'].value_counts().to_dict()}")
    
    # Validate sensor data
    sensor_processor = SensorDataProcessor(sensor_path)
    sensor_df = sensor_processor.load_sensor_data()
    sensor_df = sensor_processor.clean_data(sensor_df)
    
    print(f"✓ Sensor data loaded: {len(sensor_df)} records, {len(sensor_processor.sensor_columns)} sensors")
    print(f"✓ Time range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")
    
    # Check for sufficient data
    window_size = config['data']['window_size']
    stride = config['data']['stride']
    min_samples = window_size * 10  # Minimum samples needed
    
    if len(sensor_df) < min_samples:
        raise ValueError(f"Insufficient data: {len(sensor_df)} samples, need at least {min_samples}")
    
    print(f"✓ Sufficient data for training with window_size={window_size}")
    
    # Estimate dataset sizes
    num_windows = (len(sensor_df) - window_size) // stride + 1
    train_windows = int(num_windows * config['data']['train_ratio'])
    val_windows = int(num_windows * config['data']['val_ratio'])
    test_windows = num_windows - train_windows - val_windows
    
    print(f"✓ Estimated samples - Train: {train_windows}, Val: {val_windows}, Test: {test_windows}")
    print("Data validation completed successfully!\n")


def train_model(config: dict, resume_checkpoint: str = None):
    """Train the anomaly detection model"""
    print("=== MODEL TRAINING ===")
    
    # Set random seeds
    torch.manual_seed(config['system']['random_seed'])
    
    # Create trainer
    trainer = PipelineTrainer(config)
    
    # Resume from checkpoint if specified
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"✓ Resumed from checkpoint: {resume_checkpoint}")
    
    # Train model
    trainer.train()
    
    # Compute anomaly threshold
    threshold = trainer.compute_anomaly_threshold()
    
    print(f"✓ Training completed! Anomaly threshold: {threshold:.6f}")
    print(f"✓ Best model saved to: {config['checkpoints']['save_dir']}/best_checkpoint.pth")
    print(f"✓ Threshold saved to: {config['checkpoints']['save_dir']}/anomaly_threshold.txt\n")


def detect_anomalies(config: dict, model_path: str, threshold_path: str, mode: str = 'batch'):
    """Perform anomaly detection"""
    print("=== ANOMALY DETECTION ===")
    
    if mode == 'real-time':
        print("Real-time anomaly detection mode")
        detector = AnomalyDetector(config, model_path, threshold_path)
        
        # Example with synthetic data
        import numpy as np
        from datetime import datetime
        
        window_size = config['data']['window_size']
        num_sensors = config['model']['stream_input_dim']
        
        # Generate example sensor data
        print("Generating example sensor data...")
        sensor_data = np.random.randn(window_size, num_sensors) * 0.1  # Normal data
        
        # Add anomaly to some sensors
        sensor_data[-5:, :5] += np.random.randn(5, 5) * 2.0  # Anomalous readings
        
        is_anomaly, score, details = detector.detect_anomaly(sensor_data)
        
        print(f"✓ Anomaly detected: {is_anomaly}")
        print(f"✓ Anomaly score: {score:.6f} (threshold: {detector.anomaly_threshold:.6f})")
        
        if is_anomaly:
            report = detector.generate_anomaly_report(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                is_anomaly, score, details
            )
            print("\n" + report)
            
            # Save visualization
            viz_path = Path(config['visualization']['save_dir']) / f'anomaly_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            detector.visualize_anomaly(details, str(viz_path))
    
    elif mode == 'batch':
        print("Batch anomaly detection mode")
        
        # Create test dataset
        from src.data.preprocessing import PipelineTopologyParser, SensorDataProcessor, DataSplitter
        from src.data.dataset import StreamPipelineDataset
        
        # Process data (same as training)
        topology_parser = PipelineTopologyParser(config['data']['blueprint_path'])
        sensor_processor = SensorDataProcessor(config['data']['sensor_data_path'])
        
        nodeDF, edgeDF = topology_parser.extract_nodes_and_edges()
        nodes_df, node_id_to_index = topology_parser.process_nodes(nodeDF)
        edges_df = topology_parser.process_edges(edgeDF, node_id_to_index)
        
        sensor_df = sensor_processor.load_sensor_data()
        sensor_df = sensor_processor.clean_data(sensor_df)
        
        # Use last portion as test data
        test_start = int(len(sensor_df) * 0.85)
        test_df = sensor_df.iloc[test_start:].reset_index(drop=True)
        
        # Fit scaler (should match training)
        train_df = sensor_df.iloc[:int(len(sensor_df) * 0.7)]
        sensor_processor.create_scaler(train_df)
        test_df = sensor_processor.normalize_data(test_df)
        
        # Create test dataset
        test_dataset = StreamPipelineDataset(
            test_df, nodes_df, edges_df,
            config['data']['window_size'], 
            config['data']['stride'],
            sensor_processor.scaler, mode='test'
        )
        
        print(f"✓ Test dataset created: {len(test_dataset)} samples")
        
        # Run batch detection
        batch_detector = BatchAnomalyDetector(config, model_path, threshold_path)
        results_df = batch_detector.detect_anomalies_in_dataset(test_dataset)
        
        # Save results
        output_dir = Path(config['visualization']['save_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'anomaly_detection_results.csv'
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        num_anomalies = results_df['is_anomaly'].sum()
        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Detection Summary:")
        print(f"  - Total samples: {len(results_df)}")
        print(f"  - Anomalies detected: {num_anomalies}")
        print(f"  - Anomaly rate: {num_anomalies/len(results_df)*100:.2f}%")
        
        if num_anomalies > 0:
            print(f"  - Anomaly score range: {results_df[results_df['is_anomaly']]['anomaly_score'].min():.6f} - {results_df[results_df['is_anomaly']]['anomaly_score'].max():.6f}")


def visualize_network(config: dict):
    """Visualize the pipeline network"""
    print("=== NETWORK VISUALIZATION ===")
    
    # Use existing visualization script
    import subprocess
    import sys
    
    try:
        viz_script = Path('blueprint/GraphPlot_0708YTS4.py')
        if viz_script.exists():
            subprocess.run([sys.executable, str(viz_script)], check=True)
            print("✓ Network visualization completed")
        else:
            print("⚠ Network visualization script not found")
    except subprocess.CalledProcessError as e:
        print(f"✗ Network visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='S4 Steam Pipeline Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate data
  python main.py --mode validate
  
  # Train model
  python main.py --mode train
  
  # Resume training from checkpoint
  python main.py --mode train --resume checkpoints/last_checkpoint.pth
  
  # Real-time anomaly detection
  python main.py --mode detect --detection-mode real-time
  
  # Batch anomaly detection
  python main.py --mode detect --detection-mode batch
  
  # Visualize network
  python main.py --mode visualize
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, 
                       choices=['validate', 'train', 'detect', 'visualize'],
                       required=True,
                       help='Operation mode')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--model', type=str, default='checkpoints/best_checkpoint.pth',
                       help='Path to trained model for detection')
    parser.add_argument('--threshold', type=str, default='checkpoints/anomaly_threshold.txt',
                       help='Path to anomaly threshold file')
    parser.add_argument('--detection-mode', type=str, choices=['real-time', 'batch'],
                       default='batch', help='Detection mode for anomaly detection')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"S4 Steam Pipeline Anomaly Detection System")
    print(f"Configuration: {config_path}")
    print(f"Mode: {args.mode}")
    print("=" * 50)
    
    try:
        if args.mode == 'validate':
            validate_data(config)
        
        elif args.mode == 'train':
            validate_data(config)  # Validate before training
            train_model(config, args.resume)
        
        elif args.mode == 'detect':
            # Check if model and threshold files exist
            model_path = Path(args.model)
            threshold_path = Path(args.threshold)
            
            if not model_path.exists():
                print(f"Error: Model file not found: {model_path}")
                print("Please train the model first using: python main.py --mode train")
                sys.exit(1)
            
            if not threshold_path.exists():
                print(f"Error: Threshold file not found: {threshold_path}")
                print("Please train the model first to generate the threshold")
                sys.exit(1)
            
            detect_anomalies(config, str(model_path), str(threshold_path), args.detection_mode)
        
        elif args.mode == 'visualize':
            visualize_network(config)
        
        print("\n✓ Operation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()