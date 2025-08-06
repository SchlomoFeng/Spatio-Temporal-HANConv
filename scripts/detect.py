#!/usr/bin/env python3
"""
Detection Script for Real-time Pipeline Anomaly Detection
Steam Pipeline Network Anomaly Detection System
"""

import os
import sys
import yaml
import argparse
import torch
import time
import json
from pathlib import Path
import logging

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from data_preprocessing.topology_parser import TopologyParser
from data_preprocessing.sensor_data_cleaner import SensorDataCleaner
from data_preprocessing.graph_builder import GraphBuilder
from models.hetero_autoencoder import HeteroAutoencoder
from detection.anomaly_detector import AnomalyDetector
from detection.root_cause_analyzer import RootCauseAnalyzer
from training.data_loader import StreamingDataLoader
from training.utils import get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    handlers = [logging.StreamHandler()]
    
    if config['logging']['save_logs']:
        log_file = Path(config['output']['base_dir']) / "logs" / "detection.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def load_model(config: dict, model_path: str) -> HeteroAutoencoder:
    """Load trained model"""
    logging.info(f"Loading model from {model_path}")
    
    # Determine node feature dimensions (this should match training)
    node_feature_dims = {
        'stream': 4,  # x, y coordinates + type encoding + sensor reading
        'static': 9   # x, y coordinates + one-hot type encoding (assuming 7 types)
    }
    
    model = HeteroAutoencoder(
        node_feature_dims=node_feature_dims,
        num_sensors=36,  # Default, will be updated based on actual data
        encoder_hidden_dim=config['model']['encoder']['hidden_dim'],
        encoder_output_dim=config['model']['encoder']['output_dim'],
        encoder_num_heads=config['model']['encoder']['num_heads'],
        encoder_num_layers=config['model']['encoder']['num_layers'],
        decoder_hidden_dims=config['model']['decoder']['hidden_dims'],
        decoder_type=config['model']['decoder']['type'],
        dropout=config['model']['encoder']['dropout']
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logging.info("Model loaded successfully")
    
    return model


def setup_detection_system(config: dict, model: HeteroAutoencoder):
    """Setup the complete detection system"""
    logging.info("Setting up detection system...")
    
    # Initialize data processing components
    topology_parser = TopologyParser(config['data']['blueprint_path'])
    sensor_cleaner = SensorDataCleaner(config['data']['sensor_data_path'])
    graph_builder = GraphBuilder(topology_parser, sensor_cleaner)
    
    # Parse topology and clean initial data for calibration
    topology_data = topology_parser.parse_topology()
    sensor_cleaner.clean_sensor_data()
    
    # Setup device
    device = get_device() if config['hardware']['device'] == 'auto' else torch.device(config['hardware']['device'])
    
    # Create anomaly detector
    anomaly_detector = AnomalyDetector(
        model=model,
        device=device,
        threshold_method=config['anomaly_detection']['threshold_method'],
        threshold_value=config['anomaly_detection'].get('threshold_value')
    )
    
    # Create root cause analyzer
    sensor_mapping = graph_builder.build_node_mappings()['stream_node_mapping']
    root_cause_analyzer = RootCauseAnalyzer(
        topology_data=topology_data,
        sensor_mapping=sensor_mapping,
        anomaly_detector=anomaly_detector
    )
    
    # Create streaming data loader
    streaming_loader = StreamingDataLoader(
        graph_builder=graph_builder,
        window_size=config['data']['window_size']
    )
    
    logging.info("Detection system setup complete")
    
    return {
        'anomaly_detector': anomaly_detector,
        'root_cause_analyzer': root_cause_analyzer,
        'streaming_loader': streaming_loader,
        'graph_builder': graph_builder,
        'sensor_cleaner': sensor_cleaner,
        'topology_data': topology_data
    }


def calibrate_system(config: dict, detection_system: dict):
    """Calibrate the anomaly detection system"""
    logging.info("Calibrating anomaly detection system...")
    
    anomaly_detector = detection_system['anomaly_detector']
    graph_builder = detection_system['graph_builder']
    sensor_cleaner = detection_system['sensor_cleaner']
    
    # Use some of the cleaned data for calibration
    sensor_data = sensor_cleaner.clean_data
    if sensor_data is None:
        sensor_data = sensor_cleaner.clean_sensor_data()
    
    # Take a subset for calibration (normal operation data)
    calibration_size = min(1000, len(sensor_data))
    calibration_indices = range(0, calibration_size, 10)  # Sample every 10th point
    
    calibration_sensor_data = []
    calibration_graph_data = []
    
    for i in calibration_indices:
        # Get sensor data
        sensor_row = sensor_data.iloc[i]
        sensor_values = sensor_row[sensor_cleaner.sensor_columns].values
        calibration_sensor_data.append(torch.tensor(sensor_values, dtype=torch.float32))
        
        # Get corresponding graph data
        try:
            graph_data = graph_builder.build_hetero_data(
                window_size=config['data']['window_size'],
                current_window_idx=i
            )
            calibration_graph_data.append(graph_data)
        except:
            # Skip if graph data creation fails
            calibration_sensor_data.pop()
            continue
    
    if len(calibration_sensor_data) < 10:
        logging.warning("Not enough calibration data, using default threshold")
        return
    
    # Calibrate threshold
    try:
        anomaly_detector.calibrate_threshold(
            calibration_data=calibration_sensor_data,
            graph_data_list=calibration_graph_data,
            contamination_rate=config['anomaly_detection']['contamination_rate'],
            confidence_level=config['anomaly_detection']['confidence_level']
        )
        logging.info("System calibration completed")
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        logging.info("Using default threshold")


def batch_detection_mode(config: dict, detection_system: dict, data_file: str):
    """Run batch detection on historical data"""
    logging.info(f"Running batch detection on {data_file}")
    
    anomaly_detector = detection_system['anomaly_detector']
    root_cause_analyzer = detection_system['root_cause_analyzer']
    graph_builder = detection_system['graph_builder']
    sensor_cleaner = detection_system['sensor_cleaner']
    
    # Load and process data
    if data_file != config['data']['sensor_data_path']:
        # Load different data file
        new_cleaner = SensorDataCleaner(data_file)
        test_data = new_cleaner.clean_sensor_data()
        sensor_columns = new_cleaner.sensor_columns
    else:
        test_data = sensor_cleaner.clean_data
        sensor_columns = sensor_cleaner.sensor_columns
    
    results = []
    anomaly_count = 0
    
    # Process data in batches
    batch_size = 100
    total_samples = len(test_data)
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logging.info(f"Processing batch {start_idx}-{end_idx}/{total_samples}")
        
        for i in range(start_idx, end_idx):
            try:
                # Get sensor data
                sensor_row = test_data.iloc[i]
                sensor_values = sensor_row[sensor_columns].values
                sensor_tensor = torch.tensor(sensor_values, dtype=torch.float32)
                
                # Get graph data
                graph_data = graph_builder.build_hetero_data(
                    window_size=config['data']['window_size'],
                    current_window_idx=i
                )
                
                # Detect anomaly
                timestamp = time.time() if 'timestamp' not in sensor_row else sensor_row['timestamp']
                detection_result = anomaly_detector.detect_anomaly(
                    sensor_data=sensor_tensor,
                    graph_data=graph_data,
                    timestamp=timestamp
                )
                
                # Root cause analysis if anomaly detected
                if detection_result['is_anomaly']:
                    rca_result = root_cause_analyzer.perform_root_cause_analysis(
                        anomaly_result=detection_result,
                        sensor_names=sensor_columns,
                        include_propagation=True
                    )
                    detection_result['root_cause_analysis'] = rca_result
                    anomaly_count += 1
                
                results.append({
                    'index': i,
                    'timestamp': detection_result['timestamp'],
                    'is_anomaly': detection_result['is_anomaly'],
                    'anomaly_score': detection_result['anomaly_score'],
                    'total_error': detection_result['total_error'],
                    'root_cause_analysis': detection_result.get('root_cause_analysis', {})
                })
                
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue
    
    # Save results
    results_dir = Path(config['output']['base_dir']) / config['output']['results_dir']
    results_file = results_dir / f'batch_detection_results_{int(time.time())}.json'
    
    summary = {
        'total_samples': len(results),
        'anomaly_count': anomaly_count,
        'anomaly_rate': anomaly_count / len(results) if results else 0,
        'detection_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Batch detection completed: {anomaly_count}/{len(results)} anomalies detected")
    logging.info(f"Results saved to {results_file}")
    
    return summary


def real_time_detection_mode(config: dict, detection_system: dict):
    """Run real-time detection (simulated)"""
    logging.info("Starting real-time detection mode")
    
    anomaly_detector = detection_system['anomaly_detector']
    root_cause_analyzer = detection_system['root_cause_analyzer']
    streaming_loader = detection_system['streaming_loader']
    sensor_cleaner = detection_system['sensor_cleaner']
    
    # Simulate real-time data stream
    sensor_data = sensor_cleaner.clean_data
    sensor_columns = sensor_cleaner.sensor_columns
    
    anomaly_reports = []
    detection_count = 0
    
    try:
        for i in range(0, len(sensor_data), 10):  # Sample every 10th point for simulation
            # Simulate sensor reading
            sensor_row = sensor_data.iloc[i]
            sensor_values = sensor_row[sensor_columns].values
            sensor_tensor = torch.tensor(sensor_values, dtype=torch.float32)
            
            # Add to streaming buffer
            streaming_loader.add_sample(sensor_tensor, timestamp=time.time())
            
            # Get current window
            graph_data = streaming_loader.get_current_window()
            
            if graph_data is not None:
                # Detect anomaly
                detection_result = anomaly_detector.detect_anomaly(
                    sensor_data=sensor_tensor,
                    graph_data=graph_data,
                    timestamp=time.time()
                )
                
                detection_count += 1
                
                # Print real-time status
                status = "🚨 ANOMALY" if detection_result['is_anomaly'] else "✅ NORMAL"
                print(f"[{time.strftime('%H:%M:%S')}] {status} | "
                      f"Score: {detection_result['anomaly_score']:.3f} | "
                      f"Error: {detection_result['total_error']:.3f}")
                
                # Perform root cause analysis for anomalies
                if detection_result['is_anomaly']:
                    print(f"  🔍 Performing root cause analysis...")
                    
                    rca_result = root_cause_analyzer.perform_root_cause_analysis(
                        anomaly_result=detection_result,
                        sensor_names=sensor_columns,
                        include_propagation=True
                    )
                    
                    print(f"  📋 RCA: {rca_result['conclusion']}")
                    
                    if 'root_cause_candidates' in rca_result and rca_result['root_cause_candidates']:
                        top_candidate = rca_result['root_cause_candidates'][0]
                        print(f"  🎯 Top suspect: {top_candidate['node_name']} "
                              f"(score: {top_candidate['score']:.2f})")
                    
                    # Save anomaly report
                    anomaly_report = {
                        'timestamp': detection_result['timestamp'],
                        'detection_result': detection_result,
                        'root_cause_analysis': rca_result
                    }
                    anomaly_reports.append(anomaly_report)
                
                # Adaptive threshold update
                if (detection_count % 100 == 0 and 
                    config['anomaly_detection'].get('adaptive_threshold', False)):
                    anomaly_detector.adaptive_threshold_update()
            
            # Simulate real-time delay
            time.sleep(0.1)  # 100ms between readings
            
            # Break after some time for demo
            if detection_count >= 100:
                break
                
    except KeyboardInterrupt:
        logging.info("Real-time detection stopped by user")
    
    # Save anomaly reports
    if anomaly_reports:
        results_dir = Path(config['output']['base_dir']) / config['output']['results_dir']
        results_file = results_dir / f'realtime_anomaly_reports_{int(time.time())}.json'
        
        with open(results_file, 'w') as f:
            json.dump(anomaly_reports, f, indent=2, default=str)
        
        logging.info(f"Anomaly reports saved to {results_file}")
    
    # Print detection statistics
    stats = anomaly_detector.get_detection_statistics()
    logging.info("Detection Statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Anomaly Detection")
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, choices=['batch', 'realtime'], default='batch',
                       help='Detection mode: batch processing or real-time')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file (for batch mode)')
    parser.add_argument('--no-calibration', action='store_true',
                       help='Skip automatic calibration')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    logging.info("Starting Steam Pipeline Network Anomaly Detection")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Model: {args.model}")
    
    try:
        # Load model
        model = load_model(config, args.model)
        
        # Setup detection system
        detection_system = setup_detection_system(config, model)
        
        # Calibrate system (unless disabled)
        if not args.no_calibration:
            calibrate_system(config, detection_system)
        
        # Run detection based on mode
        if args.mode == 'batch':
            data_file = args.data or config['data']['sensor_data_path']
            results = batch_detection_mode(config, detection_system, data_file)
            print(f"\nBatch Detection Summary:")
            print(f"  Total samples: {results['total_samples']}")
            print(f"  Anomalies detected: {results['anomaly_count']}")
            print(f"  Anomaly rate: {results['anomaly_rate']:.2%}")
            
        elif args.mode == 'realtime':
            print("\n🚀 Starting real-time detection...")
            print("Press Ctrl+C to stop")
            real_time_detection_mode(config, detection_system)
        
        logging.info("Detection process completed successfully")
        
    except Exception as e:
        logging.error(f"Detection failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())