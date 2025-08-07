"""
Inference and anomaly detection module for Steam Pipeline System

This module provides:
1. Real-time anomaly detection
2. Node-level error calculation for root cause analysis
3. Top-K anomaly localization
4. Anomaly visualization and reporting
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import (
    PipelineTopologyParser, SensorDataProcessor, HeteroGraphBuilder
)
from src.data.dataset import StreamPipelineDataset, collate_hetero_batch
from src.models.han_autoencoder import create_model
from torch.utils.data import DataLoader


class AnomalyDetector:
    """Real-time anomaly detection and root cause analysis"""
    
    def __init__(self, config: Dict[str, Any], model_path: str, threshold_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load anomaly threshold
        self.anomaly_threshold = self._load_threshold(threshold_path)
        
        # Setup data processing
        self.topology_parser = PipelineTopologyParser(config['data']['blueprint_path'])
        self.sensor_processor = SensorDataProcessor(
            config['data']['sensor_data_path'],
            config['data']['scaler_type']
        )
        
        # Initialize topology and scaler (normally loaded from training)
        self._setup_topology_and_scaler()
        
        # Anomaly configuration
        self.anomaly_config = config['anomaly']
        self.smoothing_window = self.anomaly_config['smoothing_window']
        self.top_k = self.anomaly_config['top_k_anomalies']
        
        # History for smoothing
        self.anomaly_history = []
        
        print(f"Anomaly detector initialized with threshold: {self.anomaly_threshold:.6f}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = create_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_threshold(self, threshold_path: str) -> float:
        """Load anomaly threshold"""
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        return threshold
    
    def _setup_topology_and_scaler(self):
        """Setup topology and scaler (simplified for demo)"""
        # Process topology
        nodeDF, edgeDF = self.topology_parser.extract_nodes_and_edges()
        self.nodes_df, self.node_id_to_index = self.topology_parser.process_nodes(nodeDF)
        self.edges_df = self.topology_parser.process_edges(edgeDF, self.node_id_to_index)
        
        # For inference, we need the scaler fitted during training
        # In practice, this would be saved and loaded
        sensor_df = self.sensor_processor.load_sensor_data()
        sensor_df = self.sensor_processor.clean_data(sensor_df)
        
        # Use first 70% for fitting scaler (should match training)
        train_size = int(len(sensor_df) * 0.7)
        train_df = sensor_df.iloc[:train_size]
        self.sensor_processor.create_scaler(train_df)
        
        print(f"Topology loaded: {len(self.nodes_df)} nodes, {len(self.edges_df)} edges")
    
    def detect_anomaly(self, sensor_data: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect anomaly in sensor data
        Args:
            sensor_data: Time window of sensor readings (window_size, num_sensors)
        Returns:
            (is_anomaly, anomaly_score, detailed_results)
        """
        # Prepare data for model
        batch = self._prepare_batch(sensor_data)
        
        with torch.no_grad():
            # Forward pass
            output = self.model(batch)
            
            # Compute anomaly scores
            anomaly_scores = self.model.compute_anomaly_scores(output, batch['targets'])
            
            if len(anomaly_scores) == 0:
                return False, 0.0, {}
            
            # Get mean anomaly score
            mean_score = torch.mean(anomaly_scores).item()
            
            # Apply smoothing
            self.anomaly_history.append(mean_score)
            if len(self.anomaly_history) > self.smoothing_window:
                self.anomaly_history.pop(0)
            
            smoothed_score = np.mean(self.anomaly_history)
            
            # Determine if anomaly
            is_anomaly = smoothed_score > self.anomaly_threshold
            
            # Perform root cause analysis if anomaly detected
            detailed_results = {}
            if is_anomaly:
                detailed_results = self._perform_root_cause_analysis(
                    output, batch, anomaly_scores
                )
        
        return is_anomaly, smoothed_score, detailed_results
    
    def _prepare_batch(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """Prepare sensor data for model inference"""
        # Normalize sensor data
        sensor_data_normalized = self.sensor_processor.scaler.transform(sensor_data)
        
        # Create time series for Stream nodes
        num_stream_nodes = len(self.nodes_df[self.nodes_df['node_type'] == 'Stream'])
        stream_sequences = np.tile(sensor_data_normalized[np.newaxis, :, :], 
                                 (num_stream_nodes, 1, 1))
        stream_sequences = torch.tensor(stream_sequences, dtype=torch.float32).to(self.device)
        
        # Create targets (last time step)
        current_readings = sensor_data_normalized[-1]
        stream_targets = np.tile(current_readings[np.newaxis, :], (num_stream_nodes, 1))
        stream_targets = torch.tensor(stream_targets, dtype=torch.float32).to(self.device)
        
        # Static features
        node_types = ['Stream', 'VavlePro', 'Mixer', 'Tee']
        graph_builder = HeteroGraphBuilder(node_types)
        node_features = graph_builder.create_node_features(self.nodes_df)
        edge_indices = graph_builder.create_edge_indices(self.edges_df, self.nodes_df)
        
        static_features = {}
        for node_type, features in node_features.items():
            if node_type != 'Stream':
                static_features[node_type] = features.to(self.device)
        
        # Move edge indices to device
        edge_index_dict = {}
        for edge_type, edge_index in edge_indices.items():
            edge_index_dict[edge_type] = edge_index.to(self.device)
        
        batch = {
            'stream_sequences': stream_sequences,
            'static_features': static_features,
            'edge_index_dict': edge_index_dict,
            'targets': {
                'sensor_readings': stream_targets
            }
        }
        
        return batch
    
    def _perform_root_cause_analysis(self, output: Dict[str, Any], batch: Dict[str, Any], 
                                   anomaly_scores: torch.Tensor) -> Dict[str, Any]:
        """Perform root cause analysis to identify anomalous nodes"""
        results = {
            'node_anomaly_scores': {},
            'top_k_anomalous_nodes': [],
            'reconstruction_errors': {},
            'affected_sensors': []
        }
        
        # Node-level anomaly scores
        stream_nodes = self.nodes_df[self.nodes_df['node_type'] == 'Stream']
        
        for i, (_, node) in enumerate(stream_nodes.iterrows()):
            if i < len(anomaly_scores):
                score = anomaly_scores[i].item()
                results['node_anomaly_scores'][node['node_id']] = {
                    'score': score,
                    'name': node['name'],
                    'coordinates': (node['x_coord'], node['y_coord'])
                }
        
        # Find top-K anomalous nodes
        sorted_nodes = sorted(
            results['node_anomaly_scores'].items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        results['top_k_anomalous_nodes'] = sorted_nodes[:self.top_k]
        
        # Sensor-level reconstruction errors
        if 'sensor_readings' in output['reconstructed'] and 'sensor_readings' in batch['targets']:
            reconstructed = output['reconstructed']['sensor_readings']
            targets = batch['targets']['sensor_readings']
            
            # Compute per-sensor errors (averaged across Stream nodes)
            sensor_errors = torch.mean((reconstructed - targets) ** 2, dim=0)
            
            sensor_columns = self.sensor_processor.sensor_columns
            for i, sensor_name in enumerate(sensor_columns):
                if i < len(sensor_errors):
                    error = sensor_errors[i].item()
                    results['reconstruction_errors'][sensor_name] = error
            
            # Find most affected sensors
            sorted_sensors = sorted(
                results['reconstruction_errors'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            results['affected_sensors'] = sorted_sensors[:10]  # Top 10 sensors
        
        return results
    
    def generate_anomaly_report(self, timestamp: str, is_anomaly: bool, 
                              anomaly_score: float, detailed_results: Dict[str, Any]) -> str:
        """Generate detailed anomaly report"""
        report = []
        report.append(f"=== ANOMALY DETECTION REPORT ===")
        report.append(f"Timestamp: {timestamp}")
        report.append(f"Anomaly Detected: {'YES' if is_anomaly else 'NO'}")
        report.append(f"Anomaly Score: {anomaly_score:.6f}")
        report.append(f"Threshold: {self.anomaly_threshold:.6f}")
        report.append("")
        
        if is_anomaly and detailed_results:
            # Top anomalous nodes
            if 'top_k_anomalous_nodes' in detailed_results:
                report.append("=== TOP ANOMALOUS NODES (ROOT CAUSE ANALYSIS) ===")
                for i, (node_id, node_info) in enumerate(detailed_results['top_k_anomalous_nodes']):
                    report.append(f"{i+1}. Node: {node_info['name']}")
                    report.append(f"   ID: {node_id}")
                    report.append(f"   Anomaly Score: {node_info['score']:.6f}")
                    report.append(f"   Coordinates: {node_info['coordinates']}")
                    report.append("")
            
            # Affected sensors
            if 'affected_sensors' in detailed_results:
                report.append("=== MOST AFFECTED SENSORS ===")
                for i, (sensor_name, error) in enumerate(detailed_results['affected_sensors'][:5]):
                    report.append(f"{i+1}. {sensor_name}: Error = {error:.6f}")
                report.append("")
        
        return "\n".join(report)
    
    def visualize_anomaly(self, detailed_results: Dict[str, Any], 
                         save_path: Optional[str] = None) -> None:
        """Visualize anomaly detection results"""
        if not detailed_results or 'node_anomaly_scores' not in detailed_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Node anomaly scores on pipeline topology
        node_scores = detailed_results['node_anomaly_scores']
        
        # Extract coordinates and scores
        x_coords = [info['coordinates'][0] for info in node_scores.values()]
        y_coords = [info['coordinates'][1] for info in node_scores.values()]
        scores = [info['score'] for info in node_scores.values()]
        
        # Normalize coordinates
        x_coords = np.array(x_coords) / 100000
        y_coords = np.array(y_coords) / 100000
        scores = np.array(scores)
        
        # Plot nodes colored by anomaly score
        scatter = ax1.scatter(x_coords, y_coords, c=scores, cmap='Reds', 
                            s=100, alpha=0.7, edgecolors='black')
        
        # Highlight top anomalous nodes
        if 'top_k_anomalous_nodes' in detailed_results:
            top_nodes = detailed_results['top_k_anomalous_nodes']
            for node_id, node_info in top_nodes:
                x, y = np.array(node_info['coordinates']) / 100000
                ax1.scatter(x, y, s=200, c='red', marker='x', linewidths=3)
                ax1.annotate(f"#{list(node_scores.keys()).index(node_id)+1}", 
                           (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold', color='red')
        
        ax1.set_title('Pipeline Network - Node Anomaly Scores', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Coordinate (normalized)')
        ax1.set_ylabel('Y Coordinate (normalized)')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
        
        # 2. Sensor reconstruction errors
        if 'reconstruction_errors' in detailed_results:
            sensor_errors = detailed_results['reconstruction_errors']
            sensors = list(sensor_errors.keys())[:20]  # Show top 20 sensors
            errors = [sensor_errors[sensor] for sensor in sensors]
            
            # Create bar plot
            bars = ax2.bar(range(len(sensors)), errors, color='coral', alpha=0.7)
            ax2.set_title('Sensor Reconstruction Errors', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sensor Index')
            ax2.set_ylabel('Reconstruction Error')
            ax2.set_xticks(range(len(sensors)))
            ax2.set_xticklabels([f'S{i}' for i in range(len(sensors))], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Highlight top errors
            if 'affected_sensors' in detailed_results:
                top_sensors = [s[0] for s in detailed_results['affected_sensors'][:5]]
                for i, sensor in enumerate(sensors):
                    if sensor in top_sensors:
                        bars[i].set_color('red')
                        bars[i].set_alpha(0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Anomaly visualization saved to: {save_path}")
        
        plt.show()


class BatchAnomalyDetector:
    """Batch anomaly detection for historical data analysis"""
    
    def __init__(self, config: Dict[str, Any], model_path: str, threshold_path: str):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config, model_path, threshold_path)
        
    def detect_anomalies_in_dataset(self, dataset: StreamPipelineDataset) -> pd.DataFrame:
        """Detect anomalies in entire dataset"""
        results = []
        
        print(f"Processing {len(dataset)} samples...")
        
        for i in range(len(dataset)):
            sample = dataset[i]
            timestamp = sample['timestamp']
            
            # Extract sensor data from the sample
            if 'stream_sequences' in sample:
                # Get the time series data (use first Stream node as representative)
                sensor_data = sample['stream_sequences'][0].numpy()  # (window_size, num_sensors)
                
                # Detect anomaly
                is_anomaly, anomaly_score, detailed_results = self.anomaly_detector.detect_anomaly(sensor_data)
                
                # Collect top anomalous nodes
                top_nodes = []
                if 'top_k_anomalous_nodes' in detailed_results:
                    top_nodes = [node_info['name'] for _, node_info in detailed_results['top_k_anomalous_nodes']]
                
                results.append({
                    'timestamp': timestamp,
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score,
                    'top_anomalous_nodes': ', '.join(top_nodes[:3]) if top_nodes else ''
                })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
        
        return pd.DataFrame(results)


def main():
    """Main function for testing anomaly detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Steam Pipeline Anomaly Detection')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='checkpoints/best_checkpoint.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--threshold', type=str, default='checkpoints/anomaly_threshold.txt',
                       help='Path to anomaly threshold file')
    parser.add_argument('--mode', type=str, choices=['real-time', 'batch'], default='batch',
                       help='Detection mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.mode == 'real-time':
        # Real-time detection example
        detector = AnomalyDetector(config, args.model, args.threshold)
        
        # Example: detect anomaly in random data
        window_size = config['data']['window_size']
        num_sensors = config['model']['stream_input_dim']
        
        # Generate random sensor data for demo
        sensor_data = np.random.randn(window_size, num_sensors)
        
        is_anomaly, score, details = detector.detect_anomaly(sensor_data)
        
        print(f"Anomaly detected: {is_anomaly}")
        print(f"Anomaly score: {score:.6f}")
        
        if is_anomaly:
            report = detector.generate_anomaly_report(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                is_anomaly, score, details
            )
            print("\n" + report)
            
            # Visualize anomaly
            detector.visualize_anomaly(details)
    
    elif args.mode == 'batch':
        # Batch detection on test set
        print("Loading test dataset...")
        
        # Create a simple test dataset (using same approach as training script)
        from src.data.preprocessing import PipelineTopologyParser, SensorDataProcessor, DataSplitter
        from src.data.dataset import StreamPipelineDataset
        
        # Process data
        topology_parser = PipelineTopologyParser(config['data']['blueprint_path'])
        sensor_processor = SensorDataProcessor(config['data']['sensor_data_path'])
        
        nodeDF, edgeDF = topology_parser.extract_nodes_and_edges()
        nodes_df, node_id_to_index = topology_parser.process_nodes(nodeDF)
        edges_df = topology_parser.process_edges(edgeDF, node_id_to_index)
        
        sensor_df = sensor_processor.load_sensor_data()
        sensor_df = sensor_processor.clean_data(sensor_df)
        
        # Use last 15% as test data
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
        
        # Run batch detection
        batch_detector = BatchAnomalyDetector(config, args.model, args.threshold)
        results_df = batch_detector.detect_anomalies_in_dataset(test_dataset)
        
        # Save results
        output_path = 'anomaly_detection_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Print summary
        num_anomalies = results_df['is_anomaly'].sum()
        print(f"\nDetection Summary:")
        print(f"Total samples: {len(results_df)}")
        print(f"Anomalies detected: {num_anomalies}")
        print(f"Anomaly rate: {num_anomalies/len(results_df)*100:.2f}%")


if __name__ == "__main__":
    main()