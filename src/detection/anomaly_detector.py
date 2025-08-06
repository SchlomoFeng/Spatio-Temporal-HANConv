"""
Anomaly Detector
Real-time anomaly detection using trained heterogeneous autoencoder
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path
import time
import logging

from ..models.hetero_autoencoder import HeteroAutoencoder
from ..training.data_loader import StreamingDataLoader


class AnomalyDetector:
    """Real-time anomaly detector using heterogeneous autoencoder"""
    
    def __init__(self,
                 model: HeteroAutoencoder,
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 threshold_method: str = 'statistical',  # 'statistical', 'percentile', 'fixed'
                 threshold_value: Optional[float] = None):
        """
        Initialize anomaly detector
        
        Args:
            model: Trained HeteroAutoencoder model
            model_path: Path to saved model weights
            device: Device for inference
            threshold_method: Method for anomaly threshold determination
            threshold_value: Fixed threshold value (if using 'fixed' method)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        
        # Load model weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.threshold = None
        
        # Anomaly detection statistics
        self.baseline_errors = []
        self.error_statistics = {}
        
        # Real-time tracking
        self.detection_history = []
        self.error_history = []
        self.timestamp_history = []
        
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def calibrate_threshold(self,
                          calibration_data: List[torch.Tensor],
                          graph_data_list: List[Any],
                          contamination_rate: float = 0.05,
                          confidence_level: float = 0.95):
        """
        Calibrate anomaly detection threshold using normal data
        
        Args:
            calibration_data: List of normal sensor readings
            graph_data_list: List of corresponding graph data
            contamination_rate: Expected contamination rate in calibration data
            confidence_level: Confidence level for statistical threshold
        """
        self.logger.info("Calibrating anomaly detection threshold...")
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for i, (sensor_data, graph_data) in enumerate(zip(calibration_data, graph_data_list)):
                try:
                    # Prepare graph data
                    x_dict = {}
                    edge_index_dict = {}
                    
                    for node_type in graph_data.node_types:
                        if hasattr(graph_data[node_type], 'x'):
                            x_dict[node_type] = graph_data[node_type].x.to(self.device)
                    
                    for edge_type in graph_data.edge_types:
                        edge_index_dict[edge_type] = graph_data[edge_type].edge_index.to(self.device)
                    
                    # Forward pass
                    output = self.model(
                        x_dict, 
                        edge_index_dict, 
                        mapping_info=getattr(graph_data, 'mapping_info', None)
                    )
                    
                    # Calculate reconstruction error
                    sensor_target = sensor_data.to(self.device)
                    errors = self.model.get_reconstruction_errors(output, sensor_target)
                    total_error = torch.sum(errors).item()
                    
                    reconstruction_errors.append(total_error)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing calibration sample {i}: {e}")
                    continue
        
        if not reconstruction_errors:
            raise ValueError("No valid calibration data processed")
        
        self.baseline_errors = reconstruction_errors
        errors_array = np.array(reconstruction_errors)
        
        # Compute error statistics
        self.error_statistics = {
            'mean': np.mean(errors_array),
            'std': np.std(errors_array),
            'median': np.median(errors_array),
            'q75': np.percentile(errors_array, 75),
            'q90': np.percentile(errors_array, 90),
            'q95': np.percentile(errors_array, 95),
            'q99': np.percentile(errors_array, 99),
            'max': np.max(errors_array),
            'min': np.min(errors_array)
        }
        
        # Set threshold based on method
        if self.threshold_method == 'statistical':
            # Statistical method: mean + n * std
            n_sigma = 2.0 if confidence_level < 0.95 else 3.0
            self.threshold = self.error_statistics['mean'] + n_sigma * self.error_statistics['std']
            
        elif self.threshold_method == 'percentile':
            # Percentile method
            percentile = confidence_level * 100
            self.threshold = np.percentile(errors_array, percentile)
            
        elif self.threshold_method == 'fixed':
            # Fixed threshold
            if self.threshold_value is None:
                raise ValueError("threshold_value must be provided for 'fixed' method")
            self.threshold = self.threshold_value
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        self.logger.info(f"Threshold calibration complete:")
        self.logger.info(f"  Method: {self.threshold_method}")
        self.logger.info(f"  Threshold: {self.threshold:.4f}")
        self.logger.info(f"  Error statistics: {self.error_statistics}")
        
    def detect_anomaly(self,
                      sensor_data: torch.Tensor,
                      graph_data: Any,
                      timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect anomaly in single sample
        
        Args:
            sensor_data: Current sensor readings
            graph_data: Current graph data
            timestamp: Optional timestamp
            
        Returns:
            Detection results dictionary
        """
        if self.threshold is None:
            warnings.warn("Threshold not calibrated. Using default threshold.")
            self.threshold = 1.0
        
        with torch.no_grad():
            try:
                # Prepare graph data
                x_dict = {}
                edge_index_dict = {}
                
                for node_type in graph_data.node_types:
                    if hasattr(graph_data[node_type], 'x'):
                        x_dict[node_type] = graph_data[node_type].x.to(self.device)
                
                for edge_type in graph_data.edge_types:
                    edge_index_dict[edge_type] = graph_data[edge_type].edge_index.to(self.device)
                
                # Forward pass
                start_time = time.time()
                output = self.model(
                    x_dict,
                    edge_index_dict,
                    mapping_info=getattr(graph_data, 'mapping_info', None)
                )
                inference_time = time.time() - start_time
                
                # Calculate reconstruction errors
                sensor_target = sensor_data.to(self.device)
                per_sensor_errors = self.model.get_reconstruction_errors(output, sensor_target)
                total_error = torch.sum(per_sensor_errors).item()
                
                # Anomaly detection
                is_anomaly = total_error > self.threshold
                anomaly_score = total_error / self.threshold if self.threshold > 0 else total_error
                
                # Per-sensor anomaly detection
                sensor_anomalies = {}
                if len(self.baseline_errors) > 0:
                    mean_error = self.error_statistics.get('mean', 0)
                    std_error = self.error_statistics.get('std', 1)
                    sensor_threshold = mean_error / len(per_sensor_errors) + 2 * std_error / len(per_sensor_errors)
                    
                    sensor_anomalies = {
                        f'sensor_{i}': {
                            'error': per_sensor_errors[i].item(),
                            'is_anomaly': per_sensor_errors[i].item() > sensor_threshold,
                            'score': per_sensor_errors[i].item() / sensor_threshold if sensor_threshold > 0 else per_sensor_errors[i].item()
                        }
                        for i in range(len(per_sensor_errors))
                    }
                
                # Detection result
                result = {
                    'timestamp': timestamp if timestamp is not None else time.time(),
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score,
                    'total_error': total_error,
                    'threshold': self.threshold,
                    'per_sensor_errors': per_sensor_errors.cpu().numpy(),
                    'sensor_anomalies': sensor_anomalies,
                    'reconstructed': output['reconstructed'].cpu().numpy(),
                    'inference_time': inference_time
                }
                
                # Update history
                self.detection_history.append(is_anomaly)
                self.error_history.append(total_error)
                self.timestamp_history.append(result['timestamp'])
                
                # Keep history limited
                max_history = 1000
                if len(self.detection_history) > max_history:
                    self.detection_history = self.detection_history[-max_history:]
                    self.error_history = self.error_history[-max_history:]
                    self.timestamp_history = self.timestamp_history[-max_history:]
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error during anomaly detection: {e}")
                return {
                    'timestamp': timestamp if timestamp is not None else time.time(),
                    'is_anomaly': False,
                    'error': str(e),
                    'anomaly_score': 0.0,
                    'total_error': 0.0
                }
    
    def batch_detect(self,
                    sensor_data_list: List[torch.Tensor],
                    graph_data_list: List[Any],
                    timestamps: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Batch anomaly detection
        
        Args:
            sensor_data_list: List of sensor data
            graph_data_list: List of graph data
            timestamps: Optional timestamps
            
        Returns:
            List of detection results
        """
        results = []
        
        for i, (sensor_data, graph_data) in enumerate(zip(sensor_data_list, graph_data_list)):
            timestamp = timestamps[i] if timestamps else None
            result = self.detect_anomaly(sensor_data, graph_data, timestamp)
            results.append(result)
        
        return results
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.error_history:
            return {}
        
        recent_errors = self.error_history[-100:]  # Last 100 detections
        recent_anomalies = self.detection_history[-100:]
        
        stats = {
            'total_detections': len(self.detection_history),
            'anomaly_count': sum(self.detection_history),
            'anomaly_rate': sum(self.detection_history) / len(self.detection_history),
            'recent_anomaly_rate': sum(recent_anomalies) / len(recent_anomalies) if recent_anomalies else 0,
            'mean_error': np.mean(self.error_history),
            'recent_mean_error': np.mean(recent_errors),
            'max_error': np.max(self.error_history),
            'min_error': np.min(self.error_history),
            'current_threshold': self.threshold
        }
        
        return stats
    
    def reset_history(self):
        """Reset detection history"""
        self.detection_history.clear()
        self.error_history.clear()
        self.timestamp_history.clear()
        self.logger.info("Detection history reset")
    
    def update_threshold(self, new_threshold: float):
        """
        Update anomaly detection threshold
        
        Args:
            new_threshold: New threshold value
        """
        self.threshold = new_threshold
        self.logger.info(f"Threshold updated to {new_threshold:.4f}")
    
    def adaptive_threshold_update(self, window_size: int = 100, alpha: float = 0.1):
        """
        Adaptive threshold update based on recent error history
        
        Args:
            window_size: Size of recent window to consider
            alpha: Learning rate for threshold update
        """
        if len(self.error_history) < window_size:
            return
        
        recent_errors = self.error_history[-window_size:]
        recent_mean = np.mean(recent_errors)
        recent_std = np.std(recent_errors)
        
        # Compute new threshold
        if self.threshold_method == 'statistical':
            new_threshold = recent_mean + 3 * recent_std
        else:
            new_threshold = np.percentile(recent_errors, 95)
        
        # Exponential moving average update
        self.threshold = (1 - alpha) * self.threshold + alpha * new_threshold
        
        self.logger.info(f"Adaptive threshold update: {self.threshold:.4f}")


class EnsembleAnomalyDetector:
    """Ensemble-based anomaly detector using multiple models"""
    
    def __init__(self, detectors: List[AnomalyDetector]):
        """
        Initialize ensemble detector
        
        Args:
            detectors: List of individual detectors
        """
        self.detectors = detectors
        self.logger = logging.getLogger(__name__)
    
    def detect_anomaly(self,
                      sensor_data: torch.Tensor,
                      graph_data: Any,
                      timestamp: Optional[float] = None,
                      voting_method: str = 'majority') -> Dict[str, Any]:
        """
        Ensemble anomaly detection
        
        Args:
            sensor_data: Sensor data
            graph_data: Graph data
            timestamp: Timestamp
            voting_method: Voting method ('majority', 'unanimous', 'weighted')
            
        Returns:
            Ensemble detection result
        """
        individual_results = []
        
        for detector in self.detectors:
            result = detector.detect_anomaly(sensor_data, graph_data, timestamp)
            individual_results.append(result)
        
        # Combine results
        anomaly_votes = [r['is_anomaly'] for r in individual_results]
        anomaly_scores = [r['anomaly_score'] for r in individual_results]
        total_errors = [r['total_error'] for r in individual_results]
        
        if voting_method == 'majority':
            is_anomaly = sum(anomaly_votes) > len(anomaly_votes) / 2
        elif voting_method == 'unanimous':
            is_anomaly = all(anomaly_votes)
        elif voting_method == 'weighted':
            # Weight by inverse of reconstruction error
            weights = [1.0 / (err + 1e-6) for err in total_errors]
            weighted_votes = sum(vote * weight for vote, weight in zip(anomaly_votes, weights))
            total_weight = sum(weights)
            is_anomaly = weighted_votes / total_weight > 0.5
        else:
            is_anomaly = sum(anomaly_votes) > len(anomaly_votes) / 2
        
        ensemble_result = {
            'timestamp': timestamp if timestamp is not None else time.time(),
            'is_anomaly': is_anomaly,
            'anomaly_score': np.mean(anomaly_scores),
            'total_error': np.mean(total_errors),
            'individual_results': individual_results,
            'anomaly_votes': anomaly_votes,
            'vote_count': sum(anomaly_votes),
            'confidence': max(anomaly_scores) if is_anomaly else 1 - max(anomaly_scores)
        }
        
        return ensemble_result


if __name__ == "__main__":
    # Example usage
    from models.hetero_autoencoder import HeteroAutoencoder
    
    print("Testing anomaly detector...")
    
    # Create dummy model
    node_feature_dims = {'stream': 4, 'static': 7}
    model = HeteroAutoencoder(
        node_feature_dims=node_feature_dims,
        num_sensors=36
    )
    
    # Create detector
    detector = AnomalyDetector(
        model=model,
        threshold_method='statistical'
    )
    
    # Create dummy calibration data
    calibration_data = [torch.randn(36) for _ in range(100)]
    
    # Note: In real usage, you would have actual graph data
    # For testing, we create dummy graph data structure
    class DummyGraphData:
        def __init__(self):
            self.node_types = ['stream', 'static']
            self.edge_types = [('stream', 'connects', 'stream')]
            self.stream = type('', (), {'x': torch.randn(10, 4)})()
            self.static = type('', (), {'x': torch.randn(20, 7)})()
            
        def __getitem__(self, edge_type):
            return type('', (), {'edge_index': torch.randint(0, 5, (2, 3))})()
    
    graph_data_list = [DummyGraphData() for _ in range(100)]
    
    try:
        # Calibrate threshold
        detector.calibrate_threshold(calibration_data, graph_data_list)
        
        # Test detection
        test_sensor_data = torch.randn(36)
        test_graph_data = DummyGraphData()
        
        result = detector.detect_anomaly(test_sensor_data, test_graph_data)
        
        print(f"Detection result:")
        print(f"  Is anomaly: {result['is_anomaly']}")
        print(f"  Anomaly score: {result['anomaly_score']:.4f}")
        print(f"  Total error: {result['total_error']:.4f}")
        print(f"  Threshold: {result['threshold']:.4f}")
        
        # Get statistics
        stats = detector.get_detection_statistics()
        print(f"Detection statistics: {stats}")
        
        print("Anomaly detector test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()