"""
Root Cause Analyzer
Identifies the root cause of anomalies by analyzing reconstruction errors and graph structure
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from collections import defaultdict, deque
import heapq

from anomaly_detector import AnomalyDetector


class RootCauseAnalyzer:
    """Analyzer for identifying root causes of pipeline anomalies"""
    
    def __init__(self,
                 topology_data: Dict,
                 sensor_mapping: Dict[str, List[str]],
                 anomaly_detector: Optional[AnomalyDetector] = None):
        """
        Initialize root cause analyzer
        
        Args:
            topology_data: Parsed topology data from TopologyParser
            sensor_mapping: Mapping from stream nodes to sensors
            anomaly_detector: Optional anomaly detector instance
        """
        self.topology_data = topology_data
        self.sensor_mapping = sensor_mapping
        self.anomaly_detector = anomaly_detector
        
        # Build network graph
        self.graph = self._build_network_graph()
        
        # Compute graph properties
        self.centrality_measures = self._compute_centrality_measures()
        
        # Root cause analysis history
        self.rca_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _build_network_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from topology data"""
        G = nx.DiGraph()
        
        # Add nodes
        nodes_df = self.topology_data['nodes']
        for _, node in nodes_df.iterrows():
            G.add_node(
                node['node_id'],
                name=node['name'],
                type=node['type'],
                x=node['x'],
                y=node['y']
            )
        
        # Add edges
        edges_df = self.topology_data['edges']
        for _, edge in edges_df.iterrows():
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(
                    edge['source'],
                    edge['target'],
                    name=edge['name'],
                    length=edge.get('length', 0),
                    diameter=edge.get('diameter', 0)
                )
        
        self.logger.info(f"Built network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _compute_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Compute various centrality measures for nodes"""
        try:
            centrality = {
                'degree_centrality': nx.degree_centrality(self.graph),
                'in_degree_centrality': nx.in_degree_centrality(self.graph),
                'out_degree_centrality': nx.out_degree_centrality(self.graph),
                'closeness_centrality': nx.closeness_centrality(self.graph),
                'betweenness_centrality': nx.betweenness_centrality(self.graph),
            }
            
            # Compute PageRank for directed graph
            try:
                centrality['pagerank'] = nx.pagerank(self.graph)
            except:
                centrality['pagerank'] = {}
            
            self.logger.info("Centrality measures computed")
            return centrality
            
        except Exception as e:
            self.logger.warning(f"Error computing centrality measures: {e}")
            return {}
    
    def analyze_sensor_anomalies(self,
                               per_sensor_errors: np.ndarray,
                               sensor_names: List[str],
                               error_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze per-sensor reconstruction errors to identify anomalous sensors
        
        Args:
            per_sensor_errors: Array of per-sensor reconstruction errors
            sensor_names: Names of sensors corresponding to errors
            error_threshold: Threshold for identifying anomalous sensors
            
        Returns:
            Analysis results
        """
        if error_threshold is None:
            # Use statistical threshold
            mean_error = np.mean(per_sensor_errors)
            std_error = np.std(per_sensor_errors)
            error_threshold = mean_error + 2 * std_error
        
        # Identify anomalous sensors
        anomalous_sensors = []
        normal_sensors = []
        
        for i, error in enumerate(per_sensor_errors):
            sensor_info = {
                'sensor_name': sensor_names[i],
                'error': float(error),
                'error_z_score': float((error - np.mean(per_sensor_errors)) / (np.std(per_sensor_errors) + 1e-6)),
                'is_anomaly': error > error_threshold
            }
            
            if sensor_info['is_anomaly']:
                anomalous_sensors.append(sensor_info)
            else:
                normal_sensors.append(sensor_info)
        
        # Sort anomalous sensors by error magnitude
        anomalous_sensors.sort(key=lambda x: x['error'], reverse=True)
        
        analysis = {
            'total_sensors': len(sensor_names),
            'anomalous_sensors': anomalous_sensors,
            'normal_sensors': normal_sensors,
            'anomaly_count': len(anomalous_sensors),
            'anomaly_rate': len(anomalous_sensors) / len(sensor_names),
            'error_threshold': error_threshold,
            'max_error': float(np.max(per_sensor_errors)),
            'mean_error': float(np.mean(per_sensor_errors)),
            'std_error': float(np.std(per_sensor_errors))
        }
        
        return analysis
    
    def identify_root_cause_nodes(self,
                                 anomalous_sensors: List[Dict],
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Identify stream nodes that are likely root causes based on anomalous sensors
        
        Args:
            anomalous_sensors: List of anomalous sensor information
            top_k: Number of top root cause candidates to return
            
        Returns:
            List of root cause node candidates
        """
        node_scores = defaultdict(float)
        node_info = defaultdict(lambda: {'sensors': [], 'total_error': 0.0, 'sensor_count': 0})
        
        # Map sensors to nodes and accumulate scores
        for sensor_info in anomalous_sensors:
            sensor_name = sensor_info['sensor_name']
            
            # Find corresponding nodes for this sensor
            corresponding_nodes = []
            for node_id, sensor_list in self.sensor_mapping.items():
                if sensor_name in sensor_list:
                    corresponding_nodes.append(node_id)
            
            # If no direct mapping, try fuzzy matching
            if not corresponding_nodes:
                corresponding_nodes = self._fuzzy_match_sensor_to_nodes(sensor_name)
            
            # Update node scores
            for node_id in corresponding_nodes:
                if node_id in self.graph.nodes:
                    # Base score is the sensor error
                    base_score = sensor_info['error']
                    
                    # Weight by centrality measures
                    centrality_weight = 1.0
                    if self.centrality_measures:
                        for measure_name, measure_values in self.centrality_measures.items():
                            if node_id in measure_values:
                                centrality_weight += measure_values[node_id]
                    
                    final_score = base_score * centrality_weight
                    node_scores[node_id] += final_score
                    
                    # Update node info
                    node_info[node_id]['sensors'].append(sensor_info)
                    node_info[node_id]['total_error'] += sensor_info['error']
                    node_info[node_id]['sensor_count'] += 1
        
        # Create root cause candidates
        root_cause_candidates = []
        for node_id, score in node_scores.items():
            node_data = self.topology_data['nodes'][
                self.topology_data['nodes']['node_id'] == node_id
            ].iloc[0] if len(self.topology_data['nodes'][
                self.topology_data['nodes']['node_id'] == node_id
            ]) > 0 else None
            
            candidate = {
                'node_id': node_id,
                'node_name': node_data['name'] if node_data is not None else 'Unknown',
                'node_type': node_data['type'] if node_data is not None else 'Unknown',
                'score': float(score),
                'affected_sensors': node_info[node_id]['sensors'],
                'total_error': node_info[node_id]['total_error'],
                'sensor_count': node_info[node_id]['sensor_count'],
                'avg_error': node_info[node_id]['total_error'] / max(1, node_info[node_id]['sensor_count']),
                'centrality_measures': {}
            }
            
            # Add centrality measures
            for measure_name, measure_values in self.centrality_measures.items():
                if node_id in measure_values:
                    candidate['centrality_measures'][measure_name] = measure_values[node_id]
            
            root_cause_candidates.append(candidate)
        
        # Sort by score and return top-k
        root_cause_candidates.sort(key=lambda x: x['score'], reverse=True)
        return root_cause_candidates[:top_k]
    
    def _fuzzy_match_sensor_to_nodes(self, sensor_name: str, threshold: float = 0.6) -> List[str]:
        """
        Fuzzy match sensor name to node names
        
        Args:
            sensor_name: Sensor name to match
            threshold: Similarity threshold
            
        Returns:
            List of matching node IDs
        """
        matching_nodes = []
        
        # Simple fuzzy matching based on common substrings
        sensor_parts = sensor_name.lower().split('.')
        sensor_parts.extend(sensor_name.lower().split('_'))
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            
            # Check for common parts
            node_parts = node_name.split('-')
            node_parts.extend(node_name.split('_'))
            
            common_parts = set(sensor_parts) & set(node_parts)
            if len(common_parts) > 0:
                similarity = len(common_parts) / max(len(sensor_parts), len(node_parts))
                if similarity >= threshold:
                    matching_nodes.append(node_id)
        
        return matching_nodes
    
    def trace_propagation_path(self,
                             root_cause_nodes: List[str],
                             max_depth: int = 3) -> Dict[str, List[List[str]]]:
        """
        Trace how anomalies might propagate from root cause nodes
        
        Args:
            root_cause_nodes: List of potential root cause node IDs
            max_depth: Maximum propagation depth to trace
            
        Returns:
            Propagation paths for each root cause node
        """
        propagation_paths = {}
        
        for root_node in root_cause_nodes:
            if root_node not in self.graph.nodes:
                continue
            
            paths = []
            
            # BFS to find all reachable nodes within max_depth
            queue = deque([(root_node, [root_node], 0)])
            visited = set()
            
            while queue:
                current_node, path, depth = queue.popleft()
                
                if depth >= max_depth or current_node in visited:
                    continue
                
                visited.add(current_node)
                
                # Add current path
                if len(path) > 1:
                    paths.append(path.copy())
                
                # Explore neighbors
                for neighbor in self.graph.successors(current_node):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path, depth + 1))
            
            propagation_paths[root_node] = paths
        
        return propagation_paths
    
    def analyze_temporal_patterns(self,
                                anomaly_history: List[Dict],
                                time_window: float = 3600) -> Dict[str, Any]:
        """
        Analyze temporal patterns in anomaly occurrences
        
        Args:
            anomaly_history: History of anomaly detections
            time_window: Time window in seconds for pattern analysis
            
        Returns:
            Temporal pattern analysis results
        """
        if not anomaly_history:
            return {'patterns': [], 'analysis': 'No anomaly history available'}
        
        # Group anomalies by time windows
        time_bins = defaultdict(list)
        
        for anomaly in anomaly_history:
            timestamp = anomaly.get('timestamp', 0)
            bin_key = int(timestamp // time_window)
            time_bins[bin_key].append(anomaly)
        
        # Analyze patterns
        patterns = {
            'total_time_bins': len(time_bins),
            'avg_anomalies_per_bin': np.mean([len(anomalies) for anomalies in time_bins.values()]),
            'max_anomalies_per_bin': max([len(anomalies) for anomalies in time_bins.values()]),
            'bins_with_anomalies': len([bin_anomalies for bin_anomalies in time_bins.values() if len(bin_anomalies) > 0]),
        }
        
        # Find recurring patterns
        sensor_patterns = defaultdict(int)
        for bin_anomalies in time_bins.values():
            for anomaly in bin_anomalies:
                if 'sensor_anomalies' in anomaly:
                    for sensor_id, sensor_info in anomaly['sensor_anomalies'].items():
                        if sensor_info['is_anomaly']:
                            sensor_patterns[sensor_id] += 1
        
        patterns['recurring_sensor_anomalies'] = dict(
            sorted(sensor_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return patterns
    
    def perform_root_cause_analysis(self,
                                   anomaly_result: Dict,
                                   sensor_names: List[str],
                                   include_propagation: bool = True,
                                   include_temporal: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive root cause analysis
        
        Args:
            anomaly_result: Result from anomaly detection
            sensor_names: List of sensor names
            include_propagation: Whether to include propagation analysis
            include_temporal: Whether to include temporal pattern analysis
            
        Returns:
            Complete root cause analysis results
        """
        analysis_start_time = time.time()
        
        rca_result = {
            'timestamp': anomaly_result.get('timestamp'),
            'is_anomaly': anomaly_result.get('is_anomaly', False),
            'total_error': anomaly_result.get('total_error', 0.0),
            'analysis_components': []
        }
        
        if not rca_result['is_anomaly']:
            rca_result['conclusion'] = "No anomaly detected - root cause analysis not needed"
            return rca_result
        
        # Step 1: Sensor-level analysis
        per_sensor_errors = anomaly_result.get('per_sensor_errors', np.array([]))
        
        if len(per_sensor_errors) > 0 and len(sensor_names) == len(per_sensor_errors):
            sensor_analysis = self.analyze_sensor_anomalies(per_sensor_errors, sensor_names)
            rca_result['sensor_analysis'] = sensor_analysis
            rca_result['analysis_components'].append('sensor_analysis')
            
            # Step 2: Identify root cause nodes
            if sensor_analysis['anomalous_sensors']:
                root_cause_candidates = self.identify_root_cause_nodes(
                    sensor_analysis['anomalous_sensors'],
                    top_k=5
                )
                rca_result['root_cause_candidates'] = root_cause_candidates
                rca_result['analysis_components'].append('root_cause_identification')
                
                # Step 3: Propagation analysis
                if include_propagation and root_cause_candidates:
                    root_node_ids = [candidate['node_id'] for candidate in root_cause_candidates[:3]]
                    propagation_analysis = self.trace_propagation_path(root_node_ids)
                    rca_result['propagation_analysis'] = propagation_analysis
                    rca_result['analysis_components'].append('propagation_analysis')
        
        # Step 4: Temporal analysis (if requested and history available)
        if include_temporal and hasattr(self.anomaly_detector, 'detection_history'):
            # Get recent anomaly history
            recent_history = []
            if hasattr(self.anomaly_detector, 'error_history'):
                for i, (is_anomaly, error, timestamp) in enumerate(
                    zip(self.anomaly_detector.detection_history[-100:],
                        self.anomaly_detector.error_history[-100:],
                        self.anomaly_detector.timestamp_history[-100:])
                ):
                    if is_anomaly:
                        recent_history.append({
                            'timestamp': timestamp,
                            'total_error': error,
                            'is_anomaly': True
                        })
            
            if recent_history:
                temporal_analysis = self.analyze_temporal_patterns(recent_history)
                rca_result['temporal_analysis'] = temporal_analysis
                rca_result['analysis_components'].append('temporal_analysis')
        
        # Generate conclusion
        rca_result['conclusion'] = self._generate_conclusion(rca_result)
        
        # Record analysis time
        rca_result['analysis_time'] = time.time() - analysis_start_time
        
        # Add to history
        self.rca_history.append(rca_result)
        
        # Keep history limited
        if len(self.rca_history) > 100:
            self.rca_history = self.rca_history[-100:]
        
        return rca_result
    
    def _generate_conclusion(self, rca_result: Dict) -> str:
        """Generate human-readable conclusion from RCA results"""
        conclusions = []
        
        if 'sensor_analysis' in rca_result:
            sensor_analysis = rca_result['sensor_analysis']
            conclusions.append(f"Detected {sensor_analysis['anomaly_count']} anomalous sensors "
                             f"out of {sensor_analysis['total_sensors']} total sensors.")
        
        if 'root_cause_candidates' in rca_result and rca_result['root_cause_candidates']:
            top_candidate = rca_result['root_cause_candidates'][0]
            conclusions.append(f"Top root cause candidate: {top_candidate['node_name']} "
                             f"({top_candidate['node_type']}) with score {top_candidate['score']:.2f}.")
        
        if 'propagation_analysis' in rca_result:
            prop_analysis = rca_result['propagation_analysis']
            total_paths = sum(len(paths) for paths in prop_analysis.values())
            conclusions.append(f"Identified {total_paths} potential propagation paths.")
        
        return " ".join(conclusions) if conclusions else "Root cause analysis completed."
    
    def get_rca_summary(self, recent_count: int = 10) -> Dict[str, Any]:
        """
        Get summary of recent root cause analyses
        
        Args:
            recent_count: Number of recent analyses to summarize
            
        Returns:
            RCA summary statistics
        """
        recent_rca = self.rca_history[-recent_count:] if self.rca_history else []
        
        if not recent_rca:
            return {'message': 'No root cause analyses in history'}
        
        # Aggregate statistics
        total_anomalies = len([rca for rca in recent_rca if rca.get('is_anomaly')])
        
        # Most common root causes
        root_cause_frequency = defaultdict(int)
        for rca in recent_rca:
            if 'root_cause_candidates' in rca:
                for candidate in rca['root_cause_candidates']:
                    root_cause_frequency[candidate['node_name']] += 1
        
        summary = {
            'total_analyses': len(recent_rca),
            'anomaly_count': total_anomalies,
            'anomaly_rate': total_anomalies / len(recent_rca),
            'avg_analysis_time': np.mean([rca.get('analysis_time', 0) for rca in recent_rca]),
            'most_common_root_causes': dict(
                sorted(root_cause_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            )
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    from data_preprocessing.topology_parser import TopologyParser
    
    print("Testing root cause analyzer...")
    
    try:
        # Initialize topology parser (would normally use real data)
        topology_parser = TopologyParser("../../blueprint/0708YTS4.txt")
        topology_data = topology_parser.parse_topology()
        
        # Create dummy sensor mapping
        sensor_mapping = {
            'stream_node_1': ['YT.63PI_00406.PV', 'YT.63FI_00406.PV'],
            'stream_node_2': ['YT.63TI_00404.PV', 'YT.64PI_9809_2.PV'],
        }
        
        # Create root cause analyzer
        rca = RootCauseAnalyzer(topology_data, sensor_mapping)
        
        # Create dummy anomaly result
        anomaly_result = {
            'timestamp': time.time(),
            'is_anomaly': True,
            'total_error': 5.0,
            'per_sensor_errors': np.random.exponential(1.0, 36),  # Some sensors with high errors
        }
        
        # Make some sensors clearly anomalous
        anomaly_result['per_sensor_errors'][0] = 10.0  # Very high error
        anomaly_result['per_sensor_errors'][5] = 8.0   # High error
        
        sensor_names = [f'sensor_{i}' for i in range(36)]
        
        # Perform root cause analysis
        rca_result = rca.perform_root_cause_analysis(
            anomaly_result,
            sensor_names,
            include_propagation=True
        )
        
        print("Root Cause Analysis Results:")
        print(f"  Conclusion: {rca_result['conclusion']}")
        print(f"  Analysis components: {rca_result['analysis_components']}")
        
        if 'sensor_analysis' in rca_result:
            sensor_analysis = rca_result['sensor_analysis']
            print(f"  Anomalous sensors: {sensor_analysis['anomaly_count']}")
        
        if 'root_cause_candidates' in rca_result:
            candidates = rca_result['root_cause_candidates']
            print(f"  Root cause candidates: {len(candidates)}")
            if candidates:
                top = candidates[0]
                print(f"    Top candidate: {top['node_name']} (score: {top['score']:.2f})")
        
        print("Root cause analyzer test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()