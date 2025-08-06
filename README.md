# Steam Pipeline Network Anomaly Detection System

A comprehensive anomaly detection and root cause analysis system for steam pipeline networks using heterogeneous graph neural networks (HANConv). The system provides real-time monitoring, automatic anomaly detection, and precise root cause localization for industrial pipeline operations.

## 🎯 Project Overview

This system is specifically designed for monitoring the **0708YTS4 steam pipeline network**. It uses a heterogeneous graph autoencoder to learn normal operating patterns and detect anomalies in real-time, providing operators with:

- **Real-time anomaly detection** with automatic threshold adaptation
- **Root cause analysis** pinpointing problematic pipeline components
- **Visualization tools** for network topology and anomaly patterns
- **Comprehensive logging** and reporting for operational insights

## 🏗️ System Architecture

### Core Components

1. **Data Preprocessing Pipeline**
   - `topology_parser.py`: Extracts pipeline network topology (209 nodes, 206 edges)
   - `sensor_data_cleaner.py`: Processes sensor data (36 sensors, 51K+ records)
   - `graph_builder.py`: Creates heterogeneous graph representations

2. **Model Architecture**
   - `hetero_encoder.py`: HANConv-based encoder for heterogeneous graphs
   - `decoder.py`: Multiple decoder types (basic, attention, variational)
   - `hetero_autoencoder.py`: Complete autoencoder with anomaly detection

3. **Training Framework**
   - `trainer.py`: Comprehensive training pipeline with checkpointing
   - `data_loader.py`: Specialized data loaders for graph data
   - `utils.py`: Training utilities (early stopping, metrics, etc.)

4. **Detection & Diagnosis**
   - `anomaly_detector.py`: Real-time anomaly detection engine
   - `root_cause_analyzer.py`: Graph-based root cause analysis
   - `visualizer.py`: Network and anomaly visualization

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/SchlomoFeng/Spatio-Temporal-HANConv.git
cd Spatio-Temporal-HANConv
pip install -r requirements.txt
```

### Basic Usage

1. **Train the Model**
```bash
python scripts/train.py --config config/model_config.yaml
```

2. **Run Anomaly Detection**
```bash
# Batch mode
python scripts/detect.py --model checkpoints/hetero_autoencoder_best.pth --mode batch

# Real-time mode
python scripts/detect.py --model checkpoints/hetero_autoencoder_best.pth --mode realtime
```

## 📊 Data Structure

### Pipeline Topology (`blueprint/0708YTS4.txt`)
- **Format**: JSON containing network structure
- **Nodes**: 209 components (streams, valves, mixers, tees)
- **Edges**: 206 connections with physical properties
- **Attributes**: Coordinates, types, pipe specifications

### Sensor Data (`data/0708YTS4.csv`)
- **Format**: Time-series CSV with 36 sensor columns
- **Records**: 51,841 timestamped readings
- **Sensors**: Pressure (PI), Flow (FI), Temperature (TI)
- **Frequency**: 10-second intervals

## 🔧 Configuration

All system parameters are configured via `config/model_config.yaml`:

```yaml
# Model Architecture
model:
  encoder:
    hidden_dim: 64
    output_dim: 32
    num_heads: 4
    num_layers: 2

# Training Parameters
training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.001

# Anomaly Detection
anomaly_detection:
  threshold_method: "statistical"
  confidence_level: 0.95
```

## 📈 Model Performance

### Architecture Details
- **Parameters**: 61,028 total (all trainable)
- **Node Types**: Stream (dynamic) and Static (equipment)
- **Edge Types**: 4 different connection types
- **Embedding Dimension**: 32D node representations

### Training Metrics
- **Loss Function**: MSE reconstruction + KL divergence (variational)
- **Optimization**: Adam with learning rate scheduling
- **Regularization**: Dropout, weight decay, early stopping

## 🔍 Anomaly Detection Features

### Detection Methods
1. **Statistical Threshold**: μ + 3σ of reconstruction errors
2. **Percentile-based**: 95th/99th percentile thresholds  
3. **Adaptive Threshold**: Self-adjusting based on recent data

### Root Cause Analysis
- **Graph Centrality**: Uses betweenness, closeness, PageRank
- **Propagation Tracing**: Multi-hop influence analysis
- **Temporal Patterns**: Historical anomaly correlation
- **Sensor Mapping**: Direct sensor-to-component linking

## 📊 Visualization & Monitoring

### Real-time Dashboard
```bash
# View network topology
python -m src.detection.visualizer --mode network

# Plot anomaly timeline  
python -m src.detection.visualizer --mode timeline --data results/
```

### Output Examples
- **Network Graph**: Interactive topology with anomaly highlighting
- **Time Series**: Sensor trends with anomaly markers
- **Heatmaps**: Component-level error distributions
- **Reports**: Structured JSON anomaly reports

## 🔬 Advanced Usage

### Custom Model Training
```python
from src.models.hetero_autoencoder import HeteroAutoencoder
from src.training.trainer import HeteroAutoencoderTrainer

# Create custom model
model = HeteroAutoencoder(
    node_feature_dims={'stream': 4, 'static': 7},
    num_sensors=36,
    decoder_type='attention'  # or 'variational'
)

# Train with custom settings
trainer = HeteroAutoencoderTrainer(model, train_loader, val_loader)
trainer.train(num_epochs=200)
```

### Streaming Detection
```python
from src.detection.anomaly_detector import AnomalyDetector
from src.training.data_loader import StreamingDataLoader

# Setup streaming detection
detector = AnomalyDetector(model, threshold_method='adaptive')
streaming_loader = StreamingDataLoader(graph_builder)

# Process real-time data
for sensor_reading in data_stream:
    streaming_loader.add_sample(sensor_reading)
    graph_data = streaming_loader.get_current_window()
    result = detector.detect_anomaly(sensor_reading, graph_data)
```

## 📁 Directory Structure

```
Spatio-Temporal-HANConv/
├── src/
│   ├── data_preprocessing/     # Data processing modules
│   ├── models/                # Neural network architectures  
│   ├── training/              # Training pipeline
│   └── detection/             # Anomaly detection & RCA
├── config/                    # Configuration files
├── scripts/                   # Executable scripts
├── blueprint/                 # Network topology data
├── data/                      # Sensor time series data
├── outputs/                   # Generated results
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training/detection logs
│   ├── results/              # Analysis results
│   └── plots/                # Visualizations
├── docs/                      # Documentation
└── requirements.txt           # Python dependencies
```

## 🛠️ Development

### Adding New Features
1. **New Decoder Type**: Extend `decoder.py` with custom architectures
2. **Detection Method**: Add algorithms to `anomaly_detector.py`  
3. **Visualization**: Create plots in `visualizer.py`
4. **Configuration**: Update `model_config.yaml` schema

### Testing
```bash
# Test data preprocessing
python -m src.data_preprocessing.topology_parser
python -m src.data_preprocessing.sensor_data_cleaner

# Test model components
python -m src.models.hetero_autoencoder

# Test detection system
python -m src.detection.anomaly_detector
```

## 📝 API Reference

### Key Classes

- `TopologyParser`: Extracts network structure from JSON
- `SensorDataCleaner`: Preprocesses time-series sensor data
- `HeteroAutoencoder`: Main neural network model
- `AnomalyDetector`: Real-time anomaly detection engine
- `RootCauseAnalyzer`: Graph-based root cause analysis

### Configuration Options

- **Model**: Architecture, dimensions, layer counts
- **Training**: Optimization, scheduling, regularization
- **Detection**: Thresholding, adaptation, ensemble methods
- **Output**: Checkpointing, logging, visualization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Support

For technical issues or questions:
- Create GitHub issue with detailed description
- Include configuration files and error logs
- Provide sample data if possible

## 🔄 Changelog

### v1.0.0 (Current)
- Initial release with complete pipeline system
- HANConv-based heterogeneous graph autoencoder
- Real-time anomaly detection and root cause analysis
- Comprehensive visualization and reporting tools
- Full documentation and example configurations

---

**Built for Industrial IoT Pipeline Monitoring** | **Powered by PyTorch Geometric & HANConv**
