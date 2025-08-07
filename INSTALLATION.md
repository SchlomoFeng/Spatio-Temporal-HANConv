# Installation Guide

This document provides detailed installation instructions for both CPU and GPU environments.

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, Windows, or macOS
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB+ free space

### For GPU Acceleration (Optional but Recommended)
- **CUDA-compatible GPU**: NVIDIA GPU with compute capability 3.5+
- **CUDA Drivers**: Version 11.8 or higher
- **GPU Memory**: 4GB+ VRAM recommended

## Installation Options

### Option 1: CPU-Only Installation (Default)

For CPU-only usage or if you don't have a CUDA-compatible GPU:

```bash
# Clone the repository
git clone https://github.com/SchlomoFeng/Spatio-Temporal-HANConv.git
cd Spatio-Temporal-HANConv

# Install dependencies
pip install -r requirements.txt
```

### Option 2: GPU-Accelerated Installation (Recommended)

For optimal performance with CUDA-enabled GPU:

```bash
# Clone the repository
git clone https://github.com/SchlomoFeng/Spatio-Temporal-HANConv.git
cd Spatio-Temporal-HANConv

# Install CUDA-enabled dependencies
pip install -r requirements_cuda.txt
```

## Verification

After installation, verify your setup:

```bash
# Check PyTorch and CUDA installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test the system
python main.py --mode validate
```

## Environment Configuration

The system automatically detects the best available device, but you can specify it explicitly in `config/config.yaml`:

```yaml
system:
  device: "auto"  # Options: "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.
```

### Device Options:
- **`auto`**: Automatically selects GPU if available, otherwise CPU
- **`cpu`**: Forces CPU usage
- **`cuda`**: Uses the default GPU
- **`cuda:N`**: Uses specific GPU (N = 0, 1, 2, ...)

## Troubleshooting

### Common Installation Issues

#### 1. PyTorch CUDA Compatibility Issues

**Error**: `RuntimeError: Torch not compiled with CUDA support`

**Solution**:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-enabled version
pip install -r requirements_cuda.txt
```

#### 2. CUDA Version Mismatch

**Error**: `CUDA runtime is not available`

**Solutions**:
1. Check your CUDA drivers:
   ```bash
   nvidia-smi
   ```
2. Install compatible CUDA drivers from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
3. Verify GPU compatibility: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

#### 3. Memory Issues

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size in `config/config.yaml`:
   ```yaml
   training:
     batch_size: 16  # Reduce from 32
   ```
2. Reduce window size:
   ```yaml
   data:
     window_size: 30  # Reduce from 60
   ```
3. Use CPU instead:
   ```yaml
   system:
     device: "cpu"
   ```

#### 4. Environment Detection Issues

Check your environment setup:

```bash
# Run environment check
python src/utils/device_utils.py
```

This will provide detailed information about your PyTorch and CUDA setup.

### Performance Optimization

#### GPU Acceleration Tips
- Ensure `pin_memory: true` in config for GPU usage
- Use appropriate batch sizes (32-64 for most GPUs)
- Monitor GPU memory usage with `nvidia-smi`

#### CPU Optimization Tips
- Increase `num_workers` for data loading:
  ```yaml
  system:
    num_workers: 8  # Adjust based on CPU cores
  ```
- Use smaller batch sizes to reduce memory usage

## Advanced Installation

### Development Environment

For development and testing:

```bash
# Clone with development dependencies
git clone https://github.com/SchlomoFeng/Spatio-Temporal-HANConv.git
cd Spatio-Temporal-HANConv

# Install in development mode
pip install -e .
pip install -r requirements_cuda.txt

# Install optional development tools
pip install pytest black isort flake8
```

### Docker Installation (Optional)

For containerized environments:

```dockerfile
# Base image with CUDA support
FROM pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime

# Copy and install
COPY requirements_cuda.txt .
RUN pip install -r requirements_cuda.txt

# Copy application
COPY . /app
WORKDIR /app
```

### Virtual Environment Setup

Recommended for isolation:

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements_cuda.txt
```

## Getting Help

If you encounter issues:

1. **Check logs**: Look in `logs/` directory for detailed error messages
2. **Environment check**: Run `python src/utils/device_utils.py`
3. **GitHub Issues**: [Report issues](https://github.com/SchlomoFeng/Spatio-Temporal-HANConv/issues)
4. **Configuration**: Verify your `config/config.yaml` settings

## Quick Start After Installation

```bash
# 1. Validate your setup
python main.py --mode validate

# 2. Train the model
python main.py --mode train

# 3. Run anomaly detection
python main.py --mode detect
```

## Hardware Recommendations

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 5GB free space

### Recommended for GPU Acceleration
- **GPU**: NVIDIA GTX 1660+ or RTX 20/30/40 series
- **VRAM**: 6GB+
- **RAM**: 16GB+
- **CPU**: 8+ cores

### Optimal Performance
- **GPU**: NVIDIA RTX 3080+ or A100
- **VRAM**: 12GB+
- **RAM**: 32GB+
- **Storage**: SSD for faster data loading