"""
Device and PyTorch environment utilities for Steam Pipeline Anomaly Detection System

This module provides utilities for:
1. Checking PyTorch and CUDA installation status
2. Providing installation guidance 
3. Device compatibility checking
"""

import torch
import logging
from typing import Dict, Any, Optional


def check_pytorch_installation() -> Dict[str, Any]:
    """
    Check PyTorch installation status and CUDA availability.
    
    Returns:
        Dictionary with installation details and recommendations
    """
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_details': []
    }
    
    # Get GPU details if CUDA is available
    if info['cuda_available']:
        for i in range(info['device_count']):
            gpu_props = torch.cuda.get_device_properties(i)
            info['gpu_details'].append({
                'index': i,
                'name': gpu_props.name,
                'memory_gb': gpu_props.total_memory / 1024**3,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
    
    return info


def get_installation_recommendations() -> Dict[str, str]:
    """
    Get installation recommendations based on current PyTorch setup.
    
    Returns:
        Dictionary with installation recommendations
    """
    info = check_pytorch_installation()
    recommendations = {}
    
    # Check if PyTorch has CUDA support compiled in
    has_cuda_compiled = info['cuda_version'] is not None
    
    if not has_cuda_compiled:
        recommendations['pytorch_cuda'] = (
            "PyTorch was installed without CUDA support. "
            "For GPU acceleration, install CUDA-enabled PyTorch:\n"
            "  pip install -r requirements_cuda.txt"
        )
    
    if has_cuda_compiled and not info['cuda_available']:
        recommendations['cuda_runtime'] = (
            "PyTorch has CUDA support but CUDA runtime is not available. "
            "Possible solutions:\n"
            "  1. Install CUDA drivers for your GPU\n" 
            "  2. Check if your GPU is CUDA-compatible\n"
            "  3. Verify CUDA installation with 'nvidia-smi'"
        )
    
    if info['cuda_available'] and info['device_count'] == 0:
        recommendations['gpu_detection'] = (
            "CUDA is available but no GPUs detected. "
            "Check your GPU drivers and hardware configuration."
        )
    
    return recommendations


def validate_device_config(device_config: str) -> Optional[str]:
    """
    Validate device configuration against available hardware.
    
    Args:
        device_config: Device configuration string ('auto', 'cpu', 'cuda', 'cuda:N')
        
    Returns:
        Error message if validation fails, None if valid
    """
    info = check_pytorch_installation()
    
    if device_config == 'auto':
        return None  # Auto is always valid
    
    if device_config == 'cpu':
        return None  # CPU is always available
    
    if device_config == 'cuda':
        if not info['cuda_available']:
            return (
                f"CUDA device requested but CUDA is not available. "
                f"PyTorch CUDA support: {'Yes' if info['cuda_version'] else 'No'}, "
                f"CUDA runtime: {info['cuda_available']}"
            )
        return None
    
    if device_config.startswith('cuda:'):
        try:
            gpu_index = int(device_config.split(':')[1])
            
            if not info['cuda_available']:
                return f"CUDA device {gpu_index} requested but CUDA is not available"
            
            if gpu_index >= info['device_count']:
                return (
                    f"CUDA device {gpu_index} requested but only "
                    f"{info['device_count']} GPUs available"
                )
            
            return None
            
        except (ValueError, IndexError):
            return f"Invalid CUDA device format: {device_config}. Use 'cuda:N' where N is GPU index"
    
    return f"Invalid device configuration: {device_config}. Use 'auto', 'cpu', 'cuda', or 'cuda:N'"


def log_pytorch_environment(logger: logging.Logger) -> None:
    """
    Log detailed PyTorch environment information.
    
    Args:
        logger: Logger instance to use for output
    """
    info = check_pytorch_installation()
    
    logger.info("PyTorch Environment Information:")
    logger.info(f"  PyTorch version: {info['pytorch_version']}")
    logger.info(f"  CUDA support compiled: {'Yes' if info['cuda_version'] else 'No'}")
    
    if info['cuda_version']:
        logger.info(f"  CUDA version: {info['cuda_version']}")
        logger.info(f"  CUDA runtime available: {info['cuda_available']}")
        
        if info['cuda_available']:
            logger.info(f"  GPU count: {info['device_count']}")
            for gpu in info['gpu_details']:
                logger.info(f"    GPU {gpu['index']}: {gpu['name']} "
                           f"({gpu['memory_gb']:.1f} GB, CC {gpu['compute_capability']})")
        else:
            logger.info("  CUDA runtime: Not available")
    
    # Log recommendations if any
    recommendations = get_installation_recommendations()
    if recommendations:
        logger.warning("Installation Recommendations:")
        for category, message in recommendations.items():
            logger.warning(f"  {category}: {message}")


def test_device_accessibility(device: torch.device, logger: Optional[logging.Logger] = None) -> bool:
    """
    Test if a device is accessible by creating a small tensor.
    
    Args:
        device: PyTorch device to test
        logger: Optional logger for detailed output
        
    Returns:
        True if device is accessible, False otherwise
    """
    try:
        # Create a small test tensor
        test_tensor = torch.zeros(1, device=device)
        
        # Perform a simple operation
        result = test_tensor + 1
        
        # Move result back to CPU to ensure the operation completed
        _ = result.cpu()
        
        if logger:
            logger.info(f"✓ Device {device} accessibility test passed")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"✗ Device {device} accessibility test failed: {e}")
        return False


if __name__ == "__main__":
    # Demo of environment checking
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    print("PyTorch Environment Check")
    print("=" * 50)
    
    # Log environment details
    log_pytorch_environment(logger)
    
    # Test different device configurations
    test_configs = ['auto', 'cpu', 'cuda', 'cuda:0']
    
    print("\nDevice Configuration Tests:")
    print("-" * 30)
    
    for config in test_configs:
        error = validate_device_config(config)
        if error:
            print(f"  {config}: ✗ {error}")
        else:
            print(f"  {config}: ✓ Valid")
            
            # Test accessibility for valid configs
            if config == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                try:
                    device = torch.device(config)
                    accessible = test_device_accessibility(device)
                    print(f"    Accessibility: {'✓' if accessible else '✗'}")
                except:
                    print(f"    Accessibility: ✗")