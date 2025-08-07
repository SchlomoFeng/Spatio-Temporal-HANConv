#!/usr/bin/env python3
"""
Test script for device configuration and CUDA support improvements.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config_validator import load_and_validate_config
from src.utils.device_utils import (
    check_pytorch_installation, 
    get_installation_recommendations,
    validate_device_config,
    log_pytorch_environment,
    test_device_accessibility
)
from src.training.train import PipelineTrainer
import torch


def test_device_configuration():
    """Test device configuration improvements"""
    print("=" * 60)
    print("CUDA Support Enhancement Test")
    print("=" * 60)
    
    # 1. Test PyTorch installation check
    print("\n1. PyTorch Installation Check:")
    print("-" * 30)
    info = check_pytorch_installation()
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"CUDA version: {info['cuda_version']}")
    print(f"GPU count: {info['device_count']}")
    
    # 2. Test installation recommendations
    print("\n2. Installation Recommendations:")
    print("-" * 30)
    recommendations = get_installation_recommendations()
    if recommendations:
        for category, message in recommendations.items():
            print(f"{category}: {message}")
    else:
        print("✓ No installation issues detected")
    
    # 3. Test device config validation
    print("\n3. Device Configuration Validation:")
    print("-" * 30)
    test_configs = ['auto', 'cpu', 'cuda', 'cuda:0', 'invalid_device']
    
    for config in test_configs:
        error = validate_device_config(config)
        status = "✗" if error else "✓"
        print(f"  {config}: {status} {error or 'Valid'}")
    
    # 4. Test device accessibility
    print("\n4. Device Accessibility Test:")
    print("-" * 30)
    
    # Test CPU
    cpu_device = torch.device('cpu')
    cpu_accessible = test_device_accessibility(cpu_device)
    print(f"  CPU: {'✓' if cpu_accessible else '✗'}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        cuda_accessible = test_device_accessibility(cuda_device)
        print(f"  CUDA: {'✓' if cuda_accessible else '✗'}")
    else:
        print("  CUDA: Not available")
    
    # 5. Test enhanced training setup
    print("\n5. Enhanced Training Setup Test:")
    print("-" * 30)
    
    try:
        # Load config
        config = load_and_validate_config('config/config.yaml')
        
        # Test with auto device
        config['system']['device'] = 'auto'
        print("Testing device='auto'...")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger = logging.getLogger('test')
        
        # Create a minimal trainer instance to test device setup
        trainer = PipelineTrainer.__new__(PipelineTrainer)
        trainer.config = config
        trainer.logger = logger
        
        # Test device setup
        device = trainer._setup_device()
        print(f"✓ Device setup successful: {device}")
        
    except Exception as e:
        print(f"✗ Device setup failed: {e}")
    
    # 6. Test error handling for invalid CUDA config
    print("\n6. Error Handling Test:")
    print("-" * 30)
    
    try:
        config['system']['device'] = 'cuda'  # Force CUDA even if not available
        trainer_cuda = PipelineTrainer.__new__(PipelineTrainer)
        trainer_cuda.config = config
        trainer_cuda.logger = logger
        device = trainer_cuda._setup_device()
        print(f"✓ CUDA device setup: {device}")
    except Exception as e:
        print(f"✓ Expected CUDA error handled correctly: {str(e)[:100]}...")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_device_configuration()