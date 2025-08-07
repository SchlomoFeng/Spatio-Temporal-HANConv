"""
Configuration validation and type conversion utilities for Steam Pipeline Anomaly Detection System

This module provides functions to:
1. Validate configuration structure and required parameters
2. Convert string numeric values to proper types (float, int)
3. Provide detailed error messages for configuration issues
"""

import yaml
from typing import Dict, Any, List, Union
import logging
from pathlib import Path


def convert_numeric_strings(value: Any) -> Any:
    """
    Convert string representations of numbers to appropriate numeric types.
    
    Args:
        value: The value to convert (can be str, int, float, dict, list, etc.)
        
    Returns:
        Converted value with proper numeric types
    """
    if isinstance(value, str):
        # Try to convert scientific notation and decimal strings to float
        try:
            # Check if it looks like a number
            if value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                # If it contains 'e' or 'E', it's likely scientific notation
                if 'e' in value.lower():
                    return float(value)
                # If it contains a decimal point, it's a float
                elif '.' in value:
                    return float(value)
                # Otherwise, try int first, then float
                else:
                    try:
                        return int(value)
                    except ValueError:
                        return float(value)
        except ValueError:
            # If conversion fails, return original string
            pass
    elif isinstance(value, dict):
        # Recursively convert dictionary values
        return {k: convert_numeric_strings(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively convert list items
        return [convert_numeric_strings(item) for item in value]
    
    return value


def validate_and_convert_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration structure and convert string numbers to proper types.
    
    Args:
        config: Raw configuration dictionary loaded from YAML
        
    Returns:
        Validated and type-converted configuration dictionary
        
    Raises:
        ValueError: If configuration has critical issues
    """
    # First, convert all numeric strings to proper types
    config = convert_numeric_strings(config)
    
    # Define expected numeric types for specific config keys
    numeric_conversions = {
        'training': {
            'weight_decay': float,
            'min_delta': float,
            'learning_rate': float,
            'scheduler': {
                'min_lr': float,
                'factor': float
            }
        }
    }
    
    # Apply specific type conversions
    config = apply_type_conversions(config, numeric_conversions)
    
    # Validate configuration structure and values
    validation_errors = validate_config_structure(config)
    
    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
        raise ValueError(error_msg)
    
    return config


def apply_type_conversions(config: Dict[str, Any], conversions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply specific type conversions to configuration values.
    
    Args:
        config: Configuration dictionary
        conversions: Dictionary specifying type conversions
        
    Returns:
        Configuration with applied type conversions
    """
    for section, section_conversions in conversions.items():
        if section in config and isinstance(config[section], dict):
            for key, target_type in section_conversions.items():
                if key in config[section]:
                    if isinstance(target_type, dict):
                        # Nested conversion
                        if isinstance(config[section][key], dict):
                            for nested_key, nested_type in target_type.items():
                                if nested_key in config[section][key]:
                                    try:
                                        config[section][key][nested_key] = nested_type(config[section][key][nested_key])
                                    except (ValueError, TypeError) as e:
                                        logging.warning(f"Failed to convert {section}.{key}.{nested_key} to {nested_type.__name__}: {e}")
                    else:
                        # Direct conversion
                        try:
                            config[section][key] = target_type(config[section][key])
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Failed to convert {section}.{key} to {target_type.__name__}: {e}")
    
    return config


def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration structure and return list of validation errors.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check required sections
    required_sections = ['data', 'model', 'training', 'anomaly', 'system']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate system configuration early (including device settings)
    if 'system' in config:
        system_config = config['system']
        
        # Validate device configuration
        if 'device' in system_config:
            device_value = system_config['device']
            valid_devices = ['auto', 'cpu', 'cuda']
            
            # Check for cuda:N pattern
            if isinstance(device_value, str):
                if device_value.startswith('cuda:'):
                    try:
                        # Extract and validate the GPU index
                        gpu_index = int(device_value.split(':')[1])
                        if gpu_index < 0:
                            errors.append(f"system.device GPU index must be non-negative, got: {gpu_index}")
                    except (ValueError, IndexError):
                        errors.append(f"system.device has invalid CUDA index format: {device_value}")
                elif device_value not in valid_devices:
                    errors.append(f"system.device must be one of {valid_devices} or 'cuda:N', got: {device_value}")
            else:
                errors.append(f"system.device must be a string, got: {device_value} (type: {type(device_value)})")
        
        # Validate other system settings
        for key in ['num_workers', 'random_seed']:
            if key in system_config:
                value = system_config[key]
                if not isinstance(value, int) or value < 0:
                    errors.append(f"system.{key} must be a non-negative integer, got: {value} (type: {type(value)})")
        
        # Validate boolean settings
        if 'pin_memory' in system_config:
            value = system_config['pin_memory']
            if not isinstance(value, bool):
                errors.append(f"system.pin_memory must be a boolean, got: {value} (type: {type(value)})")
    
    # Validate data section
    if 'data' in config:
        data_config = config['data']
        
        # Check data split ratios
        if all(key in data_config for key in ['train_ratio', 'val_ratio', 'test_ratio']):
            total_ratio = data_config['train_ratio'] + data_config['val_ratio'] + data_config['test_ratio']
            if abs(total_ratio - 1.0) > 1e-6:
                errors.append(f"Data split ratios don't sum to 1.0: {total_ratio}")
        
        # Check positive values
        for key in ['window_size', 'stride']:
            if key in data_config:
                value = data_config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"data.{key} must be a positive number, got: {value} (type: {type(value)})")
    
    # Validate model section
    if 'model' in config:
        model_config = config['model']
        
        # Check positive dimensions
        for key in ['stream_input_dim', 'static_input_dim', 'hidden_dim', 'output_dim']:
            if key in model_config:
                value = model_config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"model.{key} must be a positive number, got: {value} (type: {type(value)})")
    
    # Validate training section
    if 'training' in config:
        training_config = config['training']
        
        # Check positive values
        for key in ['batch_size', 'epochs', 'patience']:
            if key in training_config:
                value = training_config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"training.{key} must be a positive number, got: {value} (type: {type(value)})")
        
        # Check float values
        for key in ['learning_rate', 'weight_decay', 'min_delta']:
            if key in training_config:
                value = training_config[key]
                if not isinstance(value, (int, float)):
                    errors.append(f"training.{key} must be a number, got: {value} (type: {type(value)})")
                elif value < 0:
                    errors.append(f"training.{key} must be non-negative, got: {value}")
        
        # Check scheduler configuration
        if 'scheduler' in training_config and isinstance(training_config['scheduler'], dict):
            scheduler_config = training_config['scheduler']
            for key in ['factor', 'min_lr']:
                if key in scheduler_config:
                    value = scheduler_config[key]
                    if not isinstance(value, (int, float)):
                        errors.append(f"training.scheduler.{key} must be a number, got: {value} (type: {type(value)})")
                    elif value < 0:
                        errors.append(f"training.scheduler.{key} must be non-negative, got: {value}")
    
    return errors


def load_and_validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file and validate/convert types.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Validated and type-converted configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration validation fails
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    
    if config is None:
        raise ValueError("Configuration file is empty")
    
    return validate_and_convert_config(config)


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration with type information.
    
    Args:
        config: Configuration dictionary
    """
    print("Configuration Summary:")
    print("=" * 50)
    
    def print_section(section_dict: Dict[str, Any], indent: int = 0):
        prefix = "  " * indent
        for key, value in section_dict.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_section(value, indent + 1)
            else:
                print(f"{prefix}{key}: {value} ({type(value).__name__})")
    
    for section_name, section_content in config.items():
        print(f"\n{section_name}:")
        if isinstance(section_content, dict):
            print_section(section_content, 1)
        else:
            print(f"  {section_content} ({type(section_content).__name__})")


if __name__ == "__main__":
    # Test the configuration validator
    import sys
    
    config_path = "config/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        config = load_and_validate_config(config_path)
        print("✓ Configuration validation successful!")
        print_config_summary(config)
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)