"""
Configuration loading and validation utilities
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from .defaults import (
    DEFAULT_PBA_CONFIG,
    ENVIRONMENT_CONFIGS,
    RESOURCE_RECOMMENDATIONS,
    COMPLIANCE_THRESHOLDS
)
from ..core.pba_algorithm import PBAConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration errors"""
    pass


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    environment: str = "production",
    merge_defaults: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from file or environment.

    Args:
        config_path: Path to configuration file (JSON or YAML)
        environment: Environment name (development, production, serving)
        merge_defaults: Whether to merge with default configuration

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = {}

    # Start with environment defaults if merging
    if merge_defaults:
        if environment in ENVIRONMENT_CONFIGS:
            config = ENVIRONMENT_CONFIGS[environment].copy()
            logger.info(f"Loaded {environment} environment defaults")
        else:
            config = DEFAULT_PBA_CONFIG.copy()
            logger.warning(f"Unknown environment '{environment}', using defaults")

    # Load from file if provided
    if config_path:
        file_config = _load_config_file(config_path)
        if merge_defaults:
            config.update(file_config)
        else:
            config = file_config
        logger.info(f"Loaded configuration from {config_path}")

    # Load from environment variables
    env_config = _load_from_environment()
    config.update(env_config)

    # Validate configuration
    _validate_config(config)

    return config


def _load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                return json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f) or {}
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}. "
                    "Use .json, .yml, or .yaml"
                )
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ConfigurationError(f"Failed to parse configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file: {e}")


def _load_from_environment() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {}

    # PBA configuration from environment
    env_mappings = {
        'PBA_ALPHA': ('alpha', float),
        'PBA_BETA': ('beta', float),
        'PBA_TEMPERATURE': ('temperature', float),
        'PBA_DEVICE': ('device', str),
        'PBA_BATCH_SIZE': ('max_batch_size', int),
        'PBA_VALIDATE_INPUTS': ('validate_inputs', _str_to_bool),
        'PBA_NUMERICAL_STABILITY': ('numerical_stability', _str_to_bool)
    }

    for env_var, (config_key, converter) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                config[config_key] = converter(value)
                logger.debug(f"Loaded {config_key} from environment: {config[config_key]}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid value for {env_var}: {value} ({e})")

    return config


def _str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes', 'on', 'enabled')


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    try:
        # Validate using PBAConfig dataclass
        if 'alpha' in config and not 0.0 < config['alpha'] <= 1.0:
            raise ConfigurationError(f"alpha must be in (0, 1], got {config['alpha']}")

        if 'beta' in config and config['beta'] <= 0.0:
            raise ConfigurationError(f"beta must be positive, got {config['beta']}")

        if 'temperature' in config and config['temperature'] <= 0.0:
            raise ConfigurationError(f"temperature must be positive, got {config['temperature']}")

        if 'max_batch_size' in config and config['max_batch_size'] <= 0:
            raise ConfigurationError(f"max_batch_size must be positive, got {config['max_batch_size']}")

        # Try to create PBAConfig to validate compatibility
        pba_params = {k: v for k, v in config.items() if k in PBAConfig.__annotations__}
        if pba_params:
            _ = PBAConfig(**pba_params)

        logger.debug("Configuration validation passed")

    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")


def save_config(config: Dict[str, Any], output_path: Union[str, Path], format: str = "json") -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
        output_path: Output file path
        format: Output format ("json" or "yaml")

    Raises:
        ConfigurationError: If save operation fails
    """
    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if format.lower() == "json":
                json.dump(config, f, indent=2)
            elif format.lower() in ["yaml", "yml"]:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")

        logger.info(f"Configuration saved to {output_path}")

    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {e}")


def get_resource_recommendations(model_size: str) -> Dict[str, Any]:
    """
    Get resource allocation recommendations for model size.

    Args:
        model_size: Model size category ("small", "medium", "large", "xlarge")

    Returns:
        Resource recommendations dictionary

    Raises:
        ConfigurationError: If model size is not recognized
    """
    if model_size not in RESOURCE_RECOMMENDATIONS:
        available = list(RESOURCE_RECOMMENDATIONS.keys())
        raise ConfigurationError(
            f"Unknown model size '{model_size}'. Available: {available}"
        )

    return RESOURCE_RECOMMENDATIONS[model_size].copy()


def get_compliance_thresholds(use_case: str) -> Dict[str, float]:
    """
    Get compliance thresholds for specific use case.

    Args:
        use_case: Use case category ("medical", "financial", "legal", "general")

    Returns:
        Compliance thresholds dictionary

    Raises:
        ConfigurationError: If use case is not recognized
    """
    if use_case not in COMPLIANCE_THRESHOLDS:
        available = list(COMPLIANCE_THRESHOLDS.keys())
        raise ConfigurationError(
            f"Unknown use case '{use_case}'. Available: {available}"
        )

    thresholds = COMPLIANCE_THRESHOLDS[use_case].copy()
    # Remove description from thresholds
    thresholds.pop('description', None)
    return thresholds


def create_environment_config(
    environment: str,
    model_size: str,
    use_case: str = "general",
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create optimized configuration for specific deployment scenario.

    Args:
        environment: Target environment ("development", "production", "serving")
        model_size: Model size category ("small", "medium", "large", "xlarge")
        use_case: Use case for compliance thresholds
        custom_overrides: Custom configuration overrides

    Returns:
        Optimized configuration dictionary
    """
    # Start with environment defaults
    config = load_config(environment=environment)

    # Apply resource recommendations
    recommendations = get_resource_recommendations(model_size)
    config.update({
        'max_batch_size': recommendations['recommended_batch_size'],
        '_resource_info': {
            'model_size_category': model_size,
            'memory_base_gb': recommendations['memory_base_gb'],
            'memory_with_pba_gb': recommendations['memory_base_gb'] * (1 + recommendations['memory_pba_overhead']),
            'recommended_gpu_memory_gb': recommendations['gpu_memory_gb']
        }
    })

    # Add compliance thresholds
    thresholds = get_compliance_thresholds(use_case)
    config['_compliance_thresholds'] = thresholds

    # Apply custom overrides
    if custom_overrides:
        config.update(custom_overrides)

    return config


def validate_deployment_config(
    config: Dict[str, Any],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate configuration for deployment readiness.

    Args:
        config: Configuration to validate
        model_name: Optional model name for architecture checking

    Returns:
        Validation results with recommendations

    Raises:
        ConfigurationError: If critical validation fails
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }

    # Check basic PBA parameters
    if config.get('alpha', 0.9) != 0.9:
        results['warnings'].append(
            f"Alpha parameter {config['alpha']} differs from paper-optimized value (0.9)"
        )

    if config.get('beta', 0.5) != 0.5:
        results['warnings'].append(
            f"Beta parameter {config['beta']} differs from paper-optimized value (0.5)"
        )

    # Check resource allocation
    if '_resource_info' in config:
        resource_info = config['_resource_info']
        results['recommendations'].append(
            f"Estimated memory usage: {resource_info['memory_with_pba_gb']:.1f}GB "
            f"(including {resource_info['memory_pba_overhead']*100:.0f}% PBA overhead)"
        )

    # Check compliance thresholds
    if '_compliance_thresholds' in config:
        thresholds = config['_compliance_thresholds']
        if thresholds['max_ece'] < 0.02:
            results['warnings'].append(
                "Very strict ECE threshold (<2%) may require frequent recalibration"
            )

    # Check batch size
    batch_size = config.get('max_batch_size', 32)
    if batch_size > 64:
        results['warnings'].append(
            f"Large batch size ({batch_size}) may impact latency in serving scenarios"
        )

    # Architecture compatibility check
    if model_name:
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_name)
            arch = getattr(model_config, 'architectures', [None])[0]

            from .defaults import SUPPORTED_ARCHITECTURES
            if arch not in SUPPORTED_ARCHITECTURES:
                results['warnings'].append(
                    f"Architecture {arch} not explicitly tested. "
                    f"Supported: {SUPPORTED_ARCHITECTURES[:5]}..."
                )
        except Exception:
            results['warnings'].append(f"Could not validate model architecture for {model_name}")

    results['valid'] = len(results['errors']) == 0
    return results