"""
Configuration management for b-confident

Provides configuration loading, validation, and environment setup.
"""

from .defaults import DEFAULT_PBA_CONFIG, SUPPORTED_ARCHITECTURES
from .loader import load_config, save_config, ConfigurationError

__all__ = [
    "DEFAULT_PBA_CONFIG",
    "SUPPORTED_ARCHITECTURES",
    "load_config",
    "save_config",
    "ConfigurationError"
]