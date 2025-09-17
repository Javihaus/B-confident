"""
Integration modules for various ML frameworks and serving systems.

Provides drop-in replacements and wrappers for popular frameworks
while adding uncertainty quantification capabilities.
"""

from .transformers_wrapper import (
    UncertaintyTransformersModel,
    uncertainty_generate
)

__all__ = [
    "UncertaintyTransformersModel",
    "uncertainty_generate"
]