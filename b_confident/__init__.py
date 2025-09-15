"""
B-Confident: Perplexity-Based Adjacency for Uncertainty Quantification in LLMs

This package implements the PBA methodology for enterprise-grade uncertainty
quantification in Large Language Models with regulatory compliance support.

Key Components:
    - Core PBA algorithm implementation
    - Hugging Face transformers integration
    - Regulatory compliance tools
    - Serving framework integrations
    - Calibration and validation tools
"""

__version__ = "0.1.0"
__author__ = "Javier Marin"
__email__ = "javier@jmarin.info"

# Core imports
from .core.pba_algorithm import PBAUncertainty, PBAConfig
from .core.metrics import (
    ExpectedCalibrationError,
    BrierScore,
    calculate_uncertainty_metrics
)

# Integration imports
from .integration.transformers_wrapper import (
    UncertaintyTransformersModel,
    uncertainty_generate
)

# Compliance imports
from .compliance.regulatory import (
    ComplianceReporter,
    generate_eu_ai_act_report
)

# Main API
__all__ = [
    # Core
    "PBAUncertainty",
    "PBAConfig",

    # Metrics
    "ExpectedCalibrationError",
    "BrierScore",
    "calculate_uncertainty_metrics",

    # Integration
    "UncertaintyTransformersModel",
    "uncertainty_generate",

    # Compliance
    "ComplianceReporter",
    "generate_eu_ai_act_report",

    # Version info
    "__version__",
]

# Package metadata
PACKAGE_INFO = {
    "name": "b-confident",
    "version": __version__,
    "description": "Perplexity-Based Adjacency for Uncertainty Quantification in LLMs",
    "methodology_paper": "Perplexity-Based Adjacency for Uncertainty Quantification in Large Language Models",
    "regulatory_compliance": ["EU AI Act Article 15"],
    "supported_frameworks": ["transformers", "torch", "torchserve", "fastapi", "ray-serve"],
}