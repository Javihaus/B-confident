"""
Core PBA uncertainty quantification implementation.

This module contains the fundamental PBA algorithm and configuration classes
based on the methodology described in the paper.
"""

from .pba_algorithm import PBAUncertainty, PBAConfig
from .metrics import (
    ExpectedCalibrationError,
    BrierScore,
    calculate_uncertainty_metrics,
    CalibrationResults
)

__all__ = [
    "PBAUncertainty",
    "PBAConfig",
    "ExpectedCalibrationError",
    "BrierScore",
    "calculate_uncertainty_metrics",
    "CalibrationResults"
]