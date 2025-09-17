"""
Uncertainty Calculation Debugging & Observability Framework

Addresses the challenge of debugging uncertainty calculation pipelines when
calibration drifts from expected behavior. Provides instrumented pipeline with
component-level metrics, provenance tracking, and statistical process control.

Key Features:
- Instrumented uncertainty calculation pipeline
- Component-level metrics and logging
- Uncertainty provenance tracking
- Statistical process control for drift detection
- Pipeline stage monitoring and alerting
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from contextlib import contextmanager
import json

from ..core.pba_algorithm import PBAUncertainty, PBAConfig

logger = logging.getLogger(__name__)


class DebugLevel(Enum):
    """Debug verbosity levels"""
    MINIMAL = "minimal"       # Only major pipeline stages
    STANDARD = "standard"     # Include intermediate calculations
    DETAILED = "detailed"     # All mathematical transformations
    TRACE = "trace"          # Every computation step


class PipelineStage(Enum):
    """Uncertainty calculation pipeline stages"""
    INPUT_VALIDATION = "input_validation"
    LOGITS_PROCESSING = "logits_processing"
    PROBABILITY_CALCULATION = "probability_calculation"
    PERPLEXITY_CALCULATION = "perplexity_calculation"
    UNCERTAINTY_TRANSFORMATION = "uncertainty_transformation"
    CALIBRATION_APPLICATION = "calibration_application"
    OUTPUT_VALIDATION = "output_validation"


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage"""
    stage: PipelineStage
    execution_time: float
    input_stats: Dict[str, float]
    output_stats: Dict[str, float]
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class UncertaintyProvenance:
    """Complete provenance trace for uncertainty calculation"""
    request_id: str
    timestamp: float
    input_logits_shape: Tuple[int, ...]
    input_logits_stats: Dict[str, float]
    actual_token_id: Optional[int]
    config_params: Dict[str, Any]

    stage_metrics: List[StageMetrics] = field(default_factory=list)
    final_uncertainty: float = 0.0
    total_execution_time: float = 0.0
    pipeline_warnings: List[str] = field(default_factory=list)
    pipeline_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "input_logits_shape": self.input_logits_shape,
            "input_logits_stats": self.input_logits_stats,
            "actual_token_id": self.actual_token_id,
            "config_params": self.config_params,
            "stage_metrics": [
                {
                    "stage": stage.stage.value,
                    "execution_time": stage.execution_time,
                    "input_stats": stage.input_stats,
                    "output_stats": stage.output_stats,
                    "intermediate_values": stage.intermediate_values,
                    "warnings": stage.warnings,
                    "errors": stage.errors
                }
                for stage in self.stage_metrics
            ],
            "final_uncertainty": self.final_uncertainty,
            "total_execution_time": self.total_execution_time,
            "pipeline_warnings": self.pipeline_warnings,
            "pipeline_errors": self.pipeline_errors
        }


@dataclass
class DistributionStats:
    """Statistical properties of a distribution"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    entropy: float
    kurtosis: float
    skewness: float

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DistributionStats':
        """Calculate distribution statistics from tensor"""
        flat_tensor = tensor.flatten().cpu().numpy()

        return cls(
            mean=float(np.mean(flat_tensor)),
            std=float(np.std(flat_tensor)),
            min=float(np.min(flat_tensor)),
            max=float(np.max(flat_tensor)),
            median=float(np.median(flat_tensor)),
            q25=float(np.percentile(flat_tensor, 25)),
            q75=float(np.percentile(flat_tensor, 75)),
            entropy=cls._calculate_entropy(flat_tensor),
            kurtosis=float(cls._calculate_kurtosis(flat_tensor)),
            skewness=float(cls._calculate_skewness(flat_tensor))
        )

    @staticmethod
    def _calculate_entropy(values: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        try:
            # Convert to probabilities if needed
            if np.any(values < 0) or np.sum(values) != 1.0:
                # Assume these are logits or raw values
                probs = np.exp(values - np.max(values))
                probs = probs / np.sum(probs)
            else:
                probs = values

            # Remove zeros to avoid log(0)
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs + 1e-12)))
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_kurtosis(values: np.ndarray) -> float:
        """Calculate kurtosis"""
        try:
            from scipy import stats
            return float(stats.kurtosis(values))
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_skewness(values: np.ndarray) -> float:
        """Calculate skewness"""
        try:
            from scipy import stats
            return float(stats.skew(values))
        except Exception:
            return 0.0


class StatisticalProcessController:
    """Monitor distributions for drift and anomalies"""

    def __init__(self,
                 window_size: int = 100,
                 control_limits: float = 3.0,
                 min_samples: int = 30):
        self.window_size = window_size
        self.control_limits = control_limits
        self.min_samples = min_samples

        # Historical data for each stage
        self.stage_histories: Dict[PipelineStage, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )

        # Control charts
        self.control_charts: Dict[PipelineStage, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        self._lock = threading.RLock()

    def add_measurement(self,
                       stage: PipelineStage,
                       metric_name: str,
                       value: float) -> Optional[str]:
        """
        Add measurement and check for statistical anomalies.

        Returns:
            Warning message if anomaly detected, None otherwise
        """
        with self._lock:
            history = self.stage_histories[stage][metric_name]
            history.append(value)

            # Need minimum samples to establish control limits
            if len(history) < self.min_samples:
                return None

            # Calculate control limits if not already done
            if metric_name not in self.control_charts[stage]:
                self._calculate_control_limits(stage, metric_name)

            # Check for out-of-control conditions
            return self._check_control_limits(stage, metric_name, value)

    def _calculate_control_limits(self, stage: PipelineStage, metric_name: str) -> None:
        """Calculate statistical control limits for a metric"""
        history = list(self.stage_histories[stage][metric_name])

        mean = np.mean(history)
        std = np.std(history)

        self.control_charts[stage][metric_name] = {
            "mean": mean,
            "std": std,
            "upper_control_limit": mean + self.control_limits * std,
            "lower_control_limit": mean - self.control_limits * std,
            "upper_warning_limit": mean + 2.0 * std,
            "lower_warning_limit": mean - 2.0 * std
        }

    def _check_control_limits(self,
                            stage: PipelineStage,
                            metric_name: str,
                            value: float) -> Optional[str]:
        """Check if value is within control limits"""
        limits = self.control_charts[stage][metric_name]

        if value > limits["upper_control_limit"] or value < limits["lower_control_limit"]:
            return f"CONTROL VIOLATION: {stage.value}.{metric_name} = {value:.4f} outside control limits [{limits['lower_control_limit']:.4f}, {limits['upper_control_limit']:.4f}]"

        if value > limits["upper_warning_limit"] or value < limits["lower_warning_limit"]:
            return f"WARNING: {stage.value}.{metric_name} = {value:.4f} outside warning limits [{limits['lower_warning_limit']:.4f}, {limits['upper_warning_limit']:.4f}]"

        return None

    def get_control_chart_data(self, stage: PipelineStage, metric_name: str) -> Dict[str, Any]:
        """Get control chart data for visualization"""
        with self._lock:
            history = list(self.stage_histories[stage][metric_name])
            limits = self.control_charts[stage].get(metric_name, {})

            return {
                "stage": stage.value,
                "metric": metric_name,
                "values": history,
                "limits": limits,
                "current_value": history[-1] if history else None,
                "sample_count": len(history)
            }


class InstrumentedUncertaintyCalculator:
    """
    Instrumented uncertainty calculator with full observability.

    Wraps the standard PBA uncertainty calculation with comprehensive
    debugging, metrics collection, and provenance tracking.
    """

    def __init__(self,
                 pba_config: Optional[PBAConfig] = None,
                 debug_level: DebugLevel = DebugLevel.STANDARD,
                 enable_provenance: bool = True,
                 enable_spc: bool = True):

        self.pba_config = pba_config or PBAConfig()
        self.pba_calculator = PBAUncertainty(self.pba_config)
        self.debug_level = debug_level
        self.enable_provenance = enable_provenance
        self.enable_spc = enable_spc

        # Statistical process controller
        self.spc = StatisticalProcessController() if enable_spc else None

        # Provenance storage
        self.provenance_history: deque = deque(maxlen=1000)

        # Metrics aggregation
        self.stage_metrics: Dict[PipelineStage, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self._request_counter = 0
        self._lock = threading.RLock()

        logger.info(f"Initialized InstrumentedUncertaintyCalculator with debug_level={debug_level.value}")

    def calculate_uncertainty_with_debugging(self,
                                           logits: torch.Tensor,
                                           actual_token_id: Optional[int] = None) -> Tuple[float, UncertaintyProvenance]:
        """
        Calculate uncertainty with full debugging and observability.

        Returns:
            Tuple of (uncertainty_score, provenance_trace)
        """
        with self._lock:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}_{int(time.time() * 1000000) % 1000000}"

        start_time = time.time()

        # Initialize provenance
        provenance = UncertaintyProvenance(
            request_id=request_id,
            timestamp=start_time,
            input_logits_shape=tuple(logits.shape),
            input_logits_stats=DistributionStats.from_tensor(logits).__dict__,
            actual_token_id=actual_token_id,
            config_params=self.pba_config.__dict__.copy()
        )

        try:
            # Stage 1: Input Validation
            uncertainty = self._stage_input_validation(logits, actual_token_id, provenance)

            # Stage 2: Logits Processing
            processed_logits = self._stage_logits_processing(logits, provenance)

            # Stage 3: Probability Calculation
            probabilities = self._stage_probability_calculation(processed_logits, provenance)

            # Stage 4: Perplexity Calculation
            perplexity = self._stage_perplexity_calculation(
                probabilities, processed_logits, actual_token_id, provenance
            )

            # Stage 5: Uncertainty Transformation
            uncertainty = self._stage_uncertainty_transformation(perplexity, provenance)

            # Stage 6: Calibration Application (if applicable)
            calibrated_uncertainty = self._stage_calibration_application(uncertainty, provenance)

            # Stage 7: Output Validation
            final_uncertainty = self._stage_output_validation(calibrated_uncertainty, provenance)

            provenance.final_uncertainty = final_uncertainty
            provenance.total_execution_time = time.time() - start_time

        except Exception as e:
            provenance.pipeline_errors.append(f"Pipeline error: {str(e)}")
            provenance.final_uncertainty = 1.0  # Maximum uncertainty on error
            provenance.total_execution_time = time.time() - start_time
            logger.error(f"Error in uncertainty calculation for {request_id}: {e}")

        # Store provenance
        if self.enable_provenance:
            self.provenance_history.append(provenance)

        return provenance.final_uncertainty, provenance

    def _stage_input_validation(self,
                               logits: torch.Tensor,
                               actual_token_id: Optional[int],
                               provenance: UncertaintyProvenance) -> float:
        """Stage 1: Validate inputs"""
        stage_start = time.time()
        stage = PipelineStage.INPUT_VALIDATION

        warnings = []
        errors = []
        intermediate_values = {}

        # Validate tensor properties
        if not isinstance(logits, torch.Tensor):
            errors.append(f"logits must be torch.Tensor, got {type(logits)}")

        if logits.dim() < 1:
            errors.append(f"logits must be at least 1D, got shape {logits.shape}")

        if torch.isnan(logits).any():
            errors.append("logits contains NaN values")

        if torch.isinf(logits).any():
            errors.append("logits contains infinite values")

        # Check for numerical issues
        logits_range = torch.max(logits) - torch.min(logits)
        if logits_range > 100:
            warnings.append(f"Large logits range: {logits_range:.2f} may cause numerical instability")

        intermediate_values.update({
            "logits_range": float(logits_range),
            "logits_max": float(torch.max(logits)),
            "logits_min": float(torch.min(logits)),
            "vocab_size": logits.shape[-1]
        })

        # Validate token ID
        if actual_token_id is not None:
            if actual_token_id < 0 or actual_token_id >= logits.shape[-1]:
                errors.append(f"actual_token_id {actual_token_id} out of range [0, {logits.shape[-1]})")

        execution_time = time.time() - stage_start

        # Record stage metrics
        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats=DistributionStats.from_tensor(logits).__dict__,
            output_stats={},  # No transformation in validation
            intermediate_values=intermediate_values,
            warnings=warnings,
            errors=errors
        )

        provenance.stage_metrics.append(stage_metrics)

        # Update statistical process control
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)
            self.spc.add_measurement(stage, "logits_range", float(logits_range))

        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        return 0.0  # No uncertainty calculated yet

    def _stage_logits_processing(self,
                                logits: torch.Tensor,
                                provenance: UncertaintyProvenance) -> torch.Tensor:
        """Stage 2: Process and normalize logits"""
        stage_start = time.time()
        stage = PipelineStage.LOGITS_PROCESSING

        warnings = []
        intermediate_values = {}

        # Ensure 1D tensor
        if logits.dim() > 1:
            processed_logits = logits.squeeze()
            warnings.append(f"Squeezed logits from shape {logits.shape} to {processed_logits.shape}")
        else:
            processed_logits = logits

        # Apply temperature scaling
        if self.pba_config.temperature != 1.0:
            processed_logits = processed_logits / self.pba_config.temperature
            intermediate_values["temperature_applied"] = self.pba_config.temperature

        # Check for numerical stability after processing
        processed_range = torch.max(processed_logits) - torch.min(processed_logits)
        intermediate_values.update({
            "processed_range": float(processed_range),
            "temperature_scaling": self.pba_config.temperature != 1.0
        })

        execution_time = time.time() - stage_start

        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats=DistributionStats.from_tensor(logits).__dict__,
            output_stats=DistributionStats.from_tensor(processed_logits).__dict__,
            intermediate_values=intermediate_values,
            warnings=warnings,
            errors=[]
        )

        provenance.stage_metrics.append(stage_metrics)

        # SPC monitoring
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)
            self.spc.add_measurement(stage, "processed_range", float(processed_range))

        return processed_logits

    def _stage_probability_calculation(self,
                                     processed_logits: torch.Tensor,
                                     provenance: UncertaintyProvenance) -> torch.Tensor:
        """Stage 3: Convert logits to probabilities"""
        stage_start = time.time()
        stage = PipelineStage.PROBABILITY_CALCULATION

        warnings = []
        intermediate_values = {}

        # Calculate log probabilities (more numerically stable)
        log_probs = torch.nn.functional.log_softmax(processed_logits, dim=-1)

        # Convert to probabilities
        probs = torch.exp(log_probs)

        # Check probability distribution properties
        prob_sum = torch.sum(probs)
        prob_max = torch.max(probs)
        prob_entropy = -torch.sum(probs * log_probs)

        if abs(prob_sum - 1.0) > 1e-6:
            warnings.append(f"Probability sum deviation: {prob_sum:.8f} != 1.0")

        intermediate_values.update({
            "probability_sum": float(prob_sum),
            "max_probability": float(prob_max),
            "entropy": float(prob_entropy),
            "numerical_method": "log_softmax + exp"
        })

        execution_time = time.time() - stage_start

        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats=DistributionStats.from_tensor(processed_logits).__dict__,
            output_stats=DistributionStats.from_tensor(probs).__dict__,
            intermediate_values=intermediate_values,
            warnings=warnings,
            errors=[]
        )

        provenance.stage_metrics.append(stage_metrics)

        # SPC monitoring
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)
            self.spc.add_measurement(stage, "entropy", float(prob_entropy))
            self.spc.add_measurement(stage, "max_probability", float(prob_max))

        return probs

    def _stage_perplexity_calculation(self,
                                    probs: torch.Tensor,
                                    logits: torch.Tensor,
                                    actual_token_id: Optional[int],
                                    provenance: UncertaintyProvenance) -> float:
        """Stage 4: Calculate perplexity for the target token"""
        stage_start = time.time()
        stage = PipelineStage.PERPLEXITY_CALCULATION

        warnings = []
        intermediate_values = {}

        # Determine target token
        if actual_token_id is not None:
            target_token = actual_token_id
            intermediate_values["token_selection_method"] = "actual_token_provided"
        else:
            target_token = torch.argmax(logits, dim=-1).item()
            intermediate_values["token_selection_method"] = "max_probability"
            warnings.append("No actual token provided, using max probability token")

        # Get token probability
        token_prob = probs[target_token].item()
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)[target_token].item()

        # Calculate perplexity: exp(-log P(token))
        perplexity = torch.exp(-log_prob).item()

        # Clamp perplexity to avoid numerical issues
        original_perplexity = perplexity
        perplexity = max(1.0, min(perplexity, 1000.0))

        if original_perplexity != perplexity:
            warnings.append(f"Perplexity clamped: {original_perplexity:.4f} -> {perplexity:.4f}")

        intermediate_values.update({
            "target_token_id": target_token,
            "token_probability": token_prob,
            "log_probability": log_prob,
            "raw_perplexity": original_perplexity,
            "clamped_perplexity": perplexity,
            "perplexity_clamped": original_perplexity != perplexity
        })

        execution_time = time.time() - stage_start

        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats=DistributionStats.from_tensor(probs).__dict__,
            output_stats={"perplexity": perplexity, "token_probability": token_prob},
            intermediate_values=intermediate_values,
            warnings=warnings,
            errors=[]
        )

        provenance.stage_metrics.append(stage_metrics)

        # SPC monitoring
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)
            self.spc.add_measurement(stage, "perplexity", perplexity)
            self.spc.add_measurement(stage, "token_probability", token_prob)

        return perplexity

    def _stage_uncertainty_transformation(self,
                                        perplexity: float,
                                        provenance: UncertaintyProvenance) -> float:
        """Stage 5: Transform perplexity to uncertainty"""
        stage_start = time.time()
        stage = PipelineStage.UNCERTAINTY_TRANSFORMATION

        intermediate_values = {}

        # Apply sensitivity function: f(p) = 1 - exp(-β·p)
        beta = self.pba_config.beta
        uncertainty = 1.0 - np.exp(-beta * perplexity)

        intermediate_values.update({
            "beta_parameter": beta,
            "perplexity_input": perplexity,
            "uncertainty_function": "1 - exp(-β * p)",
            "raw_uncertainty": uncertainty
        })

        execution_time = time.time() - stage_start

        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats={"perplexity": perplexity},
            output_stats={"uncertainty": uncertainty},
            intermediate_values=intermediate_values,
            warnings=[],
            errors=[]
        )

        provenance.stage_metrics.append(stage_metrics)

        # SPC monitoring
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)
            self.spc.add_measurement(stage, "uncertainty", uncertainty)

        return uncertainty

    def _stage_calibration_application(self,
                                     uncertainty: float,
                                     provenance: UncertaintyProvenance) -> float:
        """Stage 6: Apply calibration (if available)"""
        stage_start = time.time()
        stage = PipelineStage.CALIBRATION_APPLICATION

        # For now, pass-through (calibration would be applied here)
        calibrated_uncertainty = uncertainty

        intermediate_values = {
            "calibration_applied": False,
            "calibration_method": "none",
            "input_uncertainty": uncertainty,
            "output_uncertainty": calibrated_uncertainty
        }

        execution_time = time.time() - stage_start

        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats={"uncertainty": uncertainty},
            output_stats={"calibrated_uncertainty": calibrated_uncertainty},
            intermediate_values=intermediate_values,
            warnings=[],
            errors=[]
        )

        provenance.stage_metrics.append(stage_metrics)

        # SPC monitoring
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)

        return calibrated_uncertainty

    def _stage_output_validation(self,
                                uncertainty: float,
                                provenance: UncertaintyProvenance) -> float:
        """Stage 7: Validate final output"""
        stage_start = time.time()
        stage = PipelineStage.OUTPUT_VALIDATION

        warnings = []
        errors = []

        # Validate uncertainty range
        if uncertainty < 0.0 or uncertainty > 1.0:
            errors.append(f"Uncertainty {uncertainty:.6f} outside valid range [0, 1]")

        # Clamp to valid range
        final_uncertainty = min(max(uncertainty, 0.0), 1.0)

        if final_uncertainty != uncertainty:
            warnings.append(f"Uncertainty clamped: {uncertainty:.6f} -> {final_uncertainty:.6f}")

        intermediate_values = {
            "input_uncertainty": uncertainty,
            "final_uncertainty": final_uncertainty,
            "clamped": final_uncertainty != uncertainty,
            "valid_range": True if 0.0 <= final_uncertainty <= 1.0 else False
        }

        execution_time = time.time() - stage_start

        stage_metrics = StageMetrics(
            stage=stage,
            execution_time=execution_time,
            input_stats={"uncertainty": uncertainty},
            output_stats={"final_uncertainty": final_uncertainty},
            intermediate_values=intermediate_values,
            warnings=warnings,
            errors=errors
        )

        provenance.stage_metrics.append(stage_metrics)

        # SPC monitoring
        if self.spc:
            self.spc.add_measurement(stage, "execution_time", execution_time)
            self.spc.add_measurement(stage, "final_uncertainty", final_uncertainty)

        if errors:
            raise ValueError(f"Output validation failed: {'; '.join(errors)}")

        return final_uncertainty

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get aggregated pipeline metrics"""
        metrics = {}

        for stage in PipelineStage:
            stage_data = {}

            # Get SPC data if available
            if self.spc:
                for metric_name in ["execution_time", "uncertainty", "perplexity", "entropy"]:
                    try:
                        control_data = self.spc.get_control_chart_data(stage, metric_name)
                        if control_data["values"]:
                            stage_data[metric_name] = control_data
                    except KeyError:
                        continue

            if stage_data:
                metrics[stage.value] = stage_data

        return metrics

    def generate_debug_report(self, provenance: UncertaintyProvenance) -> str:
        """Generate human-readable debug report"""
        lines = [
            f"=== Uncertainty Calculation Debug Report ===",
            f"Request ID: {provenance.request_id}",
            f"Timestamp: {provenance.timestamp}",
            f"Total Time: {provenance.total_execution_time:.6f}s",
            f"Final Uncertainty: {provenance.final_uncertainty:.6f}",
            "",
            f"Input Configuration:",
            f"  Logits Shape: {provenance.input_logits_shape}",
            f"  Actual Token ID: {provenance.actual_token_id}",
            f"  Config: {provenance.config_params}",
            "",
            "Pipeline Stage Details:"
        ]

        for stage_metrics in provenance.stage_metrics:
            lines.extend([
                f"",
                f"  {stage_metrics.stage.value.upper()}:",
                f"    Execution Time: {stage_metrics.execution_time:.6f}s",
                f"    Warnings: {len(stage_metrics.warnings)}",
                f"    Errors: {len(stage_metrics.errors)}"
            ])

            if self.debug_level in [DebugLevel.DETAILED, DebugLevel.TRACE]:
                lines.append(f"    Intermediate Values:")
                for key, value in stage_metrics.intermediate_values.items():
                    lines.append(f"      {key}: {value}")

            if stage_metrics.warnings:
                lines.append(f"    Stage Warnings:")
                for warning in stage_metrics.warnings:
                    lines.append(f"      - {warning}")

            if stage_metrics.errors:
                lines.append(f"    Stage Errors:")
                for error in stage_metrics.errors:
                    lines.append(f"      - {error}")

        if provenance.pipeline_warnings:
            lines.extend(["", "Pipeline Warnings:"])
            for warning in provenance.pipeline_warnings:
                lines.append(f"  - {warning}")

        if provenance.pipeline_errors:
            lines.extend(["", "Pipeline Errors:"])
            for error in provenance.pipeline_errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)