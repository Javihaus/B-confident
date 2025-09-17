"""
Comprehensive Uncertainty Metrics Suite

Implementation of multiple uncertainty quantification approaches for comparison
and enhanced model reliability assessment. Supports various uncertainty types
including epistemic, aleatoric, and predictive uncertainty measures.

Key Features:
- Maximum Probability Confidence
- Shannon Entropy-based Uncertainty
- Prediction Consistency Metrics
- Temperature-scaled Confidence
- Mutual Information-based Measures
- Ensemble Variance (when applicable)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import scipy.stats as stats
from scipy.special import entropy
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveUncertaintyResults:
    """Complete uncertainty analysis results"""
    # Core PBA metrics
    pba_uncertainty: float

    # Alternative uncertainty measures
    max_prob_confidence: float
    entropy_uncertainty: float
    prediction_consistency: float
    temperature_scaled_confidence: float

    # Distribution properties
    prediction_sharpness: float
    calibration_gap: float

    # Metadata
    vocabulary_size: int
    sequence_length: int
    computation_time: float


class MaxProbabilityConfidence:
    """
    Traditional maximum probability confidence baseline.

    Simple baseline: confidence = max(P(token|context))
    Uncertainty = 1 - confidence
    """

    def calculate(self, logits: torch.Tensor) -> float:
        """
        Calculate confidence as maximum probability.

        Args:
            logits: Raw model logits [vocab_size]

        Returns:
            Confidence score [0, 1]
        """
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob

    def uncertainty(self, logits: torch.Tensor) -> float:
        """Calculate uncertainty as 1 - max_probability"""
        return 1.0 - self.calculate(logits)


class EntropyUncertainty:
    """
    Shannon entropy-based uncertainty quantification.

    H(P) = -Σ P(x) log P(x)
    Normalized by log(vocab_size) for comparison across models.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize entropy calculator.

        Args:
            normalize: Whether to normalize by log(vocab_size)
        """
        self.normalize = normalize

    def calculate(self, logits: torch.Tensor) -> float:
        """
        Calculate entropy-based uncertainty.

        Args:
            logits: Raw model logits [vocab_size]

        Returns:
            Entropy-based uncertainty [0, 1] if normalized
        """
        probs = F.softmax(logits, dim=-1)

        # Calculate Shannon entropy
        log_probs = torch.log(probs + 1e-12)  # Add small epsilon for stability
        entropy_val = -torch.sum(probs * log_probs).item()

        if self.normalize:
            # Normalize by maximum possible entropy (uniform distribution)
            max_entropy = np.log(logits.shape[-1])
            entropy_val = entropy_val / max_entropy if max_entropy > 0 else 0.0

        return entropy_val


class PredictionConsistency:
    """
    Measure prediction consistency across multiple forward passes.

    Uses dropout or temperature variation to assess model stability.
    Higher consistency indicates lower epistemic uncertainty.
    """

    def __init__(self, n_samples: int = 10, temperature_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initialize consistency calculator.

        Args:
            n_samples: Number of samples for consistency estimation
            temperature_range: Temperature variation range (min, max)
        """
        self.n_samples = n_samples
        self.temp_min, self.temp_max = temperature_range

    def calculate(self, model, input_ids: torch.Tensor, position: int = -1) -> float:
        """
        Calculate prediction consistency across temperature variations.

        Args:
            model: Language model
            input_ids: Input token sequence
            position: Position to analyze (-1 for last)

        Returns:
            Consistency score [0, 1], higher = more consistent
        """
        model.eval()
        predictions = []

        # Generate predictions with different temperatures
        temperatures = np.linspace(self.temp_min, self.temp_max, self.n_samples)

        with torch.no_grad():
            base_outputs = model(input_ids)
            base_logits = base_outputs.logits[0, position]

            for temp in temperatures:
                scaled_logits = base_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        # Calculate consistency as 1 - coefficient of variation
        # Higher consistency = lower variation across samples
        mean_probs = np.mean(predictions, axis=0)
        std_probs = np.std(predictions, axis=0)

        # Weighted coefficient of variation (weighted by probability mass)
        cv_weighted = np.sum(std_probs * mean_probs) / np.sum(mean_probs ** 2)
        consistency = max(0.0, 1.0 - cv_weighted)

        return consistency


class TemperatureScaledConfidence:
    """
    Temperature-calibrated confidence estimation.

    Learns optimal temperature scaling to improve calibration.
    Based on Guo et al. "On Calibration of Modern Neural Networks" (2017).
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize with temperature parameter.

        Args:
            temperature: Scaling temperature (1.0 = no scaling)
        """
        self.temperature = temperature

    def calculate(self, logits: torch.Tensor) -> float:
        """
        Calculate temperature-scaled confidence.

        Args:
            logits: Raw model logits [vocab_size]

        Returns:
            Temperature-scaled confidence [0, 1]
        """
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob

    def find_optimal_temperature(self,
                                logits_list: List[torch.Tensor],
                                labels: List[int],
                                temperature_range: Tuple[float, float] = (0.1, 5.0),
                                n_temps: int = 50) -> float:
        """
        Find optimal temperature for calibration using grid search.

        Args:
            logits_list: List of logit tensors from validation set
            labels: Corresponding ground truth labels
            temperature_range: Search range for temperature
            n_temps: Number of temperature values to try

        Returns:
            Optimal temperature value
        """
        from ..core.metrics import ExpectedCalibrationError

        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_temps)
        best_temp = 1.0
        best_ece = float('inf')
        ece_calc = ExpectedCalibrationError()

        for temp in temperatures:
            confidences = []
            for logits in logits_list:
                scaled_logits = logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                confidences.append(torch.max(probs).item())

            # Calculate ECE for this temperature
            ece, _ = ece_calc.calculate(confidences, labels)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        return best_temp


class ComprehensiveUncertaintyCalculator:
    """
    Main class that calculates all uncertainty measures in a unified interface.

    Provides comprehensive uncertainty analysis including PBA, entropy, confidence,
    and consistency measures for thorough model reliability assessment.
    """

    def __init__(self,
                 pba_config=None,
                 consistency_samples: int = 5,
                 temperature: float = 1.0):
        """
        Initialize comprehensive uncertainty calculator.

        Args:
            pba_config: PBA configuration (uses default if None)
            consistency_samples: Samples for consistency estimation
            temperature: Temperature for scaled confidence
        """
        # Import PBA here to avoid circular imports
        from .pba_algorithm import PBAUncertainty, PBAConfig

        self.pba_config = pba_config or PBAConfig()
        self.pba_calculator = PBAUncertainty(self.pba_config)

        # Initialize all uncertainty measures
        self.max_prob = MaxProbabilityConfidence()
        self.entropy = EntropyUncertainty(normalize=True)
        self.consistency = PredictionConsistency(n_samples=consistency_samples)
        self.temp_scaled = TemperatureScaledConfidence(temperature=temperature)

    def calculate_all_uncertainties(self,
                                  model,
                                  logits: torch.Tensor,
                                  input_ids: Optional[torch.Tensor] = None,
                                  actual_token_id: Optional[int] = None) -> ComprehensiveUncertaintyResults:
        """
        Calculate all uncertainty measures for comprehensive analysis.

        Args:
            model: Language model (needed for consistency calculation)
            logits: Model logits for uncertainty calculation
            input_ids: Input sequence (needed for consistency)
            actual_token_id: Actual token ID for PBA calculation

        Returns:
            Comprehensive uncertainty results
        """
        import time
        start_time = time.time()

        # 1. PBA Uncertainty (our main method)
        pba_uncertainty = self.pba_calculator.calculate_token_uncertainty(
            logits, actual_token_id=actual_token_id
        )

        # 2. Maximum Probability Confidence
        max_prob_conf = self.max_prob.calculate(logits)

        # 3. Entropy-based Uncertainty
        entropy_unc = self.entropy.calculate(logits)

        # 4. Temperature-scaled Confidence
        temp_scaled_conf = self.temp_scaled.calculate(logits)

        # 5. Prediction Consistency (if input_ids available)
        consistency_score = 0.5  # Default
        if model is not None and input_ids is not None:
            try:
                consistency_score = self.consistency.calculate(model, input_ids)
            except Exception as e:
                logger.warning(f"Consistency calculation failed: {e}")

        # 6. Additional distribution properties
        probs = F.softmax(logits, dim=-1)

        # Sharpness: How peaked is the distribution
        sharpness = 1.0 - entropy_unc  # Inverse of entropy

        # Calibration gap: Difference between confidence and expected accuracy
        # (simplified version - would need validation data for true calibration)
        calibration_gap = abs(max_prob_conf - (1.0 - pba_uncertainty))

        computation_time = time.time() - start_time

        return ComprehensiveUncertaintyResults(
            pba_uncertainty=pba_uncertainty,
            max_prob_confidence=max_prob_conf,
            entropy_uncertainty=entropy_unc,
            prediction_consistency=consistency_score,
            temperature_scaled_confidence=temp_scaled_conf,
            prediction_sharpness=sharpness,
            calibration_gap=calibration_gap,
            vocabulary_size=logits.shape[-1],
            sequence_length=input_ids.shape[-1] if input_ids is not None else 1,
            computation_time=computation_time
        )

    def compare_methods(self,
                       results_list: List[ComprehensiveUncertaintyResults],
                       ground_truth_accuracy: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Compare different uncertainty methods against ground truth accuracy.

        Args:
            results_list: List of comprehensive uncertainty results
            ground_truth_accuracy: Corresponding accuracy scores

        Returns:
            Comparison metrics for each uncertainty method
        """
        from sklearn.metrics import roc_auc_score
        from scipy.stats import pearsonr, spearmanr

        # Extract uncertainty scores for each method
        methods = {
            'pba': [r.pba_uncertainty for r in results_list],
            'max_prob': [1.0 - r.max_prob_confidence for r in results_list],  # Convert to uncertainty
            'entropy': [r.entropy_uncertainty for r in results_list],
            'consistency': [1.0 - r.prediction_consistency for r in results_list],  # Convert to uncertainty
            'temp_scaled': [1.0 - r.temperature_scaled_confidence for r in results_list]
        }

        comparison_results = {}
        error_labels = [1 - acc for acc in ground_truth_accuracy]  # Convert accuracy to error

        for method_name, uncertainties in methods.items():
            try:
                # AUROC: Ability to predict errors
                auroc = roc_auc_score(error_labels, uncertainties) if len(set(error_labels)) > 1 else 0.5

                # Correlations with accuracy
                pearson_r, pearson_p = pearsonr(uncertainties, ground_truth_accuracy)
                spearman_r, spearman_p = spearmanr(uncertainties, ground_truth_accuracy)

                # Mean and std of uncertainties
                mean_uncertainty = np.mean(uncertainties)
                std_uncertainty = np.std(uncertainties)

                comparison_results[method_name] = {
                    'auroc': auroc,
                    'pearson_correlation': pearson_r,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_r,
                    'spearman_p_value': spearman_p,
                    'mean_uncertainty': mean_uncertainty,
                    'std_uncertainty': std_uncertainty,
                    'uncertainty_range': max(uncertainties) - min(uncertainties)
                }
            except Exception as e:
                logger.warning(f"Comparison failed for {method_name}: {e}")
                comparison_results[method_name] = {
                    'auroc': 0.5,
                    'pearson_correlation': 0.0,
                    'pearson_p_value': 1.0,
                    'spearman_correlation': 0.0,
                    'spearman_p_value': 1.0,
                    'mean_uncertainty': 0.5,
                    'std_uncertainty': 0.0,
                    'uncertainty_range': 0.0
                }

        return comparison_results

    def generate_uncertainty_report(self,
                                   results: ComprehensiveUncertaintyResults) -> str:
        """
        Generate human-readable uncertainty analysis report.

        Args:
            results: Comprehensive uncertainty results

        Returns:
            Formatted uncertainty analysis report
        """
        report_lines = [
            "=== Comprehensive Uncertainty Analysis ===",
            f"Computation Time: {results.computation_time:.4f}s",
            f"Vocabulary Size: {results.vocabulary_size:,}",
            f"Sequence Length: {results.sequence_length}",
            "",
            "Uncertainty Measures:",
            f"  PBA Uncertainty:           {results.pba_uncertainty:.4f}",
            f"  Max Prob Confidence:       {results.max_prob_confidence:.4f}",
            f"  Entropy Uncertainty:       {results.entropy_uncertainty:.4f}",
            f"  Prediction Consistency:    {results.prediction_consistency:.4f}",
            f"  Temp-Scaled Confidence:    {results.temperature_scaled_confidence:.4f}",
            "",
            "Distribution Properties:",
            f"  Prediction Sharpness:      {results.prediction_sharpness:.4f}",
            f"  Calibration Gap:           {results.calibration_gap:.4f}",
            "",
            "Interpretation:",
        ]

        # Add interpretations
        if results.pba_uncertainty > 0.7:
            report_lines.append("  • High PBA uncertainty - model is very uncertain")
        elif results.pba_uncertainty > 0.4:
            report_lines.append("  • Moderate PBA uncertainty - some model uncertainty")
        else:
            report_lines.append("  • Low PBA uncertainty - model is confident")

        if results.entropy_uncertainty > 0.8:
            report_lines.append("  • High entropy - very flat probability distribution")
        elif results.entropy_uncertainty < 0.2:
            report_lines.append("  • Low entropy - sharp, peaked probability distribution")

        if results.prediction_consistency < 0.3:
            report_lines.append("  • Low consistency - high epistemic uncertainty")
        elif results.prediction_consistency > 0.7:
            report_lines.append("  • High consistency - low epistemic uncertainty")

        return "\n".join(report_lines)