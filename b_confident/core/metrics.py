"""
Uncertainty Quantification Metrics

Implementation of calibration and reliability metrics for evaluating uncertainty estimates.
Based on the evaluation methodology from the PBA paper.

Key metrics:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams
- Statistical significance testing
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResults:
    """Results from calibration analysis"""
    ece: float  # Expected Calibration Error
    brier_score: float  # Brier Score
    auroc: float  # Area Under ROC Curve
    reliability_bins: List[Tuple[float, float, int]]  # (confidence, accuracy, count) per bin
    statistical_significance: Optional[Dict[str, float]]  # p-values vs baselines
    stability_score: float  # Consistency across validation folds


class ExpectedCalibrationError:
    """
    Expected Calibration Error (ECE) implementation.

    Measures the gap between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.

    ECE = Σ |Bm|/n * |acc(Bm) - conf(Bm)|
    where Bm are confidence bins and n is total samples.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize ECE calculator.

        Args:
            n_bins: Number of confidence bins for reliability diagram
        """
        self.n_bins = n_bins

    def calculate(
        self,
        confidences: Union[List[float], np.ndarray, torch.Tensor],
        accuracies: Union[List[int], np.ndarray, torch.Tensor]
    ) -> Tuple[float, List[Tuple[float, float, int]]]:
        """
        Calculate Expected Calibration Error.

        Args:
            confidences: Predicted confidence scores [0, 1]
            accuracies: Binary correctness indicators {0, 1}

        Returns:
            Tuple of (ECE score, reliability bins info)
        """
        # Convert inputs to numpy arrays
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()
        if isinstance(accuracies, torch.Tensor):
            accuracies = accuracies.cpu().numpy()

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        if len(confidences) != len(accuracies):
            raise ValueError("confidences and accuracies must have same length")

        n_samples = len(confidences)
        ece = 0.0
        reliability_bins = []

        # Create confidence bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)

        for i in range(self.n_bins):
            # Find samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            if i == self.n_bins - 1:  # Include upper boundary for last bin
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            bin_size = in_bin.sum()

            if bin_size > 0:
                # Calculate bin confidence and accuracy
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()

                # Add to ECE
                ece += (bin_size / n_samples) * abs(bin_accuracy - bin_confidence)

                reliability_bins.append((bin_confidence, bin_accuracy, bin_size))
            else:
                reliability_bins.append((0.0, 0.0, 0))

        return ece, reliability_bins

    def compare_with_baseline(
        self,
        pba_confidences: Union[List[float], np.ndarray],
        baseline_confidences: Union[List[float], np.ndarray],
        accuracies: Union[List[int], np.ndarray]
    ) -> Dict[str, float]:
        """
        Compare PBA ECE with baseline method ECE.

        Returns statistical significance testing results.
        """
        pba_ece, _ = self.calculate(pba_confidences, accuracies)
        baseline_ece, _ = self.calculate(baseline_confidences, accuracies)

        # Perform paired t-test on per-sample calibration errors
        pba_errors = np.abs(np.array(pba_confidences) - np.array(accuracies))
        baseline_errors = np.abs(np.array(baseline_confidences) - np.array(accuracies))

        t_stat, p_value = stats.ttest_rel(pba_errors, baseline_errors)

        return {
            'pba_ece': pba_ece,
            'baseline_ece': baseline_ece,
            'improvement': (baseline_ece - pba_ece) / baseline_ece,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


class BrierScore:
    """
    Brier Score implementation for probabilistic predictions.

    Measures both calibration and sharpness as a proper scoring rule.
    Lower Brier Score indicates better probabilistic predictions.

    BS = 1/n * Σ (pi - yi)²
    where pi is predicted probability and yi is binary outcome.
    """

    def calculate(
        self,
        predicted_probs: Union[List[float], np.ndarray, torch.Tensor],
        actual_outcomes: Union[List[int], np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate Brier Score.

        Args:
            predicted_probs: Predicted probabilities [0, 1]
            actual_outcomes: Binary outcomes {0, 1}

        Returns:
            Brier Score (lower is better)
        """
        # Convert to numpy arrays
        if isinstance(predicted_probs, torch.Tensor):
            predicted_probs = predicted_probs.cpu().numpy()
        if isinstance(actual_outcomes, torch.Tensor):
            actual_outcomes = actual_outcomes.cpu().numpy()

        predicted_probs = np.array(predicted_probs)
        actual_outcomes = np.array(actual_outcomes)

        if len(predicted_probs) != len(actual_outcomes):
            raise ValueError("predicted_probs and actual_outcomes must have same length")

        # Calculate Brier Score
        brier_score = np.mean((predicted_probs - actual_outcomes) ** 2)

        return brier_score

    def decompose(
        self,
        predicted_probs: Union[List[float], np.ndarray],
        actual_outcomes: Union[List[int], np.ndarray],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Decompose Brier Score into reliability, resolution, and uncertainty components.

        BS = Reliability - Resolution + Uncertainty
        """
        predicted_probs = np.array(predicted_probs)
        actual_outcomes = np.array(actual_outcomes)

        n_samples = len(predicted_probs)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        # Overall base rate
        base_rate = np.mean(actual_outcomes)
        uncertainty = base_rate * (1 - base_rate)

        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            if i == n_bins - 1:
                in_bin = (predicted_probs >= bin_lower) & (predicted_probs <= bin_upper)
            else:
                in_bin = (predicted_probs >= bin_lower) & (predicted_probs < bin_upper)

            bin_size = in_bin.sum()

            if bin_size > 0:
                bin_prob = predicted_probs[in_bin].mean()
                bin_outcome = actual_outcomes[in_bin].mean()

                # Reliability: weighted squared difference between bin probability and bin outcome
                reliability += (bin_size / n_samples) * (bin_prob - bin_outcome) ** 2

                # Resolution: weighted squared difference between bin outcome and base rate
                resolution += (bin_size / n_samples) * (bin_outcome - base_rate) ** 2

        brier_score = reliability - resolution + uncertainty

        return {
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty
        }


def calculate_uncertainty_metrics(
    uncertainties: Union[List[float], np.ndarray, torch.Tensor],
    correctness: Union[List[int], np.ndarray, torch.Tensor],
    confidences: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
    n_bins: int = 10
) -> CalibrationResults:
    """
    Calculate comprehensive uncertainty quantification metrics.

    Args:
        uncertainties: PBA uncertainty scores [0, 1]
        correctness: Binary correctness indicators {0, 1}
        confidences: Optional confidence scores (uses 1-uncertainties if None)
        n_bins: Number of bins for calibration analysis

    Returns:
        CalibrationResults with all metrics
    """
    # Convert inputs
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.cpu().numpy()
    if isinstance(correctness, torch.Tensor):
        correctness = correctness.cpu().numpy()

    uncertainties = np.array(uncertainties)
    correctness = np.array(correctness)

    # Use 1-uncertainty as confidence if not provided
    if confidences is None:
        confidences = 1.0 - uncertainties
    else:
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()
        confidences = np.array(confidences)

    # Calculate ECE
    ece_calculator = ExpectedCalibrationError(n_bins=n_bins)
    ece, reliability_bins = ece_calculator.calculate(confidences, correctness)

    # Calculate Brier Score
    brier_calculator = BrierScore()
    brier_score = brier_calculator.calculate(confidences, correctness)

    # Calculate AUROC
    try:
        # Check if we have variance in the labels
        if len(np.unique(correctness)) < 2:
            # No discrimination possible - all labels are the same
            auroc = 0.5
        else:
            # For uncertainty quantification, we want high uncertainty to predict errors
            # So we predict error (1 - correctness) using uncertainty
            error_labels = 1 - correctness
            auroc = roc_auc_score(error_labels, uncertainties)
    except ValueError as e:
        logger.warning(f"AUROC calculation failed: {e}")
        auroc = 0.5

    # Calculate stability score (coefficient of variation)
    if len(set(uncertainties)) > 1:
        stability_score = 1.0 - (np.std(uncertainties) / np.mean(uncertainties))
        stability_score = max(0.0, min(1.0, stability_score))  # Clamp to [0, 1]
    else:
        stability_score = 1.0

    return CalibrationResults(
        ece=ece,
        brier_score=brier_score,
        auroc=auroc,
        reliability_bins=reliability_bins,
        statistical_significance=None,  # Can be populated by comparison functions
        stability_score=stability_score
    )


def cross_validation_analysis(
    uncertainty_scores_folds: List[List[float]],
    correctness_folds: List[List[int]],
    n_bins: int = 10
) -> Dict[str, Union[float, List[CalibrationResults]]]:
    """
    Perform cross-validation robustness analysis.

    Based on Table 4 from the paper: 5-fold cross-validation results.

    Args:
        uncertainty_scores_folds: List of uncertainty scores per fold
        correctness_folds: List of correctness indicators per fold
        n_bins: Number of calibration bins

    Returns:
        Dictionary with mean/std metrics and per-fold results
    """
    if len(uncertainty_scores_folds) != len(correctness_folds):
        raise ValueError("Number of folds must match for uncertainties and correctness")

    fold_results = []

    for uncertainties, correctness in zip(uncertainty_scores_folds, correctness_folds):
        fold_result = calculate_uncertainty_metrics(
            uncertainties, correctness, n_bins=n_bins
        )
        fold_results.append(fold_result)

    # Calculate statistics across folds
    ece_scores = [result.ece for result in fold_results]
    brier_scores = [result.brier_score for result in fold_results]
    auroc_scores = [result.auroc for result in fold_results]
    stability_scores = [result.stability_score for result in fold_results]

    return {
        'ece_mean': np.mean(ece_scores),
        'ece_std': np.std(ece_scores),
        'brier_mean': np.mean(brier_scores),
        'brier_std': np.std(brier_scores),
        'auroc_mean': np.mean(auroc_scores),
        'auroc_std': np.std(auroc_scores),
        'stability_mean': np.mean(stability_scores),
        'stability_std': np.std(stability_scores),
        'fold_results': fold_results,
        'n_folds': len(fold_results)
    }


def statistical_significance_test(
    pba_scores: List[float],
    baseline_scores: List[float],
    metric_name: str = "uncertainty"
) -> Dict[str, float]:
    """
    Perform statistical significance testing between PBA and baseline methods.

    Uses paired t-test as in the paper's methodology.

    Args:
        pba_scores: PBA uncertainty/confidence scores
        baseline_scores: Baseline method scores
        metric_name: Name of metric being compared

    Returns:
        Statistical test results
    """
    if len(pba_scores) != len(baseline_scores):
        raise ValueError("Score arrays must have same length")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(pba_scores, baseline_scores)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(pba_scores) + np.var(baseline_scores)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(pba_scores) - np.mean(baseline_scores)) / pooled_std
    else:
        cohens_d = 0.0

    return {
        'metric': metric_name,
        'pba_mean': np.mean(pba_scores),
        'baseline_mean': np.mean(baseline_scores),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) >= 0.8 else ('medium' if abs(cohens_d) >= 0.5 else 'small'),
        'improvement_percent': (
            (np.mean(baseline_scores) - np.mean(pba_scores)) / np.mean(baseline_scores) * 100
            if np.mean(baseline_scores) != 0 else 0.0
        )
    }