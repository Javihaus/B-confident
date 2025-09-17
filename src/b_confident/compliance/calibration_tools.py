"""
Calibration and Validation Tools

Tools for measuring uncertainty calibration on validation datasets and
continuous monitoring of calibration performance in production environments.

Required for EU AI Act Article 15 "consistent performance throughout lifecycle"
compliance.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings

from ..core.metrics import (
    CalibrationResults,
    calculate_uncertainty_metrics,
    ExpectedCalibrationError,
    BrierScore
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationDriftAlert:
    """Alert for calibration drift detection"""
    timestamp: datetime
    metric_name: str
    current_value: float
    baseline_value: float
    drift_magnitude: float
    alert_level: str  # "WARNING" or "CRITICAL"
    recommended_action: str


class CalibrationValidator:
    """
    Tools for measuring uncertainty calibration on validation datasets.

    Provides standardized procedures for testing uncertainty calibration
    during model deployment and validation that uncertainty scores correlate
    with prediction accuracy.
    """

    def __init__(self, n_bins: int = 10, random_seed: int = 42):
        """
        Initialize calibration validator.

        Args:
            n_bins: Number of bins for calibration analysis
            random_seed: Random seed for reproducible validation splits
        """
        self.n_bins = n_bins
        self.random_seed = random_seed
        self.ece_calculator = ExpectedCalibrationError(n_bins=n_bins)
        self.brier_calculator = BrierScore()

        np.random.seed(random_seed)

    def validate_calibration_on_dataset(
        self,
        uncertainty_scores: List[float],
        correctness_labels: List[int],
        dataset_name: str = "validation",
        confidence_scores: Optional[List[float]] = None
    ) -> CalibrationResults:
        """
        Validate uncertainty calibration on a labeled dataset.

        Args:
            uncertainty_scores: PBA uncertainty scores [0, 1]
            correctness_labels: Binary correctness indicators {0, 1}
            dataset_name: Name of validation dataset
            confidence_scores: Optional confidence scores (uses 1-uncertainty if None)

        Returns:
            Comprehensive calibration results
        """
        logger.info(f"Validating calibration on {dataset_name} dataset ({len(uncertainty_scores)} samples)")

        if len(uncertainty_scores) != len(correctness_labels):
            raise ValueError("uncertainty_scores and correctness_labels must have same length")

        # Calculate calibration metrics
        results = calculate_uncertainty_metrics(
            uncertainty_scores,
            correctness_labels,
            confidence_scores,
            n_bins=self.n_bins
        )

        # Log results
        logger.info(f"Calibration validation results for {dataset_name}:")
        logger.info(f"  ECE: {results.ece:.4f}")
        logger.info(f"  Brier Score: {results.brier_score:.4f}")
        logger.info(f"  AUROC: {results.auroc:.4f}")
        logger.info(f"  Stability: {results.stability_score:.4f}")

        return results

    def cross_validate_calibration(
        self,
        uncertainty_scores: List[float],
        correctness_labels: List[int],
        n_folds: int = 5
    ) -> Dict[str, Union[float, List[CalibrationResults]]]:
        """
        Perform k-fold cross-validation of calibration performance.

        Implementation of the 5-fold cross-validation methodology from the paper.

        Args:
            uncertainty_scores: PBA uncertainty scores
            correctness_labels: Binary correctness indicators
            n_folds: Number of cross-validation folds

        Returns:
            Cross-validation results with mean/std statistics
        """
        if len(uncertainty_scores) != len(correctness_labels):
            raise ValueError("Mismatched array lengths")

        n_samples = len(uncertainty_scores)
        fold_size = n_samples // n_folds
        indices = np.random.permutation(n_samples)

        fold_results = []

        for fold in range(n_folds):
            # Create fold splits
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples

            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([
                indices[:start_idx],
                indices[end_idx:]
            ])

            # Extract fold data
            test_uncertainties = [uncertainty_scores[i] for i in test_indices]
            test_correctness = [correctness_labels[i] for i in test_indices]

            # Validate on test fold
            fold_result = self.validate_calibration_on_dataset(
                test_uncertainties,
                test_correctness,
                dataset_name=f"fold_{fold}"
            )

            fold_results.append(fold_result)

        # Calculate cross-fold statistics
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
            'n_folds': n_folds,
            'total_samples': n_samples
        }

    def compare_with_baselines(
        self,
        pba_uncertainties: List[float],
        baseline_confidences: Dict[str, List[float]],
        correctness_labels: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare PBA calibration with baseline uncertainty methods.

        Args:
            pba_uncertainties: PBA uncertainty scores
            baseline_confidences: Dict of {method_name: confidence_scores}
            correctness_labels: Binary correctness indicators

        Returns:
            Comparison results for each baseline method
        """
        # Calculate PBA results
        pba_confidences = [1.0 - u for u in pba_uncertainties]
        pba_results = self.validate_calibration_on_dataset(
            pba_uncertainties, correctness_labels, "PBA", pba_confidences
        )

        comparison_results = {}

        for method_name, baseline_confs in baseline_confidences.items():
            if len(baseline_confs) != len(correctness_labels):
                logger.warning(f"Skipping {method_name}: length mismatch")
                continue

            # Calculate baseline results
            baseline_uncertainties = [1.0 - c for c in baseline_confs]
            baseline_results = self.validate_calibration_on_dataset(
                baseline_uncertainties, correctness_labels, method_name, baseline_confs
            )

            # Compare metrics
            comparison_results[method_name] = {
                'pba_ece': pba_results.ece,
                'baseline_ece': baseline_results.ece,
                'ece_improvement': (baseline_results.ece - pba_results.ece) / baseline_results.ece * 100,
                'pba_brier': pba_results.brier_score,
                'baseline_brier': baseline_results.brier_score,
                'brier_improvement': (baseline_results.brier_score - pba_results.brier_score) / baseline_results.brier_score * 100,
                'pba_auroc': pba_results.auroc,
                'baseline_auroc': baseline_results.auroc,
                'auroc_improvement': (pba_results.auroc - baseline_results.auroc) / baseline_results.auroc * 100
            }

        return comparison_results

    def generate_calibration_report(
        self,
        calibration_results: CalibrationResults,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate human-readable calibration validation report.

        Args:
            calibration_results: Calibration analysis results
            dataset_info: Optional dataset metadata

        Returns:
            Formatted calibration report
        """
        report_lines = [
            "=== Calibration Validation Report ===",
            f"Generated: {datetime.now().isoformat()}",
            ""
        ]

        if dataset_info:
            report_lines.extend([
                "Dataset Information:",
                f"  Name: {dataset_info.get('name', 'Unknown')}",
                f"  Size: {dataset_info.get('size', 'Unknown')}",
                f"  Domain: {dataset_info.get('domain', 'Unknown')}",
                ""
            ])

        report_lines.extend([
            "Calibration Metrics:",
            f"  Expected Calibration Error (ECE): {calibration_results.ece:.4f}",
            f"  Brier Score: {calibration_results.brier_score:.4f}",
            f"  Area Under ROC Curve: {calibration_results.auroc:.4f}",
            f"  Stability Score: {calibration_results.stability_score:.4f}",
            ""
        ])

        # Reliability diagram information
        report_lines.extend([
            "Reliability Bins:",
            "  Bin  | Confidence | Accuracy | Count"
        ])

        for i, (conf, acc, count) in enumerate(calibration_results.reliability_bins):
            report_lines.append(f"  {i+1:2d}   | {conf:8.3f}   | {acc:8.3f} | {count:5d}")

        report_lines.extend([
            "",
            "Calibration Assessment:",
        ])

        # Provide interpretation
        if calibration_results.ece < 0.03:
            report_lines.append("  ✓ Excellent calibration (ECE < 3%)")
        elif calibration_results.ece < 0.05:
            report_lines.append("  ✓ Good calibration (ECE < 5%)")
        elif calibration_results.ece < 0.10:
            report_lines.append("  ⚠ Acceptable calibration (ECE < 10%)")
        else:
            report_lines.append("  ✗ Poor calibration (ECE ≥ 10%)")

        if calibration_results.stability_score > 0.95:
            report_lines.append("  ✓ High stability across validation folds")
        elif calibration_results.stability_score > 0.90:
            report_lines.append("  ✓ Good stability across validation folds")
        else:
            report_lines.append("  ⚠ Lower stability - consider additional validation")

        return "\n".join(report_lines)


class ContinuousCalibrationMonitor:
    """
    Continuous monitoring of uncertainty calibration in production environments.

    Implements statistical process control for uncertainty distribution drift
    detection and automated alerts for calibration degradation.
    """

    def __init__(
        self,
        baseline_results: CalibrationResults,
        alert_thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 1000,
        min_samples: int = 100
    ):
        """
        Initialize continuous calibration monitor.

        Args:
            baseline_results: Baseline calibration results for comparison
            alert_thresholds: Custom alert thresholds
            window_size: Size of sliding window for monitoring
            min_samples: Minimum samples before triggering alerts
        """
        self.baseline_results = baseline_results
        self.window_size = window_size
        self.min_samples = min_samples

        # Default alert thresholds (relative to baseline)
        self.alert_thresholds = alert_thresholds or {
            'ece_warning': 1.5,    # 50% increase in ECE
            'ece_critical': 2.0,   # 100% increase in ECE
            'brier_warning': 1.3,  # 30% increase in Brier Score
            'brier_critical': 1.5, # 50% increase in Brier Score
            'auroc_warning': 0.9,  # 10% decrease in AUROC
            'auroc_critical': 0.8  # 20% decrease in AUROC
        }

        # Monitoring state
        self.uncertainty_buffer: List[float] = []
        self.correctness_buffer: List[int] = []
        self.alerts: List[CalibrationDriftAlert] = []

        logger.info(f"Initialized continuous calibration monitor with window size {window_size}")

    def add_samples(
        self,
        uncertainty_scores: Union[List[float], float],
        correctness_labels: Union[List[int], int]
    ) -> List[CalibrationDriftAlert]:
        """
        Add new samples to the monitoring buffer and check for drift.

        Args:
            uncertainty_scores: New uncertainty scores
            correctness_labels: Corresponding correctness labels

        Returns:
            List of new alerts triggered by these samples
        """
        # Ensure lists
        if isinstance(uncertainty_scores, (int, float)):
            uncertainty_scores = [uncertainty_scores]
        if isinstance(correctness_labels, (int, float)):
            correctness_labels = [correctness_labels]

        # Add to buffers
        self.uncertainty_buffer.extend(uncertainty_scores)
        self.correctness_buffer.extend(correctness_labels)

        # Trim buffers to window size
        if len(self.uncertainty_buffer) > self.window_size:
            excess = len(self.uncertainty_buffer) - self.window_size
            self.uncertainty_buffer = self.uncertainty_buffer[excess:]
            self.correctness_buffer = self.correctness_buffer[excess:]

        # Check for drift if we have enough samples
        new_alerts = []
        if len(self.uncertainty_buffer) >= self.min_samples:
            new_alerts = self._check_calibration_drift()

        return new_alerts

    def _check_calibration_drift(self) -> List[CalibrationDriftAlert]:
        """Check current window for calibration drift"""
        # Calculate current calibration metrics
        current_results = calculate_uncertainty_metrics(
            self.uncertainty_buffer,
            self.correctness_buffer
        )

        new_alerts = []

        # Check ECE drift
        ece_ratio = current_results.ece / self.baseline_results.ece if self.baseline_results.ece > 0 else float('inf')
        if ece_ratio >= self.alert_thresholds['ece_critical']:
            alert = CalibrationDriftAlert(
                timestamp=datetime.now(),
                metric_name="Expected Calibration Error",
                current_value=current_results.ece,
                baseline_value=self.baseline_results.ece,
                drift_magnitude=ece_ratio,
                alert_level="CRITICAL",
                recommended_action="Immediate recalibration required"
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        elif ece_ratio >= self.alert_thresholds['ece_warning']:
            alert = CalibrationDriftAlert(
                timestamp=datetime.now(),
                metric_name="Expected Calibration Error",
                current_value=current_results.ece,
                baseline_value=self.baseline_results.ece,
                drift_magnitude=ece_ratio,
                alert_level="WARNING",
                recommended_action="Schedule recalibration validation"
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check Brier Score drift
        brier_ratio = current_results.brier_score / self.baseline_results.brier_score if self.baseline_results.brier_score > 0 else float('inf')
        if brier_ratio >= self.alert_thresholds['brier_critical']:
            alert = CalibrationDriftAlert(
                timestamp=datetime.now(),
                metric_name="Brier Score",
                current_value=current_results.brier_score,
                baseline_value=self.baseline_results.brier_score,
                drift_magnitude=brier_ratio,
                alert_level="CRITICAL",
                recommended_action="Immediate model evaluation required"
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check AUROC drift
        if self.baseline_results.auroc > 0:
            auroc_ratio = current_results.auroc / self.baseline_results.auroc
            if auroc_ratio <= self.alert_thresholds['auroc_critical']:
                alert = CalibrationDriftAlert(
                    timestamp=datetime.now(),
                    metric_name="AUROC",
                    current_value=current_results.auroc,
                    baseline_value=self.baseline_results.auroc,
                    drift_magnitude=auroc_ratio,
                    alert_level="CRITICAL",
                    recommended_action="Model discrimination ability degraded"
                )
                new_alerts.append(alert)
                self.alerts.append(alert)

        return new_alerts

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of current monitoring state"""
        if not self.uncertainty_buffer:
            return {"status": "No data collected"}

        current_results = calculate_uncertainty_metrics(
            self.uncertainty_buffer,
            self.correctness_buffer
        )

        return {
            "buffer_size": len(self.uncertainty_buffer),
            "window_utilization": len(self.uncertainty_buffer) / self.window_size,
            "current_ece": current_results.ece,
            "baseline_ece": self.baseline_results.ece,
            "ece_drift_ratio": current_results.ece / self.baseline_results.ece if self.baseline_results.ece > 0 else None,
            "current_brier": current_results.brier_score,
            "current_auroc": current_results.auroc,
            "total_alerts": len(self.alerts),
            "recent_alerts": len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)]),
            "last_update": datetime.now().isoformat()
        }

    def export_alert_log(self) -> List[Dict[str, Any]]:
        """Export alert history as structured data"""
        return [
            {
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "baseline_value": alert.baseline_value,
                "drift_magnitude": alert.drift_magnitude,
                "alert_level": alert.alert_level,
                "recommended_action": alert.recommended_action
            }
            for alert in self.alerts
        ]