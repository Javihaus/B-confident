"""
Comprehensive API for PBA Uncertainty Quantification

Main API interface providing all key functionality in a unified interface.
Designed to feel natural to existing transformers users while adding
uncertainty capabilities.

Key Functions:
- uncertainty_generate(): Drop-in replacement for model.generate()
- calibrate_model(): Tools for measuring uncertainty calibration
- compliance_report(): Automated regulatory documentation
- uncertainty_metrics(): Access to all computed uncertainty measures
"""

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

from .core.pba_algorithm import PBAUncertainty, PBAConfig
from .core.metrics import calculate_uncertainty_metrics, CalibrationResults
from .integration.transformers_wrapper import (
    UncertaintyTransformersModel,
    UncertaintyGenerationResult,
    uncertainty_generate as _uncertainty_generate
)
from .compliance.regulatory import ComplianceReporter, EUAIActReport, generate_eu_ai_act_report
from .compliance.calibration_tools import CalibrationValidator, ContinuousCalibrationMonitor

logger = logging.getLogger(__name__)


def uncertainty_generate(
    model: Union[PreTrainedModel, str],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    inputs: Union[str, torch.Tensor] = "",
    max_length: int = 50,
    num_return_sequences: int = 1,
    pba_config: Optional[PBAConfig] = None,
    **generation_kwargs
) -> UncertaintyGenerationResult:
    """
    Generate text with uncertainty quantification.

    Drop-in replacement for standard model.generate() with added uncertainty scoring.
    Compatible with all major transformer architectures.

    Args:
        model: Pre-trained model or model name/path
        tokenizer: Tokenizer (loaded automatically if model is string)
        inputs: Input text or token tensor
        max_length: Maximum generation length
        num_return_sequences: Number of sequences to return
        pba_config: PBA configuration (uses optimized defaults if None)
        **generation_kwargs: Additional generation arguments

    Returns:
        Generation result with uncertainty scores

    Examples:
        >>> # Using loaded model
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> result = uncertainty_generate(model, tokenizer, "Hello world")

        >>> # Using model name (auto-loads)
        >>> result = uncertainty_generate("gpt2", inputs="Hello world", max_length=20)

        >>> # Custom PBA configuration
        >>> config = PBAConfig(alpha=0.95, beta=0.3)
        >>> result = uncertainty_generate("gpt2", inputs="Hello", pba_config=config)
    """
    # Handle string model names
    if isinstance(model, str):
        model_name = model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name)

    # Use the underlying implementation
    return _uncertainty_generate(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pba_config=pba_config,
        **generation_kwargs
    )


def calibrate_model(
    model: Union[PreTrainedModel, str, UncertaintyTransformersModel],
    validation_texts: List[str],
    validation_labels: List[int],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    pba_config: Optional[PBAConfig] = None,
    cross_validation: bool = True,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Calibrate uncertainty estimates on validation data.

    Measures uncertainty calibration and provides tools for validation that
    uncertainty scores correlate with prediction accuracy.

    Args:
        model: Model to calibrate (can be model, model name, or UncertaintyTransformersModel)
        validation_texts: List of validation input texts
        validation_labels: List of binary correctness labels (0/1)
        tokenizer: Tokenizer (auto-loaded if model is string)
        pba_config: PBA configuration
        cross_validation: Whether to perform cross-validation analysis
        n_folds: Number of cross-validation folds

    Returns:
        Comprehensive calibration results

    Example:
        >>> texts = ["The capital of France is", "2 + 2 equals"]
        >>> labels = [1, 1]  # Both correct
        >>> results = calibrate_model("gpt2", texts, labels)
        >>> print(f"ECE: {results['calibration_results'].ece:.4f}")
    """
    # Prepare model
    if isinstance(model, str):
        model_name = model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModel.from_pretrained(model_name)
        uncertainty_model = UncertaintyTransformersModel(base_model, tokenizer, pba_config)
    elif isinstance(model, UncertaintyTransformersModel):
        uncertainty_model = model
    else:
        uncertainty_model = UncertaintyTransformersModel(model, tokenizer, pba_config)

    logger.info(f"Calibrating model on {len(validation_texts)} samples")

    # Collect uncertainty scores
    uncertainty_scores = []
    confidences = []

    for text in validation_texts:
        try:
            # Generate with short length for validation
            result = uncertainty_model.uncertainty_generate(
                inputs=text,
                max_length=min(len(text.split()) + 10, 50),
                num_return_sequences=1
            )

            if result.uncertainty_scores:
                uncertainty = result.uncertainty_scores[0]
                confidence = 1.0 - uncertainty
            else:
                uncertainty = 0.5  # Neutral
                confidence = 0.5

            uncertainty_scores.append(uncertainty)
            confidences.append(confidence)

        except Exception as e:
            logger.warning(f"Error processing text '{text[:30]}...': {e}")
            uncertainty_scores.append(0.5)
            confidences.append(0.5)

    # Calculate calibration metrics
    calibration_results = calculate_uncertainty_metrics(
        uncertainty_scores, validation_labels, confidences
    )

    results = {
        "calibration_results": calibration_results,
        "n_samples": len(validation_texts),
        "model_name": getattr(uncertainty_model, 'model_name', 'unknown')
    }

    # Cross-validation analysis
    if cross_validation and len(validation_texts) >= n_folds:
        validator = CalibrationValidator()
        cv_results = validator.cross_validate_calibration(
            uncertainty_scores, validation_labels, n_folds=n_folds
        )
        results["cross_validation"] = cv_results

    # Generate calibration report
    validator = CalibrationValidator()
    results["report"] = validator.generate_calibration_report(
        calibration_results,
        dataset_info={
            "name": "validation_data",
            "size": len(validation_texts),
            "domain": "text_generation"
        }
    )

    return results


def compliance_report(
    system_name: str,
    calibration_results: Union[CalibrationResults, Dict[str, Any]],
    system_version: str = "1.0",
    evaluation_dataset: Optional[str] = None,
    model_architecture: Optional[str] = None,
    output_format: str = "markdown",
    save_path: Optional[str] = None
) -> Union[EUAIActReport, str]:
    """
    Generate automated regulatory compliance documentation.

    Creates EU AI Act Article 15 compliant reports with concrete uncertainty
    measurement evidence and regulatory mapping.

    Args:
        system_name: Name of AI system for compliance reporting
        calibration_results: Calibration results or dict from calibrate_model()
        system_version: System version identifier
        evaluation_dataset: Name of evaluation dataset
        model_architecture: Model architecture description
        output_format: Output format ("json", "markdown", "html", "report")
        save_path: Optional path to save report

    Returns:
        Compliance report (EUAIActReport object or formatted string)

    Example:
        >>> results = calibrate_model("gpt2", texts, labels)
        >>> report = compliance_report(
        ...     "MyAISystem",
        ...     results["calibration_results"],
        ...     output_format="markdown"
        ... )
        >>> print(report)
    """
    # Extract calibration results if dict provided
    if isinstance(calibration_results, dict):
        if "calibration_results" in calibration_results:
            calibration_data = calibration_results["calibration_results"]
        else:
            # Try to construct from dict
            try:
                calibration_data = CalibrationResults(**calibration_results)
            except Exception as e:
                raise ValueError(f"Invalid calibration_results format: {e}")
    else:
        calibration_data = calibration_results

    # Generate compliance report
    report = generate_eu_ai_act_report(
        system_name=system_name,
        calibration_results=calibration_data,
        system_version=system_version,
        evaluation_dataset=evaluation_dataset,
        model_architecture=model_architecture
    )

    # Return appropriate format
    if output_format == "report":
        return report
    else:
        # Format and optionally save report
        reporter = ComplianceReporter(system_name, system_version)
        formatted_report = reporter.export_compliance_documentation(
            report, format=output_format, output_path=save_path
        )
        return formatted_report


def uncertainty_metrics(
    uncertainty_scores: List[float],
    correctness_labels: List[int],
    confidence_scores: Optional[List[float]] = None,
    include_cross_validation: bool = False,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Access to all computed uncertainty measures and validation tools.

    Comprehensive uncertainty quantification metrics for research and
    production validation.

    Args:
        uncertainty_scores: PBA uncertainty scores [0, 1]
        correctness_labels: Binary correctness indicators {0, 1}
        confidence_scores: Optional confidence scores (uses 1-uncertainty if None)
        include_cross_validation: Whether to perform cross-validation analysis
        n_folds: Number of folds for cross-validation

    Returns:
        Dictionary with all uncertainty metrics and analysis

    Example:
        >>> uncertainties = [0.1, 0.8, 0.3, 0.9, 0.2]
        >>> correctness = [1, 0, 1, 0, 1]
        >>> metrics = uncertainty_metrics(uncertainties, correctness)
        >>> print(f"ECE: {metrics['ece']:.4f}")
        >>> print(f"Brier Score: {metrics['brier_score']:.4f}")
    """
    # Basic metrics
    results = calculate_uncertainty_metrics(
        uncertainty_scores, correctness_labels, confidence_scores
    )

    output = {
        "ece": results.ece,
        "brier_score": results.brier_score,
        "auroc": results.auroc,
        "stability_score": results.stability_score,
        "reliability_bins": results.reliability_bins,
        "n_samples": len(uncertainty_scores)
    }

    # Cross-validation analysis
    if include_cross_validation and len(uncertainty_scores) >= n_folds:
        from .compliance.calibration_tools import cross_validation_analysis

        # Split data into folds for analysis
        import numpy as np
        n_samples = len(uncertainty_scores)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds

        uncertainty_folds = []
        correctness_folds = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
            fold_indices = indices[start_idx:end_idx]

            fold_uncertainties = [uncertainty_scores[j] for j in fold_indices]
            fold_correctness = [correctness_labels[j] for j in fold_indices]

            uncertainty_folds.append(fold_uncertainties)
            correctness_folds.append(fold_correctness)

        cv_results = cross_validation_analysis(uncertainty_folds, correctness_folds)
        output["cross_validation"] = cv_results

    # Statistical summary
    import numpy as np
    output["statistical_summary"] = {
        "uncertainty_mean": np.mean(uncertainty_scores),
        "uncertainty_std": np.std(uncertainty_scores),
        "uncertainty_min": np.min(uncertainty_scores),
        "uncertainty_max": np.max(uncertainty_scores),
        "accuracy": np.mean(correctness_labels),
        "correlation_uncertainty_accuracy": np.corrcoef(uncertainty_scores, correctness_labels)[0, 1]
    }

    return output


def create_continuous_monitor(
    baseline_results: Union[CalibrationResults, Dict[str, Any]],
    alert_thresholds: Optional[Dict[str, float]] = None,
    window_size: int = 1000
) -> ContinuousCalibrationMonitor:
    """
    Create continuous calibration monitor for production deployment.

    Enables real-time monitoring of uncertainty calibration with automated
    alerts for calibration drift detection.

    Args:
        baseline_results: Baseline calibration results for comparison
        alert_thresholds: Custom alert thresholds
        window_size: Size of sliding window for monitoring

    Returns:
        Configured calibration monitor

    Example:
        >>> baseline = calibrate_model("gpt2", texts, labels)
        >>> monitor = create_continuous_monitor(baseline["calibration_results"])
        >>> # In production loop:
        >>> alerts = monitor.add_samples(new_uncertainties, new_labels)
    """
    # Handle dict input
    if isinstance(baseline_results, dict):
        if "calibration_results" in baseline_results:
            baseline_data = baseline_results["calibration_results"]
        else:
            baseline_data = CalibrationResults(**baseline_results)
    else:
        baseline_data = baseline_results

    return ContinuousCalibrationMonitor(
        baseline_results=baseline_data,
        alert_thresholds=alert_thresholds,
        window_size=window_size
    )


def batch_uncertainty_analysis(
    model: Union[PreTrainedModel, str, UncertaintyTransformersModel],
    input_texts: List[str],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    pba_config: Optional[PBAConfig] = None,
    max_length: int = 50,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Efficient batch processing for uncertainty analysis.

    Optimized for processing large datasets with memory management
    and progress tracking.

    Args:
        model: Model for uncertainty analysis
        input_texts: List of input texts to process
        tokenizer: Tokenizer (auto-loaded if model is string)
        pba_config: PBA configuration
        max_length: Maximum generation length
        batch_size: Processing batch size

    Returns:
        Batch analysis results with aggregated statistics

    Example:
        >>> texts = ["Hello", "World", "Test"]
        >>> results = batch_uncertainty_analysis("gpt2", texts)
        >>> print(f"Average uncertainty: {results['avg_uncertainty']:.3f}")
    """
    # Prepare model
    if isinstance(model, str):
        model_name = model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModel.from_pretrained(model_name)
        uncertainty_model = UncertaintyTransformersModel(base_model, tokenizer, pba_config)
    elif isinstance(model, UncertaintyTransformersModel):
        uncertainty_model = model
    else:
        uncertainty_model = UncertaintyTransformersModel(model, tokenizer, pba_config)

    all_uncertainties = []
    all_texts = []
    processing_times = []

    logger.info(f"Processing {len(input_texts)} texts in batches of {batch_size}")

    # Process in batches
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        batch_uncertainties = []

        import time
        batch_start = time.time()

        for text in batch_texts:
            try:
                result = uncertainty_model.uncertainty_generate(
                    inputs=text,
                    max_length=max_length,
                    num_return_sequences=1
                )

                if result.uncertainty_scores:
                    uncertainty = result.uncertainty_scores[0]
                else:
                    uncertainty = 0.5

                batch_uncertainties.append(uncertainty)
                all_texts.append(text)

            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                batch_uncertainties.append(0.5)
                all_texts.append(text)

        batch_time = time.time() - batch_start
        processing_times.append(batch_time)
        all_uncertainties.extend(batch_uncertainties)

        # Progress logging
        progress = ((i + len(batch_texts)) / len(input_texts)) * 100
        logger.info(f"Progress: {progress:.1f}% ({i + len(batch_texts)}/{len(input_texts)})")

    # Aggregate statistics
    import numpy as np
    results = {
        "uncertainties": all_uncertainties,
        "texts": all_texts,
        "n_samples": len(all_uncertainties),
        "avg_uncertainty": np.mean(all_uncertainties),
        "std_uncertainty": np.std(all_uncertainties),
        "min_uncertainty": np.min(all_uncertainties),
        "max_uncertainty": np.max(all_uncertainties),
        "total_processing_time": sum(processing_times),
        "avg_time_per_sample": sum(processing_times) / len(input_texts),
        "batch_processing_times": processing_times
    }

    # High/low uncertainty samples
    sorted_indices = np.argsort(all_uncertainties)
    results["highest_uncertainty_samples"] = [
        {"text": all_texts[i], "uncertainty": all_uncertainties[i]}
        for i in sorted_indices[-5:]  # Top 5
    ]
    results["lowest_uncertainty_samples"] = [
        {"text": all_texts[i], "uncertainty": all_uncertainties[i]}
        for i in sorted_indices[:5]  # Bottom 5
    ]

    return results