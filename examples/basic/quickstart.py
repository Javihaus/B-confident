#!/usr/bin/env python3
"""
Basic B-Confident Quickstart Example

Demonstrates core functionality of the PBA uncertainty quantification library
with simple examples that can be run immediately after installation.
"""

import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

def basic_uncertainty_generation():
    """Basic uncertainty generation with default parameters"""
    print("=== Basic Uncertainty Generation ===")

    from b_confident import uncertainty_generate

    # Simple generation with uncertainty
    result = uncertainty_generate(
        model="gpt2",
        inputs="The weather today is",
        max_length=30
    )

    print(f"Input: 'The weather today is'")
    print(f"Generated: {result.generated_texts[0]}")
    print(f"Uncertainty Score: {result.uncertainty_scores[0]:.3f}")
    print(f"Processing Time: {result.metadata['processing_time_ms']:.1f}ms")
    print()


def custom_configuration():
    """Using custom PBA configuration parameters"""
    print("=== Custom PBA Configuration ===")

    from b_confident import uncertainty_generate, PBAConfig

    # Custom configuration with higher sensitivity
    config = PBAConfig(
        alpha=0.95,  # Higher probability mass threshold
        beta=0.7,    # Higher sensitivity to perplexity
        temperature=1.2
    )

    result = uncertainty_generate(
        model="gpt2",
        inputs="Machine learning is",
        max_length=25,
        pba_config=config
    )

    print(f"Input: 'Machine learning is'")
    print(f"Generated: {result.generated_texts[0]}")
    print(f"Uncertainty Score: {result.uncertainty_scores[0]:.3f}")
    print(f"Config - α: {config.alpha}, β: {config.beta}")
    print()


def multiple_sequences():
    """Generate multiple sequences with uncertainty scores"""
    print("=== Multiple Sequence Generation ===")

    from b_confident import uncertainty_generate

    result = uncertainty_generate(
        model="gpt2",
        inputs="The future of AI is",
        max_length=20,
        num_return_sequences=3,
        temperature=0.8
    )

    print(f"Input: 'The future of AI is'")
    for i, (text, uncertainty) in enumerate(zip(result.generated_texts, result.uncertainty_scores)):
        print(f"  Sequence {i+1}: {text}")
        print(f"  Uncertainty: {uncertainty:.3f}")
        print()


def calibration_example():
    """Basic calibration validation example"""
    print("=== Calibration Validation ===")

    from b_confident import calibrate_model

    # Simple validation data (normally you'd have much more)
    validation_texts = [
        "The capital of France is",
        "2 + 2 equals",
        "The sky is",
        "Water boils at",
        "The largest planet is"
    ]

    # Binary correctness labels (1 = correct completion expected, 0 = incorrect)
    validation_labels = [1, 1, 1, 1, 1]  # All should be answerable correctly

    print("Calibrating model on validation data...")
    results = calibrate_model(
        model="gpt2",
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        cross_validation=False  # Disable for small dataset
    )

    print(f"Expected Calibration Error: {results['calibration_results'].ece:.4f}")
    print(f"Brier Score: {results['calibration_results'].brier_score:.4f}")
    print(f"AUROC: {results['calibration_results'].auroc:.4f}")
    print(f"Stability Score: {results['calibration_results'].stability_score:.4f}")
    print()


def compliance_reporting():
    """Generate EU AI Act compliance report"""
    print("=== Compliance Reporting ===")

    from b_confident import calibrate_model, compliance_report

    # Quick calibration for compliance demo
    validation_texts = [
        "The capital of Italy is",
        "10 divided by 2 equals",
        "The Earth orbits the"
    ]
    validation_labels = [1, 1, 1]

    print("Generating calibration data for compliance report...")
    calib_results = calibrate_model(
        model="gpt2",
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        cross_validation=False
    )

    # Generate compliance report
    report = compliance_report(
        system_name="DemoAISystem",
        calibration_results=calib_results["calibration_results"],
        system_version="1.0",
        evaluation_dataset="demo_validation",
        output_format="markdown"
    )

    print("EU AI Act Article 15 Compliance Report Generated:")
    print("-" * 50)
    print(report[:500] + "..." if len(report) > 500 else report)
    print()


def uncertainty_metrics_example():
    """Comprehensive uncertainty metrics analysis"""
    print("=== Uncertainty Metrics Analysis ===")

    from b_confident import uncertainty_metrics
    import random

    # Simulate some uncertainty scores and correctness labels
    # In practice, these would come from your model evaluation
    random.seed(42)  # For reproducible results

    n_samples = 50
    uncertainty_scores = [random.random() for _ in range(n_samples)]
    correctness_labels = [1 if random.random() > u else 0 for u in uncertainty_scores]  # Negative correlation

    metrics = uncertainty_metrics(
        uncertainty_scores=uncertainty_scores,
        correctness_labels=correctness_labels,
        include_cross_validation=True,
        n_folds=5
    )

    print(f"Sample Size: {metrics['n_samples']}")
    print(f"Expected Calibration Error: {metrics['ece']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Stability Score: {metrics['stability_score']:.4f}")

    print("\nStatistical Summary:")
    stats = metrics['statistical_summary']
    print(f"  Mean Uncertainty: {stats['uncertainty_mean']:.3f}")
    print(f"  Std Uncertainty: {stats['uncertainty_std']:.3f}")
    print(f"  Accuracy: {stats['accuracy']:.3f}")
    print(f"  Uncertainty-Accuracy Correlation: {stats['correlation_uncertainty_accuracy']:.3f}")

    if 'cross_validation' in metrics:
        cv = metrics['cross_validation']
        print(f"\nCross-Validation Results ({cv['n_folds']} folds):")
        print(f"  ECE: {cv['ece_mean']:.4f} ± {cv['ece_std']:.4f}")
        print(f"  Brier: {cv['brier_mean']:.4f} ± {cv['brier_std']:.4f}")
    print()


def batch_processing_example():
    """Efficient batch processing of multiple texts"""
    print("=== Batch Processing ===")

    from b_confident import batch_uncertainty_analysis

    # Sample texts for batch processing
    input_texts = [
        "The weather is",
        "Machine learning",
        "Python programming",
        "Climate change",
        "Space exploration",
        "Artificial intelligence",
        "Renewable energy",
        "Ocean conservation"
    ]

    print(f"Processing {len(input_texts)} texts in batch...")
    results = batch_uncertainty_analysis(
        model="gpt2",
        input_texts=input_texts,
        max_length=15,
        batch_size=4
    )

    print(f"Average Uncertainty: {results['avg_uncertainty']:.3f}")
    print(f"Std Uncertainty: {results['std_uncertainty']:.3f}")
    print(f"Processing Time: {results['total_processing_time']:.1f}s")
    print(f"Time per Sample: {results['avg_time_per_sample']:.2f}s")

    print("\nHighest Uncertainty Samples:")
    for sample in results['highest_uncertainty_samples'][-3:]:  # Top 3
        print(f"  '{sample['text']}' -> {sample['uncertainty']:.3f}")
    print()


def main():
    """Run all examples"""
    print("🎯 B-Confident Quickstart Examples")
    print("=" * 50)
    print()

    try:
        # Run each example
        basic_uncertainty_generation()
        custom_configuration()
        multiple_sequences()
        calibration_example()
        compliance_reporting()
        uncertainty_metrics_example()
        batch_processing_example()

        print("✅ All examples completed successfully!")
        print("\nNext Steps:")
        print("- Explore the full API in b_confident.api")
        print("- Check out serving examples in examples/serving/")
        print("- Review compliance documentation in examples/compliance/")
        print("- Read the full documentation at docs.b-confident.com")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nMake sure you have b-confident installed:")
        print("pip install b-confident")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check your installation and try again.")


if __name__ == "__main__":
    main()