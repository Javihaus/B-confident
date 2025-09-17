#!/usr/bin/env python3
"""
Command Line Interface for b-confident

Provides CLI access to all major functionality including calibration,
compliance reporting, and serving.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="B-Confident: Perplexity-Based Adjacency for Uncertainty Quantification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text with uncertainty
  b-confident generate "The weather is" --model gpt2 --max-length 30

  # Calibrate model on validation data
  b-confident calibrate --model gpt2 --validation-file validation.json

  # Generate compliance report
  b-confident compliance --system-name MyAI --calibration-file results.json

  # Start FastAPI server
  b-confident serve --model gpt2 --port 8000

  # Show model information
  b-confident info --model gpt2
        """
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--config", type=str,
        help="Path to PBA configuration file"
    )

    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda, cpu, auto)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text with uncertainty quantification"
    )
    generate_parser.add_argument(
        "text", type=str,
        help="Input text to generate from"
    )
    generate_parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name or path"
    )
    generate_parser.add_argument(
        "--max-length", type=int, default=50,
        help="Maximum generation length"
    )
    generate_parser.add_argument(
        "--num-sequences", type=int, default=1,
        help="Number of sequences to generate"
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature"
    )
    generate_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path (JSON format)"
    )

    # Calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Calibrate model uncertainty on validation data"
    )
    calibrate_parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name or path"
    )
    calibrate_parser.add_argument(
        "--validation-file", "-f", type=str, required=True,
        help="JSON file with validation data: [{'text': '...', 'correct': 0/1}, ...]"
    )
    calibrate_parser.add_argument(
        "--cross-validation", action="store_true", default=True,
        help="Perform cross-validation analysis"
    )
    calibrate_parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    calibrate_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path for calibration results"
    )

    # Compliance command
    compliance_parser = subparsers.add_parser(
        "compliance", help="Generate regulatory compliance report"
    )
    compliance_parser.add_argument(
        "--system-name", type=str, required=True,
        help="AI system name for compliance report"
    )
    compliance_parser.add_argument(
        "--calibration-file", type=str, required=True,
        help="JSON file with calibration results from 'calibrate' command"
    )
    compliance_parser.add_argument(
        "--system-version", type=str, default="1.0",
        help="System version"
    )
    compliance_parser.add_argument(
        "--dataset-name", type=str,
        help="Evaluation dataset name"
    )
    compliance_parser.add_argument(
        "--model-architecture", type=str,
        help="Model architecture description"
    )
    compliance_parser.add_argument(
        "--format", type=str, choices=["markdown", "json", "html"], default="markdown",
        help="Output format"
    )
    compliance_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path"
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start FastAPI server for uncertainty quantification"
    )
    serve_parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name or path"
    )
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind to"
    )
    serve_parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of workers"
    )
    serve_parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show model and system information"
    )
    info_parser.add_argument(
        "--model", "-m", type=str,
        help="Model name or path to analyze"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate uncertainty quantification setup"
    )
    validate_parser.add_argument(
        "--model", "-m", type=str,
        help="Model to validate (optional)"
    )

    return parser


def load_pba_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load PBA configuration from file"""
    if not config_path:
        return None

    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError("Config file must be JSON or YAML")
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None


def handle_generate(args) -> int:
    """Handle the generate command"""
    try:
        from b_confident import uncertainty_generate, PBAConfig

        # Load PBA config
        pba_config_dict = load_pba_config(args.config)
        pba_config = PBAConfig(**pba_config_dict) if pba_config_dict else None

        logger.info(f"Generating text with model: {args.model}")
        logger.info(f"Input: '{args.text}'")

        # Generate with uncertainty
        result = uncertainty_generate(
            model=args.model,
            inputs=args.text,
            max_length=args.max_length,
            num_return_sequences=args.num_sequences,
            temperature=args.temperature,
            pba_config=pba_config
        )

        # Prepare output
        output = {
            "input": args.text,
            "generated_texts": result.generated_texts,
            "uncertainty_scores": result.uncertainty_scores,
            "token_uncertainties": result.token_uncertainties,
            "metadata": result.metadata
        }

        # Display results
        print("\n=== Generation Results ===")
        for i, (text, uncertainty) in enumerate(zip(result.generated_texts, result.uncertainty_scores)):
            print(f"\nSequence {i+1}:")
            print(f"  Text: {text}")
            print(f"  Uncertainty: {uncertainty:.4f}")

        print(f"\nProcessing time: {result.metadata.get('processing_time_ms', 0):.1f}ms")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


def handle_calibrate(args) -> int:
    """Handle the calibrate command"""
    try:
        from b_confident import calibrate_model, PBAConfig
        import json

        # Load validation data
        logger.info(f"Loading validation data from {args.validation_file}")
        with open(args.validation_file, 'r') as f:
            validation_data = json.load(f)

        # Extract texts and labels
        validation_texts = [item['text'] for item in validation_data]
        validation_labels = [item['correct'] for item in validation_data]

        logger.info(f"Loaded {len(validation_texts)} validation samples")

        # Load PBA config
        pba_config_dict = load_pba_config(args.config)
        pba_config = PBAConfig(**pba_config_dict) if pba_config_dict else None

        logger.info(f"Calibrating model: {args.model}")

        # Perform calibration
        results = calibrate_model(
            model=args.model,
            validation_texts=validation_texts,
            validation_labels=validation_labels,
            pba_config=pba_config,
            cross_validation=args.cross_validation,
            n_folds=args.n_folds
        )

        # Display results
        calibration = results["calibration_results"]
        print("\n=== Calibration Results ===")
        print(f"Expected Calibration Error (ECE): {calibration.ece:.4f}")
        print(f"Brier Score: {calibration.brier_score:.4f}")
        print(f"AUROC: {calibration.auroc:.4f}")
        print(f"Stability Score: {calibration.stability_score:.4f}")

        if "cross_validation" in results:
            cv = results["cross_validation"]
            print(f"\nCross-Validation ({cv['n_folds']} folds):")
            print(f"ECE: {cv['ece_mean']:.4f} ± {cv['ece_std']:.4f}")
            print(f"Brier: {cv['brier_mean']:.4f} ± {cv['brier_std']:.4f}")

        # Assessment
        if calibration.ece < 0.03:
            print("\n[EXCELLENT] Calibration quality: ECE < 3%")
        elif calibration.ece < 0.05:
            print("\n[GOOD] Calibration quality: ECE < 5%")
        elif calibration.ece < 0.10:
            print("\n[ACCEPTABLE] Calibration quality: ECE < 10%")
        else:
            print("\n[POOR] Calibration quality: ECE >= 10% - Consider recalibration")

        # Save results
        output_file = args.output or f"calibration_{args.model.replace('/', '_')}.json"

        # Convert results to JSON-serializable format
        output = {
            "model": args.model,
            "n_samples": results["n_samples"],
            "calibration_results": {
                "ece": calibration.ece,
                "brier_score": calibration.brier_score,
                "auroc": calibration.auroc,
                "stability_score": calibration.stability_score,
                "reliability_bins": calibration.reliability_bins
            }
        }

        if "cross_validation" in results:
            output["cross_validation"] = results["cross_validation"]

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Calibration results saved to {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return 1


def handle_compliance(args) -> int:
    """Handle the compliance command"""
    try:
        from b_confident import compliance_report
        from b_confident.core.metrics import CalibrationResults
        import json

        # Load calibration results
        logger.info(f"Loading calibration results from {args.calibration_file}")
        with open(args.calibration_file, 'r') as f:
            calib_data = json.load(f)

        # Reconstruct CalibrationResults
        calib_results_dict = calib_data["calibration_results"]
        calibration_results = CalibrationResults(
            ece=calib_results_dict["ece"],
            brier_score=calib_results_dict["brier_score"],
            auroc=calib_results_dict["auroc"],
            stability_score=calib_results_dict["stability_score"],
            reliability_bins=calib_results_dict["reliability_bins"],
            statistical_significance=None
        )

        logger.info(f"Generating compliance report for: {args.system_name}")

        # Generate compliance report
        report = compliance_report(
            system_name=args.system_name,
            calibration_results=calibration_results,
            system_version=args.system_version,
            evaluation_dataset=args.dataset_name,
            model_architecture=args.model_architecture,
            output_format=args.format
        )

        # Display summary
        if args.format == "report":
            print("\n=== Compliance Report Generated ===")
            print(f"Report ID: {report.report_id}")
            print(f"Status: {report.compliance_status}")
            print(f"System: {report.system_name} v{report.system_version}")
        else:
            print("\n=== Compliance Report ===")
            if args.format == "markdown":
                print(report[:500] + "..." if len(report) > 500 else report)

        # Save to file
        if args.output:
            output_file = args.output
        else:
            ext = "json" if args.format == "json" else "md" if args.format == "markdown" else "html"
            output_file = f"compliance_report_{args.system_name.lower().replace(' ', '_')}.{ext}"

        if args.format == "report":
            # Save structured report as JSON
            with open(output_file.replace(".md", ".json"), 'w') as f:
                f.write(report.to_json(indent=2))
        else:
            with open(output_file, 'w') as f:
                f.write(report)

        logger.info(f"Compliance report saved to {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Compliance report generation failed: {e}")
        return 1


def handle_serve(args) -> int:
    """Handle the serve command"""
    try:
        from b_confident.serving import create_uncertainty_api
        import uvicorn

        logger.info(f"Starting FastAPI server for model: {args.model}")

        # Load PBA config
        pba_config_dict = load_pba_config(args.config)

        # Create app
        app = create_uncertainty_api(
            model_name_or_path=args.model,
            pba_config=pba_config_dict
        )

        # Run server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload
        )

        return 0

    except ImportError:
        logger.error("FastAPI serving not available. Install with: pip install b-confident[serving]")
        return 1
    except Exception as e:
        logger.error(f"Server failed: {e}")
        return 1


def handle_info(args) -> int:
    """Handle the info command"""
    try:
        print("=== B-Confident Information ===")
        print(f"Version: 0.1.0")
        print(f"Methodology: Perplexity-Based Adjacency (PBA)")

        # System info
        print(f"\nSystem:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  PyTorch: {torch.__version__ if torch else 'Not available'}")
        print(f"  CUDA Available: {torch.cuda.is_available() if torch else False}")

        if args.model:
            from b_confident.integration.transformers_wrapper import UncertaintyTransformersModel
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading model info for: {args.model}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                model = AutoModel.from_pretrained(args.model)
                uncertainty_model = UncertaintyTransformersModel(model, tokenizer)

                info = uncertainty_model.get_model_info()

                print(f"\nModel Information:")
                print(f"  Name: {info['model_name']}")
                print(f"  Architecture: {info['architecture']}")
                print(f"  Parameters: {info['parameters']:,}")
                print(f"  Device: {info['device']}")
                print(f"  Has Tokenizer: {info['has_tokenizer']}")
                print(f"  Supports Generation: {info['supports_generation']}")

            except Exception as e:
                logger.error(f"Failed to load model info: {e}")

        # PBA config info
        if args.config:
            config_data = load_pba_config(args.config)
            if config_data:
                print(f"\nPBA Configuration:")
                for key, value in config_data.items():
                    print(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error(f"Info command failed: {e}")
        return 1


def handle_validate(args) -> int:
    """Handle the validate command"""
    try:
        print("=== Validating B-Confident Setup ===")

        # Check imports
        print("\nChecking dependencies...")
        try:
            import torch
            print(f"[OK] PyTorch: {torch.__version__}")
        except ImportError:
            print("[ERROR] PyTorch not available")
            return 1

        try:
            import transformers
            print(f"[OK] Transformers: {transformers.__version__}")
        except ImportError:
            print("[ERROR] Transformers not available")
            return 1

        try:
            from b_confident import uncertainty_generate
            print("[OK] B-Confident core functionality")
        except ImportError as e:
            print(f"[ERROR] B-Confident import failed: {e}")
            return 1

        # Check optional dependencies
        try:
            import fastapi
            print(f"[OK] FastAPI: {fastapi.__version__} (serving support)")
        except ImportError:
            print("[WARNING] FastAPI not available (install with b-confident[serving])")

        try:
            import ray
            print(f"[OK] Ray: {ray.__version__} (distributed serving support)")
        except ImportError:
            print("[WARNING] Ray not available (install with b-confident[all])")

        # Test basic functionality
        if args.model:
            print(f"\nTesting with model: {args.model}")
            try:
                result = uncertainty_generate(
                    model=args.model,
                    inputs="Test input",
                    max_length=10
                )
                print(f"[OK] Basic generation successful")
                print(f"   Generated: {result.generated_texts[0]}")
                print(f"   Uncertainty: {result.uncertainty_scores[0]:.4f}")

            except Exception as e:
                print(f"[ERROR] Generation test failed: {e}")
                return 1

        print("\n[OK] Validation complete - setup is working correctly!")
        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle commands
    if args.command == "generate":
        return handle_generate(args)
    elif args.command == "calibrate":
        return handle_calibrate(args)
    elif args.command == "compliance":
        return handle_compliance(args)
    elif args.command == "serve":
        return handle_serve(args)
    elif args.command == "info":
        return handle_info(args)
    elif args.command == "validate":
        return handle_validate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())