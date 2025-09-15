"""
TorchServe Custom Handler for PBA Uncertainty Quantification

Provides custom TorchServe handler that adds uncertainty quantification
to existing model serving infrastructure with minimal changes.
"""

import json
import logging
import torch
from typing import Dict, List, Any, Optional
import os
import time

try:
    from ts.torch_handler.base_handler import BaseHandler
    TORCHSERVE_AVAILABLE = True
except ImportError:
    TORCHSERVE_AVAILABLE = False
    BaseHandler = object

from transformers import AutoModel, AutoTokenizer

from ..core.pba_algorithm import PBAUncertainty, PBAConfig
from ..integration.transformers_wrapper import UncertaintyTransformersModel

logger = logging.getLogger(__name__)


class PBAUncertaintyHandler(BaseHandler):
    """
    TorchServe handler for PBA uncertainty quantification.

    Integrates seamlessly with TorchServe infrastructure while adding
    uncertainty scoring capabilities. Compatible with standard TorchServe
    deployment patterns.

    Configuration via model archive properties file:
        pba_alpha=0.9
        pba_beta=0.5
        max_length=100
        temperature=1.0
    """

    def __init__(self):
        super().__init__()
        self.uncertainty_model = None
        self.tokenizer = None
        self.pba_config = None
        self.initialized = False

        # Performance tracking
        self.request_count = 0
        self.total_inference_time = 0.0
        self.total_uncertainty_time = 0.0

        if not TORCHSERVE_AVAILABLE:
            logger.warning("TorchServe not available. Handler may not work in TorchServe environment.")

    def initialize(self, context):
        """
        Initialize handler with model and configuration.

        Args:
            context: TorchServe context with model artifacts and properties
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        logger.info("Initializing PBA Uncertainty Handler")
        logger.info(f"Model directory: {model_dir}")

        try:
            # Load configuration from properties
            self._load_configuration(context)

            # Load model and tokenizer
            self._load_model(context)

            # Initialize uncertainty model
            self.uncertainty_model = UncertaintyTransformersModel(
                self.model, self.tokenizer, self.pba_config
            )

            self.initialized = True
            logger.info("PBA Uncertainty Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize handler: {e}")
            raise

    def _load_configuration(self, context):
        """Load PBA configuration from properties"""
        properties = context.model_yaml_config.get("handler", {})

        # Extract PBA parameters
        alpha = float(properties.get("pba_alpha", 0.9))
        beta = float(properties.get("pba_beta", 0.5))
        temperature = float(properties.get("temperature", 1.0))

        self.pba_config = PBAConfig(
            alpha=alpha,
            beta=beta,
            temperature=temperature,
            device=context.system_properties.get("gpu_id", "cpu")
        )

        # Store generation parameters
        self.max_length = int(properties.get("max_length", 50))
        self.num_return_sequences = int(properties.get("num_return_sequences", 1))

        logger.info(f"PBA Configuration: α={alpha}, β={beta}, temp={temperature}")

    def _load_model(self, context):
        """Load transformer model and tokenizer"""
        model_dir = context.system_properties.get("model_dir")

        # Check if we have model files in the directory
        if os.path.exists(os.path.join(model_dir, "config.json")):
            # Local model files
            model_path = model_dir
        else:
            # Model name from properties
            model_path = context.model_yaml_config.get("handler", {}).get("model_name", "gpt2")

        logger.info(f"Loading model from: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModel.from_pretrained(model_path)

        # Move to appropriate device
        device = context.system_properties.get("gpu_id")
        if device is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        logger.info(f"Model loaded on device: {self.device}")

    def preprocess(self, data):
        """
        Preprocess input data for uncertainty generation.

        Args:
            data: List of input requests

        Returns:
            Processed inputs ready for inference
        """
        if not self.initialized:
            raise RuntimeError("Handler not initialized")

        inputs = []

        for request in data:
            input_data = request.get("body") or request.get("data")

            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode('utf-8')

            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    # Treat as plain text
                    input_data = {"text": input_data}

            # Extract parameters
            text = input_data.get("text", input_data.get("prompt", ""))
            max_length = input_data.get("max_length", self.max_length)
            temperature = input_data.get("temperature", 1.0)
            num_sequences = input_data.get("num_return_sequences", self.num_return_sequences)

            inputs.append({
                "text": text,
                "max_length": max_length,
                "temperature": temperature,
                "num_return_sequences": num_sequences,
                "request_id": request.get("request_id", f"req_{len(inputs)}")
            })

        return inputs

    def inference(self, data):
        """
        Run inference with uncertainty quantification.

        Args:
            data: Preprocessed input data

        Returns:
            Inference results with uncertainty scores
        """
        if not self.initialized:
            raise RuntimeError("Handler not initialized")

        results = []

        for input_item in data:
            start_time = time.time()

            try:
                # Generate with uncertainty
                uncertainty_start = time.time()

                result = self.uncertainty_model.uncertainty_generate(
                    inputs=input_item["text"],
                    max_length=input_item["max_length"],
                    temperature=input_item["temperature"],
                    num_return_sequences=input_item["num_return_sequences"]
                )

                uncertainty_time = time.time() - uncertainty_start

                # Decode generated sequences
                generated_texts = [
                    self.tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in result.sequences
                ]

                inference_time = time.time() - start_time

                # Track performance
                self.request_count += 1
                self.total_inference_time += inference_time
                self.total_uncertainty_time += uncertainty_time

                results.append({
                    "request_id": input_item["request_id"],
                    "generated_texts": generated_texts,
                    "uncertainty_scores": result.uncertainty_scores,
                    "token_uncertainties": result.token_uncertainties,
                    "metadata": {
                        **result.metadata,
                        "inference_time_ms": inference_time * 1000,
                        "uncertainty_overhead_ms": uncertainty_time * 1000
                    }
                })

            except Exception as e:
                logger.error(f"Inference failed for request {input_item['request_id']}: {e}")
                results.append({
                    "request_id": input_item["request_id"],
                    "error": str(e),
                    "generated_texts": [],
                    "uncertainty_scores": [],
                    "token_uncertainties": []
                })

        return results

    def postprocess(self, data):
        """
        Postprocess inference results for response formatting.

        Args:
            data: Raw inference results

        Returns:
            Formatted response data
        """
        formatted_results = []

        for result in data:
            if "error" in result:
                formatted_results.append({
                    "request_id": result["request_id"],
                    "status": "error",
                    "error_message": result["error"]
                })
            else:
                # Calculate summary statistics
                avg_uncertainty = (
                    sum(result["uncertainty_scores"]) / len(result["uncertainty_scores"])
                    if result["uncertainty_scores"] else 0.0
                )

                max_uncertainty = max(result["uncertainty_scores"]) if result["uncertainty_scores"] else 0.0

                formatted_results.append({
                    "request_id": result["request_id"],
                    "status": "success",
                    "generated_texts": result["generated_texts"],
                    "uncertainty_summary": {
                        "average_uncertainty": avg_uncertainty,
                        "maximum_uncertainty": max_uncertainty,
                        "sequence_uncertainties": result["uncertainty_scores"]
                    },
                    "detailed_uncertainties": result["token_uncertainties"],
                    "performance": {
                        "inference_time_ms": result["metadata"]["inference_time_ms"],
                        "uncertainty_overhead_ms": result["metadata"]["uncertainty_overhead_ms"]
                    }
                })

        return formatted_results

    def handle(self, data, context):
        """
        Main handler entry point for TorchServe.

        Args:
            data: Input request data
            context: TorchServe context

        Returns:
            Response data with uncertainty quantification
        """
        try:
            # Preprocess
            preprocessed_data = self.preprocess(data)

            # Inference
            inference_results = self.inference(preprocessed_data)

            # Postprocess
            response = self.postprocess(inference_results)

            return response

        except Exception as e:
            logger.error(f"Handler error: {e}")
            return [{"error": str(e), "status": "error"}]

    def get_insights(self):
        """
        Get handler performance insights for monitoring.

        Returns:
            Performance metrics and handler statistics
        """
        if self.request_count == 0:
            return {"requests_processed": 0}

        avg_inference_time = self.total_inference_time / self.request_count
        avg_uncertainty_time = self.total_uncertainty_time / self.request_count
        uncertainty_overhead = (avg_uncertainty_time / avg_inference_time) * 100

        return {
            "requests_processed": self.request_count,
            "average_inference_time_ms": avg_inference_time * 1000,
            "average_uncertainty_time_ms": avg_uncertainty_time * 1000,
            "uncertainty_overhead_percent": uncertainty_overhead,
            "handler_initialized": self.initialized,
            "model_device": str(self.device) if hasattr(self, 'device') else 'unknown',
            "pba_config": self.pba_config.__dict__ if self.pba_config else None
        }


def create_torchserve_config(
    model_name: str,
    pba_alpha: float = 0.9,
    pba_beta: float = 0.5,
    max_length: int = 50,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Create TorchServe configuration for PBA uncertainty handler.

    Args:
        model_name: Hugging Face model name
        pba_alpha: PBA alpha parameter
        pba_beta: PBA beta parameter
        max_length: Maximum generation length
        temperature: Sampling temperature

    Returns:
        TorchServe configuration dictionary

    Example:
        >>> config = create_torchserve_config("gpt2", pba_alpha=0.9)
        >>> # Save as model-config.yaml for TorchServe deployment
    """
    return {
        "modelName": f"pba-{model_name}",
        "version": "1.0",
        "handler": {
            "class_name": "b_confident.serving.torchserve_handler.PBAUncertaintyHandler",
            "model_name": model_name,
            "pba_alpha": pba_alpha,
            "pba_beta": pba_beta,
            "max_length": max_length,
            "temperature": temperature,
            "num_return_sequences": 1
        },
        "runtime": "python",
        "batchSize": 1,
        "maxBatchDelay": 100,
        "responseTimeout": 120,
        "deviceType": "gpu",
        "parallelLevel": 1
    }