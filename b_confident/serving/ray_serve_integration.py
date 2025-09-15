"""
Ray Serve Integration for Distributed Uncertainty Quantification

Provides Ray Serve deployment patterns for scaling PBA uncertainty
quantification across distributed inference infrastructure.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    serve = None

import torch
from transformers import AutoModel, AutoTokenizer

from ..core.pba_algorithm import PBAUncertainty, PBAConfig
from ..integration.transformers_wrapper import UncertaintyTransformersModel
from ..compliance.calibration_tools import ContinuousCalibrationMonitor

logger = logging.getLogger(__name__)


if RAY_AVAILABLE:
    @serve.deployment(
        name="pba-uncertainty",
        num_replicas=1,
        max_concurrent_queries=10,
        ray_actor_options={"num_gpus": 1}
    )
    class PBAServeDeployment:
        """
        Ray Serve deployment for PBA uncertainty quantification.

        Provides distributed uncertainty quantification with load balancing,
        autoscaling, and consistency across multiple inference nodes.
        """

        def __init__(
            self,
            model_name: str,
            pba_config: Optional[Dict[str, Any]] = None,
            enable_monitoring: bool = True
        ):
            """
            Initialize Ray Serve deployment.

            Args:
                model_name: Hugging Face model identifier
                pba_config: PBA configuration parameters
                enable_monitoring: Enable calibration monitoring
            """
            self.model_name = model_name
            self.enable_monitoring = enable_monitoring

            # Convert dict config to PBAConfig
            if pba_config:
                self.pba_config = PBAConfig(**pba_config)
            else:
                self.pba_config = PBAConfig()

            # Initialize components
            self.model = None
            self.tokenizer = None
            self.uncertainty_model = None
            self.monitor = None

            # Performance tracking
            self.stats = {
                "requests_served": 0,
                "total_processing_time": 0.0,
                "average_uncertainty": 0.0,
                "deployment_start_time": time.time()
            }

            self._initialize_model()

            logger.info(f"Initialized PBA Ray Serve deployment for {model_name}")

        def _initialize_model(self):
            """Initialize model and supporting components"""
            try:
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModel.from_pretrained(self.model_name)

                # Create uncertainty model
                self.uncertainty_model = UncertaintyTransformersModel(
                    self.model, self.tokenizer, self.pba_config
                )

                # Initialize monitoring if enabled
                if self.enable_monitoring:
                    # Use dummy baseline for initialization
                    from ..core.metrics import CalibrationResults
                    baseline_results = CalibrationResults(
                        ece=0.03, brier_score=0.15, auroc=0.75,
                        reliability_bins=[], statistical_significance=None,
                        stability_score=0.95
                    )
                    self.monitor = ContinuousCalibrationMonitor(baseline_results)

                logger.info(f"Model {self.model_name} loaded successfully")

            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise

        async def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
            """
            Generate text with uncertainty quantification.

            Args:
                request: Generation request parameters

            Returns:
                Generation result with uncertainty scores
            """
            start_time = time.time()

            try:
                # Extract parameters
                text = request.get("text", "")
                max_length = request.get("max_length", 50)
                num_return_sequences = request.get("num_return_sequences", 1)
                temperature = request.get("temperature", 1.0)

                if not text:
                    return {"error": "No input text provided"}

                # Generate with uncertainty
                result = self.uncertainty_model.uncertainty_generate(
                    inputs=text,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature
                )

                # Decode sequences
                generated_texts = [
                    self.tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in result.sequences
                ]

                # Update statistics
                processing_time = time.time() - start_time
                self.stats["requests_served"] += 1
                self.stats["total_processing_time"] += processing_time

                if result.uncertainty_scores:
                    avg_uncertainty = sum(result.uncertainty_scores) / len(result.uncertainty_scores)
                    self.stats["average_uncertainty"] = (
                        (self.stats["average_uncertainty"] * (self.stats["requests_served"] - 1) + avg_uncertainty) /
                        self.stats["requests_served"]
                    )

                return {
                    "generated_texts": generated_texts,
                    "uncertainty_scores": result.uncertainty_scores,
                    "token_uncertainties": result.token_uncertainties,
                    "metadata": {
                        **result.metadata,
                        "processing_time_ms": processing_time * 1000,
                        "replica_id": ray.get_runtime_context().get_actor_id()
                    }
                }

            except Exception as e:
                logger.error(f"Generation error: {e}")
                return {"error": str(e)}

        async def calibrate(self, request: Dict[str, Any]) -> Dict[str, Any]:
            """
            Validate calibration on provided data.

            Args:
                request: Calibration validation request

            Returns:
                Calibration results
            """
            try:
                uncertainty_scores = request.get("uncertainty_scores", [])
                correctness_labels = request.get("correctness_labels", [])

                if len(uncertainty_scores) != len(correctness_labels):
                    return {"error": "Mismatched array lengths"}

                # Calculate calibration metrics
                from ..core.metrics import calculate_uncertainty_metrics
                results = calculate_uncertainty_metrics(
                    uncertainty_scores, correctness_labels
                )

                # Update monitor if enabled
                if self.monitor:
                    alerts = self.monitor.add_samples(uncertainty_scores, correctness_labels)
                    if alerts:
                        logger.warning(f"Calibration alerts triggered: {len(alerts)}")

                return {
                    "ece": results.ece,
                    "brier_score": results.brier_score,
                    "auroc": results.auroc,
                    "stability_score": results.stability_score,
                    "replica_id": ray.get_runtime_context().get_actor_id()
                }

            except Exception as e:
                logger.error(f"Calibration error: {e}")
                return {"error": str(e)}

        async def get_stats(self) -> Dict[str, Any]:
            """Get deployment statistics"""
            uptime = time.time() - self.stats["deployment_start_time"]
            avg_processing_time = (
                self.stats["total_processing_time"] / self.stats["requests_served"]
                if self.stats["requests_served"] > 0 else 0.0
            )

            stats = {
                **self.stats,
                "uptime_seconds": uptime,
                "average_processing_time_ms": avg_processing_time * 1000,
                "requests_per_second": self.stats["requests_served"] / uptime if uptime > 0 else 0.0,
                "replica_id": ray.get_runtime_context().get_actor_id(),
                "model_name": self.model_name,
                "pba_config": self.pba_config.__dict__
            }

            # Add monitoring stats if available
            if self.monitor:
                stats["monitoring"] = self.monitor.get_monitoring_summary()

            return stats

        async def get_health(self) -> Dict[str, Any]:
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model_loaded": self.uncertainty_model is not None,
                "replica_id": ray.get_runtime_context().get_actor_id(),
                "timestamp": time.time()
            }

else:
    # Fallback class when Ray is not available
    class PBAServeDeployment:
        def __init__(self, *args, **kwargs):
            raise ImportError("Ray Serve is required for distributed deployment. Install with: pip install ray[serve]")


class PBAServeClient:
    """
    Client for interacting with PBA Ray Serve deployment.

    Provides load balancing and consistency across multiple inference nodes.
    """

    def __init__(self, deployment_name: str = "pba-uncertainty"):
        """
        Initialize Ray Serve client.

        Args:
            deployment_name: Name of the deployed service
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed client. Install with: pip install ray[serve]")

        self.deployment_name = deployment_name
        self.handle = serve.get_deployment(deployment_name).get_handle()

        logger.info(f"Connected to PBA Ray Serve deployment: {deployment_name}")

    async def generate(
        self,
        text: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate text with uncertainty quantification.

        Args:
            text: Input text
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature

        Returns:
            Generation result with uncertainty scores
        """
        request = {
            "text": text,
            "max_length": max_length,
            "num_return_sequences": num_return_sequences,
            "temperature": temperature
        }

        return await self.handle.generate.remote(request)

    async def calibrate(
        self,
        uncertainty_scores: List[float],
        correctness_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Validate calibration across distributed nodes.

        Args:
            uncertainty_scores: Uncertainty scores to validate
            correctness_labels: Correctness labels

        Returns:
            Calibration results
        """
        request = {
            "uncertainty_scores": uncertainty_scores,
            "correctness_labels": correctness_labels
        }

        return await self.handle.calibrate.remote(request)

    async def get_cluster_stats(self) -> List[Dict[str, Any]]:
        """Get statistics from all replicas in the cluster"""
        # Get all replica handles
        replicas = serve.list_deployments()[self.deployment_name].replicas

        # Collect stats from all replicas
        stats_futures = []
        for replica in replicas:
            stats_futures.append(replica.get_stats.remote())

        return await asyncio.gather(*stats_futures)

    async def health_check(self) -> Dict[str, Any]:
        """Check health of the deployment"""
        return await self.handle.get_health.remote()


def deploy_pba_service(
    model_name: str,
    deployment_name: str = "pba-uncertainty",
    num_replicas: int = 1,
    max_concurrent_queries: int = 10,
    pba_config: Optional[Dict[str, Any]] = None,
    ray_actor_options: Optional[Dict[str, Any]] = None
) -> str:
    """
    Deploy PBA uncertainty quantification service on Ray Serve.

    Args:
        model_name: Hugging Face model identifier
        deployment_name: Name for the deployment
        num_replicas: Number of service replicas
        max_concurrent_queries: Maximum concurrent queries per replica
        pba_config: PBA configuration parameters
        ray_actor_options: Ray actor options (e.g., {"num_gpus": 1})

    Returns:
        Deployment handle name

    Example:
        >>> deploy_pba_service(
        ...     "gpt2",
        ...     num_replicas=2,
        ...     ray_actor_options={"num_gpus": 1}
        ... )
        'pba-uncertainty'
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Serve is required. Install with: pip install ray[serve]")

    # Default Ray actor options
    if ray_actor_options is None:
        ray_actor_options = {"num_gpus": 0.5}  # Share GPU across replicas

    # Configure deployment
    deployment_config = {
        "name": deployment_name,
        "num_replicas": num_replicas,
        "max_concurrent_queries": max_concurrent_queries,
        "ray_actor_options": ray_actor_options
    }

    # Create deployment
    deployment = serve.deployment(**deployment_config)(PBAServeDeployment)

    # Deploy
    deployment.deploy(
        model_name=model_name,
        pba_config=pba_config,
        enable_monitoring=True
    )

    logger.info(f"Deployed PBA service '{deployment_name}' with {num_replicas} replicas")
    return deployment_name


def create_ray_serve_config(
    model_name: str,
    num_replicas: int = 1,
    resources_per_replica: Optional[Dict[str, float]] = None,
    pba_alpha: float = 0.9,
    pba_beta: float = 0.5
) -> Dict[str, Any]:
    """
    Create Ray Serve deployment configuration for PBA uncertainty quantification.

    Args:
        model_name: Hugging Face model name
        num_replicas: Number of service replicas
        resources_per_replica: Resource allocation per replica
        pba_alpha: PBA alpha parameter
        pba_beta: PBA beta parameter

    Returns:
        Ray Serve configuration dictionary
    """
    if resources_per_replica is None:
        resources_per_replica = {"num_gpus": 0.5, "num_cpus": 1}

    return {
        "name": f"pba-{model_name}",
        "num_replicas": num_replicas,
        "max_concurrent_queries": 10,
        "ray_actor_options": resources_per_replica,
        "user_config": {
            "model_name": model_name,
            "pba_config": {
                "alpha": pba_alpha,
                "beta": pba_beta
            },
            "enable_monitoring": True
        },
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": num_replicas * 2,
            "target_num_ongoing_requests_per_replica": 5
        }
    }