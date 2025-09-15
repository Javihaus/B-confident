"""
Default configuration values for b-confident

Based on systematic parameter validation from the paper:
- α = 0.9 provides optimal balance between coverage and efficiency
- β = 0.5 provides appropriate sensitivity without oversensitivity
"""

from typing import Dict, List, Any

# Paper-validated optimal PBA parameters
DEFAULT_PBA_CONFIG: Dict[str, Any] = {
    # Core PBA parameters (from paper validation)
    "alpha": 0.9,                    # Probability mass threshold (optimal)
    "beta": 0.5,                     # Sensitivity parameter (optimal)

    # Computational parameters
    "temperature": 1.0,              # Temperature scaling for logits
    "device": None,                  # Auto-detect device if None
    "dtype": "float32",              # Computation precision

    # Memory optimization
    "batch_processing": True,        # Process tokens in batches
    "max_batch_size": 32,           # Maximum batch size

    # Validation parameters
    "validate_inputs": True,         # Validate input tensors
    "numerical_stability": True     # Apply numerical stability measures
}

# Model architecture compatibility
SUPPORTED_ARCHITECTURES: List[str] = [
    # GPT Family
    "GPT2LMHeadModel",
    "GPTNeoForCausalLM",
    "GPTNeoXForCausalLM",
    "GPTJForCausalLM",

    # LLaMA Family
    "LlamaForCausalLM",
    "CodeLlamaForCausalLM",

    # Mistral
    "MistralForCausalLM",

    # Other popular architectures
    "QWenLMHeadModel",
    "Qwen2ForCausalLM",
    "GemmaForCausalLM",

    # Smaller models
    "DistilBertForSequenceClassification",
    "BertForSequenceClassification"
]

# Environment-specific defaults
ENVIRONMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "development": {
        **DEFAULT_PBA_CONFIG,
        "validate_inputs": True,
        "batch_processing": False,  # Disable for easier debugging
        "max_batch_size": 1
    },

    "production": {
        **DEFAULT_PBA_CONFIG,
        "validate_inputs": False,   # Disable for performance
        "batch_processing": True,
        "max_batch_size": 64,
        "numerical_stability": True
    },

    "serving": {
        **DEFAULT_PBA_CONFIG,
        "batch_processing": True,
        "max_batch_size": 16,      # Balance between throughput and latency
        "validate_inputs": False   # Trust server-side validation
    }
}

# Resource allocation guidance based on model size
RESOURCE_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "small": {          # < 1B parameters
        "description": "Small models (117M - 1B parameters)",
        "memory_base_gb": 2,
        "memory_pba_overhead": 0.2,  # 20% overhead
        "recommended_batch_size": 32,
        "gpu_memory_gb": 4,
        "examples": ["gpt2", "gpt2-medium", "distilbert-base"]
    },

    "medium": {         # 1B - 7B parameters
        "description": "Medium models (1B - 7B parameters)",
        "memory_base_gb": 8,
        "memory_pba_overhead": 0.19, # 19% overhead
        "recommended_batch_size": 16,
        "gpu_memory_gb": 16,
        "examples": ["gpt2-large", "gpt2-xl", "EleutherAI/gpt-neo-2.7B"]
    },

    "large": {          # 7B - 30B parameters
        "description": "Large models (7B - 30B parameters)",
        "memory_base_gb": 24,
        "memory_pba_overhead": 0.18, # 18% overhead
        "recommended_batch_size": 8,
        "gpu_memory_gb": 40,
        "examples": ["microsoft/DialoGPT-large", "EleutherAI/gpt-j-6B"]
    },

    "xlarge": {         # > 30B parameters
        "description": "Extra large models (> 30B parameters)",
        "memory_base_gb": 80,
        "memory_pba_overhead": 0.17, # 17% overhead
        "recommended_batch_size": 4,
        "gpu_memory_gb": 80,
        "examples": ["EleutherAI/gpt-neox-20b"]
    }
}

# Compliance thresholds for different use cases
COMPLIANCE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "medical": {
        "max_ece": 0.02,           # Maximum 2% Expected Calibration Error
        "min_auroc": 0.80,         # Minimum 80% discrimination ability
        "min_stability": 0.95,     # Minimum 95% stability score
        "max_brier": 0.15,         # Maximum Brier score
        "description": "Medical diagnosis support systems"
    },

    "financial": {
        "max_ece": 0.03,           # Maximum 3% ECE
        "min_auroc": 0.75,         # Minimum 75% AUROC
        "min_stability": 0.92,     # Minimum 92% stability
        "max_brier": 0.20,         # Maximum Brier score
        "description": "Financial decision support systems"
    },

    "legal": {
        "max_ece": 0.05,           # Maximum 5% ECE
        "min_auroc": 0.70,         # Minimum 70% AUROC
        "min_stability": 0.90,     # Minimum 90% stability
        "max_brier": 0.25,         # Maximum Brier score
        "description": "Legal document analysis systems"
    },

    "general": {
        "max_ece": 0.10,           # Maximum 10% ECE
        "min_auroc": 0.65,         # Minimum 65% AUROC
        "min_stability": 0.85,     # Minimum 85% stability
        "max_brier": 0.30,         # Maximum Brier score
        "description": "General-purpose applications"
    }
}

# Serving configuration defaults
SERVING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fastapi": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "cors_origins": ["*"],
        "enable_monitoring": True,
        "request_timeout": 60,
        "max_request_size": 1024 * 1024,  # 1MB
        "rate_limiting": {
            "enabled": False,
            "requests_per_minute": 60
        }
    },

    "torchserve": {
        "model_name": "pba-uncertainty",
        "version": "1.0",
        "batch_size": 1,
        "max_batch_delay": 100,
        "response_timeout": 120,
        "device_type": "gpu",
        "parallel_level": 1
    },

    "ray_serve": {
        "num_replicas": 1,
        "max_concurrent_queries": 10,
        "ray_actor_options": {
            "num_gpus": 0.5,
            "num_cpus": 1
        },
        "autoscaling": {
            "min_replicas": 1,
            "max_replicas": 4,
            "target_num_ongoing_requests_per_replica": 5
        }
    }
}

# Monitoring and alerting defaults
MONITORING_DEFAULTS: Dict[str, Any] = {
    "calibration_monitoring": {
        "window_size": 1000,       # Samples in monitoring window
        "min_samples": 100,        # Minimum samples before alerting
        "alert_thresholds": {
            "ece_warning": 1.5,    # 50% increase triggers warning
            "ece_critical": 2.0,   # 100% increase triggers critical
            "brier_warning": 1.3,  # 30% increase triggers warning
            "brier_critical": 1.5, # 50% increase triggers critical
            "auroc_warning": 0.9,  # 10% decrease triggers warning
            "auroc_critical": 0.8  # 20% decrease triggers critical
        }
    },

    "performance_monitoring": {
        "metrics_retention_days": 30,
        "alert_on_latency": True,
        "latency_threshold_ms": 5000,
        "alert_on_error_rate": True,
        "error_rate_threshold": 0.05
    }
}

# Validation and testing defaults
VALIDATION_DEFAULTS: Dict[str, Any] = {
    "cross_validation": {
        "n_folds": 5,
        "random_seed": 42,
        "stratify": True
    },

    "calibration_validation": {
        "n_bins": 10,
        "min_samples_per_bin": 10,
        "confidence_interval": 0.95
    },

    "performance_testing": {
        "benchmark_samples": 100,
        "warmup_samples": 10,
        "timeout_seconds": 300
    }
}