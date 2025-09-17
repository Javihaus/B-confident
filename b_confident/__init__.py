"""
B-Confident: Perplexity-Based Adjacency for Uncertainty Quantification in LLMs

This package implements the PBA methodology for enterprise-grade uncertainty
quantification in Large Language Models with regulatory compliance support.

Key Components:
    - Core PBA algorithm implementation
    - Hugging Face transformers integration
    - Regulatory compliance tools
    - Serving framework integrations
    - Calibration and validation tools
"""

__version__ = "0.1.0"
__author__ = "Javier Marin"
__email__ = "javier@jmarin.info"

# Core imports
from .core.pba_algorithm import PBAUncertainty, PBAConfig
from .core.metrics import (
    ExpectedCalibrationError,
    BrierScore,
    calculate_uncertainty_metrics
)
from .core.comprehensive_metrics import (
    ComprehensiveUncertaintyCalculator,
    ComprehensiveUncertaintyResults,
    MaxProbabilityConfidence,
    EntropyUncertainty,
    PredictionConsistency,
    TemperatureScaledConfidence
)

# Integration imports
from .integration.transformers_wrapper import (
    UncertaintyTransformersModel,
    uncertainty_generate
)

# Memory management for high-throughput production
from .memory.streaming_processor import (
    StreamingUncertaintyProcessor,
    MemoryConfig,
    MemoryStats,
    create_streaming_processor,
    process_uncertainty_streaming
)

# Distributed calibration for multi-node deployments
from .distributed.calibration_manager import (
    DistributedCalibrationManager,
    CalibrationState,
    CalibrationEvent,
    CalibrationEventType,
    InMemoryMessageBroker,
    RedisMessageBroker
)

# Observability and debugging for production troubleshooting
from .observability.uncertainty_debugger import (
    InstrumentedUncertaintyCalculator,
    UncertaintyProvenance,
    DebugLevel,
    PipelineStage
)
from .observability.metrics_collector import (
    UncertaintyMetricsCollector,
    MetricsAggregator,
    AlertManager,
    AlertSeverity,
    create_standard_metrics_setup
)
from .observability.dashboard import (
    UncertaintyDashboard,
    create_uncertainty_dashboard
)

# Compliance imports
from .compliance.regulatory import (
    ComplianceReporter,
    generate_eu_ai_act_report
)
from .compliance.regulatory_mapping import (
    RegulatoryComplianceManager,
    EUAIActCompliance,
    NISTAIRMFCompliance,
    ComplianceFramework,
    RiskLevel
)

# Main API
__all__ = [
    # Core
    "PBAUncertainty",
    "PBAConfig",

    # Metrics
    "ExpectedCalibrationError",
    "BrierScore",
    "calculate_uncertainty_metrics",

    # Comprehensive Metrics
    "ComprehensiveUncertaintyCalculator",
    "ComprehensiveUncertaintyResults",
    "MaxProbabilityConfidence",
    "EntropyUncertainty",
    "PredictionConsistency",
    "TemperatureScaledConfidence",

    # Integration
    "UncertaintyTransformersModel",
    "uncertainty_generate",

    # Memory Management
    "StreamingUncertaintyProcessor",
    "MemoryConfig",
    "MemoryStats",
    "create_streaming_processor",
    "process_uncertainty_streaming",

    # Distributed Calibration
    "DistributedCalibrationManager",
    "CalibrationState",
    "CalibrationEvent",
    "CalibrationEventType",
    "InMemoryMessageBroker",
    "RedisMessageBroker",

    # Observability and Debugging
    "InstrumentedUncertaintyCalculator",
    "UncertaintyProvenance",
    "DebugLevel",
    "PipelineStage",
    "UncertaintyMetricsCollector",
    "MetricsAggregator",
    "AlertManager",
    "AlertSeverity",
    "UncertaintyDashboard",
    "create_uncertainty_dashboard",
    "create_standard_metrics_setup",

    # Compliance
    "ComplianceReporter",
    "generate_eu_ai_act_report",
    "RegulatoryComplianceManager",
    "EUAIActCompliance",
    "NISTAIRMFCompliance",
    "ComplianceFramework",
    "RiskLevel",

    # Version info
    "__version__",
]

# Package metadata
PACKAGE_INFO = {
    "name": "b-confident",
    "version": __version__,
    "description": "Perplexity-Based Adjacency for Uncertainty Quantification in LLMs",
    "methodology_paper": "Perplexity-Based Adjacency for Uncertainty Quantification in Large Language Models",
    "regulatory_compliance": [
        "EU AI Act Article 15",
        "NIST AI Risk Management Framework",
        "ISO/IEC 23053:2022",
        "IEEE 2857-2021",
        "ISO/IEC 23094-1:2023"
    ],
    "supported_frameworks": ["transformers", "torch", "torchserve", "fastapi", "ray-serve"],
    "uncertainty_methods": [
        "Perplexity-Based Adjacency (PBA)",
        "Maximum Probability Confidence",
        "Shannon Entropy",
        "Prediction Consistency",
        "Temperature-scaled Confidence"
    ],
}