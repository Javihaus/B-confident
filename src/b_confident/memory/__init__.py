"""
Memory Management Module

Provides streaming memory architecture for high-throughput batch processing
to prevent memory pressure and fragmentation in production deployments.
"""

from .streaming_processor import (
    StreamingUncertaintyProcessor,
    MemoryConfig,
    MemoryStats,
    MemoryPool,
    MemoryMonitor,
    create_streaming_processor,
    process_uncertainty_streaming
)

__all__ = [
    'StreamingUncertaintyProcessor',
    'MemoryConfig',
    'MemoryStats',
    'MemoryPool',
    'MemoryMonitor',
    'create_streaming_processor',
    'process_uncertainty_streaming'
]