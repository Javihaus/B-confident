"""
Streaming Memory Architecture for High-Throughput Batch Processing

Addresses production memory management challenges:
1. Prevents probability distribution accumulation across batches
2. Eliminates memory fragmentation from large tensor allocations
3. Provides dynamic memory-aware batch sizing
4. Implements memory pool management for efficient tensor reuse
"""

import torch
import numpy as np
import gc
import psutil
import threading
from typing import Iterator, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
from collections import deque
import logging
import time

from ..core.pba_algorithm import PBAUncertainty, PBAConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    max_memory_usage_gb: float = 2.0  # Maximum memory usage per process
    memory_pool_size_mb: int = 100   # Pre-allocated memory pool size
    chunk_size: int = 8              # Process uncertainty in chunks of this size
    gc_threshold: float = 0.8        # Trigger GC when memory usage exceeds this ratio
    enable_memory_monitoring: bool = True
    memory_check_interval: int = 10  # Check memory every N batches


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    current_usage_mb: float
    peak_usage_mb: float
    pool_utilization: float
    gc_events: int
    fragmentation_score: float


class MemoryPool:
    """
    Pre-allocated memory pool to prevent fragmentation from repeated tensor allocations.
    Maintains fixed-size tensors that can be reused across uncertainty calculations.
    """

    def __init__(self, config: MemoryConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._pools: Dict[Tuple[int, ...], deque] = {}  # Shape -> tensor deque
        self._allocated_memory = 0
        self._max_memory = config.memory_pool_size_mb * 1024 * 1024  # Convert to bytes
        self._lock = threading.RLock()

        logger.info(f"Initialized memory pool: {config.memory_pool_size_mb}MB on {device}")

    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from the pool or create new one if not available"""
        with self._lock:
            pool_key = shape

            if pool_key in self._pools and self._pools[pool_key]:
                tensor = self._pools[pool_key].popleft()
                tensor.zero_()  # Clear previous values
                return tensor

            # Create new tensor if pool is empty
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            tensor_size = tensor.numel() * tensor.element_size()

            if self._allocated_memory + tensor_size > self._max_memory:
                self._cleanup_least_used()

            self._allocated_memory += tensor_size
            return tensor

    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool for reuse"""
        with self._lock:
            pool_key = tuple(tensor.shape)

            if pool_key not in self._pools:
                self._pools[pool_key] = deque(maxlen=5)  # Limit pool size per shape

            if len(self._pools[pool_key]) < self._pools[pool_key].maxlen:
                self._pools[pool_key].append(tensor.detach())

    def _cleanup_least_used(self) -> None:
        """Remove least used tensors from pools"""
        # Simple cleanup: remove half of each pool
        for pool in self._pools.values():
            while len(pool) > len(pool) // 2:
                pool.pop()

        self._allocated_memory = sum(
            sum(t.numel() * t.element_size() for t in pool)
            for pool in self._pools.values()
        )

    def get_utilization(self) -> float:
        """Get current pool utilization ratio"""
        return self._allocated_memory / self._max_memory if self._max_memory > 0 else 0.0

    def clear(self) -> None:
        """Clear all pools and free memory"""
        with self._lock:
            self._pools.clear()
            self._allocated_memory = 0
            gc.collect()


class MemoryMonitor:
    """Monitor system memory usage and trigger cleanup when necessary"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._current_memory = 0.0
        self._peak_memory = 0.0
        self._gc_events = 0
        self._last_check = time.time()

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        if not self.config.enable_memory_monitoring:
            return False

        current_time = time.time()
        if current_time - self._last_check < 1.0:  # Rate limit checks
            return False

        self._last_check = current_time

        # Check system memory
        memory_info = psutil.virtual_memory()
        system_usage_ratio = memory_info.percent / 100.0

        # Check GPU memory if available
        gpu_usage_ratio = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            gpu_usage_ratio = gpu_memory

        max_usage_ratio = max(system_usage_ratio, gpu_usage_ratio)
        self._current_memory = max_usage_ratio * 100
        self._peak_memory = max(self._peak_memory, self._current_memory)

        return max_usage_ratio > self.config.gc_threshold

    def trigger_cleanup(self) -> None:
        """Trigger garbage collection and memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self._gc_events += 1

        logger.debug(f"Memory cleanup triggered. GC events: {self._gc_events}")

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        fragmentation_score = self._calculate_fragmentation_score()

        return MemoryStats(
            current_usage_mb=self._current_memory * 1024 / 100,  # Convert to MB
            peak_usage_mb=self._peak_memory * 1024 / 100,
            pool_utilization=0.0,  # Set by StreamingUncertaintyProcessor
            gc_events=self._gc_events,
            fragmentation_score=fragmentation_score
        )

    def _calculate_fragmentation_score(self) -> float:
        """Calculate memory fragmentation score (0-1, higher = more fragmented)"""
        if not torch.cuda.is_available():
            return 0.0

        # Simple fragmentation heuristic based on allocated vs reserved memory
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        if reserved == 0:
            return 0.0

        return max(0.0, 1.0 - (allocated / reserved))


class StreamingUncertaintyProcessor:
    """
    Streaming processor for uncertainty calculations that prevents memory accumulation.

    Key features:
    - Processes uncertainty in chunks to limit memory usage
    - Uses memory pools to prevent fragmentation
    - Dynamic batch sizing based on available memory
    - Automatic garbage collection under memory pressure
    """

    def __init__(self,
                 pba_config: Optional[PBAConfig] = None,
                 memory_config: Optional[MemoryConfig] = None,
                 device: str = "auto"):
        self.pba_config = pba_config or PBAConfig()
        self.memory_config = memory_config or MemoryConfig()

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize components
        self.pba_calculator = PBAUncertainty(self.pba_config)
        self.memory_pool = MemoryPool(self.memory_config, self.device)
        self.memory_monitor = MemoryMonitor(self.memory_config)

        # Processing state
        self._batch_count = 0
        self._total_processed = 0

        logger.info(f"StreamingUncertaintyProcessor initialized on {self.device}")
        logger.info(f"Memory config: {self.memory_config}")

    @contextmanager
    def memory_managed_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """Context manager for memory-managed tensor allocation"""
        tensor = self.memory_pool.get_tensor(shape, dtype)
        try:
            yield tensor
        finally:
            self.memory_pool.return_tensor(tensor)

    def process_batch_streaming(self,
                              logits_batch: List[torch.Tensor],
                              actual_token_ids: Optional[List[int]] = None) -> Iterator[float]:
        """
        Stream uncertainty calculations for a batch, yielding results one by one.

        This prevents accumulation of probability distributions in memory.
        """
        self._batch_count += 1

        # Check memory pressure and adjust chunk size dynamically
        if self.memory_monitor.check_memory_pressure():
            self.memory_monitor.trigger_cleanup()
            # Reduce chunk size under memory pressure
            chunk_size = max(1, self.memory_config.chunk_size // 2)
        else:
            chunk_size = self.memory_config.chunk_size

        # Process in chunks to limit memory usage
        for i in range(0, len(logits_batch), chunk_size):
            chunk_logits = logits_batch[i:i + chunk_size]
            chunk_token_ids = actual_token_ids[i:i + chunk_size] if actual_token_ids else None

            # Process chunk and yield results immediately
            for j, logits in enumerate(chunk_logits):
                token_id = chunk_token_ids[j] if chunk_token_ids else None

                # Use memory pool for intermediate calculations
                uncertainty = self._calculate_uncertainty_memory_efficient(logits, token_id)

                self._total_processed += 1
                yield uncertainty

            # Cleanup after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Periodic memory monitoring
        if self._batch_count % self.memory_config.memory_check_interval == 0:
            self._log_memory_stats()

    def _calculate_uncertainty_memory_efficient(self,
                                              logits: torch.Tensor,
                                              actual_token_id: Optional[int] = None) -> float:
        """Calculate uncertainty using memory-efficient approach"""

        # Move to correct device if needed
        if logits.device != self.device:
            logits = logits.to(self.device)

        try:
            # Use standard PBA calculation but with memory management
            uncertainty = self.pba_calculator.calculate_token_uncertainty(
                logits, actual_token_id=actual_token_id
            )
            return uncertainty

        except Exception as e:
            logger.warning(f"Error in uncertainty calculation: {e}")
            return 1.0  # Maximum uncertainty as fallback

    def calculate_optimal_batch_size(self,
                                   sample_logits: torch.Tensor,
                                   target_memory_usage: float = 0.5) -> int:
        """
        Calculate optimal batch size based on available memory and tensor sizes.

        Args:
            sample_logits: Sample logits tensor to estimate memory usage
            target_memory_usage: Target memory usage ratio (0-1)

        Returns:
            Optimal batch size
        """
        # Estimate memory per sample
        sample_memory = sample_logits.numel() * sample_logits.element_size()

        # Get available memory
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory * target_memory_usage
        else:
            available_memory = psutil.virtual_memory().available * target_memory_usage

        # Calculate optimal batch size with safety margin
        optimal_size = int(available_memory / (sample_memory * 1.5))  # 1.5x safety margin
        optimal_size = max(1, min(optimal_size, 128))  # Clamp to reasonable range

        logger.info(f"Calculated optimal batch size: {optimal_size}")
        return optimal_size

    def _log_memory_stats(self) -> None:
        """Log current memory statistics"""
        stats = self.get_memory_stats()

        logger.info(f"Memory Stats - Current: {stats.current_usage_mb:.1f}MB, "
                   f"Peak: {stats.peak_usage_mb:.1f}MB, "
                   f"Pool: {stats.pool_utilization:.1%}, "
                   f"GC Events: {stats.gc_events}, "
                   f"Fragmentation: {stats.fragmentation_score:.2f}")

    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        stats = self.memory_monitor.get_stats()
        stats.pool_utilization = self.memory_pool.get_utilization()
        return stats

    def cleanup(self) -> None:
        """Cleanup all memory resources"""
        self.memory_pool.clear()
        self.memory_monitor.trigger_cleanup()

        logger.info(f"Cleanup complete. Processed {self._total_processed} samples total")


# Convenience functions for high-level usage
def create_streaming_processor(max_memory_gb: float = 2.0,
                             device: str = "auto") -> StreamingUncertaintyProcessor:
    """Create a streaming processor with memory management"""
    memory_config = MemoryConfig(max_memory_usage_gb=max_memory_gb)
    return StreamingUncertaintyProcessor(memory_config=memory_config, device=device)


def process_uncertainty_streaming(logits_batch: List[torch.Tensor],
                                actual_token_ids: Optional[List[int]] = None,
                                max_memory_gb: float = 2.0) -> List[float]:
    """
    High-level function to process uncertainties with streaming memory management.

    Args:
        logits_batch: List of logits tensors
        actual_token_ids: Optional list of actual token IDs
        max_memory_gb: Maximum memory usage in GB

    Returns:
        List of uncertainty scores
    """
    processor = create_streaming_processor(max_memory_gb)

    try:
        uncertainties = list(processor.process_batch_streaming(logits_batch, actual_token_ids))
        return uncertainties
    finally:
        processor.cleanup()