"""
Distributed Calibration State Management System

Addresses the challenge of coordinating calibration updates across multiple inference nodes
without creating synchronization bottlenecks. Implements eventual consistency with
hierarchical calibration architecture.

Key Features:
- Eventually consistent calibration across nodes
- Event-driven calibration updates via message queues
- Hierarchical local + global calibration
- Circuit breaker pattern for fallback scenarios
- Non-blocking calibration synchronization
"""

import asyncio
import json
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationEventType(Enum):
    """Types of calibration events"""
    UPDATE_CALIBRATION = "update_calibration"
    REQUEST_SYNC = "request_sync"
    NODE_STATUS = "node_status"
    THRESHOLD_CHANGE = "threshold_change"
    RECALIBRATION_TRIGGER = "recalibration_trigger"


@dataclass
class CalibrationState:
    """Local calibration state for a node"""
    node_id: str
    uncertainty_history: List[float]
    accuracy_history: List[float]
    calibration_parameters: Dict[str, float]
    last_updated: float
    calibration_quality: float  # ECE or similar metric
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationState':
        return cls(**data)


@dataclass
class CalibrationEvent:
    """Event for calibration updates"""
    event_type: CalibrationEventType
    node_id: str
    timestamp: float
    data: Dict[str, Any]
    sequence_id: int = 0


@dataclass
class GlobalCalibrationState:
    """Global calibration state across all nodes"""
    global_parameters: Dict[str, float]
    node_states: Dict[str, CalibrationState]
    last_global_update: float
    consensus_threshold: float = 0.8  # Minimum agreement for global updates
    update_frequency: float = 300.0   # 5 minutes


class MessageBroker(ABC):
    """Abstract message broker for calibration events"""

    @abstractmethod
    async def publish(self, topic: str, event: CalibrationEvent) -> None:
        """Publish calibration event to topic"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable[[CalibrationEvent], None]) -> None:
        """Subscribe to calibration events on topic"""
        pass

    @abstractmethod
    async def get_subscribers_count(self, topic: str) -> int:
        """Get number of active subscribers for topic"""
        pass


class InMemoryMessageBroker(MessageBroker):
    """In-memory message broker for testing and single-process scenarios"""

    def __init__(self):
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()

    async def publish(self, topic: str, event: CalibrationEvent) -> None:
        """Publish event to all subscribers"""
        with self._lock:
            if topic in self._subscriptions:
                for callback in self._subscriptions[topic]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.warning(f"Error in subscriber callback: {e}")

    async def subscribe(self, topic: str, callback: Callable[[CalibrationEvent], None]) -> None:
        """Subscribe to topic"""
        with self._lock:
            self._subscriptions[topic].append(callback)

    async def get_subscribers_count(self, topic: str) -> int:
        """Get subscriber count"""
        with self._lock:
            return len(self._subscriptions.get(topic, []))


class RedisMessageBroker(MessageBroker):
    """Redis-based message broker for production distributed scenarios"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
            self._subscriptions: Dict[str, Any] = {}
            logger.info(f"Initialized Redis message broker: {redis_url}")
        except ImportError:
            logger.warning("Redis not available, falling back to in-memory broker")
            self._fallback = InMemoryMessageBroker()
            self.redis = None

    async def publish(self, topic: str, event: CalibrationEvent) -> None:
        """Publish to Redis or fallback"""
        if self.redis:
            try:
                event_data = {
                    'event_type': event.event_type.value,
                    'node_id': event.node_id,
                    'timestamp': event.timestamp,
                    'data': event.data,
                    'sequence_id': event.sequence_id
                }
                await self.redis.publish(topic, json.dumps(event_data))
            except Exception as e:
                logger.warning(f"Redis publish failed: {e}")
                if hasattr(self, '_fallback'):
                    await self._fallback.publish(topic, event)
        else:
            await self._fallback.publish(topic, event)

    async def subscribe(self, topic: str, callback: Callable[[CalibrationEvent], None]) -> None:
        """Subscribe to Redis topic"""
        if self.redis:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(topic)

                async def message_handler():
                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            try:
                                event_data = json.loads(message['data'])
                                event = CalibrationEvent(
                                    event_type=CalibrationEventType(event_data['event_type']),
                                    node_id=event_data['node_id'],
                                    timestamp=event_data['timestamp'],
                                    data=event_data['data'],
                                    sequence_id=event_data.get('sequence_id', 0)
                                )
                                await callback(event)
                            except Exception as e:
                                logger.warning(f"Error processing message: {e}")

                # Start message handler task
                asyncio.create_task(message_handler())
                self._subscriptions[topic] = pubsub

            except Exception as e:
                logger.warning(f"Redis subscribe failed: {e}")
                if hasattr(self, '_fallback'):
                    await self._fallback.subscribe(topic, callback)
        else:
            await self._fallback.subscribe(topic, callback)

    async def get_subscribers_count(self, topic: str) -> int:
        """Get subscriber count from Redis"""
        if self.redis:
            try:
                return await self.redis.pubsub_numsub(topic)
            except Exception:
                return 0
        else:
            return await self._fallback.get_subscribers_count(topic)


class CircuitBreaker:
    """Circuit breaker for calibration synchronization"""

    def __init__(self,
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 half_open_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.half_open_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

                raise e

    def get_state(self) -> str:
        return self.state


class DistributedCalibrationManager:
    """
    Main distributed calibration manager that coordinates calibration across nodes.

    Implements:
    - Eventually consistent calibration state
    - Hierarchical local + global calibration
    - Event-driven updates without synchronization bottlenecks
    - Circuit breaker pattern for reliability
    """

    def __init__(self,
                 node_id: str,
                 message_broker: Optional[MessageBroker] = None,
                 local_update_interval: float = 60.0,   # 1 minute
                 global_sync_interval: float = 300.0,   # 5 minutes
                 history_size: int = 1000):

        self.node_id = node_id
        self.broker = message_broker or InMemoryMessageBroker()
        self.local_update_interval = local_update_interval
        self.global_sync_interval = global_sync_interval
        self.history_size = history_size

        # Local state
        self.local_state = CalibrationState(
            node_id=node_id,
            uncertainty_history=[],
            accuracy_history=[],
            calibration_parameters={"beta": 0.5, "temperature": 1.0},
            last_updated=time.time(),
            calibration_quality=0.5,
            sample_count=0
        )

        # Global state cache
        self.global_state = GlobalCalibrationState(
            global_parameters={"beta": 0.5, "temperature": 1.0},
            node_states={},
            last_global_update=time.time()
        )

        # Circuit breaker for sync operations
        self.circuit_breaker = CircuitBreaker()

        # Event processing
        self._event_queue = deque(maxlen=1000)
        self._sequence_counter = 0
        self._processing_lock = threading.RLock()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False

        logger.info(f"Initialized DistributedCalibrationManager for node {node_id}")

    async def start(self) -> None:
        """Start the calibration manager and background tasks"""
        self._running = True

        # Subscribe to calibration events
        await self.broker.subscribe("calibration_events", self._handle_calibration_event)

        # Start background tasks
        self._background_tasks.append(
            asyncio.create_task(self._local_calibration_loop())
        )
        self._background_tasks.append(
            asyncio.create_task(self._global_sync_loop())
        )
        self._background_tasks.append(
            asyncio.create_task(self._process_event_queue())
        )

        logger.info(f"Started DistributedCalibrationManager for node {self.node_id}")

    async def stop(self) -> None:
        """Stop the calibration manager and cleanup"""
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info(f"Stopped DistributedCalibrationManager for node {self.node_id}")

    def update_local_calibration(self,
                                uncertainty: float,
                                accuracy: float) -> None:
        """Update local calibration with new uncertainty/accuracy pair"""
        with self._processing_lock:
            # Add to history
            self.local_state.uncertainty_history.append(uncertainty)
            self.local_state.accuracy_history.append(accuracy)

            # Maintain history size
            if len(self.local_state.uncertainty_history) > self.history_size:
                self.local_state.uncertainty_history.pop(0)
                self.local_state.accuracy_history.pop(0)

            self.local_state.sample_count += 1
            self.local_state.last_updated = time.time()

            # Recalculate calibration quality if we have enough samples
            if len(self.local_state.uncertainty_history) >= 10:
                self._recalculate_local_quality()

    def _recalculate_local_quality(self) -> None:
        """Recalculate local calibration quality (ECE)"""
        uncertainties = np.array(self.local_state.uncertainty_history)
        accuracies = np.array(self.local_state.accuracy_history)

        # Simple ECE calculation
        ece = 0.0
        n_bins = 10
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins

            in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
            if i == n_bins - 1:  # Last bin includes upper boundary
                in_bin = (uncertainties >= bin_lower) & (uncertainties <= bin_upper)

            if in_bin.sum() > 0:
                bin_uncertainty = uncertainties[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()
                bin_weight = in_bin.sum() / len(uncertainties)

                ece += bin_weight * abs(bin_uncertainty - bin_accuracy)

        self.local_state.calibration_quality = ece

        # Update calibration parameters if quality is poor
        if ece > 0.2:  # Threshold for recalibration
            self._trigger_local_recalibration()

    def _trigger_local_recalibration(self) -> None:
        """Trigger local parameter recalibration"""
        try:
            # Simple temperature scaling adjustment
            current_temp = self.local_state.calibration_parameters.get("temperature", 1.0)
            uncertainties = np.array(self.local_state.uncertainty_history)
            accuracies = np.array(self.local_state.accuracy_history)

            # Find better temperature via simple search
            best_temp = current_temp
            best_ece = self.local_state.calibration_quality

            for temp in [0.8, 0.9, 1.1, 1.2, 1.3, 1.5]:
                # Simulate scaled uncertainties
                scaled_uncertainties = 1 - np.exp(-uncertainties / temp)
                scaled_uncertainties = np.clip(scaled_uncertainties, 0, 1)

                # Calculate ECE with this temperature
                ece = self._calculate_ece(scaled_uncertainties, accuracies)

                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp

            if best_temp != current_temp:
                self.local_state.calibration_parameters["temperature"] = best_temp
                logger.info(f"Node {self.node_id} updated temperature: {current_temp} -> {best_temp}")

                # Broadcast parameter update
                asyncio.create_task(self._broadcast_parameter_update())

        except Exception as e:
            logger.warning(f"Local recalibration failed: {e}")

    def _calculate_ece(self, uncertainties: np.ndarray, accuracies: np.ndarray) -> float:
        """Calculate Expected Calibration Error"""
        ece = 0.0
        n_bins = 10

        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins

            in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
            if i == n_bins - 1:
                in_bin = (uncertainties >= bin_lower) & (uncertainties <= bin_upper)

            if in_bin.sum() > 0:
                bin_uncertainty = uncertainties[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()
                bin_weight = in_bin.sum() / len(uncertainties)

                ece += bin_weight * abs(bin_uncertainty - bin_accuracy)

        return ece

    async def _broadcast_parameter_update(self) -> None:
        """Broadcast local parameter update to other nodes"""
        event = CalibrationEvent(
            event_type=CalibrationEventType.THRESHOLD_CHANGE,
            node_id=self.node_id,
            timestamp=time.time(),
            data={
                "parameters": self.local_state.calibration_parameters.copy(),
                "quality": self.local_state.calibration_quality,
                "sample_count": self.local_state.sample_count
            },
            sequence_id=self._get_next_sequence_id()
        )

        try:
            await self.broker.publish("calibration_events", event)
        except Exception as e:
            logger.warning(f"Failed to broadcast parameter update: {e}")

    async def _handle_calibration_event(self, event: CalibrationEvent) -> None:
        """Handle incoming calibration event"""
        if event.node_id == self.node_id:
            return  # Ignore our own events

        with self._processing_lock:
            self._event_queue.append(event)

    async def _process_event_queue(self) -> None:
        """Process calibration events from queue"""
        while self._running:
            try:
                if self._event_queue:
                    with self._processing_lock:
                        event = self._event_queue.popleft()

                    await self._process_single_event(event)

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.warning(f"Error processing event queue: {e}")
                await asyncio.sleep(1.0)

    async def _process_single_event(self, event: CalibrationEvent) -> None:
        """Process a single calibration event"""
        try:
            if event.event_type == CalibrationEventType.THRESHOLD_CHANGE:
                # Update our knowledge of other node's parameters
                self.global_state.node_states[event.node_id] = CalibrationState(
                    node_id=event.node_id,
                    uncertainty_history=[],  # We don't store full history from other nodes
                    accuracy_history=[],
                    calibration_parameters=event.data.get("parameters", {}),
                    last_updated=event.timestamp,
                    calibration_quality=event.data.get("quality", 0.5),
                    sample_count=event.data.get("sample_count", 0)
                )

            elif event.event_type == CalibrationEventType.REQUEST_SYNC:
                # Another node is requesting sync - send our state
                await self._respond_to_sync_request(event.node_id)

            elif event.event_type == CalibrationEventType.NODE_STATUS:
                # Update node status in global state
                if "state" in event.data:
                    self.global_state.node_states[event.node_id] = CalibrationState.from_dict(
                        event.data["state"]
                    )

        except Exception as e:
            logger.warning(f"Error processing event {event.event_type}: {e}")

    async def _respond_to_sync_request(self, requesting_node: str) -> None:
        """Respond to sync request from another node"""
        event = CalibrationEvent(
            event_type=CalibrationEventType.NODE_STATUS,
            node_id=self.node_id,
            timestamp=time.time(),
            data={"state": self.local_state.to_dict()},
            sequence_id=self._get_next_sequence_id()
        )

        try:
            await self.broker.publish("calibration_events", event)
        except Exception as e:
            logger.warning(f"Failed to respond to sync request: {e}")

    async def _local_calibration_loop(self) -> None:
        """Local calibration update loop"""
        while self._running:
            try:
                # Perform local calibration updates
                if (time.time() - self.local_state.last_updated) > self.local_update_interval:
                    # Check if we need to recalibrate
                    if self.local_state.calibration_quality > 0.15:  # Threshold
                        self._trigger_local_recalibration()

                await asyncio.sleep(self.local_update_interval / 10)  # Check more frequently

            except Exception as e:
                logger.warning(f"Error in local calibration loop: {e}")
                await asyncio.sleep(10.0)

    async def _global_sync_loop(self) -> None:
        """Global synchronization loop"""
        while self._running:
            try:
                await asyncio.sleep(self.global_sync_interval)

                # Use circuit breaker for sync operations
                try:
                    await self.circuit_breaker.call(self._perform_global_sync)
                except Exception as e:
                    logger.warning(f"Global sync failed (circuit breaker): {e}")
                    # Fallback to local calibration only
                    continue

            except Exception as e:
                logger.warning(f"Error in global sync loop: {e}")
                await asyncio.sleep(60.0)

    async def _perform_global_sync(self) -> None:
        """Perform global synchronization with other nodes"""
        # Request sync from other nodes
        sync_request = CalibrationEvent(
            event_type=CalibrationEventType.REQUEST_SYNC,
            node_id=self.node_id,
            timestamp=time.time(),
            data={"requesting_sync": True},
            sequence_id=self._get_next_sequence_id()
        )

        await self.broker.publish("calibration_events", sync_request)

        # Wait a bit for responses
        await asyncio.sleep(5.0)

        # Calculate global consensus parameters
        await self._calculate_global_consensus()

    async def _calculate_global_consensus(self) -> None:
        """Calculate global consensus parameters from all nodes"""
        try:
            if not self.global_state.node_states:
                return  # No other nodes to sync with

            # Collect all node parameters with weights based on sample count and quality
            weighted_params: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
            total_weight = 0.0

            # Include our own state
            our_weight = self.local_state.sample_count / (1 + self.local_state.calibration_quality)
            for param, value in self.local_state.calibration_parameters.items():
                weighted_params[param].append((value, our_weight))
            total_weight += our_weight

            # Include other nodes' states
            for node_state in self.global_state.node_states.values():
                if node_state.sample_count > 0:
                    node_weight = node_state.sample_count / (1 + node_state.calibration_quality)
                    for param, value in node_state.calibration_parameters.items():
                        weighted_params[param].append((value, node_weight))
                    total_weight += node_weight

            # Calculate weighted average for each parameter
            global_params = {}
            for param, values_weights in weighted_params.items():
                if total_weight > 0:
                    weighted_sum = sum(value * weight for value, weight in values_weights)
                    global_params[param] = weighted_sum / total_weight
                else:
                    # Fallback to simple average
                    global_params[param] = sum(value for value, _ in values_weights) / len(values_weights)

            # Update global state
            old_params = self.global_state.global_parameters.copy()
            self.global_state.global_parameters = global_params
            self.global_state.last_global_update = time.time()

            logger.info(f"Node {self.node_id} updated global consensus: {old_params} -> {global_params}")

        except Exception as e:
            logger.warning(f"Global consensus calculation failed: {e}")

    def get_calibration_parameters(self, prefer_global: bool = True) -> Dict[str, float]:
        """
        Get calibration parameters, preferring global if available and recent.

        Args:
            prefer_global: Whether to prefer global parameters over local

        Returns:
            Dictionary of calibration parameters
        """
        # Check if global parameters are recent and available
        if (prefer_global and
            self.global_state.global_parameters and
            (time.time() - self.global_state.last_global_update) < (self.global_sync_interval * 2)):
            return self.global_state.global_parameters.copy()
        else:
            # Fallback to local parameters
            return self.local_state.calibration_parameters.copy()

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get comprehensive calibration statistics"""
        return {
            "node_id": self.node_id,
            "local_state": {
                "calibration_quality": self.local_state.calibration_quality,
                "sample_count": self.local_state.sample_count,
                "last_updated": self.local_state.last_updated,
                "parameters": self.local_state.calibration_parameters
            },
            "global_state": {
                "parameters": self.global_state.global_parameters,
                "last_global_update": self.global_state.last_global_update,
                "known_nodes": list(self.global_state.node_states.keys())
            },
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "event_queue_size": len(self._event_queue)
        }

    def _get_next_sequence_id(self) -> int:
        """Get next sequence ID for events"""
        self._sequence_counter += 1
        return self._sequence_counter