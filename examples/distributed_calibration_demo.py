"""
Distributed Calibration State Management Demo

Demonstrates the distributed calibration system that addresses production
challenges of coordinating calibration across multiple inference nodes.
"""

import asyncio
import time
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

from b_confident.integration import UncertaintyTransformersModel
from b_confident.distributed import (
    DistributedCalibrationManager,
    InMemoryMessageBroker,
    RedisMessageBroker
)
from b_confident.memory import MemoryConfig
from b_confident.core import PBAConfig


async def simulate_multi_node_calibration():
    """
    Simulate multiple inference nodes with distributed calibration.
    Shows how calibration parameters are synchronized across nodes.
    """
    print("=== Distributed Calibration Demo ===")
    print("Simulating 3 inference nodes with distributed calibration")

    # Create shared message broker (in production, this would be Redis)
    broker = InMemoryMessageBroker()

    # Create 3 nodes with different initial conditions
    nodes = []
    node_configs = [
        {"node_id": "node_1", "initial_beta": 0.4, "workload": "high_accuracy"},
        {"node_id": "node_2", "initial_beta": 0.6, "workload": "mixed"},
        {"node_id": "node_3", "initial_beta": 0.5, "workload": "high_uncertainty"}
    ]

    print(f"Creating {len(node_configs)} nodes with different initial conditions...")

    # Initialize calibration managers
    calibration_managers = []
    for config in node_configs:
        manager = DistributedCalibrationManager(
            node_id=config["node_id"],
            message_broker=broker,
            local_update_interval=5.0,  # Fast updates for demo
            global_sync_interval=15.0   # Fast sync for demo
        )

        # Set initial parameters
        manager.local_state.calibration_parameters["beta"] = config["initial_beta"]
        calibration_managers.append(manager)
        nodes.append(config)

        print(f"  {config['node_id']}: beta={config['initial_beta']}, workload={config['workload']}")

    # Start all calibration managers
    print("\nStarting distributed calibration managers...")
    start_tasks = [manager.start() for manager in calibration_managers]
    await asyncio.gather(*start_tasks)

    print("All nodes started. Beginning calibration simulation...")

    # Simulate different workloads generating uncertainty/accuracy pairs
    await simulate_workloads(calibration_managers, nodes)

    # Monitor calibration convergence
    await monitor_calibration_convergence(calibration_managers, duration=30)

    # Demonstrate failure scenarios
    await demonstrate_circuit_breaker(calibration_managers[0])

    # Cleanup
    print("\nStopping calibration managers...")
    stop_tasks = [manager.stop() for manager in calibration_managers]
    await asyncio.gather(*stop_tasks)

    print("Distributed calibration demo completed!")


async def simulate_workloads(managers: List[DistributedCalibrationManager],
                           node_configs: List[dict]):
    """Simulate different workloads on each node"""
    print("\nSimulating different workloads on each node...")

    # Define workload characteristics
    workload_patterns = {
        "high_accuracy": {"uncertainty_mean": 0.2, "uncertainty_std": 0.1, "accuracy_bias": 0.8},
        "mixed": {"uncertainty_mean": 0.4, "uncertainty_std": 0.15, "accuracy_bias": 0.6},
        "high_uncertainty": {"uncertainty_mean": 0.6, "uncertainty_std": 0.2, "accuracy_bias": 0.4}
    }

    # Generate samples for each node
    for manager, config in zip(managers, node_configs):
        workload_type = config["workload"]
        pattern = workload_patterns[workload_type]

        print(f"  Generating {workload_type} samples for {config['node_id']}...")

        # Generate 50 uncertainty/accuracy pairs
        np.random.seed(hash(config['node_id']) % 2**32)  # Consistent per node

        for _ in range(50):
            # Generate uncertainty
            uncertainty = np.random.normal(
                pattern["uncertainty_mean"],
                pattern["uncertainty_std"]
            )
            uncertainty = np.clip(uncertainty, 0, 1)

            # Generate correlated accuracy (higher uncertainty -> lower accuracy)
            accuracy_prob = pattern["accuracy_bias"] * (1 - uncertainty)
            accuracy = 1.0 if np.random.random() < accuracy_prob else 0.0

            # Update calibration
            manager.update_local_calibration(uncertainty, accuracy)

            # Small delay to simulate real-time processing
            await asyncio.sleep(0.01)


async def monitor_calibration_convergence(managers: List[DistributedCalibrationManager],
                                        duration: int = 30):
    """Monitor how calibration parameters converge across nodes"""
    print(f"\nMonitoring calibration convergence for {duration} seconds...")

    start_time = time.time()
    check_interval = 5.0

    while (time.time() - start_time) < duration:
        print(f"\n--- Calibration State at {time.time() - start_time:.1f}s ---")

        # Show each node's state
        for manager in managers:
            stats = manager.get_calibration_stats()
            local_params = stats["local_state"]["parameters"]
            global_params = stats["global_state"]["parameters"]

            print(f"{stats['node_id']}:")
            print(f"  Local:  beta={local_params.get('beta', 0.5):.3f}, "
                  f"temp={local_params.get('temperature', 1.0):.3f}")
            print(f"  Global: beta={global_params.get('beta', 0.5):.3f}, "
                  f"temp={global_params.get('temperature', 1.0):.3f}")
            print(f"  Quality: {stats['local_state']['calibration_quality']:.3f}, "
                  f"Samples: {stats['local_state']['sample_count']}")

        # Calculate consensus level
        global_betas = []
        for manager in managers:
            params = manager.get_calibration_parameters(prefer_global=True)
            global_betas.append(params.get("beta", 0.5))

        if global_betas:
            beta_std = np.std(global_betas)
            print(f"\nConsensus Level: β std={beta_std:.4f} "
                  f"({'HIGH' if beta_std < 0.01 else 'MEDIUM' if beta_std < 0.05 else 'LOW'} consensus)")

        await asyncio.sleep(check_interval)


async def demonstrate_circuit_breaker(manager: DistributedCalibrationManager):
    """Demonstrate circuit breaker functionality during sync failures"""
    print(f"\n--- Circuit Breaker Demonstration ---")

    original_broker = manager.broker

    # Create a failing broker to trigger circuit breaker
    class FailingBroker:
        async def publish(self, topic, event):
            raise Exception("Network failure simulation")

        async def subscribe(self, topic, callback):
            raise Exception("Network failure simulation")

        async def get_subscribers_count(self, topic):
            return 0

    print(f"Simulating network failures for {manager.node_id}...")

    # Replace broker with failing one
    manager.broker = FailingBroker()

    # Try to trigger sync operations that will fail
    for i in range(6):  # Exceed failure threshold
        try:
            await manager._perform_global_sync()
        except Exception as e:
            print(f"  Sync attempt {i+1} failed: Circuit breaker state = {manager.circuit_breaker.get_state()}")
            await asyncio.sleep(0.5)

    # Restore original broker
    manager.broker = original_broker

    print(f"Network restored. Circuit breaker state: {manager.circuit_breaker.get_state()}")
    print("Circuit breaker provides graceful degradation during network issues")


async def demonstrate_high_throughput_scenario():
    """
    Demonstrate distributed calibration integrated with high-throughput inference.
    Shows the complete production-ready system.
    """
    print("\n=== High-Throughput + Distributed Calibration Demo ===")

    # Load a small model for demo
    model_name = "gpt2"
    print(f"Loading {model_name} with distributed calibration enabled...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create uncertainty model with both streaming and distributed calibration
    uncertainty_model = UncertaintyTransformersModel(
        model=model,
        tokenizer=tokenizer,
        memory_config=MemoryConfig(max_memory_usage_gb=0.5),
        enable_streaming=True,
        enable_distributed_calibration=True,
        node_id="production_node_1"
    )

    # Start distributed calibration
    await uncertainty_model.start_distributed_calibration()

    # Process batch with distributed calibration
    test_inputs = [
        "Artificial intelligence is",
        "Machine learning algorithms",
        "Deep neural networks",
        "Natural language processing"
    ]

    print(f"Processing {len(test_inputs)} inputs with integrated calibration...")

    results = uncertainty_model.uncertainty_generate_batch_streaming(
        test_inputs,
        max_length=20
    )

    print(f"\nResults with Distributed Calibration:")
    for i, result in enumerate(results):
        raw_uncertainty = result.uncertainty_scores[0]

        # Simulate having accuracy feedback for calibration
        simulated_accuracy = 1.0 if raw_uncertainty < 0.5 else 0.0

        # Update distributed calibration
        uncertainty_model.update_distributed_calibration(raw_uncertainty, simulated_accuracy)

        # Get calibrated uncertainty
        calibrated_uncertainty = uncertainty_model.get_calibrated_uncertainty(raw_uncertainty)

        print(f"  Input: '{test_inputs[i]}'")
        print(f"  Raw uncertainty: {raw_uncertainty:.3f}")
        print(f"  Calibrated uncertainty: {calibrated_uncertainty:.3f}")
        print(f"  Streaming enabled: {result.metadata.get('memory_managed', False)}")

    # Show calibration stats
    calibration_stats = uncertainty_model.get_distributed_calibration_stats()
    if calibration_stats:
        print(f"\nDistributed Calibration Statistics:")
        print(f"  Node ID: {calibration_stats['node_id']}")
        print(f"  Calibration Quality: {calibration_stats['local_state']['calibration_quality']:.3f}")
        print(f"  Sample Count: {calibration_stats['local_state']['sample_count']}")
        print(f"  Circuit Breaker: {calibration_stats['circuit_breaker_state']}")

    # Cleanup
    await uncertainty_model.stop_distributed_calibration()
    uncertainty_model.cleanup_memory()

    print("High-throughput + distributed calibration demo completed!")


if __name__ == "__main__":
    print("Starting Distributed Calibration Demonstration...")
    print("This shows how the system coordinates calibration across inference nodes.")

    async def main():
        await simulate_multi_node_calibration()
        await demonstrate_high_throughput_scenario()

        print("\n=== Demo Complete ===")
        print("Key Benefits Demonstrated:")
        print("[OK] Eventually consistent calibration across nodes")
        print("[OK] Event-driven updates without synchronization bottlenecks")
        print("[OK] Hierarchical local + global calibration")
        print("[OK] Circuit breaker pattern for fault tolerance")
        print("[OK] Integration with streaming memory management")
        print("\nThe distributed calibration system is ready for multi-node production deployment!")

    asyncio.run(main())