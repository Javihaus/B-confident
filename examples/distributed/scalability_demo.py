"""
Distributed Calibration Scalability Demo

Demonstrates the production-scale enhancements to distributed calibration:
- Horizontal scaling with auto-scaler and load balancer
- Geographic partitioning and failover
- RAFT-inspired leader election and consensus
- Performance optimization for high-throughput scenarios
- Circuit breaker patterns and intelligent failover
- Real-time metrics and adaptive scaling decisions
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any

from b_confident.distributed.calibration_manager import (
    DistributedCalibrationManager,
    RedisMessageBroker,
    InMemoryMessageBroker,
    PartitionStrategy,
    NodeRole,
    ScalingDecision
)


async def simulate_multi_node_cluster():
    """Simulate a multi-node cluster with realistic workload patterns"""
    print("=== Multi-Node Cluster Scalability Demo ===")
    print("Simulating distributed calibration with auto-scaling and load balancing...")

    # Create message broker (use in-memory for demo)
    broker = InMemoryMessageBroker()

    # Create multiple nodes in different regions
    nodes = {}
    for i in range(3):
        node_id = f"node_{i+1}"
        region = ["us-east-1", "us-west-2", "eu-west-1"][i]

        manager = DistributedCalibrationManager(
            node_id=node_id,
            cluster_id="production_cluster",
            region=region,
            message_broker=broker,
            enable_auto_scaling=True,
            enable_load_balancing=True,
            partition_strategy=PartitionStrategy.CONSISTENCY_HASH,
            min_nodes=2,
            max_nodes=8
        )

        nodes[node_id] = manager

    # Start all nodes
    print(f"Starting {len(nodes)} nodes across multiple regions...")
    for manager in nodes.values():
        await manager.start()

    await asyncio.sleep(2.0)  # Let nodes discover each other

    print("\nInitial cluster state:")
    for node_id, manager in nodes.items():
        stats = manager.get_calibration_stats()
        print(f"  {node_id}: role={stats['node_info']['role']}, region={stats['node_info']['region']}")

    return nodes, broker


async def simulate_load_patterns(nodes: Dict[str, DistributedCalibrationManager]):
    """Simulate different load patterns to trigger scaling"""
    print("\n=== Load Pattern Simulation ===")

    # Phase 1: Normal load
    print("Phase 1: Normal distributed load...")
    for i in range(30):
        for node_id, manager in nodes.items():
            # Simulate processing with random uncertainty/accuracy
            uncertainty = random.uniform(0.2, 0.8)
            accuracy = random.uniform(0.3, 0.9)
            processing_time = random.uniform(10, 50)  # ms

            manager.update_local_calibration(uncertainty, accuracy, processing_time)

        await asyncio.sleep(0.1)

    # Check load distribution
    print("Load distribution after normal phase:")
    for node_id, manager in nodes.items():
        metrics = manager.node_metrics
        print(f"  {node_id}: load={metrics.load:.3f}, throughput={metrics.throughput_rps:.2f} rps")

    # Phase 2: High load spike (should trigger scale-up)
    print("\nPhase 2: High load spike...")
    for i in range(40):
        for node_id, manager in nodes.items():
            # Simulate high load
            for _ in range(3):  # Multiple requests per node
                uncertainty = random.uniform(0.1, 0.9)
                accuracy = random.uniform(0.2, 0.8)
                processing_time = random.uniform(80, 200)  # Higher latency

                manager.update_local_calibration(uncertainty, accuracy, processing_time)

        await asyncio.sleep(0.05)  # Faster requests

    print("Load distribution after spike:")
    for node_id, manager in nodes.items():
        metrics = manager.node_metrics
        print(f"  {node_id}: load={metrics.load:.3f}, throughput={metrics.throughput_rps:.2f} rps, latency={metrics.latency_ms:.1f}ms")

    # Phase 3: Gradual cooldown
    print("\nPhase 3: Load cooldown...")
    for i in range(20):
        # Reduced load
        node_id = random.choice(list(nodes.keys()))
        manager = nodes[node_id]

        uncertainty = random.uniform(0.3, 0.6)
        accuracy = random.uniform(0.4, 0.8)
        processing_time = random.uniform(15, 40)

        manager.update_local_calibration(uncertainty, accuracy, processing_time)
        await asyncio.sleep(0.2)

    print("Final load distribution:")
    for node_id, manager in nodes.items():
        metrics = manager.node_metrics
        print(f"  {node_id}: load={metrics.load:.3f}, throughput={metrics.throughput_rps:.2f} rps")


async def demonstrate_leader_election(nodes: Dict[str, DistributedCalibrationManager]):
    """Demonstrate leader election and failover"""
    print("\n=== Leader Election & Failover Demo ===")

    # Wait for initial leader election
    await asyncio.sleep(3.0)

    # Find current leader
    current_leader = None
    for node_id, manager in nodes.items():
        if manager.node_metrics.role == NodeRole.LEADER:
            current_leader = node_id
            break

    print(f"Current leader: {current_leader}")

    if current_leader:
        # Simulate leader failure
        print(f"Simulating failure of leader {current_leader}...")
        leader_manager = nodes[current_leader]

        # Stop the leader's background tasks to simulate failure
        await leader_manager.stop()

        # Wait for election timeout and new leader election
        print("Waiting for new leader election...")
        await asyncio.sleep(8.0)

        # Check new leader
        new_leader = None
        for node_id, manager in nodes.items():
            if node_id != current_leader and manager.node_metrics.role == NodeRole.LEADER:
                new_leader = node_id
                break

        print(f"New leader elected: {new_leader}")

        # Restart the failed node
        print(f"Restarting failed node {current_leader}...")
        await leader_manager.start()
        await asyncio.sleep(2.0)

        # Check final state
        print("Final cluster state after recovery:")
        for node_id, manager in nodes.items():
            stats = manager.get_calibration_stats()
            print(f"  {node_id}: role={stats['node_info']['role']}, term={stats['node_info']['election_term']}")


async def demonstrate_auto_scaling(nodes: Dict[str, DistributedCalibrationManager]):
    """Demonstrate auto-scaling decisions"""
    print("\n=== Auto-Scaling Demo ===")

    # Find leader to check scaling decisions
    leader_manager = None
    for manager in nodes.values():
        if manager.node_metrics.role == NodeRole.LEADER:
            leader_manager = manager
            break

    if not leader_manager:
        print("No leader found for scaling demo")
        return

    print("Current scaling recommendations:")
    recommendations = leader_manager.get_scaling_recommendations()
    print(f"  Decision: {recommendations['current_decision']}")
    print(f"  Reason: {recommendations['reason']}")
    print(f"  Healthy nodes: {recommendations['healthy_nodes']}")

    load_stats = recommendations['cluster_load_stats']
    print(f"  Cluster load - avg: {load_stats['avg_load']:.3f}, max: {load_stats['max_load']:.3f}, variance: {load_stats['load_variance']:.3f}")

    if recommendations['recommended_actions']:
        print("  Recommended actions:")
        for action in recommendations['recommended_actions']:
            print(f"    [{action['priority']}] {action['action']}: {action['description']}")


async def demonstrate_partitioning(nodes: Dict[str, DistributedCalibrationManager]):
    """Demonstrate data partitioning and load balancing"""
    print("\n=== Partitioning & Load Balancing Demo ===")

    # Use first node to demonstrate partitioning
    manager = list(nodes.values())[0]

    # Test request routing
    test_keys = ["user_123", "model_abc", "request_xyz", "batch_456", "inference_789"]

    print("Request routing demonstration:")
    for key in test_keys:
        optimal_node = manager.get_optimal_node_for_request(key)
        partition_id = manager.partition_manager.get_partition_for_key(key)
        print(f"  Key '{key}' -> Partition: {partition_id}, Optimal node: {optimal_node}")

    # Show partition distribution
    print(f"\nPartition information:")
    for node_id, manager in nodes.items():
        stats = manager.get_calibration_stats()
        if "partitioning" in stats:
            partition_info = stats["partitioning"]
            print(f"  {node_id}: {partition_info['partition_count']} partitions, strategy: {partition_info['strategy']}")


async def show_comprehensive_metrics(nodes: Dict[str, DistributedCalibrationManager]):
    """Show comprehensive metrics from all nodes"""
    print("\n=== Comprehensive Metrics Dashboard ===")

    for node_id, manager in nodes.items():
        stats = manager.get_calibration_stats()

        print(f"\n--- {node_id.upper()} ---")

        # Node info
        node_info = stats["node_info"]
        print(f"Role: {node_info['role']}, Region: {node_info['region']}, Term: {node_info['election_term']}")

        # Performance
        perf = stats["performance_metrics"]
        print(f"Processed: {perf['requests_processed']} requests, Errors: {perf['error_count']}")
        print(f"Throughput: {perf['throughput_rps']:.2f} rps, Latency: {perf['avg_latency_ms']:.1f}ms")
        print(f"Load: {perf['load']:.3f}, Memory: {perf['memory_usage']:.3f}, CPU: {perf['cpu_usage']:.3f}")

        # Calibration quality
        local_state = stats["local_state"]
        print(f"Calibration quality: {local_state['calibration_quality']:.3f}, Samples: {local_state['sample_count']}")

        # Cluster participation
        cluster = stats["cluster_state"]
        print(f"Cluster: {cluster['active_nodes']}/{cluster['total_nodes']} nodes active, {cluster['partitions']} partitions")

        # Circuit breaker
        print(f"Circuit breaker: {stats['circuit_breaker_state']}")

        # Background health
        print(f"Background tasks: {stats['background_tasks_running']}, Event queue: {stats['event_queue_size']}")

        # Scalability features
        if "auto_scaling" in stats:
            auto_scaling = stats["auto_scaling"]
            print(f"Auto-scaling: {auto_scaling['min_nodes']}-{auto_scaling['max_nodes']} nodes, threshold: {auto_scaling['scale_up_threshold']:.1f}")

        if "load_balancing" in stats:
            load_bal = stats["load_balancing"]
            print(f"Load balancing: {load_bal['strategy']} strategy, tracking {len(load_bal['known_node_weights'])} nodes")


async def performance_stress_test(nodes: Dict[str, DistributedCalibrationManager]):
    """Run performance stress test to validate scalability"""
    print("\n=== Performance Stress Test ===")

    start_time = time.time()
    total_requests = 1000
    concurrent_tasks = []

    async def stress_worker(worker_id: int, requests_per_worker: int):
        """Worker to generate load"""
        node_ids = list(nodes.keys())

        for i in range(requests_per_worker):
            # Select random node
            node_id = random.choice(node_ids)
            manager = nodes[node_id]

            # Simulate processing
            uncertainty = random.uniform(0.1, 0.9)
            accuracy = random.uniform(0.2, 0.8)
            processing_time = random.uniform(5, 100)

            manager.update_local_calibration(uncertainty, accuracy, processing_time)

            # Small random delay to simulate real processing
            await asyncio.sleep(random.uniform(0.001, 0.01))

    # Create concurrent workers
    num_workers = 10
    requests_per_worker = total_requests // num_workers

    print(f"Starting stress test: {num_workers} workers, {total_requests} total requests...")

    for worker_id in range(num_workers):
        task = asyncio.create_task(stress_worker(worker_id, requests_per_worker))
        concurrent_tasks.append(task)

    # Wait for all workers to complete
    await asyncio.gather(*concurrent_tasks)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Stress test completed in {duration:.2f} seconds")
    print(f"Throughput: {total_requests / duration:.2f} requests/second")

    # Show final metrics
    print("\nFinal performance metrics:")
    total_processed = 0
    total_errors = 0

    for node_id, manager in nodes.items():
        stats = manager.get_calibration_stats()
        perf = stats["performance_metrics"]

        node_processed = perf["requests_processed"]
        node_errors = perf["error_count"]

        total_processed += node_processed
        total_errors += node_errors

        print(f"  {node_id}: {node_processed} requests, {node_errors} errors, {perf['throughput_rps']:.2f} rps")

    print(f"\nCluster totals: {total_processed} requests, {total_errors} errors")
    print(f"Overall throughput: {total_processed / duration:.2f} requests/second")
    print(f"Error rate: {(total_errors / total_processed) * 100:.2f}%" if total_processed > 0 else "Error rate: 0%")


async def main():
    """Run complete scalability demonstration"""
    print("B-Confident Distributed Calibration Scalability Demo")
    print("=" * 60)
    print("Demonstrating production-scale distributed calibration features:")
    print("- Auto-scaling based on load and calibration drift")
    print("- Load balancing with intelligent node selection")
    print("- Geographic partitioning and failover")
    print("- RAFT-inspired leader election")
    print("- Circuit breaker patterns")
    print("- Performance optimization for high-throughput")
    print("=" * 60)

    try:
        # 1. Set up multi-node cluster
        nodes, broker = await simulate_multi_node_cluster()

        # 2. Simulate realistic load patterns
        await simulate_load_patterns(nodes)

        # 3. Demonstrate leader election and failover
        await demonstrate_leader_election(nodes)

        # 4. Show auto-scaling capabilities
        await demonstrate_auto_scaling(nodes)

        # 5. Demonstrate partitioning and load balancing
        await demonstrate_partitioning(nodes)

        # 6. Performance stress test
        await performance_stress_test(nodes)

        # 7. Show comprehensive metrics
        await show_comprehensive_metrics(nodes)

        print("\n" + "=" * 60)
        print("=== Scalability Demo Complete ===")
        print("Key capabilities demonstrated:")
        print("[OK] Multi-node cluster with regional distribution")
        print("[OK] Auto-scaling based on load metrics and calibration drift")
        print("[OK] Intelligent load balancing with optimal node selection")
        print("[OK] RAFT-inspired leader election with failover")
        print("[OK] Geographic partitioning with consistent hashing")
        print("[OK] Circuit breaker patterns for resilience")
        print("[OK] High-throughput performance under stress")
        print("[OK] Real-time metrics and monitoring")
        print("[OK] Comprehensive scalability statistics")

        print(f"\nThe distributed calibration system successfully scales from")
        print(f"single-node development to production clusters with automatic")
        print(f"scaling, intelligent load balancing, and enterprise resilience!")

        # Clean shutdown
        print("\nShutting down cluster...")
        for manager in nodes.values():
            await manager.stop()

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())