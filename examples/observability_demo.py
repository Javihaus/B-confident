"""
Uncertainty Debugging & Observability Demo

Demonstrates the comprehensive debugging and monitoring system that addresses
the opacity of uncertainty calculation pipelines, enabling easy troubleshooting
when calibration drifts from expected behavior.
"""

import torch
import numpy as np
import time
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM

from b_confident.observability import (
    InstrumentedUncertaintyCalculator,
    DebugLevel,
    UncertaintyMetricsCollector,
    MetricsAggregator,
    AlertManager,
    AlertSeverity,
    UncertaintyDashboard,
    create_uncertainty_dashboard
)
from b_confident.observability.metrics_collector import create_standard_metrics_setup
from b_confident.core import PBAConfig


def demonstrate_instrumented_calculation():
    """Show detailed instrumented uncertainty calculation with debugging"""
    print("=== Instrumented Uncertainty Calculation Demo ===")

    # Create instrumented calculator with detailed debugging
    instrumented_calc = InstrumentedUncertaintyCalculator(
        pba_config=PBAConfig(beta=0.5),
        debug_level=DebugLevel.DETAILED,
        enable_provenance=True,
        enable_spc=True
    )

    # Simulate different types of logits that could cause issues
    test_cases = [
        {
            "name": "Normal case",
            "logits": torch.tensor([1.0, 2.0, 0.5, -1.0, 0.0]),
            "token_id": 1
        },
        {
            "name": "High entropy (uncertain)",
            "logits": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            "token_id": 2
        },
        {
            "name": "Very confident prediction",
            "logits": torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]),
            "token_id": 0
        },
        {
            "name": "Large logit range (numerical issues)",
            "logits": torch.tensor([50.0, -50.0, 0.0, 25.0, -25.0]),
            "token_id": 0
        },
        {
            "name": "Edge case - small differences",
            "logits": torch.tensor([0.001, 0.002, 0.001, 0.001, 0.001]),
            "token_id": 1
        }
    ]

    print(f"Testing {len(test_cases)} different scenarios with full pipeline observability:\n")

    for i, test_case in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {test_case['name']} ---")

        uncertainty, provenance = instrumented_calc.calculate_uncertainty_with_debugging(
            test_case["logits"],
            test_case["token_id"]
        )

        print(f"Final Uncertainty: {uncertainty:.4f}")
        print(f"Total Execution Time: {provenance.total_execution_time:.6f}s")
        print(f"Pipeline Stages: {len(provenance.stage_metrics)}")

        # Show stage-by-stage breakdown
        for stage_metrics in provenance.stage_metrics:
            print(f"  {stage_metrics.stage.value}: {stage_metrics.execution_time:.6f}s")

            if stage_metrics.warnings:
                print(f"    Warnings: {stage_metrics.warnings}")

            if stage_metrics.errors:
                print(f"    Errors: {stage_metrics.errors}")

            # Show key intermediate values
            interesting_keys = ["perplexity", "token_probability", "uncertainty", "temperature_applied"]
            for key in interesting_keys:
                if key in stage_metrics.intermediate_values:
                    value = stage_metrics.intermediate_values[key]
                    print(f"    {key}: {value}")

        print()

    # Generate detailed debug report for the most interesting case
    print("=== Detailed Debug Report for Edge Case ===")
    _, edge_case_provenance = instrumented_calc.calculate_uncertainty_with_debugging(
        torch.tensor([0.001, 0.002, 0.001, 0.001, 0.001]),
        1
    )

    debug_report = instrumented_calc.generate_debug_report(edge_case_provenance)
    print(debug_report)

    return instrumented_calc


def demonstrate_statistical_process_control(instrumented_calc):
    """Show statistical process control for drift detection"""
    print("\n=== Statistical Process Control Demo ===")
    print("Simulating uncertainty calculations with gradual drift...")

    # Simulate normal operations followed by gradual drift
    base_logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5])

    # Phase 1: Normal operations (50 samples)
    print("Phase 1: Normal operations...")
    for i in range(50):
        # Add small random noise
        noise = torch.randn(5) * 0.1
        test_logits = base_logits + noise

        uncertainty, _ = instrumented_calc.calculate_uncertainty_with_debugging(
            test_logits, 0
        )

    # Phase 2: Gradual drift (30 samples with increasing bias)
    print("Phase 2: Introducing gradual drift...")
    for i in range(30):
        # Add increasing bias to simulate model drift
        drift = torch.tensor([0.0, 0.0, 0.0, 0.0, i * 0.2])  # Increasing bias on last element
        noise = torch.randn(5) * 0.1
        test_logits = base_logits + drift + noise

        uncertainty, provenance = instrumented_calc.calculate_uncertainty_with_debugging(
            test_logits, 0
        )

        # Check for SPC warnings
        for stage_metrics in provenance.stage_metrics:
            if stage_metrics.warnings:
                print(f"  SPC Warning at sample {i+51}: {stage_metrics.warnings}")

    # Phase 3: Sudden anomaly
    print("Phase 3: Sudden anomaly...")
    anomaly_logits = torch.tensor([100.0, -100.0, 50.0, -50.0, 0.0])  # Extreme values
    uncertainty, provenance = instrumented_calc.calculate_uncertainty_with_debugging(
        anomaly_logits, 0
    )

    print(f"Anomaly detected! Uncertainty: {uncertainty:.4f}")
    for stage_metrics in provenance.stage_metrics:
        if stage_metrics.warnings:
            print(f"  Stage {stage_metrics.stage.value}: {stage_metrics.warnings}")

    # Show control chart data
    if instrumented_calc.spc:
        from b_confident.observability.uncertainty_debugger import PipelineStage

        perplexity_data = instrumented_calc.spc.get_control_chart_data(
            PipelineStage.PERPLEXITY_CALCULATION, "perplexity"
        )

        if perplexity_data["values"]:
            print(f"\nPerplexity Control Chart Summary:")
            print(f"  Sample Count: {perplexity_data['sample_count']}")
            print(f"  Current Value: {perplexity_data['current_value']:.4f}")

            if perplexity_data["limits"]:
                limits = perplexity_data["limits"]
                print(f"  Control Limits: [{limits.get('lower_control_limit', 'N/A'):.4f}, {limits.get('upper_control_limit', 'N/A'):.4f}]")
                print(f"  Warning Limits: [{limits.get('lower_warning_limit', 'N/A'):.4f}, {limits.get('upper_warning_limit', 'N/A'):.4f}]")


def demonstrate_metrics_and_alerting():
    """Show metrics collection and alerting system"""
    print("\n=== Metrics Collection and Alerting Demo ===")

    # Create standard metrics setup
    aggregator, alert_manager = create_standard_metrics_setup()

    # Create multiple collectors (simulating different nodes/processes)
    collectors = {}
    for node_id in ["node_1", "node_2", "node_3"]:
        collector = UncertaintyMetricsCollector(max_history=1000)
        collectors[node_id] = collector
        aggregator.add_collector(node_id, collector)

    print(f"Created {len(collectors)} metrics collectors...")

    # Add custom alert rules
    alert_manager.add_alert_rule(
        metric_name="uncertainty",
        threshold=0.8,
        severity=AlertSeverity.WARNING,
        comparison="greater",
        cooldown_seconds=10.0  # Short cooldown for demo
    )

    alert_manager.add_alert_rule(
        metric_name="execution_time",
        threshold=0.01,  # 10ms threshold for demo
        severity=AlertSeverity.INFO,
        comparison="greater",
        cooldown_seconds=5.0
    )

    # Simulate metrics from different nodes
    print("Simulating metrics from multiple nodes...")

    for i in range(100):
        for node_id, collector in collectors.items():
            # Each node has different characteristics
            if node_id == "node_1":
                # Normal node
                uncertainty = np.random.normal(0.3, 0.1)
                exec_time = np.random.normal(0.005, 0.002)
            elif node_id == "node_2":
                # Occasionally high uncertainty
                uncertainty = np.random.normal(0.4, 0.2)
                exec_time = np.random.normal(0.007, 0.003)
            else:
                # Node with gradually increasing uncertainty (drift)
                uncertainty = np.random.normal(0.3 + i * 0.005, 0.1)
                exec_time = np.random.normal(0.006, 0.002)

            # Clamp values to reasonable ranges
            uncertainty = max(0.0, min(1.0, uncertainty))
            exec_time = max(0.001, exec_time)

            collector.record_metric("uncertainty", uncertainty)
            collector.record_metric("execution_time", exec_time)
            collector.record_metric("calibration_quality", np.random.normal(0.1, 0.05))

        # Check for alerts every 10 iterations
        if i % 10 == 0:
            alerts = alert_manager.check_metrics(aggregator)
            if alerts:
                print(f"  Generated {len(alerts)} alerts at iteration {i}")

        time.sleep(0.01)  # Small delay to simulate real-time processing

    # Show aggregated metrics
    print("\nAggregated Metrics Summary (last 30 seconds):")
    for metric_name in ["uncertainty", "execution_time", "calibration_quality"]:
        summary = aggregator.get_aggregated_summary(metric_name, window_seconds=30)
        if summary:
            print(f"  {metric_name}:")
            print(f"    Count: {summary.count}, Mean: {summary.mean:.4f}")
            print(f"    P95: {summary.p95:.4f}, Max: {summary.max:.4f}")

    # Show active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"\nActive Alerts: {len(active_alerts)}")
    for alert in active_alerts[:5]:  # Show first 5
        print(f"  [{alert.severity.value}] {alert.title}")
        print(f"    Value: {alert.metric_value:.4f}, Threshold: {alert.threshold:.4f}")

    return aggregator, alert_manager


def demonstrate_dashboard(instrumented_calc, aggregator, alert_manager):
    """Show dashboard functionality"""
    print("\n=== Dashboard Demo ===")

    # Create dashboard
    dashboard = create_uncertainty_dashboard(
        metrics_aggregator=aggregator,
        alert_manager=alert_manager,
        instrumented_calculator=instrumented_calc
    )

    print("Dashboard created with real-time monitoring capabilities...")

    # Get dashboard data
    dashboard_data = dashboard.get_dashboard_data()

    print(f"Dashboard Status: {dashboard_data['status']}")
    print(f"Total Metrics: {len(dashboard_data['metrics'])}")
    print(f"Active Alerts: {dashboard_data['alerts']['total_active']}")
    print(f"Recent Calculations: {len(dashboard_data['recent_calculations'])}")

    # Show pipeline health
    pipeline_health = dashboard_data["pipeline_health"]
    if "stage_health" in pipeline_health:
        print(f"Pipeline Health Status: {pipeline_health.get('overall_status', 'unknown')}")

        for stage_name, health in pipeline_health["stage_health"].items():
            status = health["status"]
            issues = len(health["issues"])
            print(f"  {stage_name}: {status} ({issues} issues)")

    # Generate simple HTML dashboard
    print("\nGenerating HTML dashboard...")
    html_content = dashboard.generate_html_dashboard()

    # Save to file for viewing
    with open("/tmp/uncertainty_dashboard.html", "w") as f:
        f.write(html_content)

    print("HTML dashboard saved to /tmp/uncertainty_dashboard.html")

    # Show specific calculation details
    if dashboard_data["recent_calculations"]:
        latest_calc = dashboard_data["recent_calculations"][0]
        print(f"\nLatest Calculation Details:")
        print(f"  Request ID: {latest_calc['request_id']}")
        print(f"  Uncertainty: {latest_calc['final_uncertainty']:.4f}")
        print(f"  Total Time: {latest_calc['total_time']:.6f}s")
        print(f"  Warnings: {latest_calc['warnings_count']}")

        # Get detailed calculation info
        calc_details = dashboard.get_calculation_details(latest_calc['request_id'])
        if calc_details:
            print("  Stage Execution Times:")
            for stage_name, stage_time in latest_calc['stage_times'].items():
                print(f"    {stage_name}: {stage_time:.6f}s")

    return dashboard


def demonstrate_real_model_integration():
    """Show integration with real transformer model"""
    print("\n=== Real Model Integration Demo ===")

    # Load a small model
    model_name = "gpt2"
    print(f"Loading {model_name} for real-world uncertainty debugging...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create instrumented calculator
    instrumented_calc = InstrumentedUncertaintyCalculator(
        debug_level=DebugLevel.STANDARD,
        enable_provenance=True,
        enable_spc=True
    )

    # Test with real model outputs
    test_texts = [
        "The capital of France is",
        "Machine learning is a",
        "The weather today seems",
        "In quantum physics, superposition"
    ]

    print("Processing real model outputs with full observability...")

    for text in test_texts:
        # Get model logits
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # Last position

        # Calculate uncertainty with debugging
        uncertainty, provenance = instrumented_calc.calculate_uncertainty_with_debugging(logits)

        print(f"\nText: '{text}'")
        print(f"Uncertainty: {uncertainty:.4f}")
        print(f"Processing time: {provenance.total_execution_time:.6f}s")

        # Show any warnings or issues
        total_warnings = sum(len(stage.warnings) for stage in provenance.stage_metrics)
        if total_warnings > 0:
            print(f"Warnings detected: {total_warnings}")

        # Show key intermediate values
        for stage_metrics in provenance.stage_metrics:
            if stage_metrics.stage.value == "perplexity_calculation":
                perplexity = stage_metrics.intermediate_values.get("clamped_perplexity", "N/A")
                token_prob = stage_metrics.intermediate_values.get("token_probability", "N/A")
                print(f"Token probability: {token_prob}, Perplexity: {perplexity}")
                break

    print("Real model integration completed successfully!")


def main():
    """Run complete observability demonstration"""
    print("Starting Comprehensive Observability & Debugging Demo")
    print("This demonstrates the complete solution to uncertainty calculation opacity")
    print("="*80)

    # 1. Show instrumented calculation with detailed debugging
    instrumented_calc = demonstrate_instrumented_calculation()

    # 2. Show statistical process control for drift detection
    demonstrate_statistical_process_control(instrumented_calc)

    # 3. Show metrics collection and alerting
    aggregator, alert_manager = demonstrate_metrics_and_alerting()

    # 4. Show dashboard functionality
    dashboard = demonstrate_dashboard(instrumented_calc, aggregator, alert_manager)

    # 5. Show real model integration
    demonstrate_real_model_integration()

    print("\n" + "="*80)
    print("=== Demo Complete ===")
    print("Key Capabilities Demonstrated:")
    print("[OK] Instrumented pipeline with component-level metrics")
    print("[OK] Uncertainty provenance tracking through all transformation steps")
    print("[OK] Statistical process control for automatic drift detection")
    print("[OK] Real-time metrics collection and aggregation")
    print("[OK] Automated alerting system with configurable thresholds")
    print("[OK] Web-based dashboard for monitoring and debugging")
    print("[OK] Integration with real transformer models")
    print("[OK] Comprehensive debug reports for troubleshooting")

    print("\nThe observability framework transforms the 'black box' uncertainty")
    print("calculation into a fully transparent, debuggable, and monitorable pipeline!")


if __name__ == "__main__":
    main()