"""
Real-time Uncertainty Dashboard

Provides web-based dashboard for monitoring uncertainty calculation pipelines,
viewing metrics, and debugging calibration issues in real-time.
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import logging

from .metrics_collector import MetricsAggregator, AlertManager, AlertSeverity
from .uncertainty_debugger import InstrumentedUncertaintyCalculator, PipelineStage

logger = logging.getLogger(__name__)


class UncertaintyDashboard:
    """
    Web-based dashboard for uncertainty monitoring and debugging.

    Provides real-time metrics visualization, pipeline debugging,
    and alert management interface.
    """

    def __init__(self,
                 metrics_aggregator: MetricsAggregator,
                 alert_manager: AlertManager,
                 instrumented_calculator: Optional[InstrumentedUncertaintyCalculator] = None):

        self.metrics_aggregator = metrics_aggregator
        self.alert_manager = alert_manager
        self.instrumented_calculator = instrumented_calculator

        self.dashboard_data = {
            "status": "running",
            "last_updated": time.time(),
            "metrics": {},
            "alerts": {},
            "pipeline_health": {},
            "recent_calculations": []
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        self.dashboard_data["last_updated"] = time.time()

        # Update metrics
        self.dashboard_data["metrics"] = self._get_metrics_data()

        # Update alerts
        self.dashboard_data["alerts"] = self._get_alerts_data()

        # Update pipeline health
        self.dashboard_data["pipeline_health"] = self._get_pipeline_health_data()

        # Update recent calculations
        if self.instrumented_calculator:
            self.dashboard_data["recent_calculations"] = self._get_recent_calculations()

        return self.dashboard_data

    def _get_metrics_data(self) -> Dict[str, Any]:
        """Get metrics data for dashboard"""
        metrics_data = {}

        # Get all metric names
        metric_names = self.metrics_aggregator.get_all_metric_names()

        # Time windows for different views
        time_windows = {
            "1min": 60,
            "5min": 300,
            "1hour": 3600,
            "24hour": 86400
        }

        for metric_name in metric_names:
            metrics_data[metric_name] = {}

            for window_name, window_seconds in time_windows.items():
                summary = self.metrics_aggregator.get_aggregated_summary(
                    metric_name, window_seconds
                )

                if summary:
                    metrics_data[metric_name][window_name] = {
                        "count": summary.count,
                        "mean": summary.mean,
                        "std": summary.std,
                        "min": summary.min,
                        "max": summary.max,
                        "p50": summary.p50,
                        "p95": summary.p95,
                        "p99": summary.p99
                    }

        return metrics_data

    def _get_alerts_data(self) -> Dict[str, Any]:
        """Get alerts data for dashboard"""
        active_alerts = self.alert_manager.get_active_alerts()
        alert_summary = self.alert_manager.get_alert_summary()

        # Format alerts for display
        formatted_alerts = []
        for alert in active_alerts[:20]:  # Show latest 20
            formatted_alerts.append({
                "id": alert.id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "time_ago": self._time_ago(alert.timestamp)
            })

        return {
            "summary": alert_summary,
            "active_alerts": formatted_alerts,
            "total_active": len(active_alerts)
        }

    def _get_pipeline_health_data(self) -> Dict[str, Any]:
        """Get pipeline health indicators"""
        if not self.instrumented_calculator:
            return {"status": "no_instrumentation"}

        pipeline_metrics = self.instrumented_calculator.get_pipeline_metrics()

        health_indicators = {}
        overall_status = "healthy"

        for stage_name, stage_data in pipeline_metrics.items():
            stage_health = {"status": "healthy", "issues": []}

            # Check execution time
            if "execution_time" in stage_data:
                exec_time_data = stage_data["execution_time"]
                if exec_time_data.get("values"):
                    recent_avg = sum(exec_time_data["values"][-10:]) / min(10, len(exec_time_data["values"]))

                    if recent_avg > 0.5:  # 500ms threshold
                        stage_health["status"] = "warning"
                        stage_health["issues"].append(f"High execution time: {recent_avg:.3f}s")
                        overall_status = "warning"

            # Check for control limit violations
            for metric_name, metric_data in stage_data.items():
                if "limits" in metric_data and metric_data["limits"]:
                    current_value = metric_data.get("current_value")
                    limits = metric_data["limits"]

                    if current_value is not None:
                        if (current_value > limits.get("upper_control_limit", float('inf')) or
                            current_value < limits.get("lower_control_limit", float('-inf'))):
                            stage_health["status"] = "error"
                            stage_health["issues"].append(f"{metric_name} out of control limits")
                            overall_status = "error"

            health_indicators[stage_name] = stage_health

        return {
            "overall_status": overall_status,
            "stage_health": health_indicators,
            "last_checked": time.time()
        }

    def _get_recent_calculations(self) -> List[Dict[str, Any]]:
        """Get recent uncertainty calculations with debug info"""
        if not self.instrumented_calculator.provenance_history:
            return []

        recent = list(self.instrumented_calculator.provenance_history)[-10:]
        formatted_calculations = []

        for provenance in recent:
            calculation_data = {
                "request_id": provenance.request_id,
                "timestamp": provenance.timestamp,
                "final_uncertainty": provenance.final_uncertainty,
                "total_time": provenance.total_execution_time,
                "input_shape": provenance.input_logits_shape,
                "actual_token_id": provenance.actual_token_id,
                "stage_count": len(provenance.stage_metrics),
                "warnings_count": len(provenance.pipeline_warnings),
                "errors_count": len(provenance.pipeline_errors),
                "time_ago": self._time_ago(provenance.timestamp)
            }

            # Add stage timing breakdown
            stage_times = {}
            for stage_metrics in provenance.stage_metrics:
                stage_times[stage_metrics.stage.value] = stage_metrics.execution_time

            calculation_data["stage_times"] = stage_times
            formatted_calculations.append(calculation_data)

        return formatted_calculations

    def _time_ago(self, timestamp: float) -> str:
        """Format timestamp as 'time ago' string"""
        seconds_ago = time.time() - timestamp

        if seconds_ago < 60:
            return f"{int(seconds_ago)}s ago"
        elif seconds_ago < 3600:
            return f"{int(seconds_ago/60)}m ago"
        elif seconds_ago < 86400:
            return f"{int(seconds_ago/3600)}h ago"
        else:
            return f"{int(seconds_ago/86400)}d ago"

    def get_calculation_details(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific calculation"""
        if not self.instrumented_calculator:
            return None

        # Find the calculation in history
        for provenance in self.instrumented_calculator.provenance_history:
            if provenance.request_id == request_id:
                # Convert to detailed format
                return {
                    "provenance": provenance.to_dict(),
                    "debug_report": self.instrumented_calculator.generate_debug_report(provenance)
                }

        return None

    def get_control_chart_data(self, stage: str, metric: str) -> Optional[Dict[str, Any]]:
        """Get control chart data for visualization"""
        if not self.instrumented_calculator or not self.instrumented_calculator.spc:
            return None

        try:
            stage_enum = PipelineStage(stage)
            return self.instrumented_calculator.spc.get_control_chart_data(stage_enum, metric)
        except ValueError:
            return None

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        return self.alert_manager.acknowledge_alert(alert_id)

    def export_metrics(self, time_window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Export metrics data for external analysis"""
        export_data = {
            "exported_at": time.time(),
            "time_window_seconds": time_window_seconds,
            "metrics": {}
        }

        metric_names = self.metrics_aggregator.get_all_metric_names()

        for metric_name in metric_names:
            summary = self.metrics_aggregator.get_aggregated_summary(
                metric_name, time_window_seconds
            )

            if summary:
                export_data["metrics"][metric_name] = asdict(summary)

        return export_data

    def generate_html_dashboard(self) -> str:
        """Generate simple HTML dashboard"""
        dashboard_data = self.get_dashboard_data()

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Uncertainty Calculation Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .metric { margin: 10px 0; padding: 5px; background: #f9f9f9; }
                .alert { margin: 5px 0; padding: 10px; border-radius: 3px; }
                .alert.error { background: #ffebee; border-left: 4px solid #f44336; }
                .alert.warning { background: #fff3e0; border-left: 4px solid #ff9800; }
                .alert.info { background: #e3f2fd; border-left: 4px solid #2196f3; }
                .status.healthy { color: green; }
                .status.warning { color: orange; }
                .status.error { color: red; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
            <script>
                function refreshDashboard() {
                    location.reload();
                }
                setInterval(refreshDashboard, 30000); // Refresh every 30 seconds
            </script>
        </head>
        <body>
            <h1>Uncertainty Calculation Dashboard</h1>
            <p>Last Updated: {last_updated}</p>

            <div class="section">
                <h2>System Status</h2>
                <p>Overall Health: <span class="status {overall_health}">{overall_health}</span></p>
                <p>Active Alerts: {total_alerts}</p>
            </div>

            <div class="section">
                <h2>Active Alerts</h2>
                {alerts_html}
            </div>

            <div class="section">
                <h2>Key Metrics (5 minute window)</h2>
                {metrics_html}
            </div>

            <div class="section">
                <h2>Recent Calculations</h2>
                {calculations_html}
            </div>

            <div class="section">
                <h2>Pipeline Health</h2>
                {pipeline_health_html}
            </div>
        </body>
        </html>
        """

        # Format data for HTML
        last_updated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dashboard_data["last_updated"]))
        overall_health = dashboard_data["pipeline_health"].get("overall_status", "unknown")
        total_alerts = dashboard_data["alerts"]["total_active"]

        # Format alerts
        alerts_html = ""
        if dashboard_data["alerts"]["active_alerts"]:
            for alert in dashboard_data["alerts"]["active_alerts"][:5]:
                alerts_html += f'<div class="alert {alert["severity"]}">'
                alerts_html += f'<strong>{alert["title"]}</strong><br>'
                alerts_html += f'{alert["description"]} ({alert["time_ago"]})'
                alerts_html += '</div>'
        else:
            alerts_html = '<p>No active alerts</p>'

        # Format metrics
        metrics_html = "<table><tr><th>Metric</th><th>Count</th><th>Mean</th><th>P95</th><th>Max</th></tr>"
        for metric_name, windows in dashboard_data["metrics"].items():
            if "5min" in windows:
                data = windows["5min"]
                metrics_html += f"<tr><td>{metric_name}</td><td>{data['count']}</td>"
                metrics_html += f"<td>{data['mean']:.4f}</td><td>{data['p95']:.4f}</td><td>{data['max']:.4f}</td></tr>"
        metrics_html += "</table>"

        # Format recent calculations
        calculations_html = "<table><tr><th>Request ID</th><th>Uncertainty</th><th>Time</th><th>Warnings</th></tr>"
        for calc in dashboard_data["recent_calculations"][:5]:
            calculations_html += f"<tr><td>{calc['request_id']}</td>"
            calculations_html += f"<td>{calc['final_uncertainty']:.4f}</td>"
            calculations_html += f"<td>{calc['total_time']:.4f}s</td>"
            calculations_html += f"<td>{calc['warnings_count']}</td></tr>"
        calculations_html += "</table>"

        # Format pipeline health
        pipeline_health_html = ""
        for stage_name, health in dashboard_data["pipeline_health"].get("stage_health", {}).items():
            pipeline_health_html += f'<p><strong>{stage_name}</strong>: '
            pipeline_health_html += f'<span class="status {health["status"]}">{health["status"]}</span>'
            if health["issues"]:
                pipeline_health_html += f' - {"; ".join(health["issues"])}'
            pipeline_health_html += '</p>'

        return html_template.format(
            last_updated=last_updated,
            overall_health=overall_health,
            total_alerts=total_alerts,
            alerts_html=alerts_html,
            metrics_html=metrics_html,
            calculations_html=calculations_html,
            pipeline_health_html=pipeline_health_html
        )


def create_uncertainty_dashboard(metrics_aggregator: MetricsAggregator,
                               alert_manager: AlertManager,
                               instrumented_calculator: Optional[InstrumentedUncertaintyCalculator] = None) -> UncertaintyDashboard:
    """Factory function to create dashboard with standard configuration"""
    return UncertaintyDashboard(
        metrics_aggregator=metrics_aggregator,
        alert_manager=alert_manager,
        instrumented_calculator=instrumented_calculator
    )