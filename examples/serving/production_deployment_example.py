"""
Production Deployment Examples for B-Confident

Demonstrates how to deploy uncertainty quantification in production using
TorchServe, FastAPI, and Ray Serve with streaming memory, distributed calibration,
and comprehensive observability.
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fastapi_production_deployment():
    """
    Example: Production FastAPI deployment with all features enabled
    """
    print("=== FastAPI Production Deployment Example ===")

    try:
        from b_confident.serving import create_uncertainty_api
        from b_confident.core import PBAConfig
        from b_confident.memory import MemoryConfig
        from b_confident.observability import DebugLevel

        # Production configuration
        pba_config = PBAConfig(
            beta=0.5,
            temperature=1.0,
            validate_inputs=True,
            numerical_stability=True
        )

        memory_config = MemoryConfig(
            max_memory_usage_gb=4.0,
            memory_pool_size_mb=200,
            chunk_size=16,
            gc_threshold=0.8,
            enable_memory_monitoring=True
        )

        # Create production API with all features
        app = create_uncertainty_api(
            model_name_or_path="gpt2",
            pba_config=pba_config,
            memory_config=memory_config,
            enable_monitoring=True,
            enable_streaming=True,
            enable_distributed_calibration=True,
            enable_observability=True,
            debug_level=DebugLevel.STANDARD,
            node_id="api_node_production_1",
            redis_url="redis://localhost:6379",
            cors_origins=["http://localhost:3000", "https://myapp.com"]
        )

        print("[OK] FastAPI application created with production features:")
        print("  - Streaming memory architecture for high-throughput")
        print("  - Distributed calibration with Redis coordination")
        print("  - Real-time observability and debugging")
        print("  - Comprehensive monitoring and alerting")
        print("  - CORS enabled for web integration")

        print("\nTo run the server:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4")

        print("\nAPI Endpoints available:")
        print("  POST /generate - Generate text with uncertainty")
        print("  POST /generate/batch - Batch generation with streaming")
        print("  GET /health - Health check endpoint")
        print("  GET /metrics - Prometheus-style metrics")
        print("  GET /dashboard - Real-time monitoring dashboard")
        print("  GET /calibration/stats - Distributed calibration statistics")
        print("  GET /debug/{request_id} - Detailed calculation debugging")

        return app

    except ImportError as e:
        print(f"[ERROR] FastAPI dependencies not available: {e}")
        print("Install with: pip install 'b-confident[serving]'")
        return None


def torchserve_production_deployment():
    """
    Example: TorchServe deployment with production features
    """
    print("\n=== TorchServe Production Deployment Example ===")

    try:
        from b_confident.serving import PBAUncertaintyHandler

        print("[OK] TorchServe handler available with production features:")
        print("  - Integrated streaming memory management")
        print("  - Observability with detailed request tracing")
        print("  - Compatible with existing TorchServe infrastructure")

        # Model archive configuration
        config_template = """
# model-config.properties for TorchServe
service_envelope=json
handler=b_confident.serving.PBAUncertaintyHandler
model_name=uncertainty-gpt2
model_version=1.0
batch_size=8
max_batch_delay=100
response_timeout=300

# PBA Configuration
pba_beta=0.5
pba_temperature=1.0
max_length=100
enable_streaming=true
enable_observability=true
debug_level=standard

# Memory Configuration
memory_max_usage_gb=2.0
memory_pool_size_mb=100
memory_chunk_size=8
memory_gc_threshold=0.8
"""

        print("\nModel Archive Configuration:")
        print(config_template)

        print("\nDeployment Commands:")
        print("1. Create model archive:")
        print("   torch-model-archiver --model-name uncertainty-gpt2 \\")
        print("                        --version 1.0 \\")
        print("                        --handler b_confident.serving.PBAUncertaintyHandler \\")
        print("                        --config-file model-config.properties")

        print("\n2. Start TorchServe:")
        print("   torchserve --start --model-store model_store \\")
        print("              --models uncertainty-gpt2=uncertainty-gpt2.mar \\")
        print("              --ts-config config.properties")

        print("\n3. Test deployment:")
        print("   curl -X POST http://localhost:8080/predictions/uncertainty-gpt2 \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"text\": \"The weather today is\", \"max_length\": 50}'")

        return True

    except ImportError as e:
        print(f"[ERROR] TorchServe dependencies not available: {e}")
        print("Install TorchServe and try again")
        return False


async def ray_serve_production_deployment():
    """
    Example: Ray Serve deployment with distributed features
    """
    print("\n=== Ray Serve Production Deployment Example ===")

    try:
        import ray
        from ray import serve
        from b_confident.serving import PBAServeDeployment
        from b_confident.core import PBAConfig
        from b_confident.memory import MemoryConfig

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address="auto")  # Connect to existing cluster or start local

        # Start Ray Serve
        serve.start(detached=True)

        # Production configuration
        pba_config = PBAConfig(beta=0.5, temperature=1.0)
        memory_config = MemoryConfig(max_memory_usage_gb=8.0, chunk_size=32)

        # Deploy with production features
        deployment = PBAServeDeployment.bind(
            model_name="gpt2",
            pba_config=pba_config.__dict__,
            memory_config=memory_config.__dict__,
            enable_monitoring=True,
            enable_streaming=True,
            enable_distributed_calibration=True,
            enable_observability=True,
            node_id="ray_serve_node_1",
            redis_url="redis://localhost:6379"
        )

        # Configure deployment scaling
        deployment = deployment.options(
            name="uncertainty-service",
            num_replicas=4,
            max_concurrent_queries=50,
            ray_actor_options={
                "num_gpus": 0.25,  # Share GPU across replicas
                "num_cpus": 2,
                "memory": 4 * 1024 * 1024 * 1024  # 4GB
            },
            autoscaling_config={
                "min_replicas": 2,
                "max_replicas": 10,
                "target_num_ongoing_requests_per_replica": 5
            }
        )

        # Deploy the service
        serve.deploy(deployment)

        print("[OK] Ray Serve deployment configured with production features:")
        print("  - Auto-scaling from 2 to 10 replicas based on load")
        print("  - GPU sharing for efficient resource utilization")
        print("  - Distributed calibration across all replicas")
        print("  - Streaming memory management on each node")
        print("  - Comprehensive observability and monitoring")

        print("\nService Endpoints:")
        print("  POST http://localhost:8000/generate")
        print("  POST http://localhost:8000/generate/batch")
        print("  GET  http://localhost:8000/health")
        print("  GET  http://localhost:8000/metrics")

        print("\nTesting the deployment:")
        test_request = {
            "text": "The future of AI is",
            "max_length": 50,
            "temperature": 1.0
        }

        print(f"Request: {test_request}")

        # Test the deployment
        import requests
        try:
            response = requests.post(
                "http://localhost:8000/generate",
                json=test_request,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"[OK] Response received:")
                print(f"  Generated: {result.get('generated_texts', [''])[0][:50]}...")
                print(f"  Uncertainty: {result.get('uncertainty_scores', [0.0])[0]:.4f}")
                print(f"  Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            else:
                print(f"[WARNING] Request failed with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[INFO] Could not test deployment (service may not be running): {e}")

        return True

    except ImportError as e:
        print(f"[ERROR] Ray dependencies not available: {e}")
        print("Install with: pip install 'b-confident[ray]'")
        return False
    except Exception as e:
        print(f"[WARNING] Ray Serve deployment setup failed: {e}")
        print("This may be expected if Ray cluster is not running")
        return False


def kubernetes_deployment_guide():
    """
    Example: Kubernetes deployment configurations
    """
    print("\n=== Kubernetes Production Deployment Guide ===")

    # FastAPI Kubernetes manifest
    fastapi_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: b-confident-api
  labels:
    app: b-confident-api
spec:
  replicas: 4
  selector:
    matchLabels:
      app: b-confident-api
  template:
    metadata:
      labels:
        app: b-confident-api
    spec:
      containers:
      - name: api
        image: b-confident:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "gpt2"
        - name: ENABLE_STREAMING
          value: "true"
        - name: ENABLE_DISTRIBUTED_CALIBRATION
          value: "true"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: b-confident-api-service
spec:
  selector:
    app: b-confident-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
"""

    print("Kubernetes Deployment Manifest (save as b-confident-k8s.yaml):")
    print(fastapi_manifest)

    print("\nDeployment Commands:")
    print("1. Apply the configuration:")
    print("   kubectl apply -f b-confident-k8s.yaml")
    print("\n2. Check deployment status:")
    print("   kubectl get pods -l app=b-confident-api")
    print("\n3. Get service endpoint:")
    print("   kubectl get service b-confident-api-service")
    print("\n4. View logs:")
    print("   kubectl logs -l app=b-confident-api -f")
    print("\n5. Scale deployment:")
    print("   kubectl scale deployment b-confident-api --replicas=8")


def monitoring_and_observability_setup():
    """
    Example: Monitoring and observability setup
    """
    print("\n=== Monitoring and Observability Setup ===")

    prometheus_config = """
# prometheus.yml configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'b-confident-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'b-confident-dashboard'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/dashboard/metrics'
    scrape_interval: 30s
"""

    grafana_dashboard = """
{
  "dashboard": {
    "title": "B-Confident Uncertainty Quantification",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(uncertainty_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Uncertainty Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "uncertainty_scores_bucket",
            "legendFormat": "Uncertainty"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_bytes",
            "legendFormat": "Memory"
          }
        ]
      },
      {
        "title": "Calibration Quality",
        "type": "stat",
        "targets": [
          {
            "expr": "calibration_ece",
            "legendFormat": "ECE"
          }
        ]
      }
    ]
  }
}
"""

    print("Prometheus Configuration:")
    print(prometheus_config)

    print("\nGrafana Dashboard JSON:")
    print(grafana_dashboard)

    print("\nSetup Commands:")
    print("1. Start Prometheus:")
    print("   prometheus --config.file=prometheus.yml")
    print("\n2. Start Grafana:")
    print("   grafana-server --config=grafana.ini")
    print("\n3. Import dashboard:")
    print("   - Open Grafana (http://localhost:3000)")
    print("   - Add Prometheus as data source (http://localhost:9090)")
    print("   - Import the dashboard JSON")


async def main():
    """Run all deployment examples"""
    print("B-Confident Production Deployment Examples")
    print("=" * 60)

    # FastAPI deployment
    fastapi_app = fastapi_production_deployment()

    # TorchServe deployment
    torchserve_success = torchserve_production_deployment()

    # Ray Serve deployment
    ray_success = await ray_serve_production_deployment()

    # Kubernetes guide
    kubernetes_deployment_guide()

    # Monitoring setup
    monitoring_and_observability_setup()

    print("\n" + "=" * 60)
    print("Production Deployment Summary:")
    print(f"[{'OK' if fastapi_app else 'SKIP'}] FastAPI - REST API with all production features")
    print(f"[{'OK' if torchserve_success else 'SKIP'}] TorchServe - Enterprise model serving")
    print(f"[{'OK' if ray_success else 'SKIP'}] Ray Serve - Distributed autoscaling deployment")
    print("[INFO] Kubernetes - Container orchestration manifests provided")
    print("[INFO] Monitoring - Prometheus and Grafana configurations provided")

    print("\nKey Production Features Enabled:")
    print("- Streaming memory architecture prevents memory pressure")
    print("- Distributed calibration coordinates across nodes")
    print("- Real-time observability with debugging and monitoring")
    print("- Auto-scaling based on load and resource usage")
    print("- Comprehensive health checks and metrics")
    print("- Enterprise-grade error handling and logging")

    if fastapi_app:
        print(f"\nFastAPI app created successfully!")
        print("Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4")


if __name__ == "__main__":
    asyncio.run(main())