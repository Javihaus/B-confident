#!/usr/bin/env python3
"""
FastAPI Serving Example

Complete example of deploying PBA uncertainty quantification as a REST API
using FastAPI with monitoring, compliance endpoints, and production features.

Run with:
    python fastapi_example.py
    # or
    uvicorn fastapi_example:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from b_confident.serving import create_uncertainty_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_custom_app() -> FastAPI:
    """Create a custom FastAPI app with additional endpoints"""

    # Create the base uncertainty API
    app = create_uncertainty_api(
        model_name_or_path="gpt2",
        enable_monitoring=True,
        cors_origins=["*"]  # Allow all origins for demo
    )

    @app.get("/", response_class=HTMLResponse)
    async def landing_page():
        """Custom landing page with API documentation"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PBA Uncertainty Quantification API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { background: #f0f8ff; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
                .method { color: #007acc; font-weight: bold; }
                code { background: #eee; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>[TARGET] PBA Uncertainty Quantification API</h1>
                <p>Enterprise-grade uncertainty quantification for Large Language Models</p>
                <p><strong>Model:</strong> GPT-2 | <strong>Status:</strong> 🟢 Online</p>
            </div>

            <h2>Available Endpoints</h2>

            <div class="endpoint">
                <span class="method">POST</span> <code>/generate</code><br>
                <strong>Generate text with uncertainty quantification</strong><br>
                Example: <code>{"text": "The weather today is", "max_length": 50}</code>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <code>/calibrate</code><br>
                <strong>Validate uncertainty calibration</strong><br>
                Example: <code>{"uncertainty_scores": [0.1, 0.8], "correctness_labels": [1, 0]}</code>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <code>/compliance/report</code><br>
                <strong>Generate EU AI Act compliance report</strong><br>
                Query params: <code>format=markdown&dataset_name=validation</code>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code><br>
                <strong>Health check endpoint</strong>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <code>/stats</code><br>
                <strong>Server performance statistics</strong>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <code>/monitoring/summary</code><br>
                <strong>Continuous calibration monitoring</strong>
            </div>

            <h2>Interactive Documentation</h2>
            <p>
                📖 <a href="/docs">Swagger UI Documentation</a><br>
                📋 <a href="/redoc">ReDoc Documentation</a>
            </p>

            <h2>Usage Examples</h2>
            <pre><code># Generate with uncertainty
curl -X POST "http://localhost:8000/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "The future of AI is", "max_length": 30}'

# Health check
curl http://localhost:8000/health

# Get compliance report
curl "http://localhost:8000/compliance/report?format=json"</code></pre>

            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;">
                <p>Powered by <strong>B-Confident</strong> |
                   <a href="https://github.com/javiermarin/b-confident">GitHub</a>
                </p>
            </footer>
        </body>
        </html>
        """

    @app.get("/demo")
    async def demo_endpoint():
        """Demo endpoint showing various uncertainty examples"""
        examples = [
            {
                "input": "The weather today is",
                "description": "Basic weather generation with uncertainty"
            },
            {
                "input": "Machine learning is",
                "description": "Technical topic with higher uncertainty"
            },
            {
                "input": "The capital of France is",
                "description": "Factual question with low expected uncertainty"
            }
        ]

        results = []

        # Import here to avoid circular imports
        from b_confident import uncertainty_generate

        for example in examples:
            try:
                result = uncertainty_generate(
                    model="gpt2",
                    inputs=example["input"],
                    max_length=20
                )

                results.append({
                    "input": example["input"],
                    "generated": result.generated_texts[0],
                    "uncertainty": result.uncertainty_scores[0],
                    "description": example["description"]
                })
            except Exception as e:
                results.append({
                    "input": example["input"],
                    "error": str(e),
                    "description": example["description"]
                })

        return {
            "demo_results": results,
            "explanation": "These examples show how uncertainty varies with different types of inputs",
            "methodology": "Perplexity-Based Adjacency (PBA)"
        }

    return app


async def run_performance_test():
    """Run a basic performance test on the API"""
    import httpx
    import time

    base_url = "http://localhost:8000"
    test_requests = [
        {"text": "Hello world", "max_length": 15},
        {"text": "The weather is", "max_length": 20},
        {"text": "Machine learning", "max_length": 25}
    ]

    print("\n🧪 Running Performance Test")
    print("-" * 30)

    async with httpx.AsyncClient() as client:
        start_time = time.time()

        # Test multiple requests
        tasks = []
        for req in test_requests:
            task = client.post(f"{base_url}/generate", json=req, timeout=30.0)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Processed {len(test_requests)} requests in {total_time:.2f}s")
        print(f"Average time per request: {total_time / len(test_requests):.2f}s")

        # Analyze results
        successful = 0
        total_uncertainty = 0
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"Request {i+1}: Error - {response}")
            else:
                try:
                    data = response.json()
                    if "uncertainty_scores" in data:
                        uncertainty = data["uncertainty_scores"][0]
                        total_uncertainty += uncertainty
                        successful += 1
                        print(f"Request {i+1}: Uncertainty {uncertainty:.3f}")
                    else:
                        print(f"Request {i+1}: No uncertainty data")
                except Exception as e:
                    print(f"Request {i+1}: Parse error - {e}")

        if successful > 0:
            print(f"Average uncertainty: {total_uncertainty / successful:.3f}")

        print(f"Success rate: {successful}/{len(test_requests)}")


def main():
    """Main function to run the FastAPI server"""
    if not FASTAPI_AVAILABLE:
        print("[ERROR] FastAPI not available. Install with:")
        print("pip install b-confident[serving]")
        return

    print("[DEPLOY] Starting PBA Uncertainty API Server")
    print("=" * 40)

    # Create the app
    app = create_custom_app()

    print("\n📋 Server Information:")
    print(f"Model: GPT-2")
    print(f"PBA Configuration: α=0.9, β=0.5 (paper-optimized)")
    print(f"Monitoring: Enabled")
    print(f"Compliance: EU AI Act Article 15 support")

    print("\n🌐 Available Endpoints:")
    print("- http://localhost:8000/ (Landing page)")
    print("- http://localhost:8000/docs (Swagger UI)")
    print("- http://localhost:8000/demo (Live examples)")
    print("- http://localhost:8000/generate (Main API)")
    print("- http://localhost:8000/health (Health check)")

    print("\n💡 Usage Examples:")
    print("""
# Generate text with uncertainty
curl -X POST "http://localhost:8000/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "The future of AI is", "max_length": 30}'

# Get compliance report
curl "http://localhost:8000/compliance/report?format=markdown"

# Check server health
curl http://localhost:8000/health
    """)

    print("\nStarting server on http://localhost:8000...")
    print("Press Ctrl+C to stop\n")

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()