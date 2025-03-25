#!/usr/bin/env python
"""
Analyze and visualize profiling results.

This script loads profiling data from a JSON file, analyzes it,
and generates visualizations to help identify performance bottlenecks.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_logger

logger = get_logger(__name__)


def load_profiling_data(input_file: str) -> Dict:
    """
    Load profiling data from JSON file.

    Args:
        input_file: Path to the profiling results JSON file.

    Returns:
        Dictionary with profiling data.
    """
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading profiling data: {str(e)}")
        return {}


def create_api_endpoint_comparison(data: Dict, output_dir: str):
    """
    Create comparison chart for API endpoint performance.

    Args:
        data: Profiling data dictionary.
        output_dir: Directory to save visualizations.
    """
    if "api" not in data or not data["api"]:
        logger.warning("No API profiling data found")
        return

    api_data = data["api"]
    endpoints = list(api_data.keys())

    # Extract metrics
    metrics = ["mean_ms", "median_ms", "p95_ms", "max_ms"]

    # Create dataframe for plotting
    results = []
    for endpoint in endpoints:
        for metric in metrics:
            results.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "value": api_data[endpoint][metric],
                }
            )

    df = pd.DataFrame(results)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Create grouped bar chart
    ax = sns.barplot(x="endpoint", y="value", hue="metric", data=df)

    plt.title("API Endpoint Performance Comparison")
    plt.xlabel("Endpoint")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "api_endpoint_comparison.png")
    plt.savefig(output_file)
    logger.info(f"API endpoint comparison saved to {output_file}")


def create_model_training_comparison(data: Dict, output_dir: str):
    """
    Create comparison chart for model training performance.

    Args:
        data: Profiling data dictionary.
        output_dir: Directory to save visualizations.
    """
    if "model" not in data or not data["model"]:
        logger.warning("No model profiling data found")
        return

    model_data = data["model"]

    # Filter for training operations
    training_data = {k: v for k, v in model_data.items() if k.startswith("train_")}

    if not training_data:
        logger.warning("No model training data found")
        return

    # Create dataframe for plotting
    results = []
    for operation, metrics in training_data.items():
        model_type = operation.replace("train_", "")
        results.append(
            {
                "model_type": model_type,
                "mean_ms": metrics["mean_ms"],
                "memory_mb": metrics.get("memory_mb", 0),  # Optional memory usage
                "training_time_s": metrics["mean_ms"] / 1000,  # Convert to seconds
            }
        )

    df = pd.DataFrame(results)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Sort by training time
    df = df.sort_values("training_time_s", ascending=False)

    # Create bar chart
    ax = sns.barplot(x="model_type", y="training_time_s", data=df)

    plt.title("Model Training Time Comparison")
    plt.xlabel("Model Type")
    plt.ylabel("Training Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add values on bars
    for i, v in enumerate(df["training_time_s"]):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha="center")

    # Save plot
    output_file = os.path.join(output_dir, "model_training_comparison.png")
    plt.savefig(output_file)
    logger.info(f"Model training comparison saved to {output_file}")


def create_model_inference_comparison(data: Dict, output_dir: str):
    """
    Create comparison chart for model inference performance.

    Args:
        data: Profiling data dictionary.
        output_dir: Directory to save visualizations.
    """
    if "model" not in data or not data["model"]:
        logger.warning("No model profiling data found")
        return

    model_data = data["model"]

    # Filter for inference operations
    inference_data = {k: v for k, v in model_data.items() if k.startswith("infer_")}

    if not inference_data:
        logger.warning("No model inference data found")
        return

    # Create dataframe for plotting
    results = []
    for operation, metrics in inference_data.items():
        # Parse operation name to extract model type and data size
        parts = operation.replace("infer_", "").split("_")
        model_type = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
        data_size = parts[-1] if len(parts) > 1 else "unknown"

        results.append(
            {
                "model_type": model_type,
                "data_size": data_size,
                "mean_ms": metrics["mean_ms"],
                "inference_time_ms": metrics["mean_ms"],
            }
        )

    df = pd.DataFrame(results)

    # Create plot
    plt.figure(figsize=(14, 8))

    # Create grouped bar chart for different data sizes
    ax = sns.barplot(x="model_type", y="inference_time_ms", hue="data_size", data=df)

    plt.title("Model Inference Time Comparison")
    plt.xlabel("Model Type")
    plt.ylabel("Inference Time (ms)")
    plt.xticks(rotation=45)
    plt.legend(title="Data Size")
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "model_inference_comparison.png")
    plt.savefig(output_file)
    logger.info(f"Model inference comparison saved to {output_file}")


def create_dashboard_performance_report(data: Dict, output_dir: str):
    """
    Create report for dashboard performance.

    Args:
        data: Profiling data dictionary.
        output_dir: Directory to save visualizations.
    """
    if "dashboard" not in data or not data["dashboard"]:
        logger.warning("No dashboard profiling data found")
        return

    dashboard_data = data["dashboard"]

    # Create report
    report = ["# Dashboard Performance Report\n"]
    report.append("## Component Loading Times\n")

    if dashboard_data:
        report.append("| Component | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |\n")
        report.append("|-----------|-----------|-------------|----------|----------|\n")

        for component, metrics in dashboard_data.items():
            report.append(
                f"| {component} | {metrics['mean_ms']:.2f} | {metrics['median_ms']:.2f} | "
                f"{metrics['p95_ms']:.2f} | {metrics['max_ms']:.2f} |\n"
            )
    else:
        report.append("No component timing data available.\n")

    # Save report
    output_file = os.path.join(output_dir, "dashboard_performance_report.md")
    with open(output_file, "w") as f:
        f.writelines(report)

    logger.info(f"Dashboard performance report saved to {output_file}")


def create_profiling_overview(data: Dict, output_dir: str):
    """
    Create HTML overview of all profiling results.

    Args:
        data: Profiling data dictionary.
        output_dir: Directory to save visualizations.
    """
    # Create output file path
    output_file = os.path.join(output_dir, "profiling_overview.html")

    # Create HTML content
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Supply Chain Forecaster Profiling Results</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1, h2, h3 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        ".chart-container { margin: 20px 0; }",
        ".card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }",
        ".card h3 { margin-top: 0; }",
        ".warning { color: #e63900; }",
        ".good { color: #00994d; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Supply Chain Forecaster Performance Profiling Results</h1>",
        "<p>Generated on " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>",
    ]

    # Add API section
    html.append("<h2>API Performance</h2>")

    if "api" in data and data["api"]:
        html.append("<table>")
        html.append(
            "<tr><th>Endpoint</th><th>Count</th><th>Mean (ms)</th><th>Median (ms)</th><th>P95 (ms)</th><th>Max (ms)</th><th>Status</th></tr>"
        )

        for endpoint, metrics in data["api"].items():
            # Determine status based on response time
            status = "good" if metrics["mean_ms"] < 100 else "warning"
            status_text = "✅ Good" if status == "good" else "⚠️ Slow"

            html.append(f"<tr>")
            html.append(f"<td>{endpoint}</td>")
            html.append(f"<td>{metrics['count']}</td>")
            html.append(f"<td>{metrics['mean_ms']:.2f}</td>")
            html.append(f"<td>{metrics['median_ms']:.2f}</td>")
            html.append(f"<td>{metrics['p95_ms']:.2f}</td>")
            html.append(f"<td>{metrics['max_ms']:.2f}</td>")
            html.append(f"<td class='{status}'>{status_text}</td>")
            html.append("</tr>")

        html.append("</table>")

        html.append("<div class='chart-container'>")
        html.append("<h3>API Endpoint Comparison</h3>")
        html.append(
            f"<img src='api_endpoint_comparison.png' alt='API Endpoint Comparison' style='width: 100%;'>"
        )
        html.append("</div>")
    else:
        html.append("<p>No API profiling data available.</p>")

    # Add Model section
    html.append("<h2>Model Performance</h2>")

    if "model" in data and data["model"]:
        # Training performance
        html.append("<h3>Model Training</h3>")
        training_data = {
            k: v for k, v in data["model"].items() if k.startswith("train_")
        }

        if training_data:
            html.append("<table>")
            html.append(
                "<tr><th>Model Type</th><th>Mean Time (s)</th><th>Max Time (s)</th></tr>"
            )

            for operation, metrics in training_data.items():
                model_type = operation.replace("train_", "")
                mean_time_s = metrics["mean_ms"] / 1000
                max_time_s = metrics["max_ms"] / 1000

                html.append(f"<tr>")
                html.append(f"<td>{model_type}</td>")
                html.append(f"<td>{mean_time_s:.2f}</td>")
                html.append(f"<td>{max_time_s:.2f}</td>")
                html.append("</tr>")

            html.append("</table>")

            html.append("<div class='chart-container'>")
            html.append("<h3>Model Training Comparison</h3>")
            html.append(
                f"<img src='model_training_comparison.png' alt='Model Training Comparison' style='width: 100%;'>"
            )
            html.append("</div>")
        else:
            html.append("<p>No model training data available.</p>")

        # Inference performance
        html.append("<h3>Model Inference</h3>")
        inference_data = {
            k: v for k, v in data["model"].items() if k.startswith("infer_")
        }

        if inference_data:
            html.append("<table>")
            html.append(
                "<tr><th>Model Type</th><th>Data Size</th><th>Mean Time (ms)</th><th>P95 Time (ms)</th></tr>"
            )

            for operation, metrics in inference_data.items():
                parts = operation.replace("infer_", "").split("_")
                model_type = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
                data_size = parts[-1] if len(parts) > 1 else "unknown"

                html.append(f"<tr>")
                html.append(f"<td>{model_type}</td>")
                html.append(f"<td>{data_size}</td>")
                html.append(f"<td>{metrics['mean_ms']:.2f}</td>")
                html.append(f"<td>{metrics['p95_ms']:.2f}</td>")
                html.append("</tr>")

            html.append("</table>")

            html.append("<div class='chart-container'>")
            html.append("<h3>Model Inference Comparison</h3>")
            html.append(
                f"<img src='model_inference_comparison.png' alt='Model Inference Comparison' style='width: 100%;'>"
            )
            html.append("</div>")
        else:
            html.append("<p>No model inference data available.</p>")
    else:
        html.append("<p>No model profiling data available.</p>")

    # Add Dashboard section
    html.append("<h2>Dashboard Performance</h2>")

    if "dashboard" in data and data["dashboard"]:
        html.append("<table>")
        html.append(
            "<tr><th>Component</th><th>Mean (ms)</th><th>Median (ms)</th><th>P95 (ms)</th><th>Max (ms)</th></tr>"
        )

        for component, metrics in data["dashboard"].items():
            html.append(f"<tr>")
            html.append(f"<td>{component}</td>")
            html.append(f"<td>{metrics['mean_ms']:.2f}</td>")
            html.append(f"<td>{metrics['median_ms']:.2f}</td>")
            html.append(f"<td>{metrics['p95_ms']:.2f}</td>")
            html.append(f"<td>{metrics['max_ms']:.2f}</td>")
            html.append("</tr>")

        html.append("</table>")
    else:
        html.append("<p>No dashboard profiling data available.</p>")

    # Add recommendations section
    html.append("<h2>Performance Recommendations</h2>")
    html.append("<div class='card'>")
    html.append("<h3>Key Findings and Recommendations</h3>")
    html.append("<ul>")

    # API recommendations
    if "api" in data and data["api"]:
        slow_endpoints = [
            endpoint
            for endpoint, metrics in data["api"].items()
            if metrics["mean_ms"] > 100
        ]
        if slow_endpoints:
            html.append(
                f"<li class='warning'>Slow API endpoints detected: {', '.join(slow_endpoints)}. Consider optimizing these endpoints.</li>"
            )
        else:
            html.append(
                "<li class='good'>All API endpoints are responding within acceptable time limits.</li>"
            )

    # Model recommendations
    if "model" in data and data["model"]:
        training_data = {
            k: v for k, v in data["model"].items() if k.startswith("train_")
        }
        inference_data = {
            k: v for k, v in data["model"].items() if k.startswith("infer_")
        }

        # Find slowest training model
        if training_data:
            slowest_model = max(training_data.items(), key=lambda x: x[1]["mean_ms"])
            model_type = slowest_model[0].replace("train_", "")
            train_time = slowest_model[1]["mean_ms"] / 1000  # Convert to seconds

            html.append(
                f"<li>Slowest model to train: <strong>{model_type}</strong> ({train_time:.2f} seconds). Consider optimizing or pre-training this model.</li>"
            )

        # Find slowest inference model
        if inference_data:
            large_inference = {
                k: v for k, v in inference_data.items() if k.endswith("_large")
            }
            if large_inference:
                slowest_inference = max(
                    large_inference.items(), key=lambda x: x[1]["mean_ms"]
                )
                model_type = (
                    slowest_inference[0].replace("infer_", "").replace("_large", "")
                )
                infer_time = slowest_inference[1]["mean_ms"]

                html.append(
                    f"<li>Slowest model for large dataset inference: <strong>{model_type}</strong> ({infer_time:.2f} ms). Consider optimizing or implementing batching.</li>"
                )

    # General recommendations
    html.append(
        "<li>Consider implementing model caching to reduce model loading overhead.</li>"
    )
    html.append(
        "<li>For large datasets, implement chunked processing and streaming responses.</li>"
    )
    html.append(
        "<li>Optimize dashboard loading by implementing lazy loading of components.</li>"
    )
    html.append(
        "<li>Consider adding data caching for frequently accessed datasets.</li>"
    )

    html.append("</ul>")
    html.append("</div>")

    # Close HTML
    html.append("</body>")
    html.append("</html>")

    # Write HTML to file
    with open(output_file, "w") as f:
        f.write("\n".join(html))

    logger.info(f"Profiling overview saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize profiling results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="profiling_results.json",
        help="Input file with profiling results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_visualizations",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load profiling data
    data = load_profiling_data(args.input)

    if not data:
        logger.error("No valid profiling data found")
        return

    # Create visualizations
    create_api_endpoint_comparison(data, args.output_dir)
    create_model_training_comparison(data, args.output_dir)
    create_model_inference_comparison(data, args.output_dir)
    create_dashboard_performance_report(data, args.output_dir)
    create_profiling_overview(data, args.output_dir)

    logger.info(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
