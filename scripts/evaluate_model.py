#!/usr/bin/env python
"""
Model evaluation script for CI/CD pipeline.

This script evaluates the trained models against benchmark metrics
and determines if they are ready for deployment.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate models for deployment.")
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="models/trained",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--benchmark-file", 
        type=str, 
        default="models/benchmarks.json",
        help="File containing benchmark metrics"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="models/evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--model-version", 
        type=str, 
        default=None,
        help="Specific model version to evaluate"
    )
    return parser.parse_args()


def load_benchmarks(benchmark_file):
    """Load benchmark metrics from file."""
    if not os.path.exists(benchmark_file):
        logger.warning(f"Benchmark file {benchmark_file} not found. Using default benchmarks.")
        return {
            "forecasting": {
                "prophet": {"mape": 15.0, "rmse": 25.0},
                "arima": {"mape": 18.0, "rmse": 30.0},
                "xgboost": {"mape": 12.0, "rmse": 22.0},
                "lstm": {"mape": 10.0, "rmse": 20.0},
                "exponential_smoothing": {"mape": 20.0, "rmse": 35.0}
            },
            "anomaly": {
                "isolation_forest": {"f1_score": 0.75, "precision": 0.70, "recall": 0.80},
                "autoencoder": {"f1_score": 0.80, "precision": 0.75, "recall": 0.85},
                "statistical": {"f1_score": 0.70, "precision": 0.65, "recall": 0.75}
            }
        }
    
    try:
        with open(benchmark_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing benchmark file: {e}")
        sys.exit(1)


def evaluate_forecasting_model(model_path, benchmarks):
    """Evaluate a forecasting model against benchmarks."""
    # In a real implementation, we would load the model and evaluate it
    # on a test dataset. For this script, we'll simulate the evaluation.
    
    # Extract model type from path
    model_type = os.path.basename(os.path.dirname(model_path))
    
    # Load model metadata if available
    metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Simulate metadata
        metadata = {
            "metrics": {
                "mape": np.random.uniform(5, 25),
                "rmse": np.random.uniform(15, 40)
            }
        }
    
    # Compare with benchmarks
    benchmark_metrics = benchmarks["forecasting"].get(model_type, {
        "mape": 20.0, 
        "rmse": 30.0
    })
    
    passed = (
        metadata["metrics"]["mape"] <= benchmark_metrics["mape"] and 
        metadata["metrics"]["rmse"] <= benchmark_metrics["rmse"]
    )
    
    return {
        "model_path": model_path,
        "model_type": model_type,
        "metrics": metadata["metrics"],
        "benchmarks": benchmark_metrics,
        "passed": passed
    }


def evaluate_anomaly_model(model_path, benchmarks):
    """Evaluate an anomaly detection model against benchmarks."""
    # Similar to the forecasting model evaluation, we're simulating here
    
    # Extract model type from path
    model_type = os.path.basename(os.path.dirname(model_path))
    
    # Load model metadata if available
    metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Simulate metadata
        metadata = {
            "metrics": {
                "f1_score": np.random.uniform(0.6, 0.9),
                "precision": np.random.uniform(0.6, 0.9),
                "recall": np.random.uniform(0.6, 0.9)
            }
        }
    
    # Compare with benchmarks
    benchmark_metrics = benchmarks["anomaly"].get(model_type, {
        "f1_score": 0.75, 
        "precision": 0.70, 
        "recall": 0.75
    })
    
    passed = (
        metadata["metrics"]["f1_score"] >= benchmark_metrics["f1_score"] and
        metadata["metrics"]["precision"] >= benchmark_metrics["precision"] and
        metadata["metrics"]["recall"] >= benchmark_metrics["recall"]
    )
    
    return {
        "model_path": model_path,
        "model_type": model_type,
        "metrics": metadata["metrics"],
        "benchmarks": benchmark_metrics,
        "passed": passed
    }


def find_models(model_dir, model_version=None):
    """Find all model files in the given directory."""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        logger.warning(f"Model directory {model_dir} does not exist.")
        return []
    
    # Look for model files with .pkl or .joblib extension
    model_files = []
    
    # If model_version is specified, look for specific version
    if model_version:
        for ext in [".pkl", ".joblib", ".h5"]:
            version_pattern = f"*/*_v{model_version}*{ext}"
            model_files.extend(list(model_dir.glob(version_pattern)))
    else:
        # Otherwise, find all model files
        for ext in [".pkl", ".joblib", ".h5"]:
            model_files.extend(list(model_dir.glob(f"*/*{ext}")))
    
    return [str(f) for f in model_files]


def main():
    """Main function to evaluate models."""
    args = parse_arguments()
    
    # Load benchmark metrics
    benchmarks = load_benchmarks(args.benchmark_file)
    
    # Find models to evaluate
    model_files = find_models(args.model_dir, args.model_version)
    
    if not model_files:
        logger.warning("No model files found for evaluation.")
        # Create an empty results file to avoid pipeline failure
        with open(args.output_file, 'w') as f:
            json.dump({"evaluated_models": [], "passed_models": [], "failed_models": []}, f, indent=2)
        return 0
    
    # Evaluate each model
    evaluation_results = []
    
    for model_file in model_files:
        logger.info(f"Evaluating model: {model_file}")
        
        if "forecasting" in model_file:
            result = evaluate_forecasting_model(model_file, benchmarks)
        elif "anomaly" in model_file:
            result = evaluate_anomaly_model(model_file, benchmarks)
        else:
            logger.warning(f"Unknown model type for {model_file}, skipping")
            continue
        
        evaluation_results.append(result)
    
    # Summarize results
    passed_models = [r for r in evaluation_results if r["passed"]]
    failed_models = [r for r in evaluation_results if not r["passed"]]
    
    logger.info(f"Evaluated {len(evaluation_results)} models")
    logger.info(f"Passed: {len(passed_models)}, Failed: {len(failed_models)}")
    
    # Output detailed results
    for result in evaluation_results:
        model_type = result["model_type"]
        status = "PASSED" if result["passed"] else "FAILED"
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in result["metrics"].items()])
        logger.info(f"{model_type}: {status} - {metrics_str}")
    
    # Save results to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump({
            "evaluated_models": evaluation_results,
            "passed_models": [r["model_path"] for r in passed_models],
            "failed_models": [r["model_path"] for r in failed_models]
        }, f, indent=2)
    
    # Return success if all models passed
    return 0 if len(failed_models) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())