#!/usr/bin/env python
"""
Profile application performance to identify bottlenecks.

This script runs a series of tests to profile:
1. API performance (routing, request handling, response generation)
2. Model training and inference
3. Dashboard loading and rendering
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from utils import get_logger
from utils.profiling import (
    get_profiling_stats,
    profile_cpu,
    profile_memory,
    profile_time,
    reset_profiling_stats,
)

logger = get_logger(__name__)

# Endpoints to profile
API_ENDPOINTS = [
    {"method": "GET", "url": "/health", "name": "health_check"},
    {"method": "GET", "url": "/models/", "name": "list_models"},
]

# Model types to profile
MODEL_TYPES = [
    "ProphetModel",
    "ARIMAModel",
    "XGBoostModel",
    "LSTMModel",
    "IsolationForestDetector",
]

# Sample data paths
SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "sample_data"
)
SAMPLE_DEMAND_DATA = os.path.join(SAMPLE_DATA_PATH, "sample_demand_data.csv")
SAMPLE_INVENTORY_DATA = os.path.join(SAMPLE_DATA_PATH, "sample_inventory_data.csv")


def profile_api_endpoints(
    base_url: str, iterations: int = 10, auth: Optional[Dict] = None
):
    """
    Profile API endpoints performance.

    Args:
        base_url: Base URL of the API.
        iterations: Number of times to call each endpoint.
        auth: Authentication credentials (if needed).
    """
    logger.info(f"Profiling API endpoints with {iterations} iterations per endpoint")

    headers = {}
    if auth:
        if "api_key" in auth:
            headers["X-API-Key"] = auth["api_key"]
        elif "username" in auth and "password" in auth:
            import base64

            auth_str = f"{auth['username']}:{auth['password']}"
            headers[
                "Authorization"
            ] = f"Basic {base64.b64encode(auth_str.encode()).decode()}"

    results = {}

    for endpoint in API_ENDPOINTS:
        url = f"{base_url}{endpoint['url']}"
        method = endpoint["method"]
        name = endpoint["name"]

        logger.info(f"Profiling {method} {url}")
        timings = []

        for i in tqdm(range(iterations), desc=name):
            start_time = time.time()

            try:
                if method == "GET":
                    response = requests.get(url, headers=headers)
                elif method == "POST":
                    response = requests.post(url, headers=headers)

                if response.status_code != 200:
                    logger.warning(
                        f"Request failed: {response.status_code} - {response.text}"
                    )
                    continue

            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                continue

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            timings.append(duration_ms)

        # Calculate statistics
        if timings:
            timings_array = np.array(timings)
            results[name] = {
                "count": len(timings),
                "mean_ms": float(np.mean(timings_array)),
                "median_ms": float(np.median(timings_array)),
                "min_ms": float(np.min(timings_array)),
                "max_ms": float(np.max(timings_array)),
                "p95_ms": float(np.percentile(timings_array, 95)),
                "p99_ms": float(np.percentile(timings_array, 99)),
                "std_ms": float(np.std(timings_array)),
            }

    return results


def profile_model_training(api_url: str, model_types: List[str] = None):
    """
    Profile model training performance.

    Args:
        api_url: URL of the API.
        model_types: List of model types to profile.
    """
    logger.info("Profiling model training performance")

    if model_types is None:
        model_types = MODEL_TYPES

    results = {}

    # Load sample data
    df = pd.read_csv(SAMPLE_DEMAND_DATA)

    for model_type in model_types:
        logger.info(f"Profiling training of {model_type}")

        # Prepare training parameters
        model_name = f"{model_type}_profiling_{int(time.time())}"
        feature_columns = (
            ["date", "temperature", "promotion"]
            if "date" in df.columns
            else df.columns[:-1].tolist()
        )
        target_column = "demand" if "demand" in df.columns else df.columns[-1]
        date_column = "date" if "date" in df.columns else None

        training_params = {
            "model_type": model_type,
            "model_name": model_name,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "date_column": date_column,
            "model_params": {},  # Default params
            "save_model": True,
        }

        # Convert dataframe to file for upload
        import io

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        files = {"file": ("data.csv", buffer.getvalue())}

        # Profile training
        with (
            profile_time(f"train_{model_type}", "model"),
            profile_memory(f"train_{model_type}", "model"),
        ):
            try:
                response = requests.post(
                    f"{api_url}/forecasting/train",
                    files=files,
                    data={"params": json.dumps(training_params)},
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Training successful: {result['model_name']}")

                    # Profile inference
                    profile_model_inference(api_url, model_type, model_name, df)
                else:
                    logger.warning(
                        f"Training failed: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                logger.error(f"Training error: {str(e)}")

    return get_profiling_stats()["model"]


def profile_model_inference(
    api_url: str, model_type: str, model_name: str, df: pd.DataFrame
):
    """
    Profile model inference performance.

    Args:
        api_url: URL of the API.
        model_type: Type of model.
        model_name: Name of the trained model.
        df: DataFrame with input features.
    """
    logger.info(f"Profiling inference of {model_type} model {model_name}")

    # Prepare forecast parameters
    feature_columns = (
        ["date", "temperature", "promotion"]
        if "date" in df.columns
        else df.columns[:-1].tolist()
    )
    date_column = "date" if "date" in df.columns else None

    forecast_params = {
        "model_name": model_name,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "date_column": date_column,
        "steps": 30,
        "return_conf_int": True,
        "from_deployment": False,
    }

    # Convert dataframe to file for upload
    import io

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    files = {"file": ("data.csv", buffer.getvalue())}

    # Profile inference for different data sizes
    data_sizes = [
        ("small", df.iloc[: min(100, len(df))]),
        ("medium", df.iloc[: min(1000, len(df))]),
        ("large", df),
    ]

    for size_name, size_df in data_sizes:
        if len(size_df) < 10:  # Skip if too small
            continue

        buffer = io.StringIO()
        size_df.to_csv(buffer, index=False)
        buffer.seek(0)
        files = {"file": ("data.csv", buffer.getvalue())}

        with (
            profile_time(f"infer_{model_type}_{size_name}", "model"),
            profile_memory(f"infer_{model_type}_{size_name}", "model"),
        ):
            try:
                response = requests.post(
                    f"{api_url}/forecasting/forecast",
                    files=files,
                    data={"params": json.dumps(forecast_params)},
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"Inference successful for {size_name} data ({len(size_df)} rows)"
                    )
                else:
                    logger.warning(
                        f"Inference failed: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                logger.error(f"Inference error: {str(e)}")


def profile_dashboard_loading(dashboard_url: str):
    """
    Profile dashboard loading performance.

    Args:
        dashboard_url: URL of the dashboard.
    """
    logger.info("Profiling dashboard loading performance")

    # This would typically require a headless browser or Selenium
    # For now, we'll just check if the dashboard is responsive
    try:
        response = requests.get(dashboard_url)

        if response.status_code == 200:
            logger.info(f"Dashboard is responsive: {response.status_code}")
        else:
            logger.warning(f"Dashboard may have issues: {response.status_code}")

    except Exception as e:
        logger.error(f"Dashboard connection error: {str(e)}")

    logger.info(
        "Note: For detailed dashboard profiling, use browser developer tools or Selenium"
    )


def save_profiling_results(results: Dict, output_file: str):
    """
    Save profiling results to a file.

    Args:
        results: Profiling results.
        output_file: Path to save results.
    """
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Profiling results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile application performance")
    parser.add_argument(
        "--api-url",
        type=str,
        default=f"http://{config.API_HOST}:{config.API_PORT}",
        help="API URL",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default=f"http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}",
        help="Dashboard URL",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for API profiling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiling_results.json",
        help="Output file for profiling results",
    )
    parser.add_argument(
        "--api-only", action="store_true", help="Profile only API endpoints"
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Profile only model training and inference",
    )
    parser.add_argument(
        "--dashboard-only", action="store_true", help="Profile only dashboard loading"
    )

    args = parser.parse_args()

    all_results = {}

    # Reset profiling stats
    reset_profiling_stats()

    # Profile API if requested
    if args.api_only or not (args.model_only or args.dashboard_only):
        logger.info("==== Profiling API Endpoints ====")
        api_results = profile_api_endpoints(args.api_url, args.iterations)
        all_results["api"] = api_results

    # Profile model training and inference if requested
    if args.model_only or not (args.api_only or args.dashboard_only):
        logger.info("==== Profiling Model Training and Inference ====")
        model_results = profile_model_training(args.api_url)
        all_results["model"] = model_results

    # Profile dashboard loading if requested
    if args.dashboard_only or not (args.api_only or args.model_only):
        logger.info("==== Profiling Dashboard Loading ====")
        profile_dashboard_loading(args.dashboard_url)
        # Dashboard profiling results are collected in profiling_results
        all_results["dashboard"] = get_profiling_stats().get("dashboard", {})

    # Get other profiling stats
    other_stats = get_profiling_stats()
    for category, stats in other_stats.items():
        if category not in all_results:
            all_results[category] = stats

    # Save results
    save_profiling_results(all_results, args.output)

    logger.info("Profiling completed!")


if __name__ == "__main__":
    main()
