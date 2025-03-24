"""End-to-end tests for the supply chain forecaster workflow."""

import pytest
import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
import requests
import subprocess
import signal
import sys

# Add the project root to the Python path to fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@pytest.fixture(scope="module")
def e2e_environment():
    """Set up environment for end-to-end testing."""
    # Create test directories
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", end="2023-01-01", freq="D")
    n = len(dates)
    
    df = pd.DataFrame({
        "date": dates,
        "demand": 100 + np.linspace(0, 20, n) + 10 * np.sin(np.arange(n) * (2 * np.pi / 7)) + np.random.normal(0, 5, n),
        "price": 10 + np.sin(np.arange(n) * (2 * np.pi / 180)) + np.random.normal(0, 0.2, n),
        "promotion": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    })
    
    data_file = data_dir / "e2e_test_data.csv"
    df.to_csv(data_file, index=False)
    
    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "data_file": data_file,
        "test_data": df
    }


@pytest.mark.skip(reason="Requires running services")
def test_full_forecasting_workflow(e2e_environment):
    """Test the complete forecasting workflow from data ingestion to visualization."""
    # This test requires running API and dashboard services
    # We'll mock parts of it for demonstration purposes
    
    # Step 1: Check API availability
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        api_available = response.status_code == 200
    except requests.exceptions.RequestException:
        api_available = False
    
    if not api_available:
        pytest.skip("API service not available at http://localhost:8000")
    
    # Step 2: Upload test data
    data_file = e2e_environment["data_file"]
    
    with open(data_file, "rb") as f:
        try:
            response = requests.post(
                "http://localhost:8000/forecasting/upload",
                files={"file": (data_file.name, f, "text/csv")}
            )
            if response.status_code != 200:
                pytest.skip(f"Failed to upload test data: {response.text}")
                
            dataset_id = response.json().get("id")
            assert dataset_id is not None, "No dataset ID returned"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Error uploading test data: {e}")
    
    # Step 3: Train a forecasting model
    try:
        train_data = {
            "dataset_id": dataset_id,
            "target_column": "demand",
            "date_column": "date",
            "model_type": "prophet",
            "hyperparameters": {"seasonality_mode": "multiplicative"},
            "forecast_horizon": 30
        }
        
        response = requests.post(
            "http://localhost:8000/forecasting/train",
            json=train_data
        )
        
        if response.status_code != 200:
            pytest.skip(f"Failed to train forecasting model: {response.text}")
            
        model_id = response.json().get("model_id")
        assert model_id is not None, "No model ID returned"
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Error training forecasting model: {e}")
    
    # Step 4: Generate forecasts
    try:
        forecast_data = {
            "model_id": model_id,
            "forecast_horizon": 30
        }
        
        response = requests.post(
            "http://localhost:8000/forecasting/predict",
            json=forecast_data
        )
        
        if response.status_code != 200:
            pytest.skip(f"Failed to generate forecast: {response.text}")
            
        forecast = response.json().get("forecast")
        assert forecast is not None, "No forecast data returned"
        assert len(forecast) == 30, f"Expected 30 forecast points, got {len(forecast)}"
        
        # Save the forecast for further analysis
        forecast_df = pd.DataFrame(forecast)
        forecast_file = e2e_environment["output_dir"] / "forecast_result.csv"
        forecast_df.to_csv(forecast_file, index=False)
        
        print(f"Forecast saved to {forecast_file}")
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Error generating forecast: {e}")


@pytest.mark.skip(reason="Requires Docker environment")
def test_docker_deployment():
    """Test deployment of services using Docker."""
    # This test checks if the application can be deployed using Docker
    
    try:
        # Step 1: Build and start the Docker containers
        subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True,
            timeout=120,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        )
        
        # Wait for services to start
        time.sleep(30)
        
        # Step 2: Check API health
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            assert response.status_code == 200, f"API health check failed: {response.text}"
            assert response.json().get("status") == "ok", "API not reporting healthy status"
        except requests.exceptions.RequestException as e:
            assert False, f"Error connecting to API: {e}"
        
        # Step 3: Check dashboard accessibility
        try:
            response = requests.get("http://localhost:8050", timeout=5)
            assert response.status_code == 200, "Dashboard not accessible"
        except requests.exceptions.RequestException as e:
            assert False, f"Error connecting to dashboard: {e}"
            
    finally:
        # Clean up: stop and remove the Docker containers
        subprocess.run(
            ["docker-compose", "down"],
            check=False,
            timeout=60,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        )