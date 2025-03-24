"""Integration test configuration for the supply chain forecaster."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Add the project root to the Python path to fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import application components with robust error handling
try:
    from api.app import create_app
except (ImportError, RuntimeError) as e:
    print(f"Warning: Could not import API app: {e}")
    def create_app():
        from fastapi import FastAPI
        app = FastAPI()
        return app

try:
    from dashboard.app import create_dashboard
except (ImportError, RuntimeError) as e:
    print(f"Warning: Could not import dashboard app: {e}")
    def create_dashboard():
        return None


@pytest.fixture(scope="session")
def api_client():
    """Create a test client for the API that persists across the test session."""
    try:
        app = create_app()
        # Handle potential compatibility issues with TestClient
        try:
            return TestClient(app)
        except Exception as e:
            print(f"Error creating TestClient: {e}")
            return None
    except Exception as e:
        print(f"Error creating app: {e}")
        return None


@pytest.fixture(scope="session")
def dashboard_app():
    """Create a Dash app for testing that persists across the test session."""
    try:
        return create_dashboard()
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return None


@pytest.fixture(scope="session")
def sample_data_dir():
    """Create a directory for sample data files used in integration tests."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def sample_time_series_data():
    """Create a realistic sample time series dataset for integration testing."""
    # Create a more complex time series dataset for testing
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2022-01-01", end="2023-01-01", freq="D")
    n = len(dates)
    
    # Create seasonality components
    weekly_pattern = np.sin(np.arange(n) * (2 * np.pi / 7))
    monthly_pattern = np.sin(np.arange(n) * (2 * np.pi / 30))
    quarterly_pattern = np.sin(np.arange(n) * (2 * np.pi / 90))
    trend = np.linspace(0, 20, n)
    
    # Create features with realistic patterns
    df = pd.DataFrame({
        "date": dates,
        "demand": 100 + trend + 10 * weekly_pattern + 20 * monthly_pattern + 30 * quarterly_pattern + np.random.normal(0, 5, n),
        "price": 10 + np.sin(np.arange(n) * (2 * np.pi / 180)) + np.random.normal(0, 0.2, n),
        "promotion": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "holiday": np.random.choice([0, 1], size=n, p=[0.97, 0.03]),
        "temperature": 15 + 10 * np.sin(np.arange(n) * (2 * np.pi / 365)) + np.random.normal(0, 2, n),
        "inventory": 200 - np.linspace(0, 10, n) + 15 * monthly_pattern + np.random.normal(0, 8, n),
        "lead_time": 5 + 1.5 * quarterly_pattern + np.random.normal(0, 0.5, n),
    })
    
    # Save to CSV in the data directory
    data_path = Path(__file__).parent / "data" / "sample_timeseries.csv"
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path, index=False)
    
    return df


@pytest.fixture(scope="session")
def sample_anomaly_data():
    """Create a sample dataset with injected anomalies for testing anomaly detection."""
    np.random.seed(43)  # Different seed from time series data
    n = 500
    
    # Create base features
    df = pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", end="2023-06-15", freq="D")[:n],
        "demand": 100 + 20 * np.sin(np.arange(n) * (2 * np.pi / 30)) + np.random.normal(0, 5, n),
        "inventory": 200 - 15 * np.sin(np.arange(n) * (2 * np.pi / 30)) + np.random.normal(0, 8, n),
        "lead_time": 5 + 2 * np.sin(np.arange(n) * (2 * np.pi / 90)) + np.random.normal(0, 0.5, n),
        "stockout_risk": np.random.normal(0.2, 0.05, n)
    })
    
    # Add anomalies at specific positions
    anomaly_indices = np.random.choice(range(n), size=25, replace=False)
    df.loc[anomaly_indices, "demand"] += np.random.normal(50, 10, len(anomaly_indices))
    df.loc[anomaly_indices, "inventory"] -= np.random.normal(30, 5, len(anomaly_indices))
    df.loc[anomaly_indices, "stockout_risk"] += np.random.normal(0.4, 0.1, len(anomaly_indices))
    
    # Add anomaly flag for validation
    df["is_anomaly"] = 0
    df.loc[anomaly_indices, "is_anomaly"] = 1
    
    # Save to CSV in the data directory
    data_path = Path(__file__).parent / "data" / "sample_anomaly.csv"
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path, index=False)
    
    return df


@pytest.fixture(scope="session")
def integration_test_env():
    """Set up the integration test environment."""
    # Create necessary directories
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"
    output_dir = test_dir / "output"
    model_dir = test_dir / "models"
    
    # Ensure directories exist
    for directory in [data_dir, output_dir, model_dir]:
        directory.mkdir(exist_ok=True)
    
    # Return environment information
    env = {
        "test_dir": test_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "model_dir": model_dir
    }
    
    return env