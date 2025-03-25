"""Test configuration for the supply chain forecaster."""

import os

# Add the project root to the Python path to fix imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use try/except to handle missing modules during testing
try:
    from config import config
except ImportError:
    # Create a minimal mock config for testing
    class MockConfig:
        def __init__(self):
            self.API_HOST = "localhost"
            self.API_PORT = 8000
            self.DASHBOARD_HOST = "localhost"
            self.DASHBOARD_PORT = 8050
            self.LOG_LEVEL = "INFO"
            self.DASHBOARD_DEBUG = False

    config = MockConfig()

try:
    from api.app import create_app
except (ImportError, RuntimeError):
    # Create a mock create_app function
    def create_app():
        return None


try:
    from dashboard.app import create_dashboard
except (ImportError, RuntimeError):
    # Create a mock create_dashboard function
    def create_dashboard():
        return None


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    return create_app()


@pytest.fixture
def dashboard():
    """Create a Dash app for testing."""
    return create_dashboard()


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_time_series_data():
    """Create a sample time series dataset."""
    # Create a basic time series dataset
    dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
    n = len(dates)

    # Create features
    df = pd.DataFrame(
        {
            "date": dates,
            "demand": 100
            + 20 * pd.Series(range(n)) / n
            + 10 * pd.Series(range(n)).apply(lambda x: np.sin(x * 2 * np.pi / 30))
            + np.random.normal(0, 5, n),
            "temperature": 20
            + 10 * pd.Series(range(n)).apply(lambda x: np.sin(x * 2 * np.pi / 365))
            + np.random.normal(0, 2, n),
            "promotion": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        }
    )

    return df


@pytest.fixture
def sample_anomaly_data():
    """Create a sample dataset with anomalies."""
    # Create a basic dataset
    n = 500

    # Create features
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", end="2023-06-15", freq="D")[:n],
            "demand": 100
            + 20 * pd.Series(range(n)) / n
            + 10 * pd.Series(range(n)).apply(lambda x: np.sin(x * 2 * np.pi / 30))
            + np.random.normal(0, 5, n),
            "inventory": 200
            - 20 * pd.Series(range(n)) / n
            + 15 * pd.Series(range(n)).apply(lambda x: np.cos(x * 2 * np.pi / 30))
            + np.random.normal(0, 8, n),
            "lead_time": 5
            + 2 * pd.Series(range(n)).apply(lambda x: np.sin(x * 2 * np.pi / 90))
            + np.random.normal(0, 1, n),
        }
    )

    # Add anomalies
    anomaly_indices = np.random.choice(range(n), size=int(0.05 * n), replace=False)
    df.loc[anomaly_indices, "demand"] += np.random.normal(50, 10, len(anomaly_indices))
    df.loc[anomaly_indices, "inventory"] -= np.random.normal(
        30, 5, len(anomaly_indices)
    )

    # Add anomaly flag for validation
    df["is_anomaly"] = 0
    df.loc[anomaly_indices, "is_anomaly"] = 1

    return df
