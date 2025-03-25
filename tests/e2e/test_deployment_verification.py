"""
Deployment verification tests for the CI/CD pipeline.

These tests verify that a deployment is functioning correctly by checking:
1. API endpoints are accessible and responding correctly
2. Dashboard is accessible and loads properly
3. Basic functionality works end-to-end
"""

import os
import time
from urllib.parse import urljoin

import pytest
import requests

# Set default timeout for all requests
TIMEOUT = 10

# Environment variables should be set in the CI/CD pipeline
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DASHBOARD_BASE_URL = os.environ.get("DASHBOARD_BASE_URL", "http://localhost:8050")
DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV", "local")
# Optional API key if authentication is required
API_KEY = os.environ.get("API_KEY", "")


@pytest.fixture(scope="module")
def api_client():
    """Create a session for API requests with appropriate headers."""
    session = requests.Session()
    if API_KEY:
        session.headers.update({"Authorization": f"Bearer {API_KEY}"})
    return session


def test_api_health_endpoint(api_client):
    """Test that the API health endpoint is accessible and reporting healthy."""
    url = urljoin(API_BASE_URL, "/health")

    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = api_client.get(url, timeout=TIMEOUT)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            return
        except (requests.RequestException, AssertionError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise
            print(f"Retrying API health check... ({retry_count}/{max_retries})")
            time.sleep(5)


def test_dashboard_accessibility():
    """Test that the dashboard is accessible and returns a 200 status code."""
    url = DASHBOARD_BASE_URL

    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            assert response.status_code == 200
            return
        except (requests.RequestException, AssertionError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise
            print(
                f"Retrying dashboard accessibility check... ({retry_count}/{max_retries})"
            )
            time.sleep(5)


def test_basic_forecast_endpoint(api_client):
    """Test that the forecasting endpoint is working with sample data."""
    url = urljoin(API_BASE_URL, "/forecasts/predict")

    # Create a minimal payload for testing
    payload = {
        "model_name": "DemandForecast",
        "model_type": "ExponentialSmoothing",
        "data": [
            {"date": "2023-01-01", "demand": 100},
            {"date": "2023-01-02", "demand": 110},
            {"date": "2023-01-03", "demand": 120},
            {"date": "2023-01-04", "demand": 115},
            {"date": "2023-01-05", "demand": 125},
        ],
        "horizon": 3,
        "frequency": "D",
        "target_column": "demand",
    }

    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = api_client.post(url, json=payload, timeout=TIMEOUT)
            assert response.status_code == 200
            data = response.json()
            assert "forecasts" in data
            assert len(data["forecasts"]) == 3  # Should match our horizon
            return
        except (requests.RequestException, AssertionError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise
            print(f"Retrying forecast endpoint check... ({retry_count}/{max_retries})")
            time.sleep(5)


@pytest.mark.skipif(
    DEPLOYMENT_ENV == "production", reason="Skip data-modifying tests in production"
)
def test_model_training_endpoint(api_client):
    """Test that the model training endpoint works."""
    url = urljoin(API_BASE_URL, "/models/train")

    # Create a minimal payload for testing
    payload = {
        "model_name": "TestDeploymentModel",
        "model_type": "ExponentialSmoothing",
        "data": [
            {"date": "2023-01-01", "demand": 100},
            {"date": "2023-01-02", "demand": 110},
            {"date": "2023-01-03", "demand": 120},
            {"date": "2023-01-04", "demand": 115},
            {"date": "2023-01-05", "demand": 125},
        ],
        "parameters": {"seasonal": "add", "seasonal_periods": 1},
        "target_column": "demand",
    }

    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = api_client.post(
                url, json=payload, timeout=TIMEOUT * 2
            )  # Training might take longer
            assert response.status_code == 200
            data = response.json()
            assert "model_id" in data
            assert data["status"] == "success"
            return
        except (requests.RequestException, AssertionError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise
            print(
                f"Retrying model training endpoint check... ({retry_count}/{max_retries})"
            )
            time.sleep(5)


def test_end_to_end_workflow(api_client):
    """
    Test an end-to-end workflow:
    1. Upload data
    2. Train a model
    3. Make a prediction

    This is skipped in production to avoid modifying production data.
    """
    if DEPLOYMENT_ENV == "production":
        pytest.skip("Skipping end-to-end test in production")

    # 1. Upload test data (if the API has a data upload endpoint)
    # This would depend on your API structure

    # 2. Train a model
    train_url = urljoin(API_BASE_URL, "/models/train")
    model_name = f"E2ETest_{int(time.time())}"
    train_payload = {
        "model_name": model_name,
        "model_type": "ExponentialSmoothing",
        "data": [
            {"date": "2023-01-01", "demand": 100},
            {"date": "2023-01-02", "demand": 110},
            {"date": "2023-01-03", "demand": 120},
            {"date": "2023-01-04", "demand": 115},
            {"date": "2023-01-05", "demand": 125},
        ],
        "parameters": {"seasonal": "add", "seasonal_periods": 1},
        "target_column": "demand",
    }

    response = api_client.post(train_url, json=train_payload, timeout=TIMEOUT * 2)
    assert response.status_code == 200
    train_data = response.json()
    model_id = train_data["model_id"]

    # 3. Make a prediction using the trained model
    predict_url = urljoin(API_BASE_URL, "/forecasts/predict")
    predict_payload = {
        "model_id": model_id,
        "horizon": 3,
        "return_confidence_intervals": True,
    }

    response = api_client.post(predict_url, json=predict_payload, timeout=TIMEOUT)
    assert response.status_code == 200
    predict_data = response.json()
    assert "forecasts" in predict_data
    assert len(predict_data["forecasts"]) == 3

    # 4. Cleanup (delete test model if possible)
    # This would depend on your API structure"""
