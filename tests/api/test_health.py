"""API tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_check(client):
    """Test readiness check endpoint."""
    response = client.get("/health/readiness")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_liveness_check(client):
    """Test liveness check endpoint."""
    response = client.get("/health/liveness")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_version(client):
    """Test version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()