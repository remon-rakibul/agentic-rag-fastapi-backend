"""Test health check endpoint."""
import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test that the health check endpoint works."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "healthy"]

