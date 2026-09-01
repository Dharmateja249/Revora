# pyrefly: ignore [missing-import]
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check_status_code():
    """Verify that the health check endpoint returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_payload():
    """Verify that the health check endpoint returns expected JSON payload."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "Revora"
    assert "version" in data
