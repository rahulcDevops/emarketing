"""
tests/test_api.py
Unit, contract, and deployment variant tests for LeadSentry API.
"""

from fastapi.testclient import TestClient
import pytest
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "champion_version" in data
    assert "artifact_hash" in data
    assert len(data["classes"]) == 6


def test_predict_champion_default(client: TestClient):
    payload = {"text": "Can you send over your calendar link for a demo?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "INTERESTED_DEMO"
    assert data["deployment_variant"] == "champion"
    assert "X-Request-ID" in response.headers


def test_canary_routing_header(client: TestClient):
    payload = {"text": "Stop emailing me immediately. Please unsubscribe."}
    # Test with explicit header
    response = client.post("/predict", json=payload, headers={"X-Model-Variant": "challenger"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "HARD_UNSUBSCRIBE"
    # Fallback to champion if no challenger is active, or canary if active
    assert data["deployment_variant"] in ["canary", "champion"]


def test_feedback_loop(client: TestClient):
    payload = {
        "ticket_text": "Please remove our company from your sequence.",
        "predicted_category": "HARD_UNSUBSCRIBE",
        "corrected_category": "HARD_UNSUBSCRIBE",
        "confidence": 0.98,
        "model_version": "v3",
        "is_correct": True,
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


def test_metrics_scrape(client: TestClient):
    client.post("/predict", json={"text": "I am traveling on leave until next week."})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "leadsentry_predictions_total" in response.text
