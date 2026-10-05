"""
tests/test_integration.py
End-to-end integration tests verifying real artifact compatibility with the API.
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from api.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_artifacts_exist():
    """Verify production artifacts exist on disk before integration checks."""
    model_file = ARTIFACTS_DIR / "model.joblib"
    metadata_file = ARTIFACTS_DIR / "metadata.json"
    assert model_file.exists(), "artifacts/model.joblib does not exist. Train the model first."
    assert metadata_file.exists(), "artifacts/metadata.json does not exist."


def test_realistic_triage_flow(client: TestClient):
    """Test realistic customer ticket queries and verify prediction schema invariants."""
    test_cases = [
        "My card was retained by an ATM cash machine.",
        "Why is there an unrecognised foreign transaction fee on my statement?",
        "I need to transfer money to another account abroad.",
    ]

    for ticket in test_cases:
        response = client.post("/predict", json={"text": ticket})
        assert response.status_code == 200, f"Request failed for: {ticket}"
        
        payload = response.json()
        assert isinstance(payload["category"], str)
        assert len(payload["category"]) > 0
        assert isinstance(payload["confidence"], float)
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["model_version"] == "v1"


def test_predict_rejects_payload_too_long(client: TestClient):
    """Verify input guardrails reject payloads exceeding max length constraint (2000 chars)."""
    long_ticket = "a" * 2001
    response = client.post("/predict", json={"text": long_ticket})
    assert response.status_code == 422
