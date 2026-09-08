import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_empty_payload():
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 400

@patch("src.api.pipeline")
@patch("src.api.synthesize_final_answer")
@patch("src.api.generate_reasoning_plan")
