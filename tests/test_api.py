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
def test_query_relevant_path(mock_plan, mock_synth, mock_pipeline):
    fake_doc = MagicMock()
    fake_doc.page_content = "CachyOS uses the BORE scheduler."
    mock_pipeline.db.similarity_search_with_score.return_value = [(fake_doc, 100.0)]

    mock_plan.return_value = "Plan"
    mock_synth.return_value = "Final Answer"

    response = client.post("/api/v1/query", json={"query": "what scheduler does cachyos use?"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_relevant"] is True
    assert data["response"] == "Final Answer"

@patch("src.api.pipeline")
def test_query_fallback_path(mock_pipeline):
    fake_doc = MagicMock()
    fake_doc.page_content = "Unrelated content."
    mock_pipeline.db.similarity_search_with_score.return_value = [(fake_doc, 500.0)]
    
    # Return a plain string so isinstance(..., MagicMock) evaluates correctly
    mock_pipeline.llm.invoke.return_value = "Hello! How can I help?"

    response = client.post("/api/v1/query", json={"query": "waddup"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_relevant"] is False
    assert data["response"] == "Hello! How can I help?"