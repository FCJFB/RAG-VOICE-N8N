import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.pipeline import RAGPipeline

@pytest.fixture
def mock_pipeline(mocker):
    """Fixture to mock vector DB and LLM for fast unit testing."""
    mocker.patch("src.pipeline.get_or_create_vector_db")
    mocker.patch("src.pipeline.OllamaLLM")
    
    pipeline = RAGPipeline()
    return pipeline


def test_relevant_query_triggers_reasoning_pass(mock_pipeline, mocker, capsys):
    """Low distance score (<= 350.0) must trigger two-pass reasoning."""
    # 1. Mock vector search to return a relevant doc with low distance
    fake_doc = Document(page_content="CachyOS uses the BORE scheduler by default.")
    mock_pipeline.db.similarity_search_with_score.return_value = [(fake_doc, 120.0)]

    # 2. Mock Pass 1 and Pass 2 reasoning functions
    mocker.patch("src.pipeline.generate_reasoning_plan", return_value="STATUS: SUFFICIENT CONTEXT")
    mocker.patch("src.pipeline.synthesize_final_answer", return_value="CachyOS uses BORE.")

    # 3. Execute
    mock_pipeline.process_query("what scheduler does cachyos use?")

    # 4. Verify stdout execution path
    captured = capsys.readouterr()
    assert "[Vector Search Distance Score: 120.00]" in captured.out
    assert "LLM REASONING STEPS (PASS 1)" in captured.out
    assert "FINAL RAG RESPONSE (PASS 2)" in captured.out


def test_irrelevant_query_triggers_fallback_branch(mock_pipeline, capsys):
    """High distance score (> 350.0) must bypass reasoning and trigger fallback."""
    # 1. Mock vector search to return a high distance score
    fake_doc = Document(page_content="Unrelated text chunk.")
    mock_pipeline.db.similarity_search_with_score.return_value = [(fake_doc, 520.0)]
    
    # 2. Mock LLM fallback response
    mock_pipeline.llm.invoke.return_value = "Hello! How can I help you with CachyOS?"

    # 3. Execute with casual slang or out-of-scope query
    mock_pipeline.process_query("waddup")

    # 4. Verify reasoning was BYPASSED
    captured = capsys.readouterr()
    assert "[Vector Search Distance Score: 520.00]" in captured.out
    assert "PASS 1" not in captured.out
    assert "Hello! How can I help you with CachyOS?" in captured.out