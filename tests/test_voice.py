import base64
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.voice_api import app
from src.voice_pipeline import VoicePipeline, VoicePipelineError

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["interface"] == "voice"


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_process_audio_happy_path():
    pipe = VoicePipeline()
    with patch.object(pipe, "transcribe", return_value="What is spatial filtering?"), \
         patch.object(pipe, "answer", return_value={"response": "It is a technique.", "sources": [], "is_relevant": True}), \
         patch.object(pipe, "synthesize", return_value=b"\x00\x01audio"):
        result = pipe.process_audio(b"fake-audio")

    assert result["transcript"] == "What is spatial filtering?"
    assert result["answer"] == "It is a technique."
    assert result["audio"] == b"\x00\x01audio"


def test_process_audio_empty_transcript_raises():
    pipe = VoicePipeline()
    with patch.object(pipe, "transcribe", return_value=""):
        try:
            pipe.process_audio(b"fake-audio")
        except VoicePipelineError as exc:
            assert "transcribe" in str(exc)
        else:
            raise AssertionError("Expected VoicePipelineError")


def test_transcribe_posts_audio():
    pipe = VoicePipeline(stt_url="http://stt:8000/v1")
    mock_resp = Mock()
    mock_resp.json.return_value = {"text": "hello world"}
    with patch("src.voice_pipeline.requests.post", return_value=mock_resp) as post:
        text = pipe.transcribe(b"audio", "clip.wav")

    assert text == "hello world"
    assert post.call_args.args[0] == "http://stt:8000/v1/audio/transcriptions"
    assert post.call_args.kwargs["files"]["file"][0] == "clip.wav"
    assert "model" in post.call_args.kwargs["data"]


def test_answer_posts_to_rag_api():
    pipe = VoicePipeline(rag_api_url="http://api:8000/api/v1/query")
    mock_resp = Mock()
    mock_resp.json.return_value = {"response": "the answer"}
    with patch("src.voice_pipeline.requests.post", return_value=mock_resp) as post:
        result = pipe.answer("q")

    assert result["response"] == "the answer"
    assert post.call_args.args[0] == "http://api:8000/api/v1/query"
    assert post.call_args.kwargs["json"] == {"query": "q"}


def test_synthesize_returns_audio_bytes():
    pipe = VoicePipeline(tts_url="http://tts:8880/v1")
    mock_resp = Mock()
    mock_resp.content = b"mp3bytes"
    with patch("src.voice_pipeline.requests.post", return_value=mock_resp) as post:
        audio = pipe.synthesize("hello")

    assert audio == b"mp3bytes"
    assert post.call_args.args[0] == "http://tts:8880/v1/audio/speech"


@patch("src.voice_api.pipeline")
def test_voice_ask_happy_path(mock_pipe):
    mock_pipe.process_audio.return_value = {
        "transcript": "hi",
        "answer": "hello",
        "sources": [],
        "is_relevant": True,
        "vector_distance": None,
        "audio": b"\x00\x01",
    }
    response = client.post(
        "/voice/ask",
        files={"file": ("a.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "hello"
    assert data["audio_content_type"] == "audio/mpeg"
    assert data["audio_base64"] == base64.b64encode(b"\x00\x01").decode()


@patch("src.voice_api.pipeline")
def test_voice_ask_error_path(mock_pipe):
    mock_pipe.process_audio.side_effect = VoicePipelineError("boom")
    response = client.post(
        "/voice/ask",
        files={"file": ("a.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 502
