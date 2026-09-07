"""Voice pipeline: speech-to-text -> RAG query -> text-to-speech.

Mirrors the STT/TTS flow from the LLMandN8Nfun project, but replaces the raw
Ollama LLM with the local RAG pipeline (exposed via the FastAPI query endpoint).
The three stages communicate over the OpenAI-compatible REST APIs of Speaches
(STT) and Kokoro (TTS), and the existing RAG query API.
"""
from typing import Any, Dict, Optional

import requests

from src.config import (
    RAG_API_URL,
    STT_MODEL,
    STT_URL,
    TTS_MODEL,
    TTS_URL,
    TTS_VOICE,
)


class VoicePipelineError(Exception):
    """Raised when any stage of the voice pipeline fails."""


class VoicePipeline:
    def __init__(
        self,
        stt_url: Optional[str] = None,
        tts_url: Optional[str] = None,
        rag_api_url: Optional[str] = None,
    ) -> None:
        self.stt_url = stt_url or STT_URL
        self.tts_url = tts_url or TTS_URL
        self.rag_api_url = rag_api_url or RAG_API_URL

    def transcribe(self, audio: bytes, filename: str = "audio.wav") -> str:
        """Speech-to-text via the Speaches OpenAI-compatible endpoint."""
        url = f"{self.stt_url.rstrip('/')}/audio/transcriptions"
        try:
            resp = requests.post(
                url,
                files={"file": (filename, audio, "audio/wav")},
                data={"model": STT_MODEL},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except Exception as exc:  # noqa: BLE001 - surface any transport/API error
            raise VoicePipelineError(f"STT failed: {exc}") from exc

    def answer(self, text: str) -> Dict[str, Any]:
        """Query the RAG API with the transcribed text."""
        try:
            resp = requests.post(
                self.rag_api_url,
                json={"query": text},
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise VoicePipelineError(f"RAG query failed: {exc}") from exc

    def synthesize(self, text: str) -> bytes:
        """Text-to-speech via the Kokoro OpenAI-compatible endpoint."""
        url = f"{self.tts_url.rstrip('/')}/audio/speech"
        try:
            resp = requests.post(
                url,
                json={
                    "model": TTS_MODEL,
                    "input": text,
                    "voice": TTS_VOICE,
                    "response_format": "mp3",
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            raise VoicePipelineError(f"TTS failed: {exc}") from exc

    def process_audio(self, audio: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """Run the full voice flow and return transcript, answer and audio."""
        transcript = self.transcribe(audio, filename)
        if not transcript:
            raise VoicePipelineError("Could not transcribe any speech.")

        rag_result = self.answer(transcript)
        answer_text = (rag_result.get("response") or "").strip()
        if not answer_text:
            raise VoicePipelineError("RAG returned an empty response.")

        audio_out = self.synthesize(answer_text)

        return {
            "transcript": transcript,
            "answer": answer_text,
            "audio": audio_out,
            "sources": rag_result.get("sources", []),
            "is_relevant": rag_result.get("is_relevant", False),
            "vector_distance": rag_result.get("vector_distance"),
        }
