import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DB_DIR = BASE_DIR / "chroma_db"
DOC_PATH = DATA_DIR / "knowledge.txt"

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"
MAX_COSINE_DISTANCE = 350.0

# Reads Docker container network URL if available, defaults to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Voice interface (STT -> RAG -> TTS) ---
# Speaches (Whisper) OpenAI-compatible speech-to-text endpoint.
# Note: mapped to host port 8001 in docker-compose to avoid clashing with the
# RAG API on 8000; inside the compose network it is reachable as http://speaches:8000/v1.
STT_URL = os.getenv("STT_URL", "http://127.0.0.1:8001/v1")
STT_MODEL = os.getenv("STT_MODEL", "Systran/faster-whisper-medium.en")

# Kokoro OpenAI-compatible text-to-speech endpoint.
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:8880/v1")
TTS_MODEL = os.getenv("TTS_MODEL", "kokoro")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# RAG query API the voice service calls for answers.
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/api/v1/query")