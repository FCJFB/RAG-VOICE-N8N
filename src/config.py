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