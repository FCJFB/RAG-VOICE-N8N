# Academic Lecture RAG Pipeline (Educational Prototype)

![CI Status](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/actions/workflows/test.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A hands-on educational/toy project designed to explore and demonstrate fundamental Retrieval-Augmented Generation (RAG) concepts locally. Built around academic lecture slide decks (PDF) as a sample dataset, this repository serves as a practical sandbox for learning PDF ingestion, metadata-filtered vector search, two-pass Chain-of-Thought (CoT) reasoning, containerization, and local LLM orchestration.

> Note: This project is a personal learning sandbox created to understand RAG mechanics from the ground up, not an official product or production-grade lecture assistant.

---

## Architecture Overview

The pipeline uses a single-pass hybrid routing approach: incoming user queries are embedded via mxbai-embed-large and evaluated against ChromaDB vector distance metrics. Technical queries within the cosine distance threshold trigger a two-pass reasoning pipeline, while out-of-domain or conversational queries dynamically fall back to direct LLM processing. An optional metadata filter (course_id, lecture_num) narrows the search to a specific lecture's slides.

                  +-------------------------------------+
                  |          Streamlit Web UI           |
                  |   (Port 8501, course/lecture filter)|
                  +------------------+------------------+
                                     | REST HTTP
                                     v
                  +-------------------------------------+
                  |            FastAPI Server           |
                  |             (Port 8000)             |
                  +------------------+------------------+
                                     |
                        +------------+------------+
                        v                         v
             [ ChromaDB Vector Store ]   [ Distance Evaluator ]
             (mxbai-embed-large)         (Threshold <= 350.0)
             (course/lecture metadata)          |
                        |                         |
            +-----------+-----------+             |
            |                       |             |
    [ Relevant Query ]     [ Out-of-Domain ]      |
            |                       |             |
            v                       v             |
+----------------------+  +------------------+    |
| Pass 1: CoT Reasoning|  | Direct Fallback  |<---+
+-----------+----------+  +---------+--------+
            |                       |
            v                       |
+----------------------+            |
| Pass 2: Synthesis    |            |
+-----------+----------+            |
            |                       |
            +-----------+-----------+
                        |
                        v
             +---------------------+
             | Ollama Server       |
             | (Llama 3.2 Model)   |
             +---------------------+

---

## Tech Stack

* Language & Frameworks: Python 3.11+, FastAPI, Streamlit, Pydantic
* LLM & Embeddings: Ollama (llama3.2), LangChain (mxbai-embed-large)
* Vector Database: ChromaDB
* PDF Parsing: pdfplumber, pdfminer.six
* Containerization: Docker, Docker Compose
* Testing & Quality: PyTest, pytest-cov, pytest-mock
* Automation: GNU Make, GitHub Actions CI, n8n

---

## What This Project Demonstrates

* PDF Slide Ingestion: Parses lecture PDFs page-by-page with pdfplumber, chunks the extracted text, and stores embeddings in ChromaDB with per-slide metadata (course_id, lecture_num, slide_num).
* Metadata Filtering: Narrows vector search to a specific course and lecture so answers stay grounded in the requested material.
* Cosine Distance Routing: Demonstrates how vector distance metrics (threshold 350.0) can filter out out-of-domain queries and trigger fallback prompts to prevent hallucinations.
* Two-Pass Reasoning Engine: Implements basic Chain-of-Thought processing by asking the LLM to generate a reasoning plan over retrieved context before producing the final response.
* Full-Stack Local Delivery: Orchestrates API, UI, and LLM services locally using Docker Compose to avoid external API dependency costs.
* Isolated Testing: Uses pytest-mock to test pipeline branching logic rapidly without needing live LLM inference during CI runs.

---

## Quickstart Guide

### Prerequisites

* Docker & Docker Compose installed
* GNU Make (optional)

### 1. Launch with Docker Compose

git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
make docker-up

### 2. Pull Local Models

Download the required local models into the Ollama container volume:

docker exec -it cachyos_ollama ollama pull llama3.2
docker exec -it cachyos_ollama ollama pull mxbai-embed-large

### 3. Ingest Lecture Slides

Place your lecture PDFs in the `data/` directory, then build the vector store:

docker exec -it cachyos_rag_api python -m src.ingest

Inspect the resulting index with:

docker exec -it cachyos_rag_api python -m src.inspect_db

### 4. Access Services

* Streamlit Frontend: http://localhost:8501
* FastAPI Interactive Docs: http://localhost:8000/docs
* Health Check Endpoint: http://localhost:8000/health
* Voice Interface: http://localhost:8765
* Speaches STT (Swagger): http://localhost:8001/docs
* Kokoro TTS: http://localhost:8880
* n8n Workflow Editor: http://localhost:5678

---

## Voice Interface (STT → RAG → TTS)

A browser-based voice interface (Port 8765) reuses the speech-to-text / text-to-speech pipeline from the companion project and routes it through the RAG pipeline instead of a raw LLM.

Flow: **microphone audio → Speaches (Whisper STT) → RAG query API → Kokoro (TTS) → spoken answer**.

Open http://localhost:8765, hold the "Hold to talk" button, ask a question about your lecture slides, and release to hear the answer. The page also shows the transcript, answer text, and cited sources.

When the n8n voice workflow is active (see below), the page posts each recording to the n8n webhook instead of the local `/voice/ask` endpoint. n8n then orchestrates the STT → RAG → TTS chain itself and appends every exchange — tagged with a per-browser `session_id` (kept in `localStorage`) — to `n8n_logs/voice_sessions.jsonl` on the host. This keeps the `voice_api.py` service as a thin HTML/static host while n8n sits in the middle of the data flow and logs history per session.

The stack is wired into `docker-compose.yml` (`speaches`, `kokoro`, and `voice` services) so `make docker-up` starts everything. Speaches is mapped to host port 8001 (its container still listens on 8000) to avoid clashing with the RAG API on 8000.

Speech-to-text uses the full `Systran/faster-whisper-medium.en` model (configured via `STT_MODEL` in `src/config.py`). On the GTX 1080 Ti (Pascal, compute capability 6.1) Whisper runs with `WHISPER__COMPUTE_TYPE=int8` — the only accelerated type CTranslate2 supports on that architecture (`float16`/`int8_float16` are rejected for lacking efficient FP16 compute). Both the Ollama and Speaches services reserve the GPU via `deploy.resources.reservations.devices` in `docker-compose.yml`.

Run the voice interface standalone for local development:

* Start the voice server: `make voice`
* The STT/TTS/RAG endpoints are configurable via `STT_URL`, `TTS_URL`, and `RAG_API_URL` (see `src/config.py`).

---

## n8n Workflow Orchestration

n8n (Port 5678) adds a visual, no-code orchestration layer on top of the Python pipeline. This demonstrates the same workflow that would otherwise be hand-rolled in code — webhook ingestion, HTTP calls to the RAG API, conditional branching, and text-to-speech — as a reusable, inspectable workflow.

The bundled workflow (`n8n/workflows/lecture-rag-assistant.json`) wires the services together:

```
Webhook (POST /lecture-rag)
   -> Query RAG API  (POST http://api:8000/api/v1/query)
   -> Is Relevant?   (branches on is_relevant)
        |-- true  -> Synthesize Speech (Kokoro) -> Respond Audio (MP3)
        |-- false -> Respond Text (JSON fallback)
```

Import it once the stack is running:

1. Open the n8n editor at http://localhost:5678 and complete first-time signup.
2. Workflows → Import from File → select `n8n/workflows/lecture-rag-assistant.json`.
3. Open the imported workflow and toggle it **Active** (registers the `/lecture-rag` webhook).
4. Trigger it with `curl`:
   ```
   curl -X POST http://localhost:5678/webhook/lecture-rag \
     -H "Content-Type: application/json" \
     -d '{"query": "What is spatial filtering?", "course_id": "IMAGEPROCESSING", "lecture_num": 1}'
   ```
   A relevant question returns the spoken answer as MP3; an out-of-domain question returns the text fallback JSON.

n8n persists its state in the `n8n_data` Docker volume and reaches the other services over the compose network (`api`, `kokoro`, `speaches`). To run n8n outside Docker, swap the node URLs from service names to `localhost` (e.g. `http://localhost:8000/api/v1/query`).

TODO: Is this still current?

### Voice workflow with per-session logging

A second bundled workflow (`n8n/workflows/voice-rag-assistant.json`) is what the 8765 voice UI posts to. It receives the recorded WAV (plus a `session_id`) as multipart form data and runs the full voice chain inside n8n:

```
Webhook (POST /voice-rag, multipart: file + session_id)
   -> Transcribe Audio  (Speaches Whisper STT, forwards the binary WAV)
   -> Query RAG API     (POST http://api:8000/api/v1/query)
   -> Synthesize Speech (Kokoro TTS -> MP3)
   -> Build Response + Log (Code node: base64-encode MP3 into JSON,
        and append transcript/answer to n8n_logs/voice_sessions.jsonl)
   -> Respond to Webhook (JSON: transcript, answer, sources, audio_base64)
```

Import it exactly like the text workflow, toggle it **Active**, then open http://localhost:8765 and talk to it. Each utterance lands as one JSON line in `n8n_logs/voice_sessions.jsonl`, keyed by the browser's `session_id`:

```bash
cat n8n_logs/voice_sessions.jsonl
```

The Code node needs `require('fs')` to append to the log, which is why the `n8n` service sets `NODE_FUNCTION_ALLOW_BUILTIN=fs` and bind-mounts `./n8n_logs` to `/home/node/logs` in `docker-compose.yml`. If you'd rather not use the filesystem log, remove that env var/mount and the `fs.appendFileSync(...)` line from the Code node — the flow still returns the spoken answer without logging.

The `n8n` service also sets `N8N_DEFAULT_BINARY_DATA_MODE=default` so the Code node can inline the TTS audio as base64 (`$input.first().binary.data.data`) in the webhook response. With n8n's default `filesystem-v2` storage that field is just the literal string `"filesystem-v2"` and the browser can't play the answer.

---

## Local Development & Makefile Shortcuts

If developing locally outside of Docker:

* Setup environment and dependencies: make install
* Parse PDFs and build the vector store: make ingest
* Inspect the vector store contents: make inspect
* Start FastAPI backend: make api
* Start Streamlit interface (separate window): make ui
* Execute test suite: make test

---

## Project Structure

.
├── .github/
│   └── workflows/
│       └── test.yml          # GitHub Actions CI workflow
├── data/                     # Sample lecture slide decks (PDF, git-ignored)
├── src/
│   ├── __init__.py
│   ├── api.py                # FastAPI endpoint handlers
│   ├── config.py             # Distance thresholds, model constants, paths
│   ├── ingest.py             # PDF parsing and ChromaDB ingestion
│   ├── inspect_db.py         # Vector store status inspector
│   ├── pipeline.py           # Core RAG execution logic
│   ├── reasoning.py          # Two-pass CoT prompt templates
│   ├── vector_store.py       # ChromaDB embedding and metadata-filtered search
│   ├── voice_api.py          # FastAPI voice interface (browser mic UI)
│   └── voice_pipeline.py     # STT -> RAG -> TTS orchestration
├── tests/                    # PyTest integration and unit tests
├── n8n/
│   └── workflows/
│       ├── lecture-rag-assistant.json   # n8n workflow orchestrating RAG -> TTS
│       └── voice-rag-assistant.json     # n8n voice workflow (STT/RAG/TTS + session log)
├── n8n_logs/                 # Per-session voice transcript/answer log (git-ignored)
├── app.py                    # Streamlit web UI script
├── docker-compose.yml        # Multi-service stack definition
├── Dockerfile                # Image definition for API and UI
├── Makefile                  # Local shortcut task wrapper
└── requirements.txt          # Python project dependencies

---

## Testing & CI Workflow

~~Pushing to main automatically triggers a GitHub Actions pipeline that runs unit tests, calculates code coverage via pytest-cov, and uploads the resulting report as an artifact.

Run tests locally:
pytest --cov=src --cov-report=term-missing~~
TODO: Needs overhaul, pytests are not uptodate with current pipeline

---

## License

Distributed under the MIT License. See LICENSE for details.
