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
* Automation: GNU Make, GitHub Actions CI

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
│   └── vector_store.py       # ChromaDB embedding and metadata-filtered search
├── tests/                    # PyTest integration and unit tests
├── app.py                    # Streamlit web UI script
├── docker-compose.yml        # Multi-service stack definition
├── Dockerfile                # Image definition for API and UI
├── Makefile                  # Local shortcut task wrapper
└── requirements.txt          # Python project dependencies

---

## Testing & CI Workflow

Pushing to main automatically triggers a GitHub Actions pipeline that runs unit tests, calculates code coverage via pytest-cov, and uploads the resulting report as an artifact.

Run tests locally:
pytest --cov=src --cov-report=term-missing

---

## License

Distributed under the MIT License. See LICENSE for details.
