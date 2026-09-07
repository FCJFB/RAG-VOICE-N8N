# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest

.PHONY: help venv install api ui test clean

help:
	@echo "Available commands:"
	@echo "  make api    - Start the FastAPI backend server (Port 8000)"
	@echo "  make ui     - Start the Streamlit web interface (Port 8501)"
	@echo "  make test   - Run the PyTest unit test suite"
	@echo "  make cli    - Launch interactive CLI mode"
	@echo "  make clean  - Remove bytecode and temporary cache files"

# Ensure venv exists automatically
$(PYTHON):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(PYTHON)
	$(PIP) install -r requirements.txt

api: $(PYTHON)
	$(UVICORN) src.api:app --reload --port 8000

ui: $(PYTHON)
	$(STREAMLIT) run app.py

test: $(PYTHON)
	$(PYTHON) -m pytest -v

cli: $(PYTHON)
	$(PYTHON) -m src.pipeline

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
ingest:
	./venv/bin/python -m src.ingest
inspect:
	./venv/bin/python -m src.inspect_db