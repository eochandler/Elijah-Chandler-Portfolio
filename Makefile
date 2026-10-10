PYTHON ?= python
VENV ?= .venv
PROJECT_ROOT := $(shell pwd)

setup:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

train:
	. .venv/bin/activate && PYTHONPATH=src python -m src.turbofan.train

evaluate:
	. .venv/bin/activate && PYTHONPATH=src python -m src.turbofan.evaluate

export:
	. .venv/bin/activate && PYTHONPATH=src python -m src.turbofan.export

serve:
	. .venv/bin/activate && PYTHONPATH=src uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8000}

serve-dev:
	. .venv/bin/activate && PYTHONPATH=src uvicorn app.main:app --reload --host 0.0.0.0 --port $${PORT:-8000}

mlflow-ui:
	. .venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

docker-build:
	docker build -t turbofan-rul .

docker-run:
	docker run --rm -p $${PORT:-8000}:$${PORT:-8000} --env PORT=$${PORT:-8000} turbofan-rul

.PHONY: setup train export serve serve-dev mlflow-ui test lint docker-build docker-run
