.PHONY: install lint format test up down logs clean dev

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff format .

test:
	pytest -v --cov=src/autonoma

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v

dev:
	uvicorn autonoma.serving.app:app --reload --host 0.0.0.0 --port 8000
