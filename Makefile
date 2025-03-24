.PHONY: help setup lint format test test-cov clean build-dev build-prod run-api run-dashboard run-all docs

help:
	@echo "Available commands:"
	@echo "  setup        Install development dependencies and set up pre-commit hooks"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  clean        Remove build artifacts and cache directories"
	@echo "  build-dev    Build development Docker image"
	@echo "  build-prod   Build production Docker image"
	@echo "  run-api      Run the API locally"
	@echo "  run-dashboard Run the dashboard locally"
	@echo "  run-all      Run both the API and dashboard with Docker Compose"
	@echo "  docs         Generate documentation"

setup:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:
	flake8 .
	mypy .
	black --check .
	isort --check --profile black .

format:
	black .
	isort --profile black .

test:
	pytest tests/

test-cov:
	pytest --cov=./ --cov-report=term --cov-report=html tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage htmlcov/ coverage.xml
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete

build-dev:
	docker build --target development -t supply-chain-forecaster:dev .

build-prod:
	docker build --target production -t supply-chain-forecaster:latest .

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	python -m dashboard.app

run-all:
	docker-compose up

docs:
	sphinx-build -b html docs/source docs/build