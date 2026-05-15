# Makefile for TB Drug Discovery Pipeline
# Provides standardized commands for development, testing, and deployment

.PHONY: install install-dev test lint format train clean docker-build docker-run mlflow-ui

# Python interpreter
PYTHON := python
PIP := pip

# Directories
SRC_DIR := src
TESTS_DIR := tests
SCRIPTS_DIR := scripts
CONFIG_DIR := config

# Default target
all: install test

# Installation
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

# Testing
test:
	pytest $(TESTS_DIR)/ -v --cov=$(SRC_DIR) --cov-report=term-missing \
		--ignore=$(TESTS_DIR)/test_alphafold.py \
		--ignore=$(TESTS_DIR)/test_docking.py

test-all:
	pytest $(TESTS_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html

test-property:
	pytest $(TESTS_DIR)/test_property_based.py -v --hypothesis-seed=42

test-integration:
	pytest $(TESTS_DIR)/test_integration.py -v

test-regression:
	pytest $(TESTS_DIR)/test_regression.py -v

# Code quality
lint:
	black --check --line-length=100 $(SRC_DIR)/ $(TESTS_DIR)/ $(SCRIPTS_DIR)/
	isort --check-only --profile=black $(SRC_DIR)/ $(TESTS_DIR)/ $(SCRIPTS_DIR)/
	flake8 $(SRC_DIR)/ $(TESTS_DIR)/ --max-line-length=100 --extend-ignore=E203,W503
	mypy $(SRC_DIR)/ --ignore-missing-imports --no-strict-optional

format:
	black --line-length=100 $(SRC_DIR)/ $(TESTS_DIR)/ $(SCRIPTS_DIR)/
	isort --profile=black $(SRC_DIR)/ $(TESTS_DIR)/ $(SCRIPTS_DIR)/

# Training commands
train-qsar:
	$(PYTHON) $(SCRIPTS_DIR)/train_qsar.py

train-gnn:
	$(PYTHON) $(SCRIPTS_DIR)/train_gnn.py

train-vae:
	$(PYTHON) $(SCRIPTS_DIR)/train_vae.py

train-diffusion:
	$(PYTHON) $(SCRIPTS_DIR)/train_diffusion.py

train-all: train-qsar train-gnn train-vae

# Docking
screen:
	$(PYTHON) $(SCRIPTS_DIR)/run_docking.py

# API
api:
	uvicorn $(SRC_DIR).api.app:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn $(SRC_DIR).api.app:app --host 0.0.0.0 --port 8000 --workers 4

# MLflow
mlflow-ui:
	mlflow ui --port 5000 --backend-store-uri ./mlruns

# Docker
docker-build:
	docker build -t tb-drug-discovery:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models tb-drug-discovery:latest

docker-run-dev:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models -v $(PWD)/src:/app/src tb-drug-discovery:latest

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/

# Data and models
clean-data:
	rm -rf data/processed/*
	rm -rf results/*

clean-models:
	rm -rf models/*.pt models/*.pkl models/*.joblib

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run tests (excluding slow tests)"
	@echo "  test-all      - Run all tests with coverage report"
	@echo "  lint          - Run code quality checks"
	@echo "  format        - Format code with black and isort"
	@echo "  train-qsar    - Train QSAR model"
	@echo "  train-gnn     - Train GNN model"
	@echo "  train-vae     - Train VAE generative model"
	@echo "  api           - Run API server (development)"
	@echo "  mlflow-ui     - Launch MLflow tracking UI"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  clean         - Clean generated files"
