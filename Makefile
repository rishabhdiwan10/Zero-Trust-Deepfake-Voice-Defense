.PHONY: install test lint run benchmark clean

# Install all dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests with coverage
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Lint with flake8 and check formatting with black/isort
lint:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/
	flake8 src/ tests/ scripts/ --max-line-length=100

# Format code in place
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Run the real-time pipeline (requires model checkpoints)
run:
	python -m src.pipeline.realtime_pipeline

# Run latency benchmarks across all pipeline layers
benchmark:
	python scripts/benchmark_latency.py

# Generate synthetic voice samples for testing
generate-synthetic:
	python scripts/generate_synthetic.py

# Train the CNN deepfake detector
train:
	python scripts/train.py

# Evaluate model performance
evaluate:
	python scripts/evaluate.py

# Clean build artifacts and caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "Clean complete."
