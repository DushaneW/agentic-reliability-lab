.PHONY: install test lint format typecheck check benchmark dashboard clean

install:
	uv sync
	uv pip install -e .

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy -p arl

check: lint typecheck test

benchmark:
	uv run arl benchmark run benchmarks/smoke

dashboard:
	uv run uvicorn arl.dashboard.app:app --reload --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache arl.db reports/ *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
