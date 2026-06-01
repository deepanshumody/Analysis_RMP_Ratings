.PHONY: setup test lint analysis clean

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,app]"

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check src tests

analysis:
	. .venv/bin/activate && python -m ape.questions

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
