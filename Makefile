.PHONY: help uvenv sync format lint test cov run


help:
@echo "Common commands:"
@echo " make uvenv - create/refresh local .venv using uv"
@echo " make sync - sync deps (honors uv.lock if present)"
@echo " make format - run black"
@echo " make lint - run ruff + mypy"
@echo " make test - run pytest"
@echo " make cov - run pytest with coverage"
@echo " make run - run CLI (example)"


uvenv:
uv venv --seed --python 3.12


sync:
uv sync --all-extras --group dev


format:
uv run black src tests


lint:
uv run ruff check src tests
uv run mypy src


test:
uv run pytest


cov:
uv run pytest --cov=project_name --cov-report=term-missing


run:
uv run python -m project_name.cli --help