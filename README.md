# Project Name


A reproducible scientific Python project built with [uv](https://github.com/astral-sh/uv).


## Quickstart (Windows/macOS/Linux)


```bash
# 1) Create local env and install deps
uv venv --seed --python 3.12
uv sync --all-extras --group dev


# 2) Run common tasks via nox (using uvx, no global installs needed)
uvx nox -l # list sessions
uvx nox -s run # show CLI help
uvx nox -s test # run tests
uvx nox -s format # black
uvx nox -s lint # ruff + mypy
uvx nox -s cov # coverage