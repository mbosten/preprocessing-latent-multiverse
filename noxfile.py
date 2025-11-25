import nox

PKG = "project_name"  # ← your package folder under src/

@nox.session(venv_backend="none")
def run(session):
    session.run(
        "uv", "run", "python", "-m", f"{PKG}.cli", "--help",
        external=True, env={"PYTHONPATH": "src"}
    )

@nox.session(venv_backend="none")
def format(session):
    session.run("uv", "run", "black", "src", "tests", external=True)

@nox.session(venv_backend="none")
def lint(session):
    session.run("uv", "run", "ruff", "check", "src", "tests", "--fix", external=True)
    session.run("uv", "run", "mypy", "src", external=True, env={"PYTHONPATH": "src"})

@nox.session(venv_backend="none")
def test(session):
    session.run("uv", "run", "pytest", "-q", external=True, env={"PYTHONPATH": "src"})

@nox.session(venv_backend="none")
def cov(session):
    session.run(
        "uv", "run", "pytest",
        f"--cov={PKG}", "--cov-report=term-missing",
        external=True, env={"PYTHONPATH": "src"}
    )