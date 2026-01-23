"""
Nox configuration for DeFoG testing.

Uses uv backend for fast virtual environment creation with virtualenv fallback.

Usage:
    nox                     # Run default sessions (install, tests)
    nox -s tests            # Run only tests
    nox -s tests -- -k test_model  # Run tests matching pattern

Requirements:
    pip install nox uv
"""

import nox

# Use uv backend for speed, fall back to virtualenv if uv not installed
nox.options.default_venv_backend = "uv|virtualenv"

# Default sessions to run
nox.options.sessions = ["install", "tests"]


@nox.session(python="3.9")
def install(session: nox.Session) -> None:
    """Test that the package installs correctly."""
    session.install("-e", ".")

    # Verify the package is importable
    session.run("python", "-c", "import defog.core; print('defog.core imported successfully')")
    session.run("python", "-c", "from defog.core import DeFoGModel; print('DeFoGModel imported successfully')")


@nox.session(python="3.9")
def tests(session: nox.Session) -> None:
    """Run the test suite with pytest."""
    session.install("-e", ".[dev]")

    # Run pytest with any additional arguments passed via --
    args = session.posargs or ["-v"]
    session.run("pytest", "tests/", *args)
