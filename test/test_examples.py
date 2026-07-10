"""Every script under examples/ must run to completion without error.

Each example is executed in a subprocess so a crash in one does not affect
the others, and so the examples are exercised exactly as an end user would
run them (``python examples/<name>.py``).
"""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs_cleanly(script):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(EXAMPLES_DIR.parent),
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_examples_directory_is_not_empty():
    assert EXAMPLE_SCRIPTS, "no example scripts found under examples/"
