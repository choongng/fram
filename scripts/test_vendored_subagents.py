"""Run tau-subagents' own tests against the Tau that Fram bundles.

The vendored suite runs against upstream tau-ai and against Fram's Tau, which
can diverge from it; this catches a divergence that breaks the extension Fram
ships.

Usage:
    uv run python scripts/test_vendored_subagents.py [PYTEST ARGS ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "vendor" / "tau-subagents"


def main(argv: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": str(VENDORED / "src")}
    # Running from the vendored directory makes its pyproject the pytest config,
    # so Fram's root conftest is not loaded.
    command = [sys.executable, "-m", "pytest", "tests", "-p", "no:cacheprovider"]
    return subprocess.run([*command, *argv], cwd=VENDORED, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
