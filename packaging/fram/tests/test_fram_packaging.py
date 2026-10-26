"""Tests for the fram distribution: its version, its entry point, and its wheel."""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from fram import __version__, tau_base_version
from fram.cli import main

PACKAGING = Path(__file__).resolve().parents[1]
REPO = PACKAGING.parents[1]


def _pyproject(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_fram_version_names_the_tau_base_in_pyproject() -> None:
    upstream = _pyproject(REPO / "pyproject.toml")["project"]
    assert isinstance(upstream, dict)
    assert tau_base_version() == upstream["version"], (
        "fram's version must start with the Tau release it bundles; "
        "bump fram/__init__.py after merging a Tau release"
    )
    assert len(__version__.split(".")) == 4


def test_dependencies_match_upstream() -> None:
    upstream = _pyproject(REPO / "pyproject.toml")["project"]
    fram = _pyproject(PACKAGING / "pyproject.toml")["project"]
    assert isinstance(upstream, dict) and isinstance(fram, dict)
    assert fram["dependencies"] == upstream["dependencies"]


def test_version_flag_prints_fram_and_tau_base(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--version"])
    assert capsys.readouterr().out == f"fram {__version__} (tau {tau_base_version()})\n"


def test_update_is_refused_with_reinstall_advice(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["update"])
    assert exit_info.value.code == 1
    assert "uv tool upgrade fram" in capsys.readouterr().err


def _local_index(root: Path) -> Path:
    """Build the fram wheel into a minimal file:// simple index at ``root``."""
    project = root / "fram"
    subprocess.run(
        ["uv", "build", "--wheel", "--project", str(PACKAGING), "--out-dir", str(project)],
        check=True,
        capture_output=True,
    )
    links = "".join(f'<a href="{wheel.name}">{wheel.name}</a>' for wheel in project.glob("*.whl"))
    (project / "index.html").write_text(f"<html><body>{links}</body></html>\n")
    (root / "index.html").write_text('<html><body><a href="fram/">fram</a></body></html>\n')
    return root


def test_wheel_installs_a_working_fram_from_a_local_index(tmp_path: Path) -> None:
    # A real index, not --find-links: PyPI has an unrelated 'fram' 1.0.0, and only
    # an --index is searched before PyPI (uv's first-index strategy).
    index = _local_index(tmp_path / "simple")
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tmp_path / "tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
        "TAU_HOME": str(tmp_path / "tau-home"),
    }
    subprocess.run(
        ["uv", "tool", "install", "fram", "--index", index.as_uri()],
        check=True,
        capture_output=True,
        env=env,
    )
    fram = tmp_path / "bin" / "fram"

    version = subprocess.run([fram, "--version"], capture_output=True, text=True, env=env)
    assert version.stdout == f"fram {__version__} (tau {tau_base_version()})\n"
    help_text = subprocess.run([fram, "--help"], capture_output=True, text=True, env=env)
    assert help_text.returncode == 0, help_text.stderr

    python = tmp_path / "tools" / "fram" / "bin" / "python"
    probe = (
        "import importlib.metadata as m, fram.cli, tau_coding.version as v;"
        "fram.cli.adopt_fram_identity();"
        "print(v.current_version());"
        "print([d.metadata['Name'] for d in m.distributions() if d.metadata['Name'] == 'tau-ai'])"
    )
    identity = subprocess.run([python, "-c", probe], capture_output=True, text=True, env=env)
    assert identity.stdout.split("\n")[:2] == [__version__, "[]"], identity.stderr

    tau = subprocess.run(
        ["uv", "run", "--quiet", "tau", "--version"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert tau.returncode == 0 and tau.stdout.startswith("tau "), tau.stderr
