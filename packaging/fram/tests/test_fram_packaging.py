"""Tests for the fram distribution: its version, its entry point, and its wheel."""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from fram import __version__, tau_base_version
from fram.branding import BRANDING
from fram.cli import adopt_fram_identity, main
from rich.cells import cell_len

from tau_coding.branding import ENTRY_POINT_GROUP, Branding

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


def test_fram_branding_is_a_complete_tau_branding() -> None:
    branding = Branding(**BRANDING)
    assert set(BRANDING) == set(Branding.__dataclass_fields__)
    assert (branding.product_name, branding.command_name) == ("Fram", "fram")
    assert cell_len(branding.mark) == 1


def test_wheel_declares_the_branding_entry_point() -> None:
    entry_points = _pyproject(PACKAGING / "pyproject.toml")["project"]["entry-points"]  # type: ignore[index]
    assert entry_points[ENTRY_POINT_GROUP] == {"fram": "fram.branding:BRANDING"}


def test_version_flag_prints_fram_and_tau_base(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--version"])
    assert capsys.readouterr().out == f"fram {__version__} (tau {tau_base_version()})\n"


def test_update_is_refused_with_reinstall_advice(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["update"])
    assert exit_info.value.code == 1
    assert "uv tool upgrade fram-agent" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("tau_home", "expected"),
    [(None, "home/.fram"), ("", "home/.fram"), ("/elsewhere", "/elsewhere")],
)
def test_fram_home_defaults_to_dot_fram_unless_tau_home_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tau_home: str | None, expected: str
) -> None:
    import tau_coding
    import tau_coding.version

    # adopt_fram_identity changes these process-wide; restore them afterwards.
    monkeypatch.setattr(tau_coding.version, "_DISTRIBUTION_NAME", "tau-ai")
    monkeypatch.setattr(tau_coding, "__version__", tau_coding.__version__)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TAU_NO_UPDATE_CHECK", "1")
    if tau_home is None:
        monkeypatch.delenv("TAU_HOME", raising=False)
    else:
        monkeypatch.setenv("TAU_HOME", tau_home)

    adopt_fram_identity()

    assert os.environ["TAU_HOME"] == str(tmp_path / expected)


def _local_index(root: Path) -> Path:
    """Build the fram wheel into a minimal file:// simple index at ``root``."""
    project = root / "fram-agent"
    subprocess.run(
        ["uv", "build", "--wheel", "--project", str(PACKAGING), "--out-dir", str(project)],
        check=True,
        capture_output=True,
    )
    links = "".join(f'<a href="{wheel.name}">{wheel.name}</a>' for wheel in project.glob("*.whl"))
    (project / "index.html").write_text(f"<html><body>{links}</body></html>\n")
    (root / "index.html").write_text(
        '<html><body><a href="fram-agent/">fram-agent</a></body></html>\n'
    )
    return root


def test_wheel_installs_a_working_fram_from_a_local_index(tmp_path: Path) -> None:
    # A real index, as users will name it: uv takes fram-agent from an --index before
    # looking on PyPI (uv's first-index strategy).
    index = _local_index(tmp_path / "simple")
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tmp_path / "tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
        "TAU_HOME": str(tmp_path / "tau-home"),
    }
    subprocess.run(
        ["uv", "tool", "install", "fram-agent", "--index", index.as_uri()],
        check=True,
        capture_output=True,
        env=env,
    )
    fram = tmp_path / "bin" / "fram"

    version = subprocess.run([fram, "--version"], capture_output=True, text=True, env=env)
    assert version.stdout == f"fram {__version__} (tau {tau_base_version()})\n"
    help_text = subprocess.run([fram, "--help"], capture_output=True, text=True, env=env)
    assert help_text.returncode == 0, help_text.stderr
    assert "Fram coding-agent harness" in help_text.stdout
    assert "fram install SOURCE" in help_text.stdout
    assert "tau install" not in help_text.stdout

    python = tmp_path / "tools" / "fram-agent" / "bin" / "python"
    probe = (
        "import importlib.metadata as m, fram.cli, tau_coding.version as v;"
        "fram.cli.adopt_fram_identity();"
        "print(v.current_version());"
        "print([d.metadata['Name'] for d in m.distributions() if d.metadata['Name'] == 'tau-ai']);"
        "from tau_coding.tui.app import TauTuiApp; print(TauTuiApp.TITLE);"
        "from tau_coding.paths import TauPaths; print(TauPaths().home)"
    )
    # Without TAU_HOME, fram's home is ~/.fram.
    probe_env = {name: value for name, value in env.items() if name != "TAU_HOME"}
    probe_env["HOME"] = str(tmp_path / "home")
    identity = subprocess.run([python, "-c", probe], capture_output=True, text=True, env=probe_env)
    assert identity.stdout.split("\n")[:4] == [
        __version__,
        "[]",
        "Fram",
        str(tmp_path / "home" / ".fram"),
    ], identity.stderr

    tau = subprocess.run(
        ["uv", "run", "--quiet", "tau", "--version"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert tau.returncode == 0 and tau.stdout.startswith("tau "), tau.stderr
    tau_help = subprocess.run(
        ["uv", "run", "--quiet", "tau", "--help"], cwd=REPO, capture_output=True, text=True
    )
    assert "Tau coding-agent harness" in tau_help.stdout, tau_help.stderr
