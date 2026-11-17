"""Every folder the extension uses follows `TAU_HOME` and `TAU_PROJECT_DIRNAME`.

Fram sets them to `~/.fram` and `.fram`, and a Fram install must not touch
`~/.tau` or a project `.tau/`.
"""

from pathlib import Path

import pytest

from tau_subagents.agents import load_agent_definitions
from tau_subagents.memory import resolve_memory_dir
from tau_subagents.paths import jobs_home, project_tau_dir, tau_home, transcripts_root
from tau_subagents.prompts import inherited_resource_paths
from tau_subagents.schedule_store import resolve_store_path
from tau_subagents.settings import load_subagent_settings

AGENT = "---\nname: {name}\ndescription: d\n---\nbody\n"


@pytest.fixture
def fram(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "fram-home"
    monkeypatch.setenv("TAU_HOME", str(home))
    monkeypatch.setenv("TAU_PROJECT_DIRNAME", ".fram")
    monkeypatch.delenv("TAU_SUBAGENTS_DIR", raising=False)
    monkeypatch.delenv("TAU_JOBS_DIR", raising=False)
    return home


def test_defaults_are_tau_folders(tmp_path: Path) -> None:
    assert tau_home() == Path.home() / ".tau"
    assert project_tau_dir(tmp_path) == tmp_path / ".tau"


def test_explicit_home_ignores_the_environment(fram: Path, tmp_path: Path) -> None:
    assert tau_home(tmp_path) == tmp_path / ".tau"


def test_user_folders_follow_tau_home(fram: Path) -> None:
    assert tau_home() == fram
    assert transcripts_root() == fram / "subagents"
    assert jobs_home() == fram / "jobs"
    assert inherited_resource_paths(Path("/p")).root == fram


def test_project_folders_follow_tau_project_dirname(fram: Path, tmp_path: Path) -> None:
    cwd = tmp_path / "proj"
    assert resolve_memory_dir("project", "a", cwd) == cwd / ".fram" / "agent-memory" / "a"
    assert resolve_memory_dir("local", "a", cwd) == cwd / ".fram" / "agent-memory-local" / "a"
    assert resolve_memory_dir("user", "a", cwd) == fram / "agent-memory" / "a"
    assert resolve_store_path(cwd, "s") == cwd / ".fram" / "subagent-schedules" / "s.json"


def test_agents_and_settings_come_from_fram_folders(fram: Path, tmp_path: Path) -> None:
    cwd = tmp_path / "proj"
    for folder, name in ((fram / "agents", "home"), (cwd / ".fram" / "agents", "project")):
        folder.mkdir(parents=True)
        (folder / f"{name}.md").write_text(AGENT.format(name=name))
    stale = cwd / ".tau" / "agents"
    stale.mkdir(parents=True)
    (stale / "stale.md").write_text(AGENT.format(name="stale"))
    assert set(load_agent_definitions(cwd)) >= {"home", "project"}
    assert "stale" not in load_agent_definitions(cwd)

    (fram / "subagents.json").write_text('{"maxConcurrent": 3}')
    assert load_subagent_settings(cwd).max_concurrent == 3
