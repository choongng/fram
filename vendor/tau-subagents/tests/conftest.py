"""Shared fixtures for the tau-subagents test suite."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_transcript_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point durable transcripts at a per-test directory (ADR 0003).

    Without this, any test that spawns a run would write JSONL transcripts
    into the developer's real `~/.tau/subagents/` — and the retention sweep
    on session_start could delete real files there.
    """
    root = tmp_path / "subagents-transcripts"
    monkeypatch.setenv("TAU_SUBAGENTS_DIR", str(root))
    return root


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME at a per-test directory.

    Children run with extensions enabled and discover them from
    `Path.home()/.tau/extensions`, so without this every child spawned in a
    test would load the developer's real installed extensions.
    """
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # A run from inside Fram inherits `TAU_HOME=~/.fram` and
    # `TAU_PROJECT_DIRNAME=.fram`; without this they would point the tests,
    # and their children, back at real folders.
    monkeypatch.delenv("TAU_HOME", raising=False)
    monkeypatch.delenv("TAU_PROJECT_DIRNAME", raising=False)
    return home


@pytest.fixture(autouse=True)
def _isolate_job_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the job registry at a per-test directory.

    `job_run.main` builds its store from `TAU_JOBS_DIR` rather than a command
    line flag, so this is the seam that keeps a CLI test from registering
    jobs — and running commands — under the developer's real `~/.tau/jobs`.
    """
    root = tmp_path / "jobs"
    monkeypatch.setenv("TAU_JOBS_DIR", str(root))
    return root
