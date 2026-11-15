"""The jobs half of the extension: one notice per finished job, no more.

The watcher's own transitions are `test_jobs_watch.py`'s business. What is
tested here is the seam between it and the session: that a finish becomes
exactly one notice carrying the id and the exit code, that a start becomes
none, and that the settings keys behind the sweep parse or drop as the
docstring says.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from tau_coding.extensions import ExtensionRuntime
from test_extension import (
    RecordingSession,
    _extension_module,
    _load_runtime,
)

from tau_subagents.jobs import Job, JobStore
from tau_subagents.settings import (
    JOBS_LOG_MAX_BYTES_DEFAULT,
    JOBS_RETENTION_DAYS_DEFAULT,
    SubagentSettings,
    load_subagent_settings,
)

pytestmark = pytest.mark.anyio

#: Fast enough that waiting on a tick is not a slow test, slow enough that a
#: tick cannot overlap itself.
TICK_SECONDS = 0.02


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make(store: JobStore, cwd: Path, *, label: str = "build") -> Job:
    """Register a job whose pid is this process, so it reads as running."""
    return store.create(
        label=label,
        argv=("/bin/sh", "-c", "echo hi"),
        cwd=str(cwd),
        pid=os.getpid(),
        pgid=os.getpgid(0),
        started_at=time.time(),
    )


def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the extension poll the registry on a test's clock.

    The extension is loaded under its own module name and reads this from its
    own copy of the module, so the patch has to name that copy. Without it the
    watcher ticks every two seconds and these tests look hung rather than
    merely patient.
    """
    module = _extension_module()
    monkeypatch.setattr(module, "JOBS_POLL_SECONDS", TICK_SECONDS)


def _load_extension(tmp_path: Path) -> object:
    """The extension module, loaded if this test has no session to load it."""
    try:
        return _extension_module()
    except AssertionError:
        _load_runtime(tmp_path)
        return _extension_module()


async def _wait_for(condition: Callable[[], bool], *, tries: int = 500) -> None:
    for _ in range(tries):
        if condition():
            return
        await asyncio.sleep(0.01)


def _watcher_has_seen(job_id: str) -> bool:
    """Whether the extension's live watcher has read `job_id` as running.

    The extension keeps its watcher in the closure `setup` builds, so this is
    how a test waits for a tick: the poll task is named, and a task's own
    frame is the only handle on the watcher running it.
    """
    for task in asyncio.all_tasks():
        if task.get_name() == "jobs-watch":
            watcher = task.get_coro().cr_frame.f_locals.get("self")
            return any(job.id == job_id for job in watcher.snapshot())
    return False


async def _start_watched_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ExtensionRuntime, RecordingSession, JobStore]:
    """Bind a session whose watcher polls a registry we can write to.

    `TAU_JOBS_DIR` is the seam `conftest.py` already points at
    `tmp_path/jobs`; asserting it is what keeps this honest about reading the
    same registry the extension's poller opens, rather than a look-alike.
    """
    root = Path(os.environ["TAU_JOBS_DIR"])
    runtime = _load_runtime(tmp_path)
    _fast_poll(monkeypatch)
    session = RecordingSession(tmp_path)
    runtime.bind(session)
    store = JobStore(root=root)
    await runtime.emit_session_start("startup")
    return runtime, session, store


def test_notice_text_names_id_label_status_and_rc(tmp_path: Path) -> None:
    module = _load_extension(tmp_path)
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path, label="index")
    # `finish` returns the job as the registry now reads it; the handle the
    # caller held is the pre-exit one and carries no rc.
    finished = store.finish(job, rc=2, outcome="exited", ended_at=time.time())

    text = module.format_job_notice(finished)

    assert job.id in text
    assert "(index)" in text
    assert "done" in text
    assert "rc=2" in text


def test_notice_for_a_lost_job_shows_no_rc(tmp_path: Path) -> None:
    module = _load_extension(tmp_path)
    store = JobStore(root=tmp_path / "jobs")
    # A pid nobody can own: no `done.json`, not alive, so the state is lost.
    job = store.create(
        label="orphan",
        argv=("/bin/sh", "-c", "echo hi"),
        cwd=str(tmp_path),
        pid=999_999,
        pgid=999_999,
        started_at=time.time() - 5,
    )

    text = module.format_job_notice(job)

    assert "lost" in text
    assert "rc=" not in text


async def test_a_finished_job_notifies_the_parent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime, session, store = await _start_watched_session(tmp_path, monkeypatch)

    # Wait for the watcher to have read the job as running: a job created and
    # finished between two ticks is invisible, which is the watcher's contract
    # — it reports transitions, not states.
    job = _make(store, tmp_path, label="sweep")
    await _wait_for(lambda: _watcher_has_seen(job.id))
    store.finish(job, rc=7, outcome="exited", ended_at=time.time())

    await _wait_for(lambda: session.followed_up)
    assert len(session.followed_up) == 1
    # A plain notice carries no card: text in, text out.
    assert session.followed_up_custom[0] == (None, None)
    assert job.id in session.followed_up[0]
    assert "rc=7" in session.followed_up[0]

    # One notice, not one per tick: the edge is consumed when reported.
    await asyncio.sleep(TICK_SECONDS * 5)
    assert len(session.followed_up) == 1


async def test_a_job_that_only_starts_is_not_news(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime, session, store = await _start_watched_session(tmp_path, monkeypatch)

    _make(store, tmp_path, label="quiet")
    await asyncio.sleep(TICK_SECONDS * 5)

    # A row in the strip is not a message to the parent.
    assert session.followed_up_custom == []
    assert session.followed_up == []


async def test_watch_stops_with_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, session, store = await _start_watched_session(tmp_path, monkeypatch)
    job = _make(store, tmp_path, label="late")
    await _wait_for(lambda: _watcher_has_seen(job.id))
    await runtime.emit_session_shutdown("quit")

    store.finish(job, rc=0, outcome="exited", ended_at=time.time())
    await asyncio.sleep(TICK_SECONDS * 5)
    assert session.followed_up == []


def test_jobs_settings_keys_parse_default_and_drop(tmp_path: Path) -> None:
    assert JOBS_RETENTION_DAYS_DEFAULT == 30
    assert JOBS_LOG_MAX_BYTES_DEFAULT == 0
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    (cwd / ".tau").mkdir(parents=True)
    defaults = load_subagent_settings(cwd, home=home)
    assert defaults.jobs_retention_days == 30
    assert defaults.jobs_log_max_bytes == 0

    def write(data: dict[str, object]) -> SubagentSettings:
        (cwd / ".tau" / "subagents.json").write_text(json.dumps(data))
        return load_subagent_settings(cwd, home=home)

    loaded = write({"jobsRetentionDays": 3, "jobsLogMaxBytes": 4096})
    assert loaded.jobs_retention_days == 3
    assert loaded.jobs_log_max_bytes == 4096
    # Zero is a setting, not a typo: it means the sweep or the cap is off.
    assert write({"jobsRetentionDays": 0}).jobs_retention_days == 0
    assert write({"jobsLogMaxBytes": 0}).jobs_log_max_bytes == 0

    # Out of range, wrong type, and a bool are all dropped for the default.
    for bad in (-1, 3651, "thirty", True):
        assert write({"jobsRetentionDays": bad}).jobs_retention_days == 30
    for bad in (-1, 2**40 + 1, "kilobytes", True):
        assert write({"jobsLogMaxBytes": bad}).jobs_log_max_bytes == 0
