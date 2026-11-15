"""The watcher's transitions: seed once, report once, survive a bad callback.

Everything here drives the registry directly and calls `poll_once()`, so the
tests are about the diff and not about how long they take. The two things
only the poll task can show — that `start()` ticks at all, and that `stop()`
leaves nothing behind — are the two that run a real interval.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from tau_subagents.jobs import Job, JobStore
from tau_subagents.jobs_watch import JobWatcher

pytestmark = pytest.mark.anyio

#: Short enough to wait on, long enough that one tick cannot overlap itself.
TICK_SECONDS = 0.02


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _wait_for(condition, *, tries: int = 500) -> None:  # noqa: ANN001
    for _ in range(tries):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


def _make(store: JobStore, cwd: Path, *, label: str = "build") -> Job:
    """Register a job whose pid is this test process, so it reads as running.

    The pid is ours and the argv is not what the kernel would report for it:
    `status()` answers `running` because the pid lives, and flips to `done`
    only when `done.json` says so. No child process to spawn or reap.
    """
    return store.create(
        label=label,
        argv=("/bin/sh", "-c", "echo hi"),
        cwd=str(cwd),
        pid=os.getpid(),
        pgid=os.getpgid(0),
        started_at=time.time(),
    )


def _finish(store: JobStore, job: Job, *, rc: int = 0) -> Job:
    return store.finish(job, rc=rc, outcome="exited", ended_at=time.time())


def _seeded_watcher(
    tmp_path: Path, *, interval: float = TICK_SECONDS
) -> tuple[JobStore, JobWatcher, Job]:
    """A watcher whose first read will find exactly one running job.

    `poll_once` seeds synchronously, so the tests about the poll *task* start
    from a snapshot already in place and wait only on the transition.
    """
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)
    watcher = JobWatcher(store, interval=interval)
    assert watcher.poll_once() == ([], [])
    return store, watcher, job


class _Recorder:
    """Collects every `(started, finished)` the watcher reports."""

    def __init__(self) -> None:
        self.events: list[tuple[list[str], list[str]]] = []

    def __call__(self, started: list[Job], finished: list[Job]) -> None:
        self.events.append(
            ([job.id for job in started], [job.id for job in finished])
        )

    @property
    def count(self) -> int:
        return len(self.events)


def test_first_tick_seeds_without_notices(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    recorder = _Recorder()
    watcher.subscribe(recorder)
    existing = [_make(store, tmp_path, label=f"j{i}") for i in range(4)]

    started, finished = watcher.poll_once()

    # A session that opens on four running jobs has four rows and no news.
    assert (started, finished) == ([], [])
    assert recorder.count == 0
    assert sorted(job.id for job in watcher.snapshot()) == sorted(
        job.id for job in existing
    )


def test_an_empty_registry_seeds_so_a_later_job_is_news(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    recorder = _Recorder()
    watcher.subscribe(recorder)
    assert watcher.snapshot() == []
    assert watcher.poll_once() == ([], [])

    job = _make(store, tmp_path)
    watcher.poll_once()

    # An empty first read still seeded: a job that shows up next is news.
    assert recorder.events == [([job.id], [])]


def test_a_job_appearing_is_reported_once(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    recorder = _Recorder()
    watcher.subscribe(recorder)
    # Seeded on an empty registry, so what follows is a real appearance.
    assert watcher.poll_once() == ([], [])

    job = _make(store, tmp_path)
    started, finished = watcher.poll_once()
    assert ([entry.id for entry in started], finished) == ([job.id], [])
    assert recorder.events == [([job.id], [])]

    # Nothing changed, so nothing is reported again: no edge, no notice.
    assert watcher.poll_once() == ([], [])
    assert recorder.count == 1


def test_running_to_done_is_reported_as_finished_once(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    recorder = _Recorder()
    watcher.subscribe(recorder)
    # Seed on an empty registry, then create the job: that is an appearance.
    assert watcher.poll_once() == ([], [])
    job = _make(store, tmp_path)
    watcher.poll_once()
    assert recorder.events == [([job.id], [])]

    finished_job = _finish(store, job, rc=3)
    started, finished = watcher.poll_once()

    assert started == []
    assert [entry.id for entry in finished] == [job.id]
    # The job handed over is the fresh read, not the stale running snapshot.
    assert finished[0].rc == finished_job.rc == 3
    assert finished[0].status() == "done"

    assert watcher.poll_once() == ([], [])
    assert recorder.count == 2  # the one appearance, plus this finish


def test_a_broken_callback_stops_neither_the_others_nor_the_poller(
    tmp_path: Path,
) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    first = _Recorder()
    second = _Recorder()
    calls: list[str] = []

    def explode(started: list[Job], finished: list[Job]) -> None:
        del started, finished
        calls.append("explode")
        raise RuntimeError("subscriber is broken")

    watcher.subscribe(first)
    watcher.subscribe(explode)
    watcher.subscribe(second)
    assert watcher.poll_once() == ([], [])

    job = _make(store, tmp_path)
    started, _ = watcher.poll_once()

    assert [entry.id for entry in started] == [job.id]
    assert first.count == 1 and second.count == 1
    assert calls == ["explode"]


async def test_a_watcher_without_a_loop_stays_stopped(tmp_path: Path) -> None:
    # `start()` is what needs a loop; `poll_once()` needs nothing, which is
    # why every other test here can drive the diff without one.
    watcher = JobWatcher(JobStore(root=tmp_path / "jobs"), interval=TICK_SECONDS)
    assert watcher.snapshot() == []
    watcher.stop()


async def test_the_poll_task_reports_a_finish_it_did_not_seed(tmp_path: Path) -> None:
    # A prebuilt watcher is the only way to tell "seeded, then finished" apart
    # from "saw nothing at all": `start()` on an empty registry reports the
    # appearance, and the tick we care about would be indistinguishable.
    store, watcher, existing = _seeded_watcher(tmp_path)
    recorder = _Recorder()
    watcher.subscribe(recorder)
    watcher.start()

    _finish(store, existing)
    await _wait_for(lambda: recorder.events)
    assert recorder.events[-1] == ([], [existing.id])
    await watcher.stop_async()


async def test_stop_leaves_no_task_running(tmp_path: Path) -> None:
    store, watcher, existing = _seeded_watcher(tmp_path)
    recorder = _Recorder()
    watcher.subscribe(recorder)
    watcher.start()
    task = watcher._task  # noqa: SLF001 - asserting on the task we created
    assert task is not None
    # The task has ticked: the registry it read is the one we prebuilt.
    await _wait_for(lambda: bool(watcher.snapshot()))

    await watcher.stop_async()

    assert task.done()
    assert task.cancelled()
    ours = [t for t in asyncio.all_tasks() if t.get_name() == "jobs-watch"]
    assert ours == []

    # Stopped means stopped: a job that appears now goes unreported, and the
    # task really is gone rather than merely cancelled.
    seen = recorder.count
    _make(store, tmp_path)
    await asyncio.sleep(TICK_SECONDS * 5)
    assert recorder.count == seen


def test_start_without_an_event_loop_is_a_no_op(tmp_path: Path) -> None:
    watcher = JobWatcher(
        JobStore(root=tmp_path / "jobs"), interval=TICK_SECONDS
    )
    watcher.start()
    assert watcher._task is None  # noqa: SLF001 - the no-op is the contract
    watcher.stop()


def test_unsubscribe_stops_only_that_callback(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    kept = _Recorder()
    dropped = _Recorder()
    watcher.subscribe(kept)
    unsubscribe = watcher.subscribe(dropped)
    assert watcher.poll_once() == ([], [])

    _make(store, tmp_path)
    watcher.poll_once()
    assert kept.count == 1 and dropped.count == 1

    unsubscribe()
    unsubscribe()  # idempotent: removing nothing twice is still fine

    _make(store, tmp_path, label="second")
    watcher.poll_once()
    assert kept.count == 2
    assert dropped.count == 1


def test_snapshot_is_a_copy_of_the_last_read(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    watcher = JobWatcher(store, interval=TICK_SECONDS)
    job = _make(store, tmp_path)
    watcher.poll_once()

    snapshot = watcher.snapshot()
    snapshot.clear()

    assert [entry.id for entry in watcher.snapshot()] == [job.id]
