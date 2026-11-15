"""Running jobs for real: the tee, the caps, the kill, the CLI.

These tests spawn `/bin/sh` and check what the registry and the log say
afterwards, so they are POSIX-only: process groups, `setsid` and `killpg`
have no equivalent elsewhere, and a skip is truer than a mock. Every test
that starts a process reaps it in a `finally`, so a failure cannot leave a
`sleep` churning on the host.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tau_subagents import job_run
from tau_subagents.jobs import Job, JobStore

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="jobs need POSIX process groups and killpg"
)

# A marker unique enough that `pgrep -f` cannot match another user's sleep.
MARKER = "47.13"


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    """A registry under the test's own directory — never the real `~/.tau`."""
    return JobStore(root=tmp_path / "jobs")


@pytest.fixture
def reaper() -> Iterator[list[int]]:
    """Collect child pids and guarantee the whole group dies after a test."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        _reap_group(pid)


def _reap_group(pid: int) -> None:
    """SIGKILL a job and everyone in its group, ignoring what is already gone.

    The negative id is the process group, which is how a grandchild the job
    `exec`'d or forked dies too. Best-effort by design: a test that already
    killed the job has nothing left to reap.
    """
    if pid <= 0:
        return
    for send, target in ((os.killpg, pid), (os.kill, pid)):
        with contextlib.suppress(OSError):
            send(target, signal.SIGKILL)
    _wait_gone(pid)


def _wait_gone(pid: int, *, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pgrep(needle: str) -> list[str]:
    """Pids of live processes whose command line mentions `needle`."""
    done = subprocess.run(
        ["pgrep", "-f", needle], capture_output=True, text=True
    )
    return [line for line in done.stdout.splitlines() if line]


def _run(result: Any) -> Any:
    """Run one coroutine to completion; the tests stay synchronous and simple."""
    return asyncio.run(result)


async def _spawned(
    store: JobStore, command: str, **kwargs: Any
) -> tuple[Job, asyncio.subprocess.Process]:
    return await job_run.spawn(
        store,
        label="test",
        argv=["/bin/sh", "-c", command],
        cwd=os.getcwd(),
        **kwargs,
    )


def test_stdout_and_stderr_land_in_one_log_in_order(
    store: JobStore, reaper: list[int]
) -> None:
    async def scenario() -> tuple[str, int]:
        job, process = await _spawned(
            store, "echo one; echo two; echo err 1>&2"
        )
        reaper.append(process.pid)
        outcome = await job_run.supervise_async(process, job, store)
        return outcome, job.log_path.read_text().count("\n")

    outcome, lines = _run(scenario())
    assert outcome == "exited"

    job = store.get(next(entry.id for entry in store.list()))
    assert job is not None
    text = job.log_path.read_text()
    # Both streams, one file, order preserved: stdout is unbuffered `/bin/sh`.
    assert text.splitlines() == ["one", "two", "err"]
    assert lines == 3

    done = json.loads((job.dir / "done.json").read_text())
    assert done["rc"] == 0
    assert done["outcome"] == "exited"
    assert job.status() == "done"
    # Exactly one marker: a second write would mean a double supervision.
    assert sorted(p.name for p in job.dir.iterdir()) == [
        "done.json",
        "job.json",
        "out.log",
    ]


def test_log_is_created_private_before_the_child_runs(
    store: JobStore, reaper: list[int]
) -> None:
    async def scenario() -> Path:
        job, process = await _spawned(store, "echo hi")
        reaper.append(process.pid)
        # The descriptor is handed over before the exec, so the file exists
        # with its final mode while the job is still starting.
        assert job.log_path.stat().st_mode & 0o777 == 0o600
        await job_run.supervise_async(process, job, store)
        return job.log_path

    log_path = _run(scenario())
    assert log_path.read_text() == "hi\n"


def test_a_nonzero_exit_records_the_return_code(
    store: JobStore, reaper: list[int]
) -> None:
    async def scenario() -> str:
        job, process = await _spawned(store, "exit 7")
        reaper.append(process.pid)
        return await job_run.supervise_async(process, job, store)

    assert _run(scenario()) == "exited"
    finished = store.list()[0]
    assert finished.rc == 7
    assert finished.status() == "done"
    assert finished.outcome == "exited"


def test_a_cap_kills_the_whole_group(store: JobStore, reaper: list[int]) -> None:
    async def scenario() -> tuple[str, float]:
        job, process = await _spawned(
            store, f"exec sleep {MARKER}", cap=0.4
        )
        reaper.append(process.pid)
        started = time.time()
        outcome = await job_run.supervise_async(process, job, store, cap=0.4)
        return outcome, time.time() - started

    outcome, elapsed = _run(scenario())
    assert outcome == "timeout"
    # A cap that waits for its own grace period is not a cap.
    assert elapsed < 5.0
    assert store.list()[0].status() == "done"
    _assert_no_marker_left()


def _assert_no_marker_left() -> None:
    """No live process may still mention the marker, given a short grace.

    A killed process takes a moment to disappear, and `pgrep` matching this
    test's own command line is excluded by the marker's digits.
    """
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pgrep(MARKER):
            return
        time.sleep(0.1)
    raise AssertionError(f"processes still match {MARKER}: {_pgrep(MARKER)}")


def test_an_oversized_log_kills_the_job(store: JobStore, reaper: list[int]) -> None:
    async def scenario() -> tuple[str, int]:
        # Slow on purpose. A job that dumps 200 KB finishes inside the
        # first 250 ms poll, and a job that finished has an outcome of
        # `exited`, which is correct. One write a quarter-second, each big
        # enough to breach the cap, is still running when the cap is checked.
        job, process = await _spawned(
            store,
            "i=0; while [ $i -lt 200 ]; do head -c 16384 /dev/zero "
            "| tr '\\0' 'x'; sleep 0.2; i=$((i+1)); done",
            log_max_bytes=4096,
        )
        reaper.append(process.pid)
        outcome = await job_run.supervise_async(
            process, job, store, log_max_bytes=8192
        )
        await asyncio.sleep(0.05)
        return outcome, job.log_path.stat().st_size

    outcome, size = _run(scenario())
    assert outcome == "log_overflow"
    # The cap is enforced by polling, so the log overshoots by whatever the
    # job wrote since the last look — but stays far under 2 MB.
    assert size < 2_000_000
    assert store.list()[0].status() == "done"


def test_kill_job_stops_a_running_job(store: JobStore, reaper: list[int]) -> None:
    async def scenario() -> tuple[bool, int, int, Job]:
        # A bare `sleep 61`, with none of `MARKER`'s digits: a job whose
        # command line contained them would be reaped by the timeout test's
        # `pgrep`, and would then report itself as already dead.
        job, process = await _spawned(store, "exec sleep 61")
        reaper.append(process.pid)
        pid = process.pid
        signalled = job_run.kill_job(job)
        rc = await process.wait()
        await asyncio.sleep(0.05)
        return signalled, rc, pid, store.get(job.id)

    signalled, rc, pid, after = _run(scenario())
    assert signalled is True
    assert rc < 0
    # `done` if a supervisor had written the marker, `lost` if only the
    # process is gone — this test has no supervisor, so either is right.
    assert after.status() in ("done", "lost")
    assert after.status() != "running"
    assert _wait_gone(pid)


def test_wait_for_finish_returns_the_finished_job(
    store: JobStore, reaper: list[int]
) -> None:
    async def scenario() -> tuple[str | None, int | None]:
        job, process = await _spawned(store, "exit 4")
        reaper.append(process.pid)
        waiter = asyncio.create_task(
            job_run.wait_for_finish(store, job.id, timeout=5)
        )
        outcome = await job_run.supervise_async(process, job, store)
        finished = await waiter
        return outcome, finished.rc if finished else None

    outcome, rc = _run(scenario())
    assert outcome == "exited"
    assert rc == 4


def test_wait_for_finish_gives_up_on_a_timeout(
    store: JobStore, reaper: list[int]
) -> None:
    async def scenario() -> tuple[Job | None, float]:
        job, process = await _spawned(store, "exec sleep 62")
        reaper.append(process.pid)
        started = time.time()
        result = await job_run.wait_for_finish(store, job.id, timeout=0.2)
        elapsed = time.time() - started
        job_run.kill_job(job, how="kill")
        await process.wait()
        return result, elapsed

    result, elapsed = _run(scenario())
    assert result is None
    # Long enough to have polled once, nowhere near the job's five seconds.
    assert 0.15 <= elapsed < 2.0


def test_cli_supervises_and_borrows_the_jobs_return_code(
    store: JobStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TAU_JOBS_DIR", str(tmp_path / "cli-jobs"))
    assert job_run.main(["--label", "x", "--", "/bin/sh", "-c", "exit 3"]) == 3

    listed = JobStore(root=tmp_path / "cli-jobs").list()
    assert [entry.label for entry in listed] == ["x"]
    assert listed[0].rc == 3
    assert listed[0].status() == "done"


def test_cli_detaches_and_the_job_finishes_anyway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "detach-jobs"
    monkeypatch.setenv("TAU_JOBS_DIR", str(root))
    started = time.time()
    assert (
        job_run.main(["--detach", "--", "/bin/sh", "-c", "sleep 4; echo bye"])
        == 0
    )
    # Returning before the job ends is the whole point of `--detach`, and the
    # job has to outlive the call by enough that a supervisor still holding
    # the report pipe, or a caller still waiting on the supervisor, would
    # show up here rather than pass on a fast machine.
    assert time.time() - started < 1.0

    registry = JobStore(root=root)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        jobs = registry.list()
        if jobs and jobs[0].status() == "done":
            break
        time.sleep(0.1)
    finished = registry.list()[0]
    assert finished.status() == "done"
    assert finished.log_path.read_text() == "bye\n"


def test_cli_usage_errors_exit_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TAU_JOBS_DIR", str(tmp_path / "usage"))
    # No separator, an empty command, and two kinds of bad cap.
    assert job_run.main(["--label", "x", "/bin/true"]) == 2
    assert job_run.main(["--"]) == 2
    assert job_run.main(["--cap", "not-a-number", "--", "/bin/true"]) == 2
    assert job_run.main(["--cap", "0", "--", "/bin/true"]) == 2
    assert job_run.main(["--env", "NOEQUALS", "--", "/bin/true"]) == 2
    assert job_run.main(["--bogus", "--", "/bin/true"]) == 2
    assert JobStore(root=tmp_path / "usage").list() == []


def test_cli_prints_the_id_and_log_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "print-jobs"
    monkeypatch.setenv("TAU_JOBS_DIR", str(root))
    assert job_run.main(["--", "/bin/sh", "-c", "echo printed"]) == 0
    out = capsys.readouterr().out.strip().split("\t")
    assert len(out) == 2
    assert Path(out[1]).read_text() == "printed\n"
    # The default label is the command's basename and its first argument.
    assert JobStore(root=root).list()[0].label == "sh -c"
