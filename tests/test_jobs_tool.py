"""The `jobs` tool's five verbs, `/jobs`, and the text both render from.

`TAU_JOBS_DIR` is the seam: `paths.jobs_home` reads it, `tests/conftest.py`
already points it at a per-test directory, and a `JobStore()` built with no
root lands there. So a test that spawns through the tool registers a job in
its own directory and runs a real command, and the only thing faked is the
session's identity.

Two facts shape the waits here. `job_run.spawn_detached` blocks until the
job's own supervisor exits, so a test that wants a *running* job starts one
in a thread and watches the registry rather than the call. And a job is only
settled when `done.json` says so — the process dying is not the registry's
evidence — hence `_settle` after every kill.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tau_subagents.job_run import spawn_detached, wait_for_finish
from tau_subagents.jobs import Job, JobStore
from tau_subagents.jobs_tool import (
    ACTIONS,
    DEFAULT_LIST_LIMIT,
    LABEL_COLUMN,
    DEFAULT_LOG_TAIL_BYTES,
    DEFAULT_WAIT_SECONDS,
    MAX_LOG_TAIL_BYTES,
    MAX_WAIT_SECONDS,
    JobActions,
    build_jobs_tool,
    format_job_list,
    format_job_log,
    format_job_row,
    jobs_command_text,
)

pytestmark = pytest.mark.anyio

#: A live job long enough that its parent returns while it runs.
LONG_SLEEP_SECONDS = 25

#: The instant the format fixtures are measured from, taken once at import.
#: A finished job's elapsed time is `ended_at - started_at`, both stored, so
#: the row does not move while the test runs — and an integer offset plus a
#: truncated subtraction is the same integer however long the write takes.
#: A *running* job's elapsed time is measured to the clock, so its row is
#: never asserted to the second.
NOW = time.time()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def store() -> JobStore:
    """A store on the directory `conftest.py` pointed `TAU_JOBS_DIR` at."""
    return JobStore()


@pytest.fixture
def actions(store: JobStore) -> JobActions:
    """The verbs, against a session whose cwd is this test's temp directory."""
    return JobActions(
        store,
        cwd=lambda: Path.cwd(),
        session_id=lambda: "session-1",
        log_max_bytes=lambda: 0,
    )


@pytest.fixture
def live_job(store: JobStore):  # noqa: ANN201 - yields a Job, reaps after
    """A real sleep job, started synchronously and reaped when the test ends.

    `spawn_detached` blocks until its supervisor exits, so a fixture that ran
    it in a coroutine would sit on the loop for the job's whole life. The
    `cap` is the backstop: a test that forgets to kill the job still leaves
    the group to die on its own deadline rather than at `sleep 25`'s own.
    """
    job = spawn_detached(
        store,
        label="sleeper",
        argv=["/bin/sh", "-c", f"sleep {LONG_SLEEP_SECONDS}"],
        cwd=Path.cwd(),
        cap=LONG_SLEEP_SECONDS + 5,
    )
    yield job
    _reap(job)


def _reap(job: Job) -> None:
    """Signal the group until its pid is gone, so no test leaves a sleeper."""
    for _ in range(100):
        if _pid_gone(job.pid):
            return
        try:
            os.killpg(job.pgid, signal.SIGKILL)
        except OSError:
            return
        time.sleep(0.02)


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


async def _spawn(
    actions: JobActions, argv: list[str], **arguments: Any
) -> tuple[str, str]:
    """Run the `spawn` action off the loop; return its text and the job id.

    `to_thread` because the spawn under it forks, and its read of the report
    pipe is not something to sit on in a coroutine.
    """
    result = await asyncio.to_thread(
        lambda: actions.call({"action": "spawn", "argv": argv, **arguments})
    )
    text = (await result).text
    job_id = _find_id(text)
    assert job_id is not None, text
    return text, job_id


def _find_id(text: str) -> str | None:
    """The `<yyyymmdd>-<hhmmss>-<4hex>` token in a result's text."""
    return next((token for token in text.split() if _looks_like_id(token)), None)


def _looks_like_id(token: str) -> bool:
    parts = token.split("-")
    return len(parts) == 3 and all(part.isdigit() or _is_hex(part) for part in parts)


def _is_hex(token: str) -> bool:
    try:
        int(token, 16)
    except ValueError:
        return False
    return len(token) == 4


async def _settle(store: JobStore, job_id: str, *, timeout: float = 8.0) -> Job:
    """Wait for the marker: the registry's own evidence that a job is over."""
    job = await asyncio.wait_for(
        wait_for_finish(store, job_id, timeout=timeout), timeout + 2.0
    )
    assert job is not None, f"job {job_id} never wrote done.json"
    return job


def _manifest(job: Job) -> dict[str, Any]:
    return json.loads((job.dir / "job.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- spawn


async def test_spawn_returns_an_id_whose_directory_exists_and_finishes(
    store: JobStore, actions: JobActions
) -> None:
    text, job_id = await _spawn(
        actions, ["/bin/sh", "-c", "echo hi"], label="greeter"
    )
    assert "Started job" in text

    job = store.get(job_id)
    assert job is not None
    assert job.dir.is_dir()
    assert job.log_path.is_file()
    assert job.label == "greeter"
    assert str(job.log_path) in text

    finished = await _settle(store, job_id)
    assert finished.status() == "done"
    assert finished.rc == 0
    assert finished.log_path.read_text() == "hi\n"


async def test_spawn_detaches_and_returns_well_before_the_job_ends(
    tmp_path: Path,
) -> None:
    """A detached spawn of a 25-second job must not wait for it.

    Measured against the spawn's own clock, not against the test's wall clock:
    the detached supervisor is the job's parent and lives as long as it does,
    so it outlives the spawner on purpose. Waiting for the script to exit would
    be waiting for the job — and the spawner's stdout is closed by then, so it
    is written to a file the test reads back.
    """
    script = tmp_path / "timed_spawn.py"
    out = tmp_path / "elapsed.txt"
    script.write_text(
        "import os, time\n"
        "from tau_subagents.job_run import spawn_detached\n"
        "from tau_subagents.jobs import JobStore\n"
        "started = time.time()\n"
        "spawn_detached(JobStore(), label='slow', cwd=os.getcwd(), cap=30,\n"
        "               argv=['/bin/sh', '-c', 'sleep 25'])\n"
        "elapsed = time.time() - started\n"
        "open(os.environ['ELAPSED_FILE'], 'w').write('%.3f' % elapsed)\n",
        encoding="utf-8",
    )
    env = {**os.environ, "ELAPSED_FILE": str(out)}

    async def scenario() -> float:
        # A thread, because the call forks and the test process has a loop.
        return await asyncio.to_thread(
            lambda: subprocess.run(
                [sys.executable, str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        )

    started = time.monotonic()
    completed = await scenario()
    elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr
    # The spawn's own reading is the claim; the test's is the sanity check on
    # the fixture, and both must hold for the regression to be caught.
    assert float(out.read_text(encoding="utf-8")) < 1.0
    assert elapsed < 2.0

    registry = JobStore()
    running = [job for job in registry.list() if job.label == "slow"]
    assert len(running) == 1
    assert running[0].status() == "running"


async def test_spawn_records_the_session_and_the_defaults(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "true"])
    manifest = _manifest(store.get(job_id))
    assert manifest["parent_session"] == "session-1"
    # No cap means no deadline; no label means the command named itself.
    assert manifest["deadline"] is None
    # The name `job_run._default_label` gives the same command: basename plus
    # its first argument, so `-c` is part of it.
    assert manifest["label"] == "sh -c"
    assert manifest["cwd"] == str(Path.cwd().resolve())


async def test_spawn_cap_sets_a_deadline_and_zero_is_an_error(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(
        actions, ["/bin/sh", "-c", "true"], cap=30, label="capped"
    )
    manifest = _manifest(store.get(job_id))
    assert manifest["deadline"] == pytest.approx(
        manifest["started_at"] + 30, abs=2.0
    )

    bad = await actions.call({"action": "spawn", "argv": ["/bin/true"], "cap": 0})
    assert bad.details == {"is_error": True}
    assert "cap must be greater than zero" in bad.text
    assert len(store.list()) == 1  # the rejected spawn registered nothing


async def test_spawn_applies_the_configured_log_cap(store: JobStore) -> None:
    """`jobsLogMaxBytes` reaches the supervisor, which enforces it by polling.

    A job that writes its oversized log in one go is `exited` before the first
    poll, which is correct — so this job keeps writing, and the cap has to
    catch it while it is still running. The assertion is that the cap is
    carried at all: without it, the same job runs to its own end.
    """
    capped = JobActions(
        store,
        cwd=lambda: Path.cwd(),
        session_id=lambda: None,
        log_max_bytes=lambda: 4096,
    )
    _, job_id = await _spawn(
        capped,
        [
            "/bin/sh",
            "-c",
            "i=0; while [ $i -lt 200 ]; do head -c 16384 /dev/zero"
            " | tr '\\0' 'x'; sleep 0.2; i=$((i+1)); done",
        ],
        label="chatty",
    )
    finished = await _settle(store, job_id)
    assert finished.outcome == "log_overflow"
    assert finished.log_path.stat().st_size < 2_000_000


async def test_spawn_honours_cwd_and_a_relative_one_is_anchored(
    store: JobStore, actions: JobActions, tmp_path: Path
) -> None:
    _, job_id = await _spawn(
        actions, ["/bin/sh", "-c", "pwd"], label="where", cwd=str(tmp_path)
    )
    finished = await _settle(store, job_id)
    assert finished.log_path.read_text().strip() == str(tmp_path.resolve())

    here = Path.cwd().resolve()
    (here / "relative-cwd-target").mkdir(exist_ok=True)
    _, relative_id = await _spawn(
        actions, ["/bin/sh", "-c", "pwd"], cwd="relative-cwd-target", label="sub"
    )
    relative = await _settle(store, relative_id)
    assert relative.log_path.read_text().strip() == str(here / "relative-cwd-target")


# ----------------------------------------------------------------- list


async def test_list_shows_the_row_with_its_status_and_rc(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "true"], label="ok")
    await _settle(store, job_id)
    body = (await actions.call({"action": "list"})).text
    assert body.startswith("1 job(s), newest first:")
    assert job_id in body
    assert "done" in body
    assert "rc=0" in body
    assert "log=0B" in body


async def test_list_is_newest_first_and_honours_the_limit(
    store: JobStore, actions: JobActions
) -> None:
    for name in ("first", "second"):
        # A second apart, because ids sort by construction and share a second.
        await asyncio.sleep(1.05)
        await asyncio.to_thread(
            spawn_detached,
            store,
            label=name,
            argv=["/bin/sh", "-c", "true"],
            cwd=Path.cwd(),
        )
    # "first"/"second" would sort against each other by name, so the two rows
    # are checked by their ids against the rows' own order.
    body = (await actions.call({"action": "list"})).text
    rows = [line for line in body.splitlines() if _looks_like_id(line.split()[0])]
    assert len(rows) == 2
    assert rows[0].split()[0] > rows[1].split()[0]

    one = (await actions.call({"action": "list", "limit": 1})).text
    assert "1 job(s), newest first:" in one
    assert len(one.splitlines()) == 3  # header, one row, the hint

    bad = await actions.call({"action": "list", "limit": 0})
    assert bad.details == {"is_error": True}
    assert "limit must be at least 1" in bad.text


# ------------------------------------------------------------------ log


async def test_log_returns_the_whole_log_when_it_fits(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(
        actions, ["/bin/sh", "-c", "echo one; echo two"], label="two lines"
    )
    await _settle(store, job_id)
    body = (await actions.call({"action": "log", "id": job_id})).text
    assert body.endswith("one\ntwo")
    assert "bytes above" not in body


async def test_log_smaller_than_the_file_starts_with_the_skipped_rule(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(
        actions, ["/bin/sh", "-c", "printf abcdefghij"], label="ten bytes"
    )
    await _settle(store, job_id)
    lines = (await actions.call({"action": "log", "id": job_id, "tail": 4})).text
    body = lines.splitlines()
    assert body[1] == "…6 bytes above"
    assert body[2] == "ghij"

    whole = (await actions.call({"action": "log", "id": job_id, "tail": 4096})).text
    assert "bytes above" not in whole
    assert whole.endswith("abcdefghij")

    # A tail past the ceiling is the ceiling, which still holds the file.
    clamped = (await actions.call({"action": "log", "id": job_id, "tail": 10**7})).text
    assert clamped.endswith("abcdefghij")


async def test_log_treats_a_missing_log_as_a_fact(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "echo gone"], label="eraser")
    await _settle(store, job_id)
    job = store.get(job_id)
    job.log_path.unlink()
    body = (await actions.call({"action": "log", "id": job_id})).text
    assert "No log file at" in body
    assert str(job.log_path) in body


async def test_log_rejects_a_tail_that_is_not_a_count(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "true"], label="quiet")
    await _settle(store, job_id)
    for bad_tail in (0, -1, "abc", True):
        result = await actions.call({"action": "log", "id": job_id, "tail": bad_tail})
        assert result.details == {"is_error": True}, bad_tail


# ----------------------------------------------------------------- wait


async def test_wait_on_a_finished_job_returns_it(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "echo done"], label="final")
    body = (await actions.call({"action": "wait", "id": job_id, "timeout": 5})).text
    assert job_id in body
    assert "done" in body
    assert "Outcome: exited." in body


async def test_wait_on_a_running_job_times_out_gracefully(
    store: JobStore, actions: JobActions, live_job: Job
) -> None:
    assert live_job.status() == "running"
    started = time.monotonic()
    result = await actions.call({"action": "wait", "id": live_job.id, "timeout": 1})
    elapsed = time.monotonic() - started
    assert result.details != {"is_error": True}
    assert f"{live_job.id}  running" in result.text
    assert "Still running after 1s." in result.text
    assert 0.9 < elapsed < 6.0


async def test_wait_clamps_the_timeout_and_rejects_a_bad_one(
    store: JobStore, actions: JobActions
) -> None:
    assert DEFAULT_WAIT_SECONDS == 60.0 and MAX_WAIT_SECONDS == 300.0
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "true"], label="brief")
    await _settle(store, job_id)
    # Past the ceiling is the ceiling, and a finished job answers it at once.
    body = (await actions.call({"action": "wait", "id": job_id, "timeout": 10**4})).text
    assert "Outcome: exited." in body

    missing = await actions.call({"action": "wait"})
    assert "Missing id" in missing.text
    for bad in (-1, "soon"):
        result = await actions.call({"action": "wait", "id": job_id, "timeout": bad})
        assert result.details == {"is_error": True}, bad


# ----------------------------------------------------------------- kill


async def test_kill_stops_a_sleep_job_and_leaves_no_process(
    store: JobStore, actions: JobActions, live_job: Job
) -> None:
    body = (await actions.call({"action": "kill", "id": live_job.id})).text
    assert "Sent TERM to job" in body
    assert "process group" in body

    finished = await _settle(store, live_job.id)
    assert finished.status() == "done"
    await asyncio.sleep(0.2)
    assert _pid_gone(live_job.pid)
    assert not _group_exists(live_job.pgid)


async def test_kill_a_job_that_was_already_gone_is_a_reported_no_op(
    store: JobStore, actions: JobActions
) -> None:
    _, job_id = await _spawn(actions, ["/bin/sh", "-c", "true"], label="brief")
    before = await _settle(store, job_id)
    body = (await actions.call({"action": "kill", "id": job_id, "how": "kill"})).text
    assert format_job_row(before) in body
    assert "the process group is already gone" in body
    assert "Status now: done" in body
    assert before.dir.is_dir()  # a no-op kill removes nothing


async def test_kill_rejects_an_unknown_how(
    actions: JobActions, live_job: Job
) -> None:
    bad = await actions.call({"action": "kill", "id": live_job.id, "how": "burn"})
    assert bad.details == {"is_error": True}
    assert "Pass 'term' or 'kill'" in bad.text


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except OSError:
        return False
    return True


# ------------------------------------------------------------ bad input


async def test_unknown_id_is_an_error_result_and_not_an_exception(
    actions: JobActions,
) -> None:
    for action in ("log", "wait", "kill"):
        result = await actions.call({"action": action, "id": "no-such-job"})
        assert result.details == {"is_error": True}, action
        assert "Unknown job id: 'no-such-job'" in result.text


async def test_unknown_action_and_missing_argv(actions: JobActions) -> None:
    unknown = await actions.call({"action": "terminate"})
    assert unknown.details == {"is_error": True}
    assert "Unknown action: 'terminate'" in unknown.text
    assert ", ".join(ACTIONS) in unknown.text

    omitted = await actions.call({})
    assert "Unknown action" in omitted.text

    missing = await actions.call({"action": "spawn"})
    assert "Missing argv" in missing.text
    assert '["make", "-j4"]' in missing.text

    shell = await actions.call({"action": "spawn", "argv": "make -j4"})
    assert "not one shell string" in shell.text

    empty = await actions.call({"action": "spawn", "argv": []})
    assert "at least one entry" in empty.text

    for action in ("log", "wait", "kill"):
        assert "Missing id" in (await actions.call({"action": action})).text


async def test_spawn_failure_is_a_reported_error_and_not_an_exception(
    actions: JobActions,
) -> None:
    """A command the OS refuses must come back as a result, not a traceback."""
    missing = await actions.call({"action": "spawn", "argv": ["/no/such/command"]})
    assert missing.details == {"is_error": True}
    assert "Could not start job" in missing.text
    assert len(actions._store.list()) == 0  # noqa: SLF001 - the registry is the claim

    no_such_dir = await actions.call(
        {"action": "spawn", "argv": ["/bin/true"], "cwd": "/no/such/directory"}
    )
    assert no_such_dir.details == {"is_error": True}


async def test_call_is_awaitable_and_never_raises(actions: JobActions) -> None:
    """Every action, given nonsense, returns a result rather than raising.

    The tool's contract with tau is that an argument mistake is data. A
    handler that raises turns a typo into a failed turn.
    """
    for arguments in (
        {"action": "spawn", "argv": 3},
        {"action": "list", "limit": "many"},
        {"action": "log", "id": 42},
        {"action": "wait", "id": None},
        {"action": "kill", "id": [], "how": 1},
        {"action": None},
        {},
    ):
        result = await actions.call(arguments)
        assert isinstance(result.text, str), arguments
        assert result.text, arguments


# ----------------------------------------------------------- one schema


def test_the_tool_is_one_jobs_tool_with_an_action_enum() -> None:
    tool = build_jobs_tool(lambda *args, **kwargs: None)
    assert tool.name == "jobs"
    assert tool.parameters["required"] == ["action"]
    assert tool.parameters["properties"]["action"]["enum"] == list(ACTIONS)
    assert {
        "argv",
        "label",
        "cwd",
        "cap",
        "log_max_bytes",
        "id",
        "tail",
        "timeout",
        "limit",
        "how",
    } <= set(tool.parameters["properties"])


# ------------------------------------------------------- formatting voices


def _fixture_store(root: Path) -> tuple[Job, Job]:
    """Two jobs written straight to a registry: one done, one still running.

    Filed an exact number of seconds before `NOW`, which is what makes a
    finished row's elapsed column assertable: it is `ended_at - started_at`,
    both stored, so it does not move while the test runs. The snapshot is read
    back after `finish` because a job's `rc` comes from the marker, and a
    caller who kept the pre-marker object would be holding a job that has not
    exited yet.
    """
    registry = JobStore(root=root)
    done = registry.create(
        label="build",
        argv=("/bin/sh", "-c", "make"),
        cwd=str(root),
        pid=os.getpid(),
        pgid=os.getpgid(0),
        # Ran from NOW-64 to NOW-1: an exact 63 seconds, so the row reads
        # `63s` whatever the test then takes to run.
        started_at=NOW - 64.0,
        job_id="20260101-000001-aaaa",
    )
    registry.finish(done, rc=0, outcome="exited", ended_at=NOW - 1.0)
    done = registry.get(done.id)
    running = registry.create(
        label="a very long label indeed for a job that is still going",
        argv=("sleep", "100"),
        cwd=str(root),
        pid=os.getpid(),
        pgid=os.getpgid(0),
        started_at=NOW - 900.0,
        job_id="20260101-000002-bbbb",
    )
    return done, running


def _deadline_store(root: Path) -> Job:
    """One running job whose deadline has passed by a minute at `NOW`."""
    registry = JobStore(root=root)
    return registry.create(
        label="late",
        argv=("sleep", "100"),
        cwd=str(root),
        pid=os.getpid(),
        pgid=os.getpgid(0),
        started_at=NOW - 120.0,
        deadline=NOW - 60.0,
        job_id="20260101-000003-cccc",
    )


def test_format_job_row_is_one_stable_line(tmp_path: Path) -> None:
    done, running = _fixture_store(tmp_path / "registry")
    assert format_job_row(done) == (
        "20260101-000001-aaaa  done  build  63s  rc=0  log=0B"
    )
    # A label is cut at LABEL_COLUMN, never wrapped, and a job with no marker
    # shows no rc. Its own elapsed seconds are not asserted — a running job is
    # measured to the clock — only that the row still ends in one field.
    row = format_job_row(running)
    assert row.startswith("20260101-000002-bbbb  running  a very long label")
    assert "…" in row  # 46 characters of label, cut to 39 plus an ellipsis
    assert row.endswith("s  log=0B")
    assert len(row.split("  ")[2]) == LABEL_COLUMN
    assert "\n" not in row
    assert "rc=" not in row
    # `status` is the seam a caller passes a clock through. Overdue is a
    # deadline question, and this job was never given one — so a `now` a day
    # ahead still reads as running, which is the honest answer.
    assert "overdue" not in row
    assert "running" in format_job_row(running, now=NOW + 86_400)
    assert "overdue" in format_job_row(
        _deadline_store(tmp_path / "late"), now=NOW + 60.0
    )


def test_format_job_list_is_a_stable_table(tmp_path: Path) -> None:
    done, running = _fixture_store(tmp_path / "registry")
    table = format_job_list([running, done])
    assert table.splitlines()[0] == "2 job(s), newest first:"
    assert table.splitlines()[-1] == "From a shell: tau-job --help."
    assert table == "\n".join(
        [
            "2 job(s), newest first:",
            format_job_row(running),
            format_job_row(done),
            "From a shell: tau-job --help.",
        ]
    )
    # Read back off disk, the same registry gives the same rows.
    read_back = format_job_list(JobStore(root=tmp_path / "registry").list())
    assert read_back == table
    assert len(format_job_list([done]).splitlines()) == 3
    assert DEFAULT_LIST_LIMIT == 50


def test_format_job_log_marks_the_bytes_it_skipped(tmp_path: Path) -> None:
    done, _ = _fixture_store(tmp_path / "registry")
    done.log_path.write_bytes(b"abcdefghij")
    assert format_job_log(done, b"ghij", 10) == (
        f"Log of job {done.id} (build):\n…6 bytes above\nghij"
    )
    assert (
        format_job_log(done, b"abcdefghij", 10)
        == f"Log of job {done.id} (build):\nabcdefghij"
    )  # nothing skipped, so no rule line
    empty = format_job_log(done, b"", 0)
    assert empty == f"Log of job {done.id} (build):\n(the log is empty)"
    assert "No log file at" in format_job_log(done, b"", 0, missing=True)
    assert DEFAULT_LOG_TAIL_BYTES == 4096 and MAX_LOG_TAIL_BYTES == 65_536


# ------------------------------------------------------------ the command


def test_jobs_command_lists_and_tails_without_any_ui(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = JobStore(root=root)
    done, running = _fixture_store(root)
    done.log_path.write_bytes(b"line one\nline two\n")
    finished_row = format_job_row(done)

    table = jobs_command_text(registry)
    assert table.startswith("2 job(s), newest first:\n")
    assert table.endswith("From a shell: tau-job --help.")
    assert finished_row in table
    assert running.id in table  # the running row is there, seconds and all
    # The table is rows only; a log appears when one job is asked for.
    assert "line one" not in table

    one = jobs_command_text(registry, done.id)
    assert one.splitlines()[0] == finished_row
    assert "line one" in one and "line two" in one
    assert "tau-job --help" not in one

    unknown = jobs_command_text(registry, "no-such-job")
    assert "Unknown job id: 'no-such-job'" in unknown
    assert done.id in unknown  # the suggestion names what does exist
