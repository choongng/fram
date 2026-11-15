"""The on-disk job store: round-trip, the five states, tail, sweep."""

import json
import os
from pathlib import Path

import pytest

from tau_subagents.jobs import Job, JobStore


def _always_alive(_pid: int) -> bool:
    return True


def _never_alive(_pid: int) -> bool:
    return False


def _fake_cmdline(*parts: str) -> object:
    return lambda _pid: list(parts)


def _make(
    store: JobStore,
    tmp_path: Path,
    *,
    label: str = "build",
    argv: tuple[str, ...] = ("/bin/sh", "-c", "echo hi"),
    started_at: float = 1_700_000_000.0,
    **kwargs: float | str | None,
) -> Job:
    return store.create(
        label=label,
        argv=argv,
        cwd=str(tmp_path),
        pid=os.getpid(),
        pgid=os.getpgid(0),
        started_at=started_at,
        **kwargs,
    )


def test_create_get_list_round_trip(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path, label="compile", unit="rust")

    fetched = store.get(job.id)
    assert fetched is not None
    assert fetched.id == job.id
    assert fetched.label == "compile"
    assert fetched.argv == ("/bin/sh", "-c", "echo hi")
    assert fetched.cwd == str(tmp_path)
    assert fetched.unit == "rust"
    assert fetched.log_path == job.dir / "out.log"
    assert fetched.rc is None and fetched.outcome is None

    listed = store.list()
    assert [entry.id for entry in listed] == [job.id]
    assert store.get("no-such-id") is None


def test_ids_sort_chronologically_newest_first(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    older = _make(store, tmp_path, started_at=1_700_000_000.0)
    newer = _make(store, tmp_path, started_at=1_700_000_060.0)

    assert older.id < newer.id
    assert [entry.id for entry in store.list()] == [newer.id, older.id]
    assert [entry.id for entry in store.list(limit=1)] == [newer.id]


def test_created_directories_and_files_are_private(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    assert (job.dir / "job.json").stat().st_mode & 0o777 == 0o600
    assert (job.dir / "out.log").stat().st_mode & 0o777 == 0o600
    # The job directory and its project parent are both 0700.
    assert job.dir.stat().st_mode & 0o777 == 0o700
    assert job.dir.parent.stat().st_mode & 0o777 == 0o700


def test_manifest_is_versioned_with_the_field_names(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path, deadline=1_700_000_100.0)

    payload = json.loads((job.dir / "job.json").read_text())
    assert payload["v"] == 1
    assert payload["id"] == job.id
    assert payload["argv"] == ["/bin/sh", "-c", "echo hi"]
    assert payload["deadline"] == 1_700_000_100.0
    assert payload["parent_session"] is None


def test_status_running_for_a_live_pid(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    assert job.status(read_cmdline=_fake_cmdline(*job.argv)) == "running"
    # A real, live pid with no fake at all: this process is definitely here.
    assert job.status() == "running"


def test_status_overdue_once_past_the_deadline(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path, deadline=1_700_000_010.0)

    assert job.status(1_700_000_005.0, read_cmdline=None) == "running"
    assert job.status(1_700_000_011.0, read_cmdline=None) == "overdue"


def test_status_lost_when_the_pid_is_gone(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    assert job.status(is_alive=_never_alive) == "lost"


def test_status_lost_when_the_pid_has_been_recycled(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    hijacked = job.status(
        is_alive=_always_alive,
        read_cmdline=_fake_cmdline("/usr/bin/other", "--flag"),
    )
    assert hijacked == "lost"
    # A reader with no opinion (no /proc) leaves the job running.
    assert (
        job.status(is_alive=_always_alive, read_cmdline=lambda _pid: None)
        == "running"
    )


def test_status_done_after_finish_carries_the_rc(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    finished = store.finish(job, rc=3, outcome="exited", ended_at=1_700_000_002.0)

    assert finished.status() == "done"
    assert finished.rc == 3
    assert finished.outcome == "exited"
    assert finished.ended_at == 1_700_000_002.0
    # The marker wins over the pid: done.json is the only evidence of an exit.
    assert store.get(job.id).status(is_alive=_never_alive) == "done"
    assert finished.elapsed() == 2.0
    assert json.loads((job.dir / "done.json").read_text())["v"] == 1
    assert (job.dir / "done.json").stat().st_mode & 0o777 == 0o600


def test_status_unreadable_for_a_malformed_manifest(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)
    (job.dir / "job.json").write_text("{not json")

    broken = store.get(job.id)
    assert broken is not None
    assert broken.status() == "unreadable"
    assert "manifest" in (broken.error or "")
    # A missing manifest is unreadable too, and listing never raises.
    listed = store.list()
    assert [entry.status() for entry in listed] == ["unreadable"]
    (job.dir / "job.json").unlink()
    assert store.get(job.id).status() == "unreadable"


def test_list_keeps_an_unreadable_job_alongside_the_readable_ones(
    tmp_path: Path,
) -> None:
    store = JobStore(root=tmp_path / "jobs")
    good = _make(store, tmp_path, label="good", started_at=1_700_000_000.0)
    broken = _make(store, tmp_path, label="broken", started_at=1_700_000_060.0)
    # A manifest that parses but has no usable argv: the row keeps its label.
    (broken.dir / "job.json").write_text('{"v": 1, "id": 7, "label": "broken"}')

    listed = store.list()
    assert [entry.label for entry in listed] == ["broken", "good"]
    assert listed[0].status() == "unreadable"
    assert listed[1].id == good.id


def test_tail_reads_with_a_cursor(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    # create() touches the log, so an empty file is size 0, not missing.
    assert store.tail(job) == (b"", 0)
    job.log_path.write_bytes(b"first\n")
    chunk, size = store.tail(job, start=0)
    assert (chunk, size) == (b"first\n", 6)

    job.log_path.write_bytes(b"first\nsecond\n")
    # The cursor the previous call implies reads only what was appended.
    chunk, size = store.tail(job, start=6)
    assert (chunk, size) == (b"second\n", 13)
    # A start past the end is empty bytes with the real size.
    assert store.tail(job, start=999) == (b"", 13)
    # max_bytes caps the read so a caller keeps a cursor across polls.
    assert store.tail(job, start=0, max_bytes=5)[0] == b"first"


def test_tail_of_a_missing_log_is_empty(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)
    job.log_path.unlink()

    assert store.tail(job) == (b"", 0)


def test_last_line_skips_blanks_and_truncates(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)

    assert store.last_line(job) is None
    job.log_path.write_bytes(b"\n\n  \n")
    assert store.last_line(job) is None

    job.log_path.write_bytes(b"one\ntwo\n\n")
    assert store.last_line(job) == "two"

    job.log_path.write_bytes(("x" * 200 + " tail\n").encode())
    line = store.last_line(job)
    assert line is not None
    assert len(line) == 120
    assert line.startswith("x" * 120)

    # Undecodable bytes are replaced rather than raising.
    job.log_path.write_bytes(b"\xff\xfe done\n")
    assert store.last_line(job) == "\ufffd\ufffd done"


def test_sweep_removes_an_old_finished_job_and_keeps_a_running_one(
    tmp_path: Path,
) -> None:
    store = JobStore(root=tmp_path / "jobs")
    finished = _make(store, tmp_path, label="finished")
    running = _make(store, tmp_path, label="running")
    store.finish(finished, rc=0, outcome="exited", ended_at=1_700_000_001.0)

    old = 1_700_000_000.0
    os.utime(finished.dir / "done.json", (old, old))
    # The running job's directory is older still, and must survive.
    os.utime(running.dir, (old - 100, old - 100))

    assert store.sweep(retention_days=1) == 1

    assert not finished.dir.exists()
    assert running.dir.is_dir()
    assert store.get(running.id) is not None
    # Nothing left to remove, and a disabled retention removes nothing.
    assert store.sweep(retention_days=1) == 0
    assert store.sweep(retention_days=0) == 0


def test_sweep_leaves_an_unreadable_job_to_age_by_directory_mtime(
    tmp_path: Path,
) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)
    (job.dir / "job.json").write_text("{not json")

    # Fresh by directory mtime: left alone, and nothing raised.
    assert store.sweep(retention_days=1) == 0
    assert job.dir.is_dir()

    old = 1_700_000_000.0
    os.utime(job.dir, (old, old))
    assert store.sweep(retention_days=1) == 1
    assert not job.dir.exists()


@pytest.mark.parametrize("bad", ["", "[]", "null", "{}"])
def test_malformed_manifests_are_rows_not_errors(
    tmp_path: Path, bad: str
) -> None:
    store = JobStore(root=tmp_path / "jobs")
    job = _make(store, tmp_path)
    (job.dir / "job.json").write_text(bad)

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].status() == "unreadable"
