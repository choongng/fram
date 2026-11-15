"""The on-disk job registry: where a job lives, and what state it is in.

A job is a command the session started outside its own process tree, and the
registry is the only thing that still knows about it once the process that
spawned it has exited. So a job is its files and nothing else: one manifest
written at spawn, one log the OS appends to, and one marker written at exit.
A `Job` is a snapshot of those three, which is why the store is synchronous
and has no asyncio, no UI, and no tau imports — the CLI, the model-facing
tool, and the strip widgets all read the same truth off the same directory.

    ~/.tau/jobs/<cwd-slug>-<hash>/<id>/job.json     written once at spawn
                                  /out.log          stdout and stderr, appended
                                  /done.json        written once at exit

The directory-per-job shape is the whole design. A log written to a pipe
needs a live reader, so a registry built on pipes would need a process kept
alive to stay readable; the fd-level redirect instead costs one directory of
files and lets the spawner exit as soon as it has written the manifest. In
exchange, the registry can see a job it did not start — a launch script, a
build left behind by a crash — because all it needs is the manifest.

Two limits are worth naming. Liveness is `os.kill(pid, 0)`, which is POSIX:
it says only what the kernel will signal, and a process we do not own answers
`EPERM` because we may signal it and not because it is ours. And a pid is a
recycled number, so a dead job can look alive; where `/proc` exists we compare
the recorded argv against `/proc/<pid>/cmdline` and call the job `lost` on a
mismatch. Without `/proc`, only the pid is checked, and a row can be wrong in
that one way.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Literal

from .paths import jobs_home, project_dir_name

MANIFEST_VERSION = 1
MANIFEST_NAME = "job.json"
LOG_NAME = "out.log"
DONE_NAME = "done.json"

# The whole vocabulary, with no `unknown` for a caller to reason about: a
# manifest we cannot read is a row that says so.
JobStatus = Literal["running", "overdue", "done", "lost", "unreadable"]

LAST_LINE_WIDTH = 120
_SECONDS_PER_DAY = 86_400

# Injected by callers that need to fake an answer: a recycled pid cannot be
# conjured in a test, and the real `/proc` must never be a test's dependency.
Liveness = Callable[[int], bool]
CmdlineReader = Callable[[int], list[str] | None]


@dataclass(frozen=True, slots=True)
class Job:
    """One job, as its manifest and done marker said at the last read.

    Plain data, so a widget can hold a snapshot between polls. Paths are
    absolute; the log is read through `JobStore.tail`, never by a caller who
    opens it.
    """

    id: str
    label: str
    argv: tuple[str, ...]
    cwd: str
    pid: int
    pgid: int
    started_at: float
    deadline: float | None
    unit: str | None
    parent_session: str | None
    dir: Path
    log_path: Path
    rc: int | None = None
    ended_at: float | None = None
    outcome: str | None = None
    error: str | None = None

    def status(
        self,
        now: float | None = None,
        *,
        is_alive: Liveness | None = None,
        read_cmdline: CmdlineReader | None = None,
    ) -> JobStatus:
        """Return the job's state: running, overdue, done, lost, unreadable.

        `done.json` is the only evidence of an exit, and the pid decides the
        rest. A running job past its deadline is `overdue` rather than a new
        state — it is still running, and the caller still kills it the same
        way. `is_alive` and `read_cmdline` are the seams for liveness and for
        the recycled-pid check; pass them in a test, leave them out in code.
        """
        if self.error is not None:
            return "unreadable"
        done = self.dir / DONE_NAME
        if done.is_file():
            return "done"
        alive = (is_alive or _pid_is_alive)(self.pid) and _cmdline_matches(
            self, read_cmdline
        )
        if not alive:
            return "lost"
        if self.deadline is not None and (now or _now()) > self.deadline:
            return "overdue"
        return "running"

    def elapsed(self, now: float | None = None) -> float:
        """Seconds from spawn to finish, or to now for a job still running."""
        end = self.ended_at if self.ended_at is not None else (now or _now())
        return max(0.0, end - self.started_at)


class JobStore:
    """Reads and writes the job directories under one root.

    The store makes every path, so a caller cannot point a job's log at
    somewhere it does not own. Every method is blocking and best-effort about
    other people's files: a manifest it cannot parse comes back as an
    `unreadable` job rather than raising, because a row that says so beats a
    listing that does not exist.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else jobs_home()

    def create(
        self,
        *,
        label: str,
        argv: tuple[str, ...] | list[str],
        cwd: str | Path,
        pid: int,
        pgid: int,
        started_at: float,
        deadline: float | None = None,
        unit: str | None = None,
        parent_session: str | None = None,
        job_id: str | None = None,
    ) -> Job:
        """Register a job that has been spawned; write its manifest and log.

        The log is created here so a tail of a job that has written nothing is
        an empty file rather than a missing one. `job.json` is written last,
        since a directory with no manifest is not a job.

        Pass `job_id` (from `prepare`) when the directory and its log already
        exist, which is how a spawner that opened the log before the child got
        a pid keeps the manifest beside it. Leave it out to mint a fresh id.
        """
        argv = tuple(argv)
        cwd = str(cwd)
        job_id = job_id or new_job_id(started_at)
        directory = self._job_dir_for(job_id, cwd)
        # The root and the project directory are private too, so the whole
        # tree is 0700 rather than only the leaf a job's files live in.
        _mkdir_private(self.root)
        _mkdir_private(directory.parent)
        _mkdir_private(directory)
        try:
            _touch_private(directory / LOG_NAME)
            _write_json_atomic(
                directory / MANIFEST_NAME,
                {
                    "v": MANIFEST_VERSION,
                    "id": job_id,
                    "label": label,
                    "argv": list(argv),
                    "cwd": cwd,
                    "pid": pid,
                    "pgid": pgid,
                    "started_at": started_at,
                    "deadline": deadline,
                    "unit": unit,
                    "parent_session": parent_session,
                },
            )
        except OSError:
            # A job with no manifest is not a job, so it leaves no directory.
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return _job_from_manifest(
            job_id,
            directory,
            label=label,
            argv=argv,
            cwd=cwd,
            pid=pid,
            pgid=pgid,
            started_at=started_at,
            deadline=deadline,
            unit=unit,
            parent_session=parent_session,
        )

    def prepare(self, job_id: str, cwd: str | Path) -> Path:
        """Make a job's directory and its empty log; return the log path.

        A spawner cannot use `create` alone: the child's stdout is the log, so
        the file has to exist before the child, and the manifest wants a pid
        the child does not have yet. This is the half that comes first. The
        directories are private and the log is `0600`, exactly as `create`
        leaves them, so a job that dies in between leaves an empty directory
        and no manifest — an `unreadable` row, which is the truth about a job
        whose spawn never finished.
        """
        directory = self._job_dir_for(job_id, cwd)
        _mkdir_private(self.root)
        _mkdir_private(directory.parent)
        _mkdir_private(directory)
        _touch_private(directory / LOG_NAME)
        return directory / LOG_NAME

    def get(self, job_id: str) -> Job | None:
        """Return the job for `job_id`, or None when there is no such job."""
        for directory in self._project_dirs():
            candidate = directory / job_id
            if not candidate.is_dir():
                continue
            return _read_job(candidate)
        return None

    def list(self, *, limit: int = 50) -> list[Job]:
        """Return jobs newest first, including any with an unreadable manifest.

        Ids sort chronologically by construction, so sorting them descending
        is the timeline. A directory that holds no manifest is still a row,
        because a half-written job is worth seeing more than a clean listing.
        """
        jobs = [
            job
            for directory in self._project_dirs()
            for job in (_read_job(child) for child in _job_dirs(directory))
        ]
        jobs.sort(key=lambda job: job.id, reverse=True)
        return jobs[:limit] if limit > 0 else []

    def finish(self, job: Job, *, rc: int, outcome: str, ended_at: float) -> Job:
        """Write `done.json` for a job that has exited; return its new snapshot.

        Atomic, because a half-written marker would report an exit we cannot
        read. Nothing here looks at the process: the caller is the supervisor
        that waited for it.
        """
        payload = {
            "v": MANIFEST_VERSION,
            "rc": rc,
            "outcome": outcome,
            "ended_at": ended_at,
        }
        _write_json_atomic(job.dir / DONE_NAME, payload)
        return _read_job(job.dir)

    def tail(
        self,
        job: Job,
        *,
        start: int = 0,
        max_bytes: int = 65_536,
    ) -> tuple[bytes, int]:
        """Read the log from `start` to at most `max_bytes`; return bytes, size.

        The size is the file's current size so the caller can keep the returned
        cursor for the next poll and know whether it read everything. A `start`
        past the end is not an error — the log has not grown — and returns
        empty bytes with the real size. A missing log is `(b"", 0)`.
        """
        try:
            with job.log_path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if start >= size:
                    return b"", size
                handle.seek(start)
                return handle.read(max_bytes), size
        except FileNotFoundError:
            return b"", 0

    def last_line(self, job: Job) -> str | None:
        """Return the last non-empty line of the log, truncated for a row.

        Reads only the window at the end of the file, so the row costs the
        same whether the job has written a line or a gigabyte. Undecodable
        bytes are replaced rather than dropped: a row shows something.
        """
        window = 4096
        try:
            size = job.log_path.stat().st_size
        except OSError:
            return None
        if size == 0:
            return None
        offset = max(0, size - window)
        try:
            with job.log_path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
        except OSError:
            return None
        for line in reversed(chunk.splitlines()):
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                return text[:LAST_LINE_WIDTH]
        return None

    def sweep(self, *, retention_days: int) -> int:
        """Delete finished job directories older than the retention; return count.

        Age is the marker's mtime, falling back to the directory's for a job
        whose marker we cannot read. A job that is running or overdue is never
        touched, and nothing here raises: a sweep runs unattended and a
        directory it cannot read is simply left for another day.
        """
        if retention_days <= 0:
            return 0
        cutoff = _now() - retention_days * _SECONDS_PER_DAY
        removed = 0
        for directory in self._project_dirs():
            for child in _job_dirs(directory):
                if self._sweep_one(child, cutoff):
                    removed += 1
        return removed

    def _sweep_one(self, directory: Path, cutoff: float) -> bool:
        try:
            marker = directory / DONE_NAME
            age = marker.stat().st_mtime if marker.is_file() else _mtime(directory)
            if age >= cutoff:
                return False
            # Whatever state a readable manifest implies decides it, so a job
            # still running past its own retention is kept rather than erased.
            if (directory / MANIFEST_NAME).is_file():
                if _read_job(directory).status() in ("running", "overdue"):
                    return False
            shutil.rmtree(directory)
        except OSError:
            return False
        return True

    def _project_dirs(self) -> list[Path]:
        """Return the project directories under the root, oldest path first."""
        try:
            return sorted(
                child for child in self.root.iterdir() if child.is_dir()
            )
        except OSError:
            return []

    def _job_dir_for(self, job_id: str, cwd: str | Path) -> Path:
        return self.root / project_dir_name(Path(cwd)) / job_id


def new_job_id(started_at: float | None = None) -> str:
    """Return a job id that sorts chronologically: `"%Y%m%d-%H%M%S-<4hex>"`.

    The random suffix separates jobs created in the same second; four hex
    chars is wide enough for a project's parallel jobs and short enough to
    read in a row.
    """
    stamp = datetime.fromtimestamp(
        started_at if started_at is not None else _now()
    ).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def _pid_is_alive(pid: int) -> bool:
    """Whether the kernel will signal `pid` (POSIX liveness, not ownership)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, which is still a live process.
        return True
    except OSError:
        return False
    return True


def read_cmdline(pid: int) -> list[str] | None:
    """Return `/proc/<pid>/cmdline` split into argv, or None when unreadable.

    None means "no opinion", not "mismatch": off Linux, or for a process we
    cannot read, the caller falls back to the pid alone.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return [
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    ]


def _cmdline_matches(job: Job, read: CmdlineReader | None) -> bool:
    """Whether the pid still runs the command the manifest recorded."""
    if read is None:
        return True
    actual = read(job.pid)
    if actual is None:
        return True
    recorded = list(job.argv)
    if not recorded:
        return True
    # The kernel may resolve the path and drop the interpreter's spelling, so
    # the first entry is compared by basename and the rest exactly.
    if os.path.basename(actual[0]) != os.path.basename(recorded[0]):
        return False
    return actual[1:] == recorded[1:]


def _job_from_manifest(
    job_id: str,
    directory: Path,
    *,
    label: str,
    argv: tuple[str, ...],
    cwd: str,
    pid: int,
    pgid: int,
    started_at: float,
    deadline: float | None,
    unit: str | None,
    parent_session: str | None,
) -> Job:
    return Job(
        id=job_id,
        label=label,
        argv=argv,
        cwd=cwd,
        pid=pid,
        pgid=pgid,
        started_at=started_at,
        deadline=deadline,
        unit=unit,
        parent_session=parent_session,
        dir=directory,
        log_path=directory / LOG_NAME,
    )


def _read_job(directory: Path) -> Job:
    """Build the job for one directory, never raising on what is in it."""
    manifest = directory / MANIFEST_NAME
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        # Nothing could be read, so the row has only the directory to name it.
        return _unreadable(directory, f"manifest unreadable: {exc}")
    if not isinstance(data, dict):
        return _placeholder(directory, data, "manifest is not an object")
    # The label and id are read before anything that can fail, so an
    # unreadable job can still show what it was called and be pointed at.
    label = str(data.get("label") or "")
    job_id = _manifest_id(data.get("id"), directory)
    try:
        raw_argv = data["argv"]
        if not isinstance(raw_argv, list) or not all(
            isinstance(part, str) for part in raw_argv
        ):
            raise TypeError("argv must be a list of strings")
        job = _job_from_manifest(
            job_id,
            directory,
            label=label,
            argv=tuple(raw_argv),
            cwd=str(data.get("cwd") or ""),
            pid=int(data["pid"]),
            pgid=int(data["pgid"]),
            started_at=float(data["started_at"]),
            deadline=_optional_float(data.get("deadline")),
            unit=_optional_str(data.get("unit")),
            parent_session=_optional_str(data.get("parent_session")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _placeholder(directory, data, f"manifest malformed: {exc}")
    done = directory / DONE_NAME
    if done.is_file():
        marker = _read_marker(done)
        if marker is None:
            return _unreadable(directory, "done marker malformed", label=label)
        rc, ended_at, outcome = marker
        return _with_marker(job, rc, ended_at, outcome)
    return job


def _manifest_id(value: object, directory: Path) -> str:
    """The manifest's id, or the directory name when there is no usable one."""
    if isinstance(value, str) and value:
        return value
    return directory.name


def _read_marker(
    path: Path,
) -> tuple[int | None, float | None, str | None] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        rc = int(data["rc"])
    except (KeyError, TypeError, ValueError):
        return None
    ended_at = _optional_float(data.get("ended_at"))
    outcome = _optional_str(data.get("outcome"))
    return rc, ended_at, outcome


def _with_marker(
    job: Job, rc: int | None, ended_at: float | None, outcome: str | None
) -> Job:
    values = {field.name: getattr(job, field.name) for field in fields(Job)}
    values.update({"rc": rc, "ended_at": ended_at, "outcome": outcome})
    return Job(**values)


def _placeholder(directory: Path, data: object, error: str) -> Job:
    """A row for a manifest we could parse but not use."""
    found = data if isinstance(data, dict) else {}
    return _unreadable(
        directory,
        error,
        job_id=_manifest_id(found.get("id"), directory),
        label=str(found.get("label") or ""),
    )


def _unreadable(
    directory: Path, error: str, *, job_id: str | None = None, label: str = ""
) -> Job:
    """A row that says so.

    The id falls back to the directory name, which is what a caller can still
    point at; the label is kept when the manifest gave one, so the row shows
    what the job was called even when it cannot say what state it is in.
    """
    return Job(
        id=job_id or directory.name,
        label=label or directory.name,
        argv=(),
        cwd="",
        pid=0,
        pgid=0,
        started_at=_mtime(directory),
        deadline=None,
        unit=None,
        parent_session=None,
        dir=directory,
        log_path=directory / LOG_NAME,
        error=error,
    )


def _job_dirs(directory: Path) -> list[Path]:
    try:
        return sorted(child for child in directory.iterdir() if child.is_dir())
    except OSError:
        return []


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _touch_private(path: Path) -> None:
    with path.open("ab"):
        pass
    os.chmod(path, 0o600)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write JSON through a temp file in the same directory, then replace.

    A reader never sees a partial manifest, and a failed write leaves the
    previous file in place rather than a truncated one.
    """
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _now() -> float:
    return datetime.now().timestamp()
