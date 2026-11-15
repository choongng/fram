"""Running a job: the tee'd spawn, the supervisor, and the `tau-job` CLI.

`jobs.py` owns what a job *is* — three files in a directory. This module owns
the process behind them: start a command with its output already redirected,
watch it to the end, and write the marker that says how it ended. A job here
is any command the session started outside its own process tree, and it is
deliberately dumb: argv, a log, a cap. No tau core, no UI, no settings.

The tee is a descriptor redirect and not a pipe. A pipe needs a live reader:
the moment the supervisor goes away the job takes `EPIPE` on its next write
and dies. With `stdout` pointed at the log the kernel appends without our
help, so the job keeps producing output whether or not anyone is watching,
and the supervisor stays alive only for its own bookkeeping.

    stdout=log  stderr=STDOUT  stdin=DEVNULL  start_new_session=True

The log is opened `"ab"`, mode `0600`, before the child is created, so there
is no window in which the child could be born holding the spawner's
descriptors. The manifest is written after the child is up, because a
manifest naming a pid that never ran is a lie the registry cannot check.

What survives the supervisor is the log, the manifest, and — if it got that
far — `done.json`. What does not survive is the cap: only a live supervisor
sends the TERM at a deadline. A job whose supervisor was killed runs on, and
its row reads `lost`, which is the truth rather than a guess.

Process groups, sessions, `os.killpg` and the fork in `spawn_detached` are
POSIX; on any other platform neither this module nor the registry works.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jobs import DONE_NAME, MANIFEST_NAME, Job, JobStore, new_job_id
from .paths import jobs_home

# Polling is a supervisor's only clock: a deadline is checked, never signalled.
POLL_SECONDS = 0.25
# Grace between the TERM at a deadline and the KILL that follows it.
TERM_GRACE_SECONDS = 30.0
# `tau-job`'s exit code when the job itself never started.
SPAWN_FAILED_RC = 1


class UsageError(ValueError):
    """A command line `main` can explain in one line and exit 2 over."""


async def spawn(
    store: JobStore,
    *,
    label: str,
    argv: list[str] | tuple[str, ...],
    cwd: str | Path,
    cap: float | None = None,
    unit: str | None = None,
    env: dict[str, str] | None = None,
    parent_session: str | None = None,
    log_max_bytes: int = 0,
) -> tuple[Job, asyncio.subprocess.Process]:
    """Start a job's command tee'd into its log, and register it.

    Returns the manifest and the live child. The log is opened before the
    child exists; the manifest is written only once the child is up. If
    `store.create` raises, the child's group is killed before the exception
    escapes — a job with no manifest is a job nobody is watching, and it must
    not run on unregistered.

    `start_new_session` makes the child a group and session leader, which is
    what lets `kill_job` signal everything the job spawned and not the
    spawner. `log_max_bytes` is accepted here so one call can carry a spawn's
    whole configuration to its supervisor; `supervise_async` is where the cap
    is enforced, because a spawn cannot know how big the log will get.
    """
    del log_max_bytes
    argv = list(argv)
    # Resolved, because the store names a job's project directory from the
    # resolved path: a relative cwd here and a resolved one there would file
    # the manifest in one bucket and the job in another.
    cwd = str(Path(cwd).expanduser().resolve())
    # One timestamp decides both the manifest's `started_at` and the id its
    # directory is named after, so the log `prepare` makes and the manifest
    # `create` writes are the same job's. That ordering is the whole reason
    # there are two calls: the child's stdout is the log, and the manifest
    # wants a pid the child does not have until it is running.
    started_at = time.time()
    job_id = new_job_id(started_at)
    log_path = store.prepare(job_id, cwd)
    process: asyncio.subprocess.Process | None = None
    try:
        # The file object, not a descriptor: the spawn machinery moves a file
        # object onto the child's stdout itself, where a raw descriptor would
        # stay at whatever number it happened to have. Append mode, so a log
        # this directory already holds is never truncated.
        with log_path.open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        job = store.create(
            job_id=job_id,
            label=label,
            argv=tuple(argv),
            cwd=cwd,
            pid=process.pid,
            pgid=process.pid,
            started_at=started_at,
            deadline=(started_at + cap) if cap is not None else None,
            unit=unit,
            parent_session=parent_session,
        )
    except BaseException:
        if process is not None:
            _kill_group(process.pid, how="kill")
        _discard_unregistered(log_path)
        raise
    return job, process


async def supervise_async(
    process: asyncio.subprocess.Process,
    job: Job,
    store: JobStore,
    *,
    cap: float | None = None,
    log_max_bytes: int = 0,
) -> str:
    """Wait for a job to end, enforce its caps, write `done.json` once.

    Returns the outcome: `exited` for a normal return, `signalled` for a
    signal, `timeout` when the deadline passed, `log_overflow` when the log
    grew past `log_max_bytes`. A deadline sends `SIGTERM` to the group once
    and `SIGKILL` `TERM_GRACE_SECONDS` later, so a job that traps TERM still
    ends; an oversized log is a disk problem and is killed outright.

    The marker is written exactly once, after the wait resolves, whatever
    path through the loop the exit took. A deadline never fires for a job
    that has already exited — the outcome is what killed it, not what was
    nearly true when it did.

    Two clocks read one duration, and nothing compares them. This loop's
    deadline is monotonic and starts when supervision starts, because a
    signal must not fire early because the wall clock stepped. The manifest
    records the same duration as an absolute wall-clock instant
    (`started_at + cap`), which is what `Job.status()` weighs against now to
    call a job `overdue`. Enforcement acts, the registry judges; a row may
    read `overdue` while this loop is still inside its poll interval or its
    TERM grace, and that is the honest reading of a job past its deadline and
    not yet dead.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + cap if cap is not None else None
    term_sent_at: float | None = None
    overflow = False
    rc: int | None = None
    while True:
        try:
            # `TimeoutError` is `asyncio.TimeoutError` here, so a wait that
            # is merely not yet over is the only thing this catches.
            rc = await asyncio.wait_for(process.wait(), timeout=POLL_SECONDS)
            break
        except TimeoutError:
            pass
        now = loop.time()
        if overflow:
            continue
        if log_max_bytes > 0 and _log_size(job) > log_max_bytes:
            _kill_group(job.pgid, how="kill")
            overflow = True
            continue
        if deadline is not None and now >= deadline:
            if term_sent_at is None:
                _kill_group(job.pgid, how="term")
                term_sent_at = now
            elif now - term_sent_at >= TERM_GRACE_SECONDS:
                _kill_group(job.pgid, how="kill")
    outcome = _outcome(rc=rc, overflow=overflow, cap=cap, term_sent=term_sent_at)
    # `rc` is None only if the child vanished without a waitable status; -1
    # records "no return code" rather than inventing a clean one.
    store.finish(
        job,
        rc=rc if rc is not None else -1,
        outcome=outcome,
        ended_at=time.time(),
    )
    return outcome


def spawn_detached(
    store: JobStore,
    *,
    label: str,
    argv: list[str] | tuple[str, ...],
    cwd: str | Path,
    cap: float | None = None,
    unit: str | None = None,
    env: dict[str, str] | None = None,
    parent_session: str | None = None,
    log_max_bytes: int = 0,
) -> Job:
    """Start a job whose supervisor outlives this process; return its manifest.

    Returns as soon as the supervisor reports the job's id, without waiting
    for the job. The caller's process is then free to exit while the job
    keeps running. `tau-job --detach` and the `jobs` tool's `spawn` are its
    callers.

    It reaps nothing. The supervisor is the job's parent, so it lives exactly
    as long as the job, and waiting for it would be waiting for the job — the
    one thing this function exists to avoid. When the supervisor is finished
    its parent is free to ignore the pid, and an exited one is adopted and
    reaped by `init`. Only a supervisor that reported nothing is waited on,
    since then the caller is about to raise and there is no job to keep.

    Must be called from synchronous code with no event loop running: it
    forks, and a child of a process that has an asyncio loop inherits a loop
    that belongs to its parent. `main` therefore creates the loop only for
    the foreground path, never before forking.

    No re-execution. A `setsid python -m ...` supervisor would re-import the
    world, re-parse the intent, and could never read the job's exit status,
    because the job would not be its child. Forking keeps the supervisor as
    the real parent — which is also why the child cannot reuse a process
    object handed down from a spawn: that object belongs to whoever waited on
    the pid. So the child `setsid`s out of the caller's terminal, then runs
    `spawn` and `supervise_async` inside a fresh `asyncio.run` of its own.
    The child is then the spawner, `process.wait()` is genuine, and the exit
    status is readable.

    The id crosses back over a pipe as `<job_id>\\n<pid>\\n`. A supervisor
    that fails to start the job writes nothing, so an empty read is the
    failure signal. Both ends are `O_CLOEXEC`, which is what makes that
    signal arrive when it should: the supervisor holds the write end while it
    supervises, so without close-on-exec the job inherits a writer, the read
    cannot reach its end-of-file until the job exits, and `--detach` returns
    at the end of the command instead of at its start. Closing it early is not
    an alternative — the supervisor is the one that writes the id.
    """
    # `os.pipe()` sets `O_CLOEXEC` on both ends (CPython 3.4+, POSIX only).
    # `set_inheritable(False)` is what `subprocess` keys its
    # close-descriptors-on-exec behaviour off, so the two calls together are
    # what keeps this pipe out of the job — unlike the job's own log, which
    # is opened without it precisely because it must survive the exec.
    readable, writable = os.pipe()
    os.set_inheritable(readable, False)
    os.set_inheritable(writable, False)
    pid = os.fork()
    if pid == 0:
        _supervise_detached(
            readable,
            writable,
            store,
            label=label,
            argv=argv,
            cwd=cwd,
            cap=cap,
            unit=unit,
            env=env,
            parent_session=parent_session,
            log_max_bytes=log_max_bytes,
        )
    os.close(writable)
    try:
        report = _read_report(readable)
    finally:
        os.close(readable)
    if not report:
        # An empty report means the supervisor died before spawning, so it is
        # already gone and this only reaps it. When it did report, there is
        # nothing to wait for: the supervisor lives exactly as long as the
        # job, so reaping it here would turn `--detach` into a foreground run.
        # An exited supervisor is adopted by `init`, which reaps it.
        os.waitpid(pid, 0)
        raise OSError("detached supervisor reported no job")
    job = store.get(report[0])
    if job is None:
        raise OSError(f"detached job {report[0]} is not in the registry")
    return job


def kill_job(job: Job, *, how: str = "term") -> bool:
    """Signal a job's process group; report whether anything was signalled.

    `how` is `term` or `kill`. Signalling the group is the point of starting
    the job in its own session: a job that spawned a build tree dies with it.
    A group that is already gone returns False, which is what the caller
    wanted and not an error.

    One guard: a manifest that names *our own* group must never be signalled
    as a group, or the caller kills itself and everyone sharing its terminal.
    Such a job falls back to signalling the recorded pid alone.
    """
    if how not in ("term", "kill"):
        raise ValueError(f"unknown kill style: {how!r}")
    if job.pgid > 0 and job.pgid != os.getpgrp():
        return _kill_group(job.pgid, how=how)
    return job.pid > 0 and _kill_pid(job.pid, how=how)


async def wait_for_finish(
    store: JobStore,
    job_id: str,
    *,
    timeout: float | None = None,
    poll: float = POLL_SECONDS,
) -> Job | None:
    """Poll the registry until the job reads `done`; return it, or None.

    The registry is what is polled and never the process: the supervisor's
    `done.json` is the evidence of an exit, and a caller holding no process
    handle — a different process, a different day — gets the same answer. The
    refreshed `Job` is returned so the caller does not read it a second time.
    """
    loop = asyncio.get_running_loop()
    limit = None if timeout is None else loop.time() + timeout
    while True:
        job = store.get(job_id)
        if job is not None and job.status() == "done":
            return job
        if limit is None:
            await asyncio.sleep(poll)
        else:
            remaining = limit - loop.time()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(poll, remaining))


def main(argv: list[str] | None = None) -> int:
    """`tau-job` entry point: parse, spawn, supervise, report.

    Prints `<id>\\t<log path>` on stdout by default, so a script can capture
    the pair, or the manifest and the finished job as one JSON object under
    `--json`. The foreground exit code is the job's own return code, which is
    what lets `tau-job` compose like `timeout`; `--detach` prints the id and
    exits 0 as soon as the supervisor reports it. A missing `--`, an empty
    command, or a bad option is a usage error: one line on stderr, exit 2.
    """
    try:
        spec = _parse_args(sys.argv[1:] if argv is None else argv)
    except UsageError as exc:
        print(f"tau-job: {exc}", file=sys.stderr)
        return 2
    store = _store()
    try:
        if spec.detach:
            job = spawn_detached(
                store,
                label=spec.label,
                argv=spec.argv,
                cwd=spec.cwd,
                cap=spec.cap,
                env=spec.env,
                log_max_bytes=spec.log_max_bytes,
            )
            print(job.id, flush=True)
            return 0
        return asyncio.run(_run_foreground(store, spec))
    except UsageError as exc:  # pragma: no cover - parser already ran
        print(f"tau-job: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"tau-job: {exc}", file=sys.stderr)
        return SPAWN_FAILED_RC


async def _run_foreground(store: JobStore, spec: _Spec) -> int:
    """Spawn, print what was spawned, supervise, return the exit code."""
    job, process = await spawn(
        store,
        label=spec.label,
        argv=spec.argv,
        cwd=spec.cwd,
        cap=spec.cap,
        env=spec.env,
        log_max_bytes=spec.log_max_bytes,
    )
    if not spec.as_json:
        print(f"{job.id}\t{job.log_path}", flush=True)
    outcome = await supervise_async(
        process, job, store, cap=spec.cap, log_max_bytes=spec.log_max_bytes
    )
    if spec.as_json:
        print(_json_report(store, job.id, outcome), flush=True)
    return _exit_code(outcome, process.returncode)


def _supervise_detached(
    readable: int, writable: int, store: JobStore, **kwargs: Any
) -> None:  # pragma: no cover - body runs only in the forked child
    """Run the detached supervisor, then leave without unwinding the parent.

    Never returns: the child exits by `os._exit`, so it cannot run the
    parent's `finally` blocks, flush its buffers twice, or keep its
    descriptors alive. The read end is closed immediately — the child has
    nothing to read — and the write end is closed by `_detached_run` as soon
    as it has written the id, so the parent is not left waiting on a writer
    that has nothing more to say. A supervisor that dies before writing holds
    no writer either, and the parent's empty read is the failure signal.

    Its own three standard descriptors go to `/dev/null` before anything else.
    A supervisor outlives the process that
    forked it, and whoever spawned one is often holding those descriptors open
    to read from — a pipe a caller captures, a socket, a pty. Left inherited,
    they keep that pipe's writer count above zero for the whole life of the
    job, so a caller waiting for end-of-file on its own child waits until the
    job ends and the detach is undone. `subprocess.run(..., capture_output=*)`
    is the ordinary case, not a hostile one.
    """
    os.close(readable)
    code = SPAWN_FAILED_RC
    try:
        _detach_std_streams()
        os.setsid()
        rc = asyncio.run(_detached_run(store, report_to=writable, **kwargs))
        code = rc if rc is not None else SPAWN_FAILED_RC
    except BaseException:
        code = SPAWN_FAILED_RC
    os._exit(code)


async def _detached_run(
    store: JobStore,
    *,
    report_to: int,
    cap: float | None = None,
    log_max_bytes: int = 0,
    **kwargs: Any,
) -> int:  # pragma: no cover - child-side path
    """Spawn in the child's own loop, report the id, supervise to the end.

    The caps stay in this signature rather than being dug back out of a
    bag of keyword arguments, because they are what supervision needs and a
    `.get()` would lose both the type and the default.

    The report is written and its descriptor closed before the supervising
    wait, so the pipe has no writer of ours while the job runs: the parent is
    waiting for an end-of-file, and every microsecond it waits is a
    millisecond the caller is not free.
    """
    job, process = await spawn(
        store, cap=cap, log_max_bytes=log_max_bytes, **kwargs
    )
    _report_and_close(report_to, f"{job.id}\n{process.pid}\n")
    outcome = await supervise_async(
        process, job, store, cap=cap, log_max_bytes=log_max_bytes
    )
    return _exit_code(outcome, process.returncode)


def _report_and_close(report_to: int, text: str) -> None:  # pragma: no cover
    """Write the report, then release the write end.

    Paired because the two are one decision: the parent is blocked on this
    pipe's end-of-file, and it reaches that only when no writer is left. A
    separate close in a `finally` is a correctness trap of its own — a second
    close of a descriptor the interpreter may have handed to another file.
    """
    os.write(report_to, text.encode())
    os.close(report_to)


def _detach_std_streams() -> None:  # pragma: no cover - child-side path
    """Point the child's stdin, stdout and stderr at `/dev/null`.

    Done with `os.dup2` on an opened `/dev/null` rather than by reassigning
    `sys.stdout`, because the descriptor is what a grandchild inherits; a
    Python-level stub would leave the real one open. Any failure is ignored:
    a supervisor that cannot open `/dev/null` has lost its terminal but has a
    job to watch, and that is not a reason to kill it or the job.
    """
    try:
        null = os.open(os.devnull, os.O_RDWR)
    except OSError:
        return
    try:
        for standard in (0, 1, 2):
            os.dup2(null, standard)
    except OSError:  # pragma: no cover - only if /dev/null is unkind
        pass
    if null > 2:
        os.close(null)


def _outcome(
    *,
    rc: int | None,
    overflow: bool,
    cap: float | None,
    term_sent: float | None,
) -> str:
    """Name how a job ended, given what the supervisor did and the rc it saw.

    An overflow kills with SIGKILL, so the log rule wins over the rc it
    produces. A negative rc after we sent the deadline's TERM is the deadline
    taking effect; a negative rc with no TERM of ours is somebody else's
    signal, which is `signalled`.
    """
    if overflow:
        return "log_overflow"
    if term_sent is not None and rc is not None and rc < 0:
        return "timeout" if cap is not None else "signalled"
    if rc is None or rc < 0:
        return "signalled"
    return "exited"


def _exit_code(outcome: str, rc: int | None) -> int:
    """The code `tau-job` exits with, matching how the job ended."""
    if outcome == "timeout":
        return 124
    if rc is None:
        return 1
    return rc if rc >= 0 else 128 + abs(rc)


def _json_report(store: JobStore, job_id: str, outcome: str) -> str:
    """Render the manifest and the finished marker as one JSON object."""
    job = store.get(job_id)
    manifest: object = {}
    done: object = {}
    if job is not None:
        manifest = _read_json(job.dir / MANIFEST_NAME)
        done = _read_json(job.dir / DONE_NAME)
    return json.dumps({"job": manifest, "finished": done, "outcome": outcome})


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}


def _read_report(handle: int) -> list[str]:
    """Read the supervisor's `<id>\\n<pid>\\n` report; nothing on a failure."""
    chunk = b""
    while not chunk.endswith(b"\n"):
        try:
            part = os.read(handle, 128)
        except OSError:
            return []
        if not part:
            break
        chunk += part
    return [line for line in chunk.decode(errors="replace").splitlines() if line]


def _store() -> JobStore:
    """The registry `main` reads, which is `jobs_home()`.

    `paths.jobs_home` is the one place `TAU_JOBS_DIR` is read: a store takes
    its root as an argument, so pointing the variable at a temporary directory
    moves the CLI, the extension's watcher, and a test together. That seam is
    what keeps `main(argv) -> int` free of a required flag: a command line
    that needed `--root` could not be the default shape.
    """
    return JobStore(root=jobs_home())


def _discard_unregistered(log_path: Path) -> None:
    """Remove a job directory whose manifest was never written.

    A spawn that failed before `JobStore.create` leaves a directory with a log
    in it and nothing to say who it belongs to. That is not a job, and a row
    for it would outlive the attempt.
    """
    directory = log_path.parent
    if (directory / MANIFEST_NAME).exists():
        return
    shutil.rmtree(directory, ignore_errors=True)


def _log_size(job: Job) -> int:
    try:
        return job.log_path.stat().st_size
    except OSError:
        return 0


def _kill_group(pgid: int, *, how: str) -> bool:
    return _signal(os.killpg, pgid, how)


def _kill_pid(pid: int, *, how: str) -> bool:
    return _signal(os.kill, pid, how)


def _signal(send: Any, target: int, how: str) -> bool:
    """Send TERM or KILL to one target; report whether it was there to take it."""
    sig = signal.SIGTERM if how == "term" else signal.SIGKILL
    try:
        send(target, sig)
    except OSError:
        # Includes ESRCH (already gone) and EPERM (not ours to kill).
        return False
    return True


@dataclass
class _Spec:
    """The parsed `tau-job` command line."""

    label: str = ""
    argv: list[str] = field(default_factory=list)
    cwd: str = "."
    cap: float | None = None
    env: dict[str, str] | None = None
    log_max_bytes: int = 0
    detach: bool = False
    as_json: bool = False


# Options that stand alone, mapped to the attribute they set.
_FLAGS = {"--detach": "detach", "--json": "as_json"}


def _parse_args(raw: list[str]) -> _Spec:
    """Parse the command line, raising `UsageError` for any mistake.

    Hand-written because argv after `--` belongs to the job and must not be
    touched: argparse would raise its own errors and exit the process rather
    than return 2, and an absent `--` is our condition, not its parsing.
    """
    spec = _Spec()
    label = ""
    index = 0
    while index < len(raw):
        option = raw[index]
        if option == "--":
            spec.argv = list(raw[index + 1 :])
            break
        if option in _FLAGS:
            setattr(spec, _FLAGS[option], True)
            index += 1
            continue
        value, index = _option_value(raw, index)
        if option == "--label":
            label = value
        elif option == "--cap":
            spec.cap = _positive_float(value, "--cap")
        elif option == "--cwd":
            spec.cwd = value
        elif option == "--env":
            key, sep, val = value.partition("=")
            if not sep or not key:
                raise UsageError("--env wants K=V")
            spec.env = spec.env if spec.env is not None else {}
            spec.env[key] = val
        elif option == "--log-max-bytes":
            spec.log_max_bytes = _non_negative_int(value, "--log-max-bytes")
        else:
            raise UsageError(f"unknown option: {option}")
    if not spec.argv:
        raise UsageError("usage: tau-job [options] -- command [args...]")
    spec.label = label or _default_label(spec.argv)
    return spec


def _option_value(raw: list[str], index: int) -> tuple[str, int]:
    """Return the value following the option at `index`, and the next index.

    `--` is never an option's value, even in the position where one belongs:
    it opens the job's argv, and eating it here would drop the separator and
    swallow the command. So `--cap -- /bin/true` is a `--cap` with no value
    rather than a silently misparsed job.
    """
    if index + 1 >= len(raw) or raw[index + 1] == "--":
        raise UsageError(f"option {raw[index]} needs a value")
    return raw[index + 1], index + 2


def _positive_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise UsageError(f"{name} must be a number of seconds") from None
    if value <= 0:
        raise UsageError(f"{name} must be greater than zero")
    return value


def _non_negative_int(text: str, name: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise UsageError(f"{name} must be a whole number of bytes") from None
    if value < 0:
        raise UsageError(f"{name} must not be negative")
    return value


def _default_label(argv: list[str]) -> str:
    """Name a job after its command when `--label` was not given.

    The basename plus its first argument is what a row can read at a glance;
    sixty characters is what a strip row has room for.
    """
    return f"{os.path.basename(argv[0])} {' '.join(argv[1:2])}".strip()[:60]


if __name__ == "__main__":
    raise SystemExit(main())
