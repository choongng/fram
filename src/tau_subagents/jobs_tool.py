"""The model-facing `jobs` tool and the `/jobs` command.

One tool with an `action` enum, so the model has one thing to remember and
five verbs: `spawn`, `list`, `log`, `wait`, `kill`. `/jobs` is the same
information rendered for a human through the plain-text command path, so it
works in print mode where there are no components to mount.

Both are thin. `jobs.py` owns what a job is, `job_run.py` owns the process,
and this module only decides what a caller is told about them. The text is
built by the small pure `format_*` functions below, and the tool and the
command call the same ones, so the two cannot drift into two vocabularies for
one registry. `extension.format_job_notice` is the third voice — a notice says
something happened, a row says what a job is now — and it stays there.

Nothing here imports `extension.py`. That would be a cycle, and it would mean
the tool could only run inside the extension; taking the store, the session's
cwd, its id and its settings as arguments keeps this testable against a
temporary registry.

A tool handler never raises. Every mistake — an unknown id, an unknown action,
a missing `argv`, a cap of zero — comes back as a result whose text names what
to pass instead, because a raised handler loses its `details` and reaches the
model as a generic tool failure.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tau_agent.tools import AgentTool, AgentToolResult

from .job_run import kill_job, spawn_detached, wait_for_finish
from .jobs import Job, JobStore

#: Rows `list` shows when the caller does not name a number.
DEFAULT_LIST_LIMIT = 50
#: Bytes of log the `log` action returns when the caller does not ask.
DEFAULT_LOG_TAIL_BYTES = 4096
#: Bytes of log `/jobs <id>` shows: what reads well without paging.
COMMAND_LOG_TAIL_BYTES = 2048
#: A `log` tail no caller may ask past, whatever it hopes for.
MAX_LOG_TAIL_BYTES = 65_536
#: `wait` seconds by default, and the ceiling. A model that asks for an hour
#: is asking to hold the turn hostage; it can call again.
DEFAULT_WAIT_SECONDS = 60.0
MAX_WAIT_SECONDS = 300.0
#: How wide a label is cut for a row. A build command is often longer than a
#: terminal, and one wrapped row reads as two jobs.
LABEL_COLUMN = 40

#: Marks an error result in `details`; the text is what the model reads.
RESULT_PREFIX = "jobs:"

#: The verb list, spelled once for the schema, the dispatch and its errors.
ACTIONS: tuple[str, ...] = ("spawn", "list", "log", "wait", "kill")


class JobActions:
    """The five verbs, against one registry and one session's identity.

    The session's cwd, id and log cap arrive as callables rather than values,
    because a session's id is only what it is some time after `setup()` runs.
    A getter that raises costs one field of a row, never the whole call: each
    is read inside a guard at the point where its absence is survivable.
    """

    def __init__(
        self,
        store: JobStore,
        *,
        cwd: Callable[[], str],
        session_id: Callable[[], str | None],
        log_max_bytes: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._cwd = cwd
        self._session_id = session_id
        self._log_max_bytes = log_max_bytes

    async def call(self, arguments: Mapping[str, Any]) -> AgentToolResult:
        """Dispatch one action; anything wrong comes back as error text."""
        action = str(arguments.get("action") or "")
        handlers: Mapping[
            str, Callable[[Mapping[str, Any]], Awaitable[AgentToolResult]]
        ] = {
            "spawn": self.spawn,
            "list": self.list,
            "log": self.log,
            "wait": self.wait,
            "kill": self.kill,
        }
        handler = handlers.get(action)
        if handler is None:
            return failure(
                f"Unknown action: {action!r}. Pass one of: {', '.join(ACTIONS)}."
            )
        return await handler(arguments)

    async def spawn(self, arguments: Mapping[str, Any]) -> AgentToolResult:
        """Detach a command into the registry and report where it lives.

        `spawn_detached` forks a supervisor and returns as soon as that
        supervisor reports the id, so this call costs milliseconds whatever
        the command then takes; `jobs.py`'s own note on why a foreground spawn
        is the wrong default applies twice over here. It forks, and a fork
        inherits its parent's asyncio loop, so the call is taken to a thread
        with no loop of its own.

        `log_max_bytes` comes from the `jobsLogMaxBytes` setting unless the
        caller overrides it. That is where the setting is finally enforced:
        the value has meant nothing since it was parsed, because nothing
        carried it to the supervisor that kills the group for outgrowing it.
        """
        argv, argv_error = _string_list(arguments.get("argv"), "argv")
        if argv_error is not None or argv is None:
            return failure(argv_error or "Missing argv.")
        cap, cap_error = _positive_number(arguments.get("cap"), "cap")
        if cap_error is not None:
            return failure(cap_error)
        log_max_bytes, bytes_error = _non_negative_int(
            arguments.get("log_max_bytes"),
            "log_max_bytes",
            default=self._configured_log_cap(),
        )
        if bytes_error is not None:
            return failure(bytes_error)
        cwd = _session_cwd(self._cwd, arguments.get("cwd"))
        label = _job_label(arguments.get("label"), argv)
        try:
            job = await asyncio.to_thread(
                spawn_detached,
                self._store,
                label=label,
                argv=argv,
                cwd=cwd,
                cap=cap,
                parent_session=self._session(),
                log_max_bytes=log_max_bytes,
            )
        except (OSError, ValueError) as exc:
            return failure(f"Could not start job: {exc}")
        return success(
            f"Started job {job.id} ({job.label}).\n"
            f"Log: {job.log_path}\n"
            f"Status: {job.status()}. This session is told when it finishes;"
            " use wait to follow it or log to read its output."
        )

    async def list(self, arguments: Mapping[str, Any]) -> AgentToolResult:
        """Return one row per job, newest first."""
        limit, error = _positive_int(
            arguments.get("limit"), "limit", default=DEFAULT_LIST_LIMIT
        )
        if error is not None:
            return failure(error)
        jobs = await asyncio.to_thread(self._store.list, limit=limit)
        return success(format_job_list(jobs))

    async def log(self, arguments: Mapping[str, Any]) -> AgentToolResult:
        """Return the last bytes of one job's log, and what was skipped."""
        job, error = self._need_job(arguments.get("id"))
        if error is not None or job is None:
            return error or failure("Missing id.")
        tail, tail_error = _bounded_int(
            arguments.get("tail"),
            "tail",
            default=DEFAULT_LOG_TAIL_BYTES,
            maximum=MAX_LOG_TAIL_BYTES,
        )
        if tail_error is not None:
            return failure(tail_error)
        chunk, size = await asyncio.to_thread(_tail_of, job, self._store, tail)
        return success(
            format_job_log(job, chunk, size, missing=not job.log_path.is_file())
        )

    async def wait(self, arguments: Mapping[str, Any]) -> AgentToolResult:
        """Follow a job to its end, or report where it is at the deadline."""
        job_id, id_error = _required_str(arguments.get("id"), "id")
        if id_error is not None:
            return failure(id_error)
        timeout, error = _bounded_number(
            arguments.get("timeout"),
            "timeout",
            default=DEFAULT_WAIT_SECONDS,
            maximum=MAX_WAIT_SECONDS,
        )
        if error is not None:
            return failure(error)
        finished = await wait_for_finish(self._store, job_id, timeout=timeout)
        if finished is not None:
            return success(format_job_finished(finished))
        current = self._store.get(job_id)
        if current is None:
            return failure(unknown_job_message(job_id, self._store))
        return success(
            f"{format_job_row(current)}\n"
            f"Still {current.status()} after {timeout:g}s."
            f"\nLog: {current.log_path}"
        )

    async def kill(self, arguments: Mapping[str, Any]) -> AgentToolResult:
        """Signal a job's process group and report what the job reads now."""
        job, error = self._need_job(arguments.get("id"))
        if error is not None or job is None:
            return error or failure("Missing id.")
        how = str(arguments.get("how") or "term")
        if how not in ("term", "kill"):
            return failure(f"Unknown how: {how!r}. Pass 'term' or 'kill'.")
        signalled = kill_job(job, how=how)
        refreshed = self._store.get(job.id) or job
        if not signalled:
            return success(
                f"{format_job_row(refreshed)}\n"
                "Nothing was signalled: the process group is already gone."
                f"\nStatus now: {refreshed.status()}."
            )
        return success(
            f"Sent {how.upper()} to job {job.id}'s process group."
            f"\nStatus now: {refreshed.status()}."
            "\nWait for it to report how it ended."
        )

    def _need_job(
        self, value: object
    ) -> tuple[Job | None, AgentToolResult | None]:
        """The job an id names, or the error naming what to pass instead."""
        job_id, error = _required_str(value, "id")
        if error is not None:
            return None, failure(error)
        job = self._store.get(job_id)
        if job is None:
            return None, failure(unknown_job_message(job_id, self._store))
        return job, None

    def _configured_log_cap(self) -> int:
        """`jobsLogMaxBytes`, or no cap when there is no settings reader."""
        if self._log_max_bytes is None:
            return 0
        try:
            return self._log_max_bytes()
        except Exception:  # noqa: BLE001 - a setting we cannot read is no cap
            return 0

    def _session(self) -> str | None:
        try:
            return self._session_id()
        except Exception:  # noqa: BLE001 - an id is a label, not a dependency
            return None


def success(text: str) -> AgentToolResult:
    """A plain-text result with an empty `details`, as the loop wants."""
    return AgentToolResult(content=text, details={})


def failure(message: str) -> AgentToolResult:
    """An error result in the shape `_tool_result` uses, never an exception.

    `extension._tool_result` has no `is_error` parameter — the failure lives
    in the text and that is what the model reads — so the flag is carried in
    `details` for anything that wants to branch on it.
    """
    return AgentToolResult(
        content=f"{RESULT_PREFIX} {message}",
        details={"is_error": True},
    )


def unknown_job_message(job_id: str, store: JobStore) -> str:
    """An unknown id, plus the ids that would have been recognised."""
    try:
        known = ", ".join(job.id for job in store.list(limit=5)) or "none"
    except Exception:  # noqa: BLE001 - the suggestion is a courtesy
        known = "unknown"
    return (
        f"Unknown job id: {job_id!r}. Known jobs, newest first: {known}."
        " Pass action='list' for the table."
    )


def format_job_row(job: Job, *, now: float | None = None) -> str:
    """One line for one job: id, status, label, elapsed, rc, log size.

    The vocabulary is the registry's — `running`, `overdue`, `done`, `lost`,
    `unreadable` — and a label is cut rather than wrapped, so a row is a row.
    An `rc` is shown only when a marker recorded one, and an unreadable
    manifest says so in the row instead of inventing a state.
    """
    label = job.label.strip() or "(unlabelled)"
    if len(label) > LABEL_COLUMN:
        label = f"{label[: LABEL_COLUMN - 1].rstrip()}…"
    row = f"{job.id}  {job.status()}  {label}  {int(job.elapsed(now=now))}s"
    if job.rc is not None:
        row += f"  rc={job.rc}"
    row += f"  log={_log_size(job)}B"
    if job.error:
        row += f"  ({job.error})"
    return row


def format_job_list(jobs: Sequence[Job], *, now: float | None = None) -> str:
    """The table the `list` action and `/jobs` print: newest first."""
    if not jobs:
        return (
            "No jobs in this project's registry. Start one with the jobs"
            " tool (action='spawn') or from a shell: tau-job --help."
        )
    lines = [f"{len(jobs)} job(s), newest first:"]
    lines.extend(format_job_row(job, now=now) for job in jobs)
    lines.append("From a shell: tau-job --help.")
    return "\n".join(lines)


def format_job_log(
    job: Job,
    chunk: bytes,
    size: int,
    *,
    missing: bool = False,
) -> str:
    """Render a log tail with the rule for the bytes above it.

    `chunk` is what `JobStore.tail` read, `size` the file's whole size, so
    the line saying how much was skipped is arithmetic and not a guess. A
    missing log is reported as a fact about the filesystem: a job whose
    directory was swept still has a row, and its log is simply not there.
    """
    heading = f"Log of job {job.id} ({job.label}):"
    if missing:
        return f"{heading}\nNo log file at {job.log_path}."
    if size == 0:
        return f"{heading}\n(the log is empty)"
    skipped = max(0, size - len(chunk))
    body = chunk.decode("utf-8", errors="replace").rstrip("\n")
    lines = [heading]
    if skipped > 0:
        lines.append(f"…{skipped} bytes above")
    lines.append(body or "(no output in this window)")
    return "\n".join(lines)


def format_job_finished(job: Job) -> str:
    """A finished job as the `wait` action reports it."""
    return (
        f"{format_job_row(job)}\n"
        f"Outcome: {job.outcome or job.status()}."
        f"\nLog: {job.log_path}"
    )


def jobs_command_text(store: JobStore, argument: str = "") -> str:
    """What `/jobs [id]` prints: the table, or one job and the tail of its log.

    Plain text and nothing else, because the command path is synchronous end
    to end and print mode has no components to render into. The tail here is
    the human-sized one, and the registry is read on the calling thread — the
    same read the tool does in a thread, and small enough to be a readdir per
    job directory.
    """
    argument = argument.strip()
    if not argument:
        return format_job_list(store.list(limit=DEFAULT_LIST_LIMIT))
    job = store.get(argument)
    if job is None:
        return unknown_job_message(argument, store)
    chunk, size = _tail_of(job, store, COMMAND_LOG_TAIL_BYTES)
    return "\n".join(
        [
            format_job_row(job),
            "",
            format_job_log(job, chunk, size, missing=not job.log_path.is_file()),
        ]
    )


def build_jobs_tool(
    execute_fn: Callable[..., Awaitable[AgentToolResult]],
) -> AgentTool:
    """The `jobs` tool: one action enum, and the arguments each verb reads.

    The handler is passed in rather than captured here, because the actions
    it dispatches through are built in `extension.setup` and this module must
    not import that one.
    """
    return AgentTool(
        name="jobs",
        label="Jobs",
        description=(
            "Start and follow background jobs: commands that outlive this"
            " tool call, kept in a registry on disk so they survive the"
            " session that started them. Pick one action."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(ACTIONS),
                    "description": (
                        "spawn: start a detached command and return its id"
                        " and log path immediately. list: one row per job,"
                        " newest first. log: the last bytes of a job's log."
                        " wait: block until a job finishes or the timeout"
                        " passes. kill: signal a job's process group."
                    ),
                },
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "spawn only, required: the command and its arguments"
                        ' as a list of strings (["make", "-j4"]). There is no'
                        " shell, so no pipes or redirects."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "spawn only: what the row calls the job. Default:"
                        " the command's basename and its first argument."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "spawn only: directory to run the command in."
                        " Default: this session's working directory."
                    ),
                },
                "cap": {
                    "type": "number",
                    "minimum": 0.001,
                    "description": (
                        "spawn only: seconds the job may run before it is"
                        " sent SIGTERM and then killed. Omit for no deadline."
                    ),
                },
                "log_max_bytes": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "spawn only: bytes the log may reach before the job's"
                        " group is killed. Default: the jobsLogMaxBytes"
                        " setting, which is 0 (no cap) unless configured."
                    ),
                },
                "id": {
                    "type": "string",
                    "description": (
                        "log, wait and kill only: the id a spawn returned."
                    ),
                },
                "tail": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LOG_TAIL_BYTES,
                    "description": (
                        f"log only: how many bytes to return from the end of"
                        f" the log (default {DEFAULT_LOG_TAIL_BYTES}, maximum"
                        f" {MAX_LOG_TAIL_BYTES})."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": MAX_WAIT_SECONDS,
                    "description": (
                        f"wait only: seconds to block (default"
                        f" {DEFAULT_WAIT_SECONDS:g}, maximum"
                        f" {MAX_WAIT_SECONDS:g}). A job still running when it"
                        " passes is reported as still running, not as an"
                        " error."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        f"list only: how many jobs to show (default"
                        f" {DEFAULT_LIST_LIMIT})."
                    ),
                },
                "how": {
                    "type": "string",
                    "enum": ["term", "kill"],
                    "description": (
                        "kill only: SIGTERM (default) or SIGKILL, sent to the"
                        " job's whole process group."
                    ),
                },
            },
            "required": ["action"],
        },
        execute_fn=execute_fn,
        prompt_snippet="Start and follow detached background jobs and their logs.",
    )


# ------------------------------------------------------------ argument reading
#
# Every reader below returns `(value, error)` rather than raising, so a
# handler can answer a bad argument in the same breath it names the good
# shape. The messages say what to pass and not merely what was wrong, on the
# theory that the reader is a model deciding what to send next.


def _required_str(value: object, name: str) -> tuple[str, str | None]:
    """The string a caller must have passed, and the error saying so."""
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    return "", f"Missing {name}: pass the job id a spawn returned."


def _string_list(value: object, name: str) -> tuple[list[str] | None, str | None]:
    """A non-empty list of strings, or what to pass instead of it.

    A single shell string is named out rather than split, because splitting
    `"rm -rf $dir"` by hand would quietly decide what the caller's quotes
    meant — and there is no shell on the other end to do it honestly.
    """
    if isinstance(value, str) and value.strip():
        return None, (
            f"{name} must be a list of strings, not one shell string:"
            ' pass ["make", "-j4"]. There is no shell, so no pipes.'
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, (
            f"Missing {name}: pass the command as a list of strings,"
            ' e.g. ["make", "-j4"].'
        )
    items = list(value)
    if not items:
        return None, f"{name} must name a command: pass at least one entry."
    if not all(isinstance(item, str) and item for item in items):
        return None, f"{name} must be a list of non-empty strings."
    return items, None


def _positive_number(
    value: object | None, name: str
) -> tuple[float | None, str | None]:
    """An omitted or positive number; anything else names the mistake."""
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name} must be a number of seconds."
    if value <= 0:
        return None, f"{name} must be greater than zero; omit it for no cap."
    return float(value), None


def _bounded_number(
    value: object | None, name: str, *, default: float, maximum: float
) -> tuple[float, str | None]:
    """A number clamped to `maximum`: asking for more is not an error."""
    if value is None:
        return default, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default, f"{name} must be a number of seconds."
    if value < 0:
        return default, f"{name} must not be negative."
    return min(float(value), maximum), None


def _positive_int(
    value: object | None, name: str, *, default: int
) -> tuple[int, str | None]:
    """A positive count, or the message naming what a count looks like."""
    if value is None:
        return default, None
    number, error = _coerce_int(value, name, unit="row")
    if error is not None:
        return default, error
    if number <= 0:
        return default, f"{name} must be at least 1."
    return number, None


def _bounded_int(
    value: object | None, name: str, *, default: int, maximum: int
) -> tuple[int, str | None]:
    """A byte count clamped to `maximum`; zero would ask for no output."""
    if value is None:
        return default, None
    number, error = _coerce_int(value, name, unit="byte")
    if error is not None:
        return default, error
    if number <= 0:
        return default, f"{name} must be at least 1 byte."
    return min(number, maximum), None


def _non_negative_int(
    value: object | None, name: str, *, default: int
) -> tuple[int, str | None]:
    """A byte count that may be zero, where zero means the configured none."""
    if value is None:
        return default, None
    number, error = _coerce_int(value, name, unit="byte")
    if error is not None:
        return default, error
    if number < 0:
        return default, f"{name} must not be negative; use 0 for no cap."
    return number, None


def _coerce_int(value: object, name: str, *, unit: str) -> tuple[int, str | None]:
    """An int from whatever the caller sent; a bool is never a count."""
    if isinstance(value, bool):
        return 0, f"{name} must be a whole number of {unit}s."
    try:
        return int(value), None  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0, f"{name} must be a whole number of {unit}s."


# ---------------------------------------------------------------- other bits


def _session_cwd(getter: Callable[[], Path | str], override: object) -> Path:
    """Where a job runs: the directory it names, else the session's.

    A relative `cwd` is anchored at the session's directory rather than left
    for the child to resolve against wherever this process happens to be —
    `spawn_detached` resolves what it is given, and the two only agree if
    this does the anchoring.
    """
    try:
        base = Path(getter())
    except Exception:  # noqa: BLE001 - a session with no readable cwd
        base = Path.cwd()
    if isinstance(override, str) and override.strip():
        candidate = Path(override.strip()).expanduser()
        return candidate if candidate.is_absolute() else base / candidate
    return base


def _job_label(value: object, argv: Sequence[str]) -> str:
    """The caller's label, or the basename and first argument it stands for.

    The same sixty characters `job_run._default_label` gives a `tau-job` row,
    so a job reads the same however it was started.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    first = f" {argv[1]}" if len(argv) > 1 else ""
    return f"{os.path.basename(argv[0])}{first}".strip()[:60]


def _tail_of(job: Job, store: JobStore, max_bytes: int) -> tuple[bytes, int]:
    """Read the last `max_bytes` of a log, and report the file's whole size.

    `JobStore.tail` reads forward from a cursor, because the watcher wants
    what is new; this wants what is last, and a job that has written more than
    one read's worth is the normal case, not the exception. So the offset is
    worked out here: back up `max_bytes` from the end, then read forward. A
    log shorter than the read starts at zero and no bytes are skipped, which
    is what makes the "bytes above" line say nothing when there is nothing
    above. Bytes are counted, not lines, so a job that writes one enormous
    line cannot make this read the whole file.
    """
    size = _log_size(job)
    if size <= 0:
        return b"", 0
    start = max(0, size - max_bytes)
    chunk, reread = store.tail(job, start=start, max_bytes=max_bytes)
    return chunk, reread or size


def _log_size(job: Job) -> int:
    try:
        return job.log_path.stat().st_size
    except OSError:
        return 0
