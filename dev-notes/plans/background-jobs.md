# Background jobs: tee'd spawn, on-disk registry, strip rows

## Overview

Add a job: any command the session starts outside its own process tree.
The spawner redirects the command's descriptors to a log file and writes a
small manifest next to it. A registry reads those files, so a run stays
visible after the process that started it exits. The strip shows a row per
job, and a log view tails the file.

We do this because a detached run is invisible today. Tau core has no job
concept: the bash tool pipes stdout and waits. `tau -p` in the background
leaves nothing but a session directory.

## Goals

- Start a job from the CLI (`tau-job`) or from the model (`jobs` tool).
- Capture stdout and stderr in one file from the first byte, with no reader
  process required.
- List jobs, read their logs, wait for them, and kill them, from the model,
  from `/jobs`, and from the strip.
- Tell the parent session once, at its next turn, when a job finishes.
- Keep every part working with the parent dead, and in print mode.

## Current State

- `src/tau_subagents/output_file.py` owns durable transcripts at
  `~/.tau/subagents/<slug>-<hash>/<session>/tasks/<id>.jsonl`, and
  `encode_cwd` / `project_dir_name` name the project directory.
- `src/tau_subagents/ui/controller.py` mounts the fleet strip through
  `components.set_slot_widget(STRIP_KEY, ..., placement="below_prompt")` and
  owns one nav state machine in a pre-dispatch key interceptor.
  `ui/strip.py` renders one row per run; `ui/viewer.py` opens a transcript
  through `components.open_main_view`.
- `src/tau_coding/tools.py` (core) runs a shell with `stdout=PIPE` and waits.
  Nothing survives the tool call.

## Design

**Where this goes, and what it hides.** Four new modules.
`jobs.py` owns the on-disk shape: where a job lives, what its manifest says,
what state it is in, and how to read the tail of its log. It is synchronous
and has no asyncio, no UI, and no model-facing code, so the CLI, the tools,
and the widgets all read the same truth. `job_run.py` owns the process:
spawn with the tee, supervise to the end, write `done.json`, enforce a cap,
kill a group. `jobs_watch.py` is the one poller: it snapshots the registry on
a timer and tells subscribers what changed; the notification queue and the
strip both subscribe, so no other part of the extension polls.
`jobs_tool.py` holds the model-facing tool and `/jobs`.

**The approach that lost.** A registry in the session process, as the strip
has for runs, with a log path handed to whoever asks. It loses twice. It
cannot see a job the session did not start, which is the case that matters
(a launch script, a build started before a crash, a run whose supervisor
died). And a log written to a pipe needs a live reader: kill the supervisor
and the job gets `EPIPE`, so a registry that survives would have to keep a
process alive to stay readable. The fd-level redirect costs us one small
directory of files and buys both properties. A second rejected shape:
a supervisor that copies the pipe and rotates the log. It makes the log the
disk problem of a process that can die. We cap by killing the job group
instead, and leave the file intact.

**The interface.** One `JobStore` with `create(...) -> Job`, `get(id)`,
`list()`, `tail(job, ...)`, `sweep(...)`, and a `Job` that answers
`status(now) -> JobStatus`. A caller never opens a manifest, never computes a
path, and never guesses a state from mtimes. A `Job` snapshot is plain data,
so a widget can hold one between polls. `JobStatus` is the whole vocabulary:
`running`, `overdue`, `done`, `lost`, `unreadable`. There is no `unknown`
state to reason about: a manifest we cannot read is a row that says so.

**What the next change touches.** Most likely: a second kind of job source,
such as jobs discovered from tau session directories, or jobs owned by core.
That touches `jobs_watch.py` (feed it a second source) and nothing else; the
manifest, the tools, and the rows do not change, because they read `Job`. A
change to the log format (structured events instead of bytes) touches
`ui/log_view.py` and `Job.tail`, and no writer.

**Errors defined away.** A log file cannot be lost or interleaved: one job,
one descriptor, opened once at spawn. A job cannot appear finished while its
process lives, because `done.json` is the only evidence of an exit and the
pid decides the rest. A dead supervisor cannot misreport: no `done.json` plus
no live pid is `lost`, which is the truth. A job cannot write to an arbitrary
path: the store makes every path. A kill cannot leave children behind: the
job is its own process group, and we signal the group.

## Changes Required

### 1. Paths (`src/tau_subagents/paths.py`, new; `output_file.py`)

Move `encode_cwd`, `project_dir_name`, and the `~/.tau/subagents` resolver to
`paths.py`, and add `jobs_home(home=None) -> Path` returning `~/.tau/jobs`.
`output_file.py` imports them back, so its API and tests do not change.

### 2. Registry (`src/tau_subagents/jobs.py`, new)

```python
@dataclass(frozen=True, slots=True)
class Job:
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
    rc: int | None = None          # from done.json
    ended_at: float | None = None
    outcome: str | None = None     # exited|timeout|signalled|log_overflow
    error: str | None = None       # set for an unreadable manifest

    def status(self, now: float | None = None) -> JobStatus: ...
    def elapsed(self, now: float | None = None) -> float: ...

class JobStore:
    def __init__(self, root: Path | None = None) -> None: ...
    def create(self, *, label, argv, cwd, pid, pgid, started_at,
               deadline=None, unit=None, parent_session=None) -> Job: ...
    def get(self, job_id: str) -> Job | None: ...
    def list(self, *, limit: int = 50) -> list[Job]: ...   # newest first
    def finish(self, job, *, rc, outcome, ended_at) -> None: ...
    def tail(self, job, *, start: int = 0,
             max_bytes: int = 65_536) -> tuple[bytes, int]: ...  # bytes, size
    def last_line(self, job) -> str | None: ...
    def sweep(self, *, retention_days: int) -> int: ...
```

Layout, `0700` directories and `0600` files:

```
~/.tau/jobs/<slug>-<hash>/<id>/job.json     written once at spawn
                              out.log       stdout+stderr, appended by the OS
                              done.json     written once at exit
```

`<id>` sorts chronologically: `"%Y%m%d-%H%M%S-<4hex>"`. `job.json` is
`{"v": 1, ...}` in the field names above. Liveness is `os.kill(pid, 0)`;
`EPERM` means alive. On Linux a mismatching `/proc/<pid>/cmdline` means the
pid was reused, and the job is `lost`. `sweep` removes finished job
directories older than the retention, never a running one.

### 3. Spawn and supervise (`src/tau_subagents/job_run.py`, new)

```python
def spawn(store, *, label, argv, cwd, cap=None, unit=None,
          env=None, log_max_bytes=0) -> tuple[Job, Any]:
    """Open the log, Popen in a new session, register the manifest."""

async def supervise_async(process, job, store, *, cap, log_max_bytes) -> str:
    """Await the process, enforce cap and log cap, write done.json."""

def supervise(process, job, store, *, cap, log_max_bytes) -> int:
    """Sync form for the CLI: wait, enforce, write, return rc."""

def kill_job(job, *, how: str = "term") -> bool:
    """Signal the group: TERM, then KILL after 5s."""

async def wait_for_finish(store, job_id, *, timeout=None) -> Job | None: ...

def main(argv=None) -> int:
    """`tau-job` entry: parse, spawn, supervise (or detach), print."""
```

The tee is `stdout=log, stderr=STDOUT, stdin=DEVNULL, start_new_session=True`,
with the log opened `"ab"` before the fork. A cap sends `SIGTERM` to the
group at the deadline and `SIGKILL` 30s later, with
`outcome="timeout"`. A log over `log_max_bytes` sends `SIGKILL` to the group
with `outcome="log_overflow"`, because the alternative is the disk. `--detach`
re-executes `python -m tau_subagents.job_run --supervise <job-dir>` under
`setsid`, so a script can fire a job and exit.

CLI:

```
tau-job [--label L] [--cap SECONDS] [--cwd DIR] [--name PATH]
        [--env K=V] [--log-max-bytes N] [--detach] [--json] -- argv...
```

### 4. Watcher (`src/tau_subagents/jobs_watch.py`, new)

```python
class JobWatcher:
    def __init__(self, store, *, interval: float = 2.0) -> None: ...
    def subscribe(self, cb: Callable[[list[Job], list[Job]], None]) -> None:
        """Call cb(started, finished) on each change; return unsubscribe."""
    def start(self) -> None: ...     # idempotent; no-op without an event loop
    def stop(self) -> None: ...
    def snapshot(self) -> list[Job]: ...
```

One `asyncio` task, one poll of the directory, and a diff on `status()`.
It holds no state the registry does not already have.

### 5. Model surface (`src/tau_subagents/jobs_tool.py`, new; `extension.py`)

One tool, `jobs`, with an `action` enum:

| action | arguments | returns |
| --- | --- | --- |
| `spawn` | `argv`, `label`, `cwd`, `cap` | id, log path |
| `list` | `limit` | one row per job |
| `log` | `id`, `tail` | last N bytes of the log |
| `wait` | `id`, `timeout` | the finished job |
| `kill` | `id` | whether it stopped |

`/jobs [id]` lists, or tails one job, and prints a plain table so it works
in print mode. A finished job reaches the model once, at the next turn
boundary, through the same outbox the subagent notices use, and once to the
terminal with `tau.notify`.

### 6. Strip rows and log view (`ui/strip.py`, `ui/controller.py`,
`ui/log_view.py`)

Job rows follow the run rows in the one strip widget, so one nav machine owns
both. A job row is a status glyph, the label, elapsed, and the last log line.
`enter` on a job row opens `ui/log_view.py` through `open_main_view`.

`JobLogView` keeps a byte cursor, reads only appended bytes with
`store.tail(job, start=cursor)`, renders the last `MAX_LINES` lines, follows
by default, and pauses on `f`. `g` and `G` go to the top of what is held and
back to the tail; `q` closes. It never reads the whole file. A log bigger
than the view's window shows a `…N bytes above` rule.

### 7. Settings (`src/tau_subagents/settings.py`)

`jobsRetentionDays` (default 30) and `jobsLogMaxBytes` (default 0, none).
Same merge and range-drop rules as the keys above it.

## Testing Plan

```bash
uv run pytest -q tests/test_jobs_store.py tests/test_job_run.py
uv run pytest -q tests/test_jobs_watch.py tests/test_jobs_tool.py
uv run pytest -q tests/test_jobs_ui.py
uv run pytest -q            # full suite
```

Store: create/get/list round-trip, id sort order, `status` for every state
including pid reuse (a fake `/proc` path) and an unreadable manifest, tail
with a cursor past the end, and `sweep` leaving a running job alone.
`job_run`: a real `/bin/sh` job writes the log and `done.json`; a cap kills
the group (assert a sleep grandchild dies); `--detach` returns and the job
still finishes. Watcher: one callback per transition. Tool: each action, plus
print-mode `/jobs`. UI: rows render for a fixture store, and `JobLogView`
follows an appended chunk (Textual pilot, as in `tests/test_ui.py`).

Manual: `tau-job --label sleep --cap 3 -- /bin/sh -c 'echo a; sleep 2; echo
b'`, then `/jobs` and `jobs log <id>` in a session; start a job that outlives
the session and check the row shows `lost`.

## Implementation Order

1. `paths.py` move, `jobs.py`, `job_run.py`, `pyproject.toml` script, tests.
2. `jobs_watch.py`, settings, notification at the turn boundary, tests.
3. `jobs_tool.py`, `/jobs`, registration in `extension.py`, tests.
4. `ui/log_view.py`, strip rows, controller wiring, tests.
5. README, `dev-notes/adr/0006-...`, this plan's status.

Each step leaves the suite green.

## Files Modified

- New: `src/tau_subagents/{paths,jobs,job_run,jobs_watch,jobs_tool}.py`,
  `src/tau_subagents/ui/log_view.py`, five test modules, the ADR, this plan.
- Changed: `output_file.py` (imports), `settings.py`, `extension.py`,
  `ui/strip.py`, `ui/controller.py`, `ui/__init__.py`, `pyproject.toml`,
  `README.md`, `dev-notes/README.md`.

## Definition of Done

- A job started by `tau-job --detach` from a shell appears in `/jobs` of a
  session in the same project, with its log tailable from the strip.
- A job whose supervisor was `kill -9` shows `lost`, not `running`.
- No job, and no `tau-job`, keeps a reader process alive.
- The full suite passes; print mode raises nothing.

## Open Questions

- Does `~/.tau/jobs` belong to us, or should it sit under
  `~/.tau/subagents/jobs` until core wants the concept?

## Out of Scope

- Discovering runs that were never registered (no manifest). A transcript
  pane for a `tau -p` job depends on that, so it is out too.
- Log rotation, structured log rendering, and job dependencies.
- Windows. `start_new_session` and process groups are POSIX; other platforms
  keep the CLI and the registry, and lose `kill` and the cap.

## Risks and Mitigations

- A job can fill the disk. `jobsLogMaxBytes` kills the group; the default is
  off, and the row shows the log size when it is large.
- A 2s poll adds a readdir per interval. The watcher runs only while a
  session is live, and reads manifests for jobs it has not seen.
- Pid reuse can call a dead job alive. We compare `/proc/<pid>/cmdline`
  where it exists, and the row for a job with no `done.json` says `lost`
  after the pid is gone either way.
