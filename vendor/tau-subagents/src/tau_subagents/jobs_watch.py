"""The one poller for the job registry: snapshot, diff, tell subscribers.

`jobs.py` can say what every job is right now; it cannot say what just
changed. That is this module's whole job. One asyncio task reads the registry
on a timer, diffs it against what it saw last time, and reports the two
things anyone can act on: a job that appeared, and a job that finished. The
notification path and the strip both subscribe, so nothing else in the
extension polls, and no subscriber has to keep its own copy of the world.

Statelessness is the point. The registry on disk is the truth; the watcher
holds only the previous snapshot, so restarting it costs nothing and two
watchers would disagree only about what they had already announced.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from .jobs import Job, JobStore

#: Statuses a live job answers with. Anything else is over: it finished,
#: went missing, or its manifest stopped making sense.
LIVE_STATUSES = ("running", "overdue")

Unsubscribe = Callable[[], None]


async def _await_cancelled(task: "asyncio.Task[None]") -> None:
    """Wait for a cancelled task to finish dying; never raise for its cancel.

    `cancel()` only requests the stop, so awaiting is what makes "the poll
    task is gone" true rather than almost true. The suppression lives here
    because a cancelled poller is the normal way this ends.
    """
    with contextlib.suppress(asyncio.CancelledError):
        await task


class JobWatcher:
    """Watches a `JobStore` for jobs that appeared or finished.

    The first read seeds and reports nothing: a session that opens on four
    running jobs gets four rows and no notices, because it never saw them
    start. Call `poll_once()` before `start()` to seed off the loop, as the
    extension does; leave it out and the first tick seeds instead. Either
    way only a watched change is reported, which is also why two watchers
    cannot double-announce a transition between them.
    """

    def __init__(self, store: JobStore, *, interval: float = 2.0) -> None:
        # Two seconds: a finished job reported well after the fact, and a
        # readdir nobody notices. The caller names it, so a test does not
        # have to sleep through the default to see a tick.
        self._store = store
        self._interval = interval
        self._callbacks: list[Callable[[list[Job], list[Job]], None]] = []
        self._snapshot: list[Job] = []
        self._last: dict[str, str] = {}
        self._tracking = False
        self._task: asyncio.Task[None] | None = None

    def subscribe(
        self, cb: Callable[[list[Job], list[Job]], None]
    ) -> Unsubscribe:
        """Register `cb(started, finished)`; return its unsubscribe.

        The callback runs on the event loop, so it must not block. One that
        raises is skipped and the rest of the fan-out goes on as if it had
        returned.
        """
        self._callbacks.append(cb)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._callbacks.remove(cb)

        return unsubscribe

    def start(self) -> None:
        """Start polling; a no-op when already running or with no loop.

        Called from `session_start`, which print mode reaches without an
        event loop of its own. Raising there would break the session for a
        feature whose absence is invisible, so the watcher simply stays
        stopped until a loop exists.
        """
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._run(), name="jobs-watch")

    def stop(self) -> None:
        """Cancel the poll task; from a coroutine, `stop_async` is the form.

        A task cannot be awaited from a loop that is already running — that
        is `stop_async`'s situation, not a sync caller's — so this cancels
        and returns. What it leaves behind is bounded: a cancelled task wakes
        at its next sleep, holds no lock, and notifies nobody.
        """
        task = self._task
        self._task = None
        if task is None or asyncio.current_task() is task:
            return
        task.cancel()
        if task.done():
            with contextlib.suppress(asyncio.CancelledError):
                task.result()

    async def stop_async(self) -> None:
        """`stop()` for a coroutine: the poll task is gone when it returns."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await _await_cancelled(task)

    def snapshot(self) -> list[Job]:
        """The jobs read at the last tick; empty before the first."""
        return list(self._snapshot)

    def poll_once(self) -> tuple[list[Job], list[Job]]:
        """Read the registry once and report `(started, finished)`.

        The tick body without the sleep, so a session can seed its snapshot
        and a test can drive a transition with no wall-clock wait either.
        """
        started, finished = self._apply(self._store.list())
        if started or finished:
            self._dispatch(started, finished)
        return started, finished

    def _dispatch(self, started: list[Job], finished: list[Job]) -> None:
        """Hand one change to every subscriber; a broken one is skipped."""
        for callback in tuple(self._callbacks):
            try:
                callback(started, finished)
            except Exception:  # noqa: BLE001 - one cannot stop the rest
                continue

    async def _run(self) -> None:
        """Poll until cancelled; only the registry read is suppressible.

        A blanket `except Exception` around the sleep would look harmless and
        not be: `CancelledError` has derived from `BaseException` since 3.8,
        so it slips past an `Exception` clause and the poller cannot be killed.
        """
        while True:
            await asyncio.sleep(self._interval)
            started, finished = await self._tick()
            if started or finished:
                self._dispatch(started, finished)

    async def _tick(self) -> tuple[list[Job], list[Job]]:
        """One read and diff, off the loop; a failed read reports nothing.

        `on_session_start`'s transcript sweep is the precedent for taking the
        blocking file IO to a thread: a registry of thousands of directories
        must not stall the loop a conversation is running on.
        """
        try:
            jobs = await asyncio.to_thread(self._store.list)
        except Exception:  # noqa: BLE001 - a registry we cannot read is no news
            return [], []
        return self._apply(jobs)

    def _apply(self, jobs: list[Job]) -> tuple[list[Job], list[Job]]:
        """Record one reading and return what it changed.

        A job is started when the previous reading did not have it, and
        finished when it stops answering `running` or `overdue` — `done`,
        `lost`, and `unreadable` are all worth telling someone about. A job
        that disappears while still live is not a finish: the sweep removing
        a directory is not news about a process.

        The first reading seeds and reports nothing, so a session that opens
        on four running jobs gets four rows and no notices — it never saw
        them start. A registry that is empty also seeds, which is why the
        state is a flag rather than an empty previous reading.
        """
        started: list[Job] = []
        finished: list[Job] = []
        if self._tracking:
            for job in jobs:
                was = self._last.get(job.id)
                status = job.status()
                if was is None:
                    started.append(job)
                elif was in LIVE_STATUSES and status not in LIVE_STATUSES:
                    finished.append(job)
        self._snapshot = jobs
        self._last = {job.id: job.status() for job in jobs}
        self._tracking = True
        return started, finished
