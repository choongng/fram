"""Stall timer for streamed responses.

Total response time is not limited. A response is ended when the server has
stopped sending: no stream event for
`TAU_STREAM_STALL_SECONDS` (default: `STREAM_STALL_START_SECONDS`, doubling
per retry up to `STREAM_STALL_CAP_SECONDS`). A stall
is raised as `StreamStallError`, a read timeout, so a provider's transport
retry handles it like any other dropped connection: retried when no content
has arrived yet, an error once some has. A slow but steady stream resets the
timer with every event and is never stopped by it.

The wait ramps with each retry because the first request after an idle can
land on a model server still loading its weights: that request may neither
error nor complete for as long as the load takes, and only a later one goes
through. The timer starts low (60 s ends a stuck server's first request
quickly) and doubles per attempt to the cap, so the retries outlast a cold
load without a fixed timeout sized for one server's load time.
`TAU_STREAM_STALL_SECONDS` overrides the ramp with one fixed value;
`"0"` or `"off"` turns the timer off.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from os import environ

import anyio
import httpx

STREAM_STALL_SECONDS_ENV = "TAU_STREAM_STALL_SECONDS"
#: The stall wait for a first attempt: seconds with no stream event before a
#: response is called stalled. It has to outlast the silence before a first
#: token: a local server sends nothing while it prefills a long prompt. 60 s
#: leaves room for that and still ends a stuck server's first request quickly.
STREAM_STALL_START_SECONDS = 60.0

#: The ramp's cap. A cold model server can spend many minutes loading weights
#: before its first token, so the last retries wait up to ten minutes.
STREAM_STALL_CAP_SECONDS = 600.0


class StreamStallError(httpx.ReadTimeout):
    """The server sent nothing for `seconds`."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"the server sent nothing for {seconds:g} s")
        self.seconds = seconds


def ramped_stream_stall_seconds(attempt: int) -> float:
    """The stall wait for `attempt`: start doubled per attempt, capped."""
    return float(min(STREAM_STALL_CAP_SECONDS, STREAM_STALL_START_SECONDS * (2 ** max(attempt, 0))))


def stream_stall_seconds(attempt: int = 0, *, floor_seconds: float = 0.0) -> float | None:
    """The stall timer's seconds for `attempt`, or None when it is off.

    The environment's value, when set, applies as-is to every attempt; the
    default ramps per attempt from `STREAM_STALL_START_SECONDS` to
    `STREAM_STALL_CAP_SECONDS`, and never waits less than `floor_seconds`, so
    a longer configured request timeout is not undercut.
    """
    raw = environ.get(STREAM_STALL_SECONDS_ENV, "").strip().lower()
    if raw == "off":
        return None
    if raw:
        try:
            seconds = float(raw)
        except ValueError:
            seconds = None
        if seconds is not None:
            return seconds if seconds > 0 else None
    return max(ramped_stream_stall_seconds(attempt), floor_seconds)


async def stall_guarded[T](source: AsyncGenerator[T], seconds: float | None) -> AsyncGenerator[T]:
    """Pass `source`'s items on, raising `StreamStallError` when none arrives
    for `seconds`. The timer covers only the wait for the next item, so a
    consumer that is slow to take one is never counted against the server."""
    async with aclosing(source):
        while True:
            try:
                if seconds is None:
                    item = await anext(source)
                else:
                    with anyio.move_on_after(seconds) as scope:
                        item = await anext(source)
                    if scope.cancelled_caught:
                        raise StreamStallError(seconds)
            except StopAsyncIteration:
                return
            yield item
