"""The application's choice, per request, of whether a looping reply is cut.

A provider that reads a tool call's arguments as they stream can end a reply
once they repeat one block of text. The cut is a heuristic, so the application
decides whether it runs. The choice belongs to one request: two conversations
can share a provider and choose differently, and a request can wait in a queue
between being made and being sent, so it is not kept on the provider. The
caller marks the request where it reads the stream, and the provider asks here
when it starts to parse the response.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar

_CUT_LOOPING_TOOL_ARGUMENTS: ContextVar[bool] = ContextVar(
    "tau_cut_looping_tool_arguments", default=True
)


def looping_tool_arguments_cut() -> bool:
    """Return whether the response being read now is cut when its tool-call
    arguments loop. True unless the caller chose otherwise."""
    return _CUT_LOOPING_TOOL_ARGUMENTS.get()


@contextmanager
def cutting_looping_tool_arguments(enabled: bool) -> Iterator[None]:
    """Cut (or leave alone) looping tool-call arguments in responses read
    inside the block."""
    token = _CUT_LOOPING_TOOL_ARGUMENTS.set(enabled)
    try:
        yield
    finally:
        # An async generator closed from another context (aclose at GC) cannot
        # reset a token made in its own; that context is gone anyway.
        with suppress(ValueError):
            _CUT_LOOPING_TOOL_ARGUMENTS.reset(token)
