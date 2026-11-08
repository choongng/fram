"""Provider-neutral hint that the conversation making a request is long-lived.

Anthropic's default prompt-cache breakpoint lives five minutes; a 1-hour TTL
costs more to write. An interactive session waits on its user for longer than
five minutes between requests, so the long cache pays for it. A worker (a
sub-agent or a print-mode run) makes its requests seconds apart and ends, so
the short cache is cheaper.

A conversation must not ask for 1 hour while a 5-minute prefix is still live:
Anthropic serves a 5-minute request from a prefix written at either TTL, but a
1-hour request only from a prefix written at 1 hour, so one 1-hour request
after a run of 5-minute ones rewrites everything they added. A long-lived
conversation therefore asks for 1 hour from its first request, and the loop
records that here for each request; a provider that offers a longer cache
reads it, and others ignore it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar

_LONG_LIVED: ContextVar[bool] = ContextVar("tau_long_lived_conversation", default=False)


def conversation_long_lived() -> bool:
    """Return whether the request being made now belongs to a long-lived conversation."""
    return _LONG_LIVED.get()


@contextmanager
def long_lived_conversation(long_lived: bool) -> Iterator[None]:
    """Mark requests made inside the block as from a long-lived conversation (or not)."""
    token = _LONG_LIVED.set(long_lived)
    try:
        yield
    finally:
        # An async generator closed from another context (aclose at GC) cannot
        # reset a token made in its own; that context is gone anyway.
        with suppress(ValueError):
            _LONG_LIVED.reset(token)
