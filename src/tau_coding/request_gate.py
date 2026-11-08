"""Queue same-process model requests for providers that serve one at a time.

A local inference server started with one slot (llama.cpp ``--parallel 1``)
processes one request at a time. Two requests from the same Tau process, such as
the main agent and a background subagent, then take turns in the slot, and each
evicts the other's cached prompt prefix, so both reprocess their whole context
on every turn. With ``max_concurrent_requests`` set on the provider, Tau holds
the extra request locally until the first finishes instead of sending both.

The limit is per provider name and per process. It cannot see requests from
other processes or other clients of the same server.
"""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop, get_running_loop
from collections.abc import AsyncIterator, Callable, MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from tau_agent.messages import AgentMessage
from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
)
from tau_agent.tools import AgentTool


@dataclass(frozen=True, slots=True)
class RequestQueued:
    """A request is waiting for a free slot on a concurrency-limited provider."""

    provider_name: str
    session_id: str | None
    limit: int
    ahead: int
    """Requests in flight or already waiting when this one arrived."""


type RequestQueuedObserver = Callable[[RequestQueued], None]

_OBSERVERS: list[RequestQueuedObserver] = []


def observe_queued_requests(observer: RequestQueuedObserver) -> Callable[[], None]:
    """Call ``observer`` whenever a gated request has to wait; return an unsubscribe.

    Observers are process-wide because providers are created in several places
    (sessions, subagents, discovery) that do not share a host. An observer
    filters by ``session_id`` to report only its own requests.
    """
    _OBSERVERS.append(observer)

    def unsubscribe() -> None:
        if observer in _OBSERVERS:
            _OBSERVERS.remove(observer)

    return unsubscribe


class _Gate:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.waiting = 0


_GATES: MutableMapping[AbstractEventLoop, dict[str, _Gate]] = WeakKeyDictionary()


def _gate(provider_name: str, limit: int) -> _Gate:
    """Return this loop's gate for a provider, replacing it if the limit changed.

    Gates are cached per event loop because ``asyncio.Semaphore`` binds to the
    running loop on first contention. Requests already holding a replaced gate
    finish under it; new requests use the new limit.
    """
    gates = _GATES.setdefault(get_running_loop(), {})
    gate = gates.get(provider_name)
    if gate is None or gate.limit != limit:
        gate = _Gate(limit)
        gates[provider_name] = gate
    return gate


class ConcurrencyLimitedProvider:
    """Model provider wrapper that queues requests beyond a per-provider limit.

    A request takes its slot when its stream is first iterated and gives it
    back as soon as the terminal done or error event arrives, so a consumer
    that stops reading at that event does not keep the slot. A stream that
    fails, is cancelled, or is closed early releases it too. Every other
    attribute is forwarded to the wrapped provider so capability checks such as
    model discovery keep working.
    """

    def __init__(self, inner: ModelProvider, *, provider_name: str, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        self._inner = inner
        self._provider_name = provider_name
        self._limit = limit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream one response once a slot is free, reporting when it had to wait."""

        async def iterator() -> AsyncIterator[AssistantMessageEvent]:
            gate = _gate(self._provider_name, self._limit)
            if gate.semaphore.locked():
                _notify(
                    RequestQueued(
                        provider_name=self._provider_name,
                        session_id=session_id,
                        limit=self._limit,
                        ahead=gate.active + gate.waiting,
                    )
                )
            gate.waiting += 1
            try:
                await gate.semaphore.acquire()
            finally:
                gate.waiting -= 1
            gate.active += 1
            held = True

            def release() -> None:
                nonlocal held
                if held:
                    held = False
                    gate.active -= 1
                    gate.semaphore.release()

            stream = self._inner.stream_response(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                signal=signal,
                session_id=session_id,
            )
            try:
                async for event in stream:
                    if isinstance(event, AssistantDoneEvent | AssistantErrorEvent):
                        release()
                    yield event
            finally:
                release()
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()

        return iterator()


def _notify(notice: RequestQueued) -> None:
    for observer in tuple(_OBSERVERS):
        # A reporting failure must never block or fail the request.
        with suppress(Exception):
            observer(notice)
