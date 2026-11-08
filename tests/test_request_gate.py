import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage, UserMessage
from tau_agent.messages import AgentMessage
from tau_agent.provider import CancellationToken
from tau_agent.provider_events import AssistantMessageEvent
from tau_agent.tools import AgentTool
from tau_ai import OpenAICompatibleProvider
from tau_coding.credentials import FileCredentialStore
from tau_coding.provider_config import OpenAICompatibleProviderConfig, ProviderConfigError
from tau_coding.provider_runtime import create_model_provider
from tau_coding.request_gate import (
    ConcurrencyLimitedProvider,
    RequestQueued,
    observe_queued_requests,
)


class HeldProvider:
    """Provider whose responses finish only when the test lets them."""

    def __init__(self) -> None:
        self.started: list[str | None] = []
        self.finish = asyncio.Event()
        self.closed = False

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
        async def iterator() -> AsyncIterator[AssistantMessageEvent]:
            self.started.append(session_id)
            yield assistant_start()
            await self.finish.wait()
            yield assistant_done(message=AssistantMessage(content="ok"))

        return iterator()

    async def aclose(self) -> None:
        self.closed = True


def _gated(inner: HeldProvider, name: str = "llama", limit: int = 1) -> ConcurrencyLimitedProvider:
    return ConcurrencyLimitedProvider(inner, provider_name=name, limit=limit)


def _stream(
    provider: ConcurrencyLimitedProvider, session_id: str
) -> AsyncIterator[AssistantMessageEvent]:
    return provider.stream_response(
        model="m",
        system="s",
        messages=[UserMessage(content="hi")],
        tools=[],
        session_id=session_id,
    )


async def _drain(stream: AsyncIterator[AssistantMessageEvent]) -> list[AssistantMessageEvent]:
    return [event async for event in stream]


@pytest.fixture
def queued() -> Iterator[list[RequestQueued]]:
    notices: list[RequestQueued] = []
    unsubscribe = observe_queued_requests(notices.append)
    yield notices
    unsubscribe()


@pytest.mark.anyio
async def test_second_request_waits_for_the_first_and_is_reported(
    queued: list[RequestQueued],
) -> None:
    inner = HeldProvider()
    # Two provider objects for the same provider name share one queue, like a
    # session and a subagent that each built their own provider.
    first_provider, second_provider = _gated(inner), _gated(inner)

    first = asyncio.create_task(_drain(_stream(first_provider, "main")))
    await asyncio.sleep(0)
    second = asyncio.create_task(_drain(_stream(second_provider, "subagent")))
    for _ in range(5):
        await asyncio.sleep(0)

    assert inner.started == ["main"]
    assert queued == [RequestQueued(provider_name="llama", session_id="subagent", limit=1, ahead=1)]

    inner.finish.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert inner.started == ["main", "subagent"]


@pytest.mark.anyio
async def test_requests_within_the_limit_or_on_other_providers_do_not_wait(
    queued: list[RequestQueued],
) -> None:
    inner = HeldProvider()
    streams = [
        _stream(_gated(inner, limit=2), "a"),
        _stream(_gated(inner, limit=2), "b"),
        _stream(_gated(inner, name="other", limit=2), "c"),
    ]
    tasks = [asyncio.create_task(_drain(stream)) for stream in streams]
    for _ in range(5):
        await asyncio.sleep(0)

    assert inner.started == ["a", "b", "c"]
    assert queued == []
    inner.finish.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)


@pytest.mark.anyio
async def test_slot_is_released_at_the_done_event_and_on_early_close() -> None:
    inner = HeldProvider()
    inner.finish.set()
    provider = _gated(inner)

    # A consumer that stops reading at the done event must not keep the slot.
    stream = _stream(provider, "a")
    async for event in stream:
        if event.type == "done":
            break
    await asyncio.wait_for(_drain(_stream(provider, "b")), timeout=5)

    # Nor must one that closes the stream after the first event.
    inner.finish.clear()
    early = _stream(provider, "c")
    await anext(early)
    await early.aclose()  # type: ignore[attr-defined]
    inner.finish.set()
    await asyncio.wait_for(_drain(_stream(provider, "d")), timeout=5)
    assert inner.started == ["a", "b", "c", "d"]


@pytest.mark.anyio
async def test_cancelled_waiting_request_does_not_take_a_slot(
    queued: list[RequestQueued],
) -> None:
    inner = HeldProvider()
    provider = _gated(inner)
    first = asyncio.create_task(_drain(_stream(provider, "a")))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(_drain(_stream(provider, "b")))
    for _ in range(3):
        await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    inner.finish.set()
    await asyncio.wait_for(first, timeout=5)
    await asyncio.wait_for(_drain(_stream(provider, "c")), timeout=5)
    assert inner.started == ["a", "c"]


@pytest.mark.anyio
async def test_observer_failures_do_not_block_the_request() -> None:
    def broken(notice: RequestQueued) -> None:
        raise RuntimeError("observer bug")

    unsubscribe = observe_queued_requests(broken)
    try:
        inner = HeldProvider()
        provider = _gated(inner)
        first = asyncio.create_task(_drain(_stream(provider, "a")))
        await asyncio.sleep(0)
        second = asyncio.create_task(_drain(_stream(provider, "b")))
        await asyncio.sleep(0)
        inner.finish.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    finally:
        unsubscribe()


@pytest.mark.anyio
async def test_wrapper_forwards_close_and_attributes() -> None:
    inner = HeldProvider()
    provider = _gated(inner)

    assert provider.finish is inner.finish
    await provider.aclose()
    assert inner.closed


def test_create_model_provider_gates_only_when_max_concurrent_requests_is_set(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAU_TEST_LOCAL_API_KEY", "key")
    store = FileCredentialStore(tmp_path / "credentials.json")
    config = OpenAICompatibleProviderConfig(
        name="local",
        api_key_env="TAU_TEST_LOCAL_API_KEY",
        models=("qwen",),
        default_model="qwen",
    )

    plain = create_model_provider(config, credential_store=store)
    gated = create_model_provider(
        OpenAICompatibleProviderConfig(
            name="local",
            api_key_env="TAU_TEST_LOCAL_API_KEY",
            models=("qwen",),
            default_model="qwen",
            max_concurrent_requests=1,
        ),
        credential_store=store,
    )

    assert isinstance(plain, OpenAICompatibleProvider)
    assert isinstance(gated, ConcurrencyLimitedProvider)


def test_max_concurrent_requests_must_be_positive() -> None:
    with pytest.raises(ProviderConfigError):
        OpenAICompatibleProviderConfig(name="local", max_concurrent_requests=0)


@pytest.mark.anyio
async def test_slot_is_released_when_the_stream_fails_midway() -> None:
    class FailingProvider(HeldProvider):
        def stream_response(self, **kwargs: object) -> AsyncIterator[AssistantMessageEvent]:  # type: ignore[override]
            async def iterator() -> AsyncIterator[AssistantMessageEvent]:
                yield assistant_start()
                raise RuntimeError("connection lost")

            return iterator()

    with pytest.raises(RuntimeError, match="connection lost"):
        await _drain(_stream(_gated(FailingProvider()), "first"))

    inner = HeldProvider()
    inner.finish.set()
    await asyncio.wait_for(_drain(_stream(_gated(inner), "second")), timeout=1)
    assert inner.started == ["second"]
