"""Provider transport retries surface as transient status, not silence."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pi_event_helpers import assistant_done, assistant_start, text_delta
from tau_agent import AssistantMessage, ProviderRetryEvent, UserMessage
from tau_agent.loop import run_agent_loop
from tau_agent.provider_events import AssistantRetryEvent
from tau_ai import FakeProvider
from tau_ai._provider_events import ProviderRetryEvent as TransportRetryEvent
from tau_ai.stream import canonicalize_provider_stream
from tau_coding.rendering.plain import FinalTextRenderer
from tau_coding.tui import TuiEventAdapter, TuiState


def _transport_retry(attempt: int = 0) -> TransportRetryEvent:
    return TransportRetryEvent(
        attempt=attempt + 1,
        max_attempts=5,
        delay_seconds=0.25,
        message=f"Retrying provider request {attempt + 2}/5 after network error in 0.25s.",
    )


async def _source(*events: object) -> AsyncIterator[object]:
    for event in events:
        yield event


@pytest.mark.anyio
async def test_canonicalize_surfaces_transport_retries_as_stream_events() -> None:
    stream = canonicalize_provider_stream(
        _source(_transport_retry()), api="openai-completions", provider="llama", model="m"
    )
    events = [event async for event in stream]
    assert [event for event in events if isinstance(event, AssistantRetryEvent)] == [
        AssistantRetryEvent(
            attempt=1,
            max_attempts=5,
            delay_seconds=0.25,
            message="Retrying provider request 2/5 after network error in 0.25s.",
        )
    ]


@pytest.mark.anyio
async def test_agent_loop_emits_provider_retry_status_events() -> None:
    provider = FakeProvider(
        [
            [
                AssistantRetryEvent(
                    attempt=1,
                    max_attempts=5,
                    delay_seconds=0.25,
                    message="Retrying provider request 2/5 after the server sent nothing for 60 s.",
                ),
                assistant_start(),
                text_delta("Hi"),
                assistant_done(AssistantMessage(content="Hi", model="fake")),
            ]
        ]
    )
    events = [
        event
        async for event in run_agent_loop(
            provider=provider,
            model="fake",
            system="You are a test.",
            messages=[UserMessage(content="hello")],
            tools=[],
        )
    ]
    retries = [event for event in events if isinstance(event, ProviderRetryEvent)]
    assert [
        (event.attempt, event.max_attempts, event.delay_seconds, event.message) for event in retries
    ] == [(1, 5, 0.25, "Retrying provider request 2/5 after the server sent nothing for 60 s.")]
    # The run still completes normally around the status events.
    assert any(
        getattr(event, "message", None) is not None
        and getattr(getattr(event, "message", None), "text", None) == "Hi"
        for event in events
    )


def test_final_text_renderer_logs_retries_to_the_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = FinalTextRenderer()
    renderer.render(
        ProviderRetryEvent(attempt=1, max_attempts=5, delay_seconds=0.25, message="retrying")
    )
    assert capsys.readouterr().err == "… retrying\n"


def test_tui_adapter_shows_retries_as_status() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ProviderRetryEvent(attempt=1, max_attempts=5, delay_seconds=0.25, message="retrying")
    )
    assert [(item.role, item.text) for item in state.items] == [("status", "… retrying")]
