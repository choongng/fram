"""Anthropic prompt-cache TTL fits how long the conversation lives.

The default ephemeral breakpoint lives five minutes; a long-lived conversation
(an interactive main session) asks for the 1-hour TTL on every request, so its
idle gaps do not expire the cache. The TTL each request used is recorded on its
usage. Requests go through a fake httpx transport that captures the payload.
"""

import json
from typing import Any

import httpx
import pytest

from anthropic_transport_helpers import OK_SSE
from tau_agent import UserMessage
from tau_agent.cache_hints import long_lived_conversation
from tau_agent.messages import AssistantMessage, current_timestamp_ms
from tau_ai import AnthropicConfig, AnthropicProvider, AssistantDoneEvent
from tau_ai.env import CACHE_RETENTION_NONE

_MODEL = "claude-test"


def _provider(captured: list[dict[str, Any]], **config: Any) -> AnthropicProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, text=OK_SSE, headers={"content-type": "text/event-stream"})

    return AnthropicProvider(
        AnthropicConfig(
            api_key="test", base_url="https://example.test/v1", max_retries=0, **config
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _request(provider: AnthropicProvider, **kwargs: Any) -> AssistantMessage:
    done: AssistantMessage | None = None
    async for event in provider.stream_response(
        model=_MODEL,
        system="You are an AI assistant analyzing a large document.",
        messages=[UserMessage(content="Summarize section 1.")],
        tools=[],
        **kwargs,
    ):
        if isinstance(event, AssistantDoneEvent):
            done = event.message
    assert done is not None
    return done


def _cache_controls(value: Any) -> list[Any]:
    """Every cache_control marker anywhere in a request payload."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "cache_control":
                found.append(item)
            else:
                found.extend(_cache_controls(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_cache_controls(item))
    return found


@pytest.mark.anyio
async def test_cache_ttl_default_request_sends_plain_ephemeral() -> None:
    captured: list[dict[str, Any]] = []
    await _request(_provider(captured))

    markers = _cache_controls(captured[0])
    assert markers
    assert all(marker == {"type": "ephemeral"} for marker in markers)


@pytest.mark.anyio
async def test_cache_ttl_long_lived_conversation_uses_one_hour() -> None:
    captured: list[dict[str, Any]] = []
    await _request(_provider(captured), long_lived=True)

    markers = _cache_controls(captured[0])
    assert markers
    assert all(marker == {"type": "ephemeral", "ttl": "1h"} for marker in markers)


@pytest.mark.anyio
async def test_cache_ttl_long_lived_request_records_one_hour_on_usage() -> None:
    captured: list[dict[str, Any]] = []
    message = await _request(_provider(captured), long_lived=True)

    assert message.usage.cache_ttl == "1h"


@pytest.mark.anyio
async def test_cache_ttl_default_request_records_short_ttl_on_usage() -> None:
    captured: list[dict[str, Any]] = []
    message = await _request(_provider(captured))

    assert message.usage.cache_ttl is not None
    assert message.usage.cache_ttl != "1h"


@pytest.mark.anyio
async def test_cache_ttl_disabled_caching_sends_no_cache_control_when_long_lived() -> None:
    captured: list[dict[str, Any]] = []
    await _request(_provider(captured, cache_retention=CACHE_RETENTION_NONE), long_lived=True)

    assert _cache_controls(captured[0]) == []


async def _request_after(
    provider: AnthropicProvider, earlier: AssistantMessage, **kwargs: Any
) -> None:
    messages = [UserMessage(content="Summarize section 1."), earlier, UserMessage(content="Next.")]
    async for _ in provider.stream_response(
        model=_MODEL, system="s", messages=messages, tools=[], **kwargs
    ):
        pass


def _earlier_turn(
    *, minutes_ago: float, cache_ttl: str | None = "5m", model: str = _MODEL
) -> AssistantMessage:
    earlier = AssistantMessage(
        content="Section 1 says...",
        api="anthropic-messages",
        provider="anthropic",
        model=model,
        timestamp=current_timestamp_ms() - int(minutes_ago * 60_000),
    )
    earlier.usage.cache_ttl = cache_ttl
    return earlier


# A 1h request reads only a prefix written at 1h, so a conversation that used
# 1h keeps it: switching back and forth would rewrite the prompt each time.
@pytest.mark.anyio
async def test_cache_ttl_keeps_one_hour_recorded_in_history() -> None:
    captured: list[dict[str, Any]] = []
    await _request_after(
        _provider(captured), _earlier_turn(minutes_ago=0.1, cache_ttl="1h"), long_lived=False
    )

    assert all(
        marker == {"type": "ephemeral", "ttl": "1h"} for marker in _cache_controls(captured[0])
    )


@pytest.mark.anyio
async def test_cache_ttl_does_not_inherit_one_hour_from_another_model() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(captured)

    await _request_after(
        provider, _earlier_turn(minutes_ago=1, cache_ttl="1h", model="another-model")
    )

    assert all("ttl" not in control for control in _cache_controls(captured[0]))


@pytest.mark.anyio
async def test_cache_ttl_stays_short_while_the_conversation_is_warm() -> None:
    captured: list[dict[str, Any]] = []
    await _request_after(_provider(captured), _earlier_turn(minutes_ago=1), long_lived=False)

    assert all(marker == {"type": "ephemeral"} for marker in _cache_controls(captured[0]))


# Idle past a 5-minute entry's life (waiting on background work), the short
# cache is gone and this request rewrites the prompt anyway: it writes at 1h,
# so the next wait hits.
@pytest.mark.anyio
async def test_cache_ttl_moves_to_one_hour_after_an_idle_gap() -> None:
    captured: list[dict[str, Any]] = []
    await _request_after(_provider(captured), _earlier_turn(minutes_ago=6), long_lived=False)

    assert all(
        marker == {"type": "ephemeral", "ttl": "1h"} for marker in _cache_controls(captured[0])
    )


@pytest.mark.anyio
async def test_cache_ttl_never_asks_for_one_hour_where_the_model_cannot_keep_it() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(captured, supports_long_cache_retention=False)
    await _request(provider, long_lived=True)
    await _request_after(provider, _earlier_turn(minutes_ago=6, cache_ttl="1h"))

    for payload in captured:
        assert all(marker == {"type": "ephemeral"} for marker in _cache_controls(payload))


@pytest.mark.anyio
async def test_cache_ttl_reads_long_lived_hint_from_the_agent_loop() -> None:
    captured: list[dict[str, Any]] = []
    with long_lived_conversation(True):
        await _request(_provider(captured))
    await _request(_provider(captured))

    assert all(marker.get("ttl") == "1h" for marker in _cache_controls(captured[0]))
    assert all(marker == {"type": "ephemeral"} for marker in _cache_controls(captured[1]))
