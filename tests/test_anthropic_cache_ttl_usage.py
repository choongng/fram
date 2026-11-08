"""The TTL each Anthropic request used, and what it leaves unchanged.

The TTL a request asked for is recorded on its usage beside the cache read and
write counts, so a hit or a miss can be read against the TTL that was used
(Anthropic's cache diagnostics: "Reading diagnostics alongside usage").
"""

import json
from typing import Any

import httpx
import pytest

from tau_agent import UserMessage
from tau_agent.messages import AssistantMessage
from tau_ai import AnthropicConfig, AnthropicProvider, AssistantDoneEvent
from tau_ai.anthropic import CACHE_TTL_LONG
from tau_ai.env import CACHE_RETENTION_NONE

_MODEL = "claude-test"
_SYSTEM = "You are an AI assistant analyzing a large document. <document>...</document>"


def _sse(usage: dict[str, int]) -> str:
    start = {"type": "message_start", "message": {"usage": usage}}
    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": usage.get("output_tokens", 1)},
    }
    return (
        f"data: {json.dumps(start)}\n\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"ok"}}\n\n'
        f"data: {json.dumps(delta)}\n\n"
        'data: {"type":"message_stop"}\n\n'
    )


def _provider(
    captured: list[dict[str, Any]], usage: dict[str, int] | None = None, **config: Any
) -> AnthropicProvider:
    body = _sse(usage or {"input_tokens": 1, "output_tokens": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

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
        system=_SYSTEM,
        messages=[UserMessage(content="Summarize section 1.")],
        tools=[],
        **kwargs,
    ):
        if isinstance(event, AssistantDoneEvent):
            done = event.message
    assert done is not None
    return done


def _cache_controls(value: Any) -> list[Any]:
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


def _strip_cache_controls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache_controls(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_controls(v) for v in value]
    return value


@pytest.mark.anyio
async def test_cache_ttl_usage_default_turn_sends_plain_ephemeral() -> None:
    captured: list[dict[str, Any]] = []
    await _request(_provider(captured))

    markers = _cache_controls(captured[0])
    assert markers
    assert all(marker == {"type": "ephemeral"} for marker in markers)


@pytest.mark.anyio
async def test_cache_ttl_usage_cache_write_usage_is_recorded_with_one_hour_ttl() -> None:
    captured: list[dict[str, Any]] = []
    usage = {
        "input_tokens": 42,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 41850,
        "output_tokens": 210,
    }
    message = await _request(_provider(captured, usage), long_lived=True)

    assert message.usage.input == 42
    assert message.usage.cache_read == 0
    assert message.usage.cache_write == 41850
    assert message.usage.output == 210
    assert message.usage.cache_ttl == CACHE_TTL_LONG


@pytest.mark.anyio
async def test_cache_ttl_usage_cache_hit_usage_is_recorded_with_default_ttl() -> None:
    captured: list[dict[str, Any]] = []
    usage = {
        "input_tokens": 42,
        "cache_read_input_tokens": 41850,
        "cache_creation_input_tokens": 0,
        "output_tokens": 210,
    }
    message = await _request(_provider(captured, usage), long_lived=False)

    assert message.usage.cache_read == 41850
    assert message.usage.cache_write == 0
    assert message.usage.cache_ttl is not None
    assert message.usage.cache_ttl != CACHE_TTL_LONG
    assert all(marker == {"type": "ephemeral"} for marker in _cache_controls(captured[0]))


@pytest.mark.anyio
async def test_cache_ttl_usage_ttl_choice_changes_only_cache_control() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(captured)

    await _request(provider, long_lived=False)
    await _request(provider, long_lived=True)

    assert _cache_controls(captured[0])
    assert _cache_controls(captured[1])
    assert _strip_cache_controls(captured[0]) == _strip_cache_controls(captured[1])


@pytest.mark.anyio
async def test_cache_ttl_usage_late_request_marks_every_breakpoint_one_hour() -> None:
    captured: list[dict[str, Any]] = []
    await _request(_provider(captured), long_lived=True)

    markers = _cache_controls(captured[0])
    assert markers
    assert all(marker.get("ttl") == CACHE_TTL_LONG for marker in markers)
    assert all(marker.get("type") == "ephemeral" for marker in markers)


@pytest.mark.anyio
async def test_cache_ttl_usage_disabled_caching_sends_no_cache_control_when_not_late() -> None:
    captured: list[dict[str, Any]] = []
    await _request(_provider(captured, cache_retention=CACHE_RETENTION_NONE), long_lived=False)

    assert _cache_controls(captured[0]) == []
